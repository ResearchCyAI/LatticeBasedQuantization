"""
Simple MLP for binary classification using Coset core APIs with SCALAR quantization.
This version uses scalar quantization (lsq_scalar_quantize) instead of E8 lattice quantization.
Each element in 8-D blocks is quantized independently using LSQ scalar quantization.
"""
'''
Scalar quantization on 8-D blocks: each element is quantized independently.
Uses 8x8 identity generator matrix (no transformation, each dimension independent).
Each row is tiled into 8-D blocks (768→96 tiles for input_size=768).
Scale those blocks by row-wise beta, then quantize each element independently.
'''
import os
import pandas as pd
import numpy as np
import torch
import copy
import math
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import time
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import gc
import json
os.environ["TRANSFORMERS_NO_FLASH_ATTENTION"] = "1"
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:256"
from typing import Optional, Tuple
# Import coset core modules
import coset
try:
    from coset.core.layers import HNLQLinearQAT, ste_quantize
    from coset.core.scalar.codecs import lsq_scalar_quantize
    print("✓ Successfully imported coset core modules (SCALAR QUANTIZATION version)")
except ImportError as e:
    print(f"✗ Failed to import coset core modules: {e}")
    print("Make sure you have installed coset with: pip install -e .")
    exit(1)
# Check CUDA availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
import random
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# --- Scalar quantization parameters ---
q_taken=2
M_taken=2
weight_clip_value=2.0

def compute_bits(q: int, M: int) -> int:
    """Compute effective bits from q and M: bits = floor(M * log2(q))"""
    return int(math.floor(M * math.log2(q)))

def compute_bits_actual(q: int, M: int) -> float:
    """Compute actual bits value from q and M: bits = M * log2(q) (not floored)"""
    return M * math.log2(q)

def get_q_m_combinations(target_bits: list) -> list:
    """
    Generate all (q, M) combinations that achieve target bit widths.
    Args:
        target_bits: List of target bit widths (e.g., [1, 2, 3, 4, 8])
    Returns:
        List of tuples (bits, q, M) sorted by bits, then q, then M
    """
    combinations = []
    # q values: 2, 3, 4, 5, 6, 7, 8
    q_values = [2, 3, 4, 5, 6, 7, 8]
    # M values: 1, 2, 3, 4
    M_values = [1, 2, 3, 4]
    for q in q_values:
        for M in M_values:
            bits = compute_bits(q, M)
            bits_actual = compute_bits_actual(q, M)
            if bits in target_bits:
                combinations.append((bits, q, M, bits_actual))
    # Remove duplicates and sort
    combinations = sorted(set(combinations), key=lambda x: (x[0], x[1], x[2]))
    return combinations

def create_scalar_lattice(device: Optional[torch.device] = None, block_dim: int = 8):
    """
    Create a minimal lattice-like object for scalar quantization with 8x8 generator matrix.
    HNLQLinearQAT requires a lattice object with name, compute_delta0, and get_generators.
    We create a simple object with these attributes/methods without inheriting from Lattice.
    
    Args:
        device: Device to place tensors on
        block_dim: Dimension of the block (default 8 for 8x8 generator matrix)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Simple object with required attributes and methods
    class ScalarLatticeObj:
        def __init__(self):
            self.name = "Scalar"
            self.d = block_dim  # 8-D blocks
            self.device = device
            self.r_pack = 0.5  # Packing radius (approximate for scalar quantization)
        
        def compute_delta0(self, q: int, M: int, rho: float = 0.95) -> float:
            """Compute Delta0 for scalar quantization."""
            return 1.0  # Simplified for scalar quantization
        
        def get_generators(self):
            """Return 8x8 identity generator matrices (each dimension independent)."""
            G = torch.eye(block_dim, device=self.device)
            return G, G
    
    return ScalarLatticeObj()

class MeanCenter(nn.Module):
    """
    Subtracts the batch mean per feature during training.
    Uses an EMA running mean at eval time. No scaling, no affine.
    """
    def __init__(self, momentum: float = 0.1, eps: float = 1e-5):
        super().__init__()
        self.momentum = momentum
        self.eps = eps
        self.register_buffer("running_mean", None)   # [D]
        self.last_mu = None                          # for logging/regularization

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, D]
        if self.running_mean is None:
            # lazily initialize buffer shape
            with torch.no_grad():
                rm = x.mean(dim=0)                  # [D]
            self.running_mean = rm.detach().clone()

        if self.training:
            mu = x.mean(dim=0, keepdim=True)        # [1, D]
            # update running mean (no grad)
            with torch.no_grad():
                self.running_mean.mul_(1 - self.momentum).add_(self.momentum * mu.squeeze(0))
            self.last_mu = mu.detach()              # [1, D]
            return x - mu                           # zero-center per feature
        else:
            mu = self.running_mean.view(1, -1)
            return x - mu


def make_scalar_quantize_fn_adapter(M: int = 2):
    """
    Returns a quantize_fn for scalar quantization using Euclidean distance.
    Uses 8x8 identity generator matrix (no transformation).
    Each element in the 8-D block is quantized independently using LSQ scalar quantization.
    
    Args:
        M: Hierarchical levels (effective bits = floor(M * log2(q)))
    """
    def _adapter(x: torch.Tensor, q: int) -> torch.Tensor:
        """
        Scalar quantization on 8-D blocks: quantize each element independently.
        Input x is [N, 8] where each 8-D vector is quantized element-wise.
        Uses the existing lsq_scalar_quantize function from coset which:
        - Computes effective bits: bits = floor(M * log2(q))
        - Codebook: integers from -q_max to q_max where q_max = 2^(bits-1) - 1
        - Euclidean distance: |x - codebook_value|, find nearest (round and clip) for each element
        - 8x8 identity generator matrix: no transformation applied, each dimension independent
        """
        # Use the existing LSQ scalar quantization function from coset
        # This properly handles quantization bounds based on q and M
        # For 8-D blocks, each element is quantized independently (scalar quantization)
        y = lsq_scalar_quantize(x, q=q, M=M)
        return y
    return _adapter
    
class QuantizedMLP(nn.Module):
    def __init__(
        self,
        input_size=768,
        hidden_size=512,
        output_size=1,
        q=4, M=2,
        warmup_epochs=5,                 # warmup epochs before enabling quantization
        use_warmup=True,
        tiling="row",                    # or "block"
        block_size=8,                    # scalar quantization on 8-D blocks
        quantize_activations=False,      # set True to enable LSQ-A
        act_bit_width=8,
        weight_clip_value=weight_clip_value,
        enable_diagnostics=False,
        Delta0=1.5
    ):
        super().__init__()
        self.use_warmup = use_warmup
        self.input_padding = (8 - (input_size % 8)) % 8
        eff_in = input_size + self.input_padding
        
        # NEW: add centering
        self.center = MeanCenter(momentum=0.1)#-----------------------
        
        # Create a minimal lattice-like object for scalar quantization (required by HNLQLinearQAT)
        scalar_lattice = create_scalar_lattice(device=device, block_dim=8)
        
        self.fc1 = GlobalScaledHNLQLinearQAT(
            in_features=input_size,
            out_features=hidden_size,
            lattice=scalar_lattice,  # Scalar lattice with 8x8 identity generator matrix
            quantize_fn=make_scalar_quantize_fn_adapter(M=M),  # Pass M to the adapter
            q=q, M=M,  # Use the parameters passed to QuantizedMLP, not global variables
            tiling=tiling,
            block_size=8,  # Scalar quantization on 8-D blocks (each element in block quantized independently)
            warmup_epochs=warmup_epochs,  # Use the parameter passed to QuantizedMLP
            enable_diagnostics=enable_diagnostics,
            weight_clip_value=weight_clip_value,
            quantize_activations=False,
            act_bit_width=act_bit_width,
            Delta0=1.5,
            a_init=1.0,
            b_init=0.0,
            init_method='normal',
            init_kwargs={'mean': 0.0, 'std': 1.0}  # Increased variance: std=0.5 for higher weight initialization variance
        )

        # Initialize fc3 with higher variance as well
        self.fc3 = nn.Linear(hidden_size, output_size)
        nn.init.normal_(self.fc3.weight, mean=0.0, std=1.0)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        if self.input_padding > 0:
            pad = torch.zeros(x.size(0), self.input_padding, device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=1)

        x = self.center(x)#--------------------------
        x = self.relu(self.fc1(x))
        x = self.fc3(x)
        return x
class GlobalScaledHNLQLinearQAT(HNLQLinearQAT):
    

    def __init__(self, *args, a_init: float = 1.0, b_init: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)

        # beta_global is fixed at 1.0 and not trainable
        self.register_buffer('beta_global', torch.tensor(1.0))
        # alpha_global is fixed at 0.0 and not trainable
        self.register_buffer('alpha_global', torch.tensor(0.0))

    
    def _quantize_weights(self):
        """
        Quantize weights using the new pipeline:
        1. Global affine scaling: W_s = β_global * W + α_g
        2. Row-wise scaling: W_r = β_r * W_s where β_r = Y_max/(C_b * sigma_r)
        3. Quantization: W_q = Q_scalar(W_r) - each element in 8-D blocks quantized independently
        4. Save quantized blocks for diagnostics
        5. Undo both scalings
        """
        W = self.weight  # [out_features, in_features]

        # ---- 0) Cold start: no quantization ----
        if not self.quantization_enabled:
            if self.enable_diagnostics:
                self._weight_history.append(W.detach().clone())
            return W

        # ---- 1) Global affine scaling: W_s = β_global * W + α_g ----
        beta_global = self.beta_global  # learnable parameter
        alpha_g = self.alpha_global      # learnable parameter
        W_s = beta_global * W + alpha_g  # [out_features, in_features]

        # ---- 2) Row-wise scaling: W_r = β_r * W_s ----
        # Compute row-wise standard deviation sigma_r
        sigma_r = W_s.std(dim=1, keepdim=True) + 1e-8  # [out_features, 1]
        
        #Delta0_base = 1.5
        #qM = float(self.q ** self.M)
        #Y_max = Delta0_base * (qM - 1) / 2.0
        # === SQ-specific: derive bits and scalar clip ===
        bits = int(math.floor(self.M * math.log2(self.q)))  # e.g. q=2, M=2 -> bits=2
        k_max = 2 ** (bits - 1) - 1                    # e.g. bits=2 -> k_max = 1

        # choose how many sigmas you want inside the clip
        #k_sigma = 3.0                                       # ~99.7% for Gaussian

        # Real-valued max before quantizer (what you want σ_r to map to)
        Y_max = float(k_max)    # with Δ=1 for SQ
        # Compute β_r = Y_max / (C_b * sigma_r) where C_b = 1
        C_b = 5.0
        beta_r = Y_max / (C_b * sigma_r)  # [out_features, 1]
        
        # Apply row-wise scaling
        W_r = beta_r * W_s  # [out_features, in_features]

        # ---- 3) Quantization: W_q = Q_scalar(W_r) ----
        # Tile into 8-D blocks
        W_blocks = W_r.view(self.out_features, self.blocks_per_row, self.block_size)
        W_blocks_flat = W_blocks.reshape(-1, self.block_size)  # [N_blocks, 8]

        # Scalar quantization on 8-D blocks: each element quantized independently using Euclidean distance
        # IMPORTANT: After scaling (beta_r), each of the 8 elements in a block is quantized independently
        # 8x8 identity generator matrix means no transformation, just round each element to nearest integer
        # This is NOT 8 scaling factors per block - it's 1 scaling factor per row, then 8 independent quantizations
        W_quantized_blocks = ste_quantize(W_blocks_flat, self.quantize_fn, self.q)

        # Reshape back
        W_quantized_blocks_reshaped = W_quantized_blocks.reshape(
            self.out_features, self.blocks_per_row, self.block_size
        )
        W_quantized = W_quantized_blocks_reshaped.reshape(
            self.out_features, self.in_features
        )

        # ---- 4) Save quantized blocks for overload checking ----
        self._last_W_aff_q_blocks = W_quantized_blocks.detach()   # [N_blocks, 8]
        self._last_W_aff_q = W_quantized.detach()                 # [out_features, in_features]

        # ---- 5) Undo both scalings ----
        # Undo row-wise scaling: W_q / β_r
        W_quant_undo_row = W_quantized / beta_r  # [out_features, in_features]

        # Undo global affine scaling: (W_q - α_g) / β_global
        W_out = (W_quant_undo_row - alpha_g) / (beta_global + 1e-8)

        # Apply weight clipping to stabilize training
        W_out = torch.clamp(W_out, -self.weight_clip_value, self.weight_clip_value)

        # Diagnostics: compare original vs final quantized
        if self.enable_diagnostics:
            self._weight_history.append(W.detach().clone())
            self._quantization_errors.append(torch.norm(W - W_out).item())

        return W_out


def load_embeddings_from_folder(emb_dir, batch_size=64):
    """
    Load train/val/test embeddings and labels from a single folder.
    Expects: DataSet/embeddings_<name>/ with
      train_emb.npy, val_emb.npy, test_emb.npy
      train_labels.npy, val_labels.npy, test_labels.npy
    (Same layout as produced by FP32EmbeddingsCreation.py)
    """
    train_emb = np.load(os.path.join(emb_dir, "train_emb.npy")).astype(np.float32)
    val_emb = np.load(os.path.join(emb_dir, "val_emb.npy")).astype(np.float32)
    test_emb = np.load(os.path.join(emb_dir, "test_emb.npy")).astype(np.float32)
    train_labels = np.load(os.path.join(emb_dir, "train_labels.npy"))
    val_labels = np.load(os.path.join(emb_dir, "val_labels.npy"))
    test_labels = np.load(os.path.join(emb_dir, "test_labels.npy"))

    def _make_loader(X, y, shuffle=False):
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)
        return DataLoader(
            TensorDataset(X_t, y_t),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=True,
        )

    train_loader = _make_loader(train_emb, train_labels, shuffle=True)
    val_loader = _make_loader(val_emb, val_labels, shuffle=False)
    test_loader = _make_loader(test_emb, test_labels, shuffle=False)
    return train_loader, val_loader, test_loader


def prepare_dataloaders(csv_path, batch_size=64, data_fraction=0.1):
    """
    Load dataloaders from DataSet/embeddings/ using standardized filenames
    (train_emb.npy, val_emb.npy, test_emb.npy, train_labels.npy, etc.).
    """
    code_dir = os.path.dirname(os.path.abspath(__file__))
    csv_stem = os.path.splitext(os.path.basename(csv_path))[0]
    emb_dir = os.path.join(code_dir, "DataSet", f"embeddings")
    if not os.path.isdir(emb_dir):
        raise FileNotFoundError(
            f"Embeddings folder not found: {emb_dir}. "
            "Run FP32EmbeddingsCreation.py on the CSV first to create embeddings."
        )
    dataset_name = csv_stem
    train_loader, val_loader, test_loader = load_embeddings_from_folder(emb_dir, batch_size=batch_size)
    return dataset_name, train_loader, val_loader, test_loader

@torch.no_grad()
def evaluate(model, data_loader, criterion, device, threshold=0.5):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []

    for data, target in data_loader:
        data, target = data.to(device), target.to(device)

        logits = model(data)
        loss = criterion(logits.view(-1), target.view(-1))
        total_loss += loss.item()

        probs = torch.sigmoid(logits.view(-1))
        preds = (probs >= threshold).float()

        all_preds.append(preds.detach().cpu())
        all_targets.append(target.detach().cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()

    # Compute all metrics
    avg_loss = total_loss / max(len(data_loader), 1)
    acc = accuracy_score(all_targets, all_preds) * 100
    prec = precision_score(all_targets, all_preds, zero_division=0)
    rec = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)

    return {
        "loss": avg_loss,
        "acc": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1
    }

def train_model(
    model,
    train_loader,
    val_loader,
    test_loader,
    epochs=50,
    lr=1e-3,
    threshold=0.5,
    patience=6,        # ← stop if no improvement for this many epochs
    min_delta=0.0,     # ← required improvement in val F1 to reset patience
    warmup_epochs=0,   # ← epochs to ignore for early-stopping decisions
):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    print(f"\nTraining quantized MLP (SCALAR QUANTIZATION) for up to {epochs} epochs...")
    print(f"Model configuration: {getattr(model, 'config', None)}")

    best_val_f1 = -1.0
    best_epoch = 0
    best_model_state = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0
    stopped_early = False
    
    # Clear any existing diagnostics to free memory
    if hasattr(model, 'fc1') and hasattr(model.fc1, '_weight_history'):
        if model.fc1._weight_history is not None:
            model.fc1._weight_history.clear()
    if hasattr(model, 'fc1') and hasattr(model.fc1, '_quantization_errors'):
        if model.fc1._quantization_errors is not None:
            model.fc1._quantization_errors.clear()
    
    # Initial memory cleanup
    torch.cuda.empty_cache()
    gc.collect()
    
    # Track training history for plotting
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': [],
        # fc1 layer stats
        'beta_global': [],
        'alpha_global': [],
        'sigma_p5': [],
        'sigma_p95': [],
        'sigma_mean': [],
        'sigma_median': [],
        # fc3 layer stats
        'fc3_weight_mean': [],
        'fc3_weight_std': [],
        'fc3_weight_min': [],
        'fc3_weight_max': [],
    }

    for epoch in range(epochs):
        # ---- update QAT warmup state ----
        # If warmup disabled → enable quantization from epoch 0
        if hasattr(model, "fc1") and hasattr(model.fc1, "update_epoch"):
            if model.use_warmup:
                model.fc1.update_epoch(epoch)
            else:
                model.fc1.quantization_enabled = True    # <-- warmup bypass

        # ===== Train =====
        model.train()
        start_time = time.time()
        running_loss, correct, total = 0.0, 0, 0
        # if hasattr(model.fc1, "update_epoch"):
        #    model.fc1.update_epoch(epoch)
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            logits = model(data)
            

            loss = criterion(logits.view(-1), target.view(-1))
            loss.backward()
            '''with torch.no_grad():
                g1 = (model.fc1.weight.grad.norm().item()
                    if model.fc1.weight.grad is not None else 0.0)
                g3 = (model.fc3.weight.grad.norm().item()
                    if model.fc3.weight.grad is not None else 0.0)
            print(f"  grad||fc1||={g1:.3e}  grad||fc3||={g3:.3e}")'''

            optimizer.step()

            running_loss += loss.item()
            preds = (torch.sigmoid(logits.view(-1)) >= threshold).float()
            correct += (preds == target.view(-1)).sum().item()
            total += target.size(0)
            
            # Clear intermediate tensors to free memory
            del logits, loss, preds

        train_time = time.time() - start_time
        train_avg_loss = running_loss / max(len(train_loader), 1)
        train_acc = 100.0 * correct / max(total, 1)

        # ===== Validate =====
        val_metrics = evaluate(model, val_loader, criterion, device, threshold)
        val_loss, val_acc, val_f1 = val_metrics["loss"], val_metrics["acc"], val_metrics["f1"]
        
        # ===== Track history =====
        history['train_loss'].append(train_avg_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        history['beta_global'].append(model.fc1.beta_global.item())
        history['alpha_global'].append(model.fc1.alpha_global.item())
        
        # Track sigma_r (row-wise standard deviation) statistics for fc1
        with torch.no_grad():
            W = model.fc1.weight
            beta_global = model.fc1.beta_global
            alpha_g = model.fc1.alpha_global
            W_s = beta_global * W + alpha_g
            sigma_r = W_s.std(dim=1)  # [out_features]
            # Compute statistics
            sigma_r_cpu = sigma_r.cpu()
            history['sigma_p5'].append(torch.quantile(sigma_r_cpu, 0.05).item())
            history['sigma_p95'].append(torch.quantile(sigma_r_cpu, 0.95).item())
            history['sigma_mean'].append(sigma_r_cpu.mean().item())
            history['sigma_median'].append(torch.median(sigma_r_cpu).item())
        
        # Track fc3 weight statistics
        with torch.no_grad():
            fc3_weight = model.fc3.weight.cpu()
            history['fc3_weight_mean'].append(fc3_weight.mean().item())
            history['fc3_weight_std'].append(fc3_weight.std().item())
            history['fc3_weight_min'].append(fc3_weight.min().item())
            history['fc3_weight_max'].append(fc3_weight.max().item())

        # ===== Track best and early stopping =====
        improved = (val_f1 - best_val_f1) > min_delta
        if improved:
            best_val_f1 = val_f1
            best_epoch = epoch + 1
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        # else:
        #     # don't count toward patience until warmup is done
        #     if (epoch + 1) > warmup_epochs:
        #         epochs_no_improve += 1

        print(f"Epoch {epoch+1}/{epochs} in {train_time:.2f}s")
        print(f"  Train: loss={train_avg_loss:.4f} acc={train_acc:.2f}%")
        print(
            f"  Val  : loss={val_loss:.4f} acc={val_acc:.2f}% f1={val_f1:.4f} "
            f"{'⬅︎ best' if improved else ''}"
        )
        # if (epoch + 1) > warmup_epochs:
        #     print(f"  EarlyStop patience: {epochs_no_improve}/{patience}")
        print("-" * 60)

        # Early stopping logic commented out - model will run for all epochs
        # if (epoch + 1) > warmup_epochs and epochs_no_improve >= patience:
        #     print(f"⏹️ Early stopping triggered at epoch {epoch+1}. Best at epoch {best_epoch} (F1={best_val_f1:.4f}).")
        #     stopped_early = True
        #     break
        
        # Periodic memory cleanup
        if (epoch + 1) % 10 == 0:
            torch.cuda.empty_cache()
            gc.collect()
            if torch.cuda.is_available():
                print(f"  Memory cleanup: {torch.cuda.memory_allocated(device) / 1024**2:.1f} MB allocated")
        
        if epoch % 5 == 0:
            beta_val = model.fc1.beta_global.item()
            alpha_val = model.fc1.alpha_global.item()
            beta_grad = model.fc1.beta_global.grad.item() if model.fc1.beta_global.grad is not None else 0.0
            alpha_grad = model.fc1.alpha_global.grad.item() if model.fc1.alpha_global.grad is not None else 0.0
            #print(f"[debug] fc1 global β_global={beta_val:.4f} (grad={beta_grad:.6e}), α_g={alpha_val:.4f} (grad={alpha_grad:.6e})")
            print(f"[debug] epoch={epoch+1}, quant_enabled={model.fc1.quantization_enabled}, "
          f"β_global={model.fc1.beta_global.item():.4f}, α_g={model.fc1.alpha_global.item():.4f}")
    # ===== Final evaluation with the best model (in-memory state) =====
    model.load_state_dict(best_model_state)
    test_metrics  = evaluate(model, test_loader, criterion, device, threshold)
    train_metrics = evaluate(model, train_loader, criterion, device, threshold)
    valid_metrics = evaluate(model, val_loader,   criterion, device, threshold)

    print("********************************************")
    print("Train performance: ")
    print(
        f'loss={train_metrics["loss"]:.4f}, '
        f'acc={train_metrics["acc"]:.2f}%, '
        f'prec={train_metrics["precision"]:.3f}, '
        f'rec={train_metrics["recall"]:.3f}, '
        f'f1={train_metrics["f1"]:.3f}'
    )
    print("Validation performance: ")
    print(
        f'loss={valid_metrics["loss"]:.4f}, '
        f'acc={valid_metrics["acc"]:.2f}%, '
        f'prec={valid_metrics["precision"]:.3f}, '
        f'rec={valid_metrics["recall"]:.3f}, '
        f'f1={valid_metrics["f1"]:.3f}'
    )
    print("Test performance: ")
    print(
        f'Best Val F1={best_val_f1:.4f} (epoch {best_epoch}) | Test: '
        f'loss={test_metrics["loss"]:.4f}, '
        f'acc={test_metrics["acc"]:.2f}%, '
        f'prec={test_metrics["precision"]:.3f}, '
        f'rec={test_metrics["recall"]:.3f}, '
        f'f1={test_metrics["f1"]:.3f}'
    )
    # if stopped_early:
    #     print("Note: Training stopped early due to no improvement.")
    print("-" * 50)
    
    # Final memory cleanup
    torch.cuda.empty_cache()
    gc.collect()

    return {
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "train_metrics": train_metrics,
        "val_metrics": valid_metrics,
        "test_metrics": test_metrics,
        "best_model": model,
        "stopped_early": stopped_early,
        "history": history,
    }

def save_metrics_to_csv(all_results, save_path="experiment_metrics.csv"):
    """
    Save all experiment metrics (train, validation, test) to a CSV file.
    All floating point metrics are rounded to 6 decimal places.
    
    Args:
        all_results: List of dictionaries containing results for each experiment
        save_path: Path to save the CSV file
    """
    if not all_results:
        print("No results to save to CSV.")
        return
    
    # Prepare data for CSV with 6 decimal precision for floating point values
    rows = []
    for res in all_results:
        row = {
            'bits_actual': round(res.get('bits_actual', res['bits']), 6),  # Actual bits value
            'q': res['q'],
            'M': res['M'],
            # Train metrics (rounded to 6 decimal places)
            'train_loss': round(res['train_metrics']['loss'], 6),
            'train_acc': round(res['train_metrics']['acc'], 6),
            'train_precision': round(res['train_metrics']['precision'], 6),
            'train_recall': round(res['train_metrics']['recall'], 6),
            'train_f1': round(res['train_metrics']['f1'], 6),
            # Validation metrics (rounded to 6 decimal places)
            'val_loss': round(res['val_metrics']['loss'], 6),
            'val_acc': round(res['val_metrics']['acc'], 6),
            'val_precision': round(res['val_metrics']['precision'], 6),
            'val_recall': round(res['val_metrics']['recall'], 6),
            'val_f1': round(res['val_metrics']['f1'], 6),
            # Test metrics (rounded to 6 decimal places)
            'test_loss': round(res['test_metrics']['loss'], 6),
            'test_acc': round(res['test_metrics']['acc'], 6),
            'test_precision': round(res['test_metrics']['precision'], 6),
            'test_recall': round(res['test_metrics']['recall'], 6),
            'test_f1': round(res['test_metrics']['f1'], 6),
            # Additional info (rounded to 6 decimal places for floating point)
            'best_val_f1': round(res['best_val_f1'], 6),
            'best_epoch': res['best_epoch'],
            'overload_count': res.get('overload_count', 0),
            'overload_pct': round(res.get('overload_pct', 0.0), 6),
        }
        rows.append(row)
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False, float_format='%.6f')
    print(f"✓ All experiment metrics saved to: {save_path}")


def main():
    """Main function to run the MLP example with SCALAR quantization for all q,M combinations across all datasets."""
    print("Scalar Quantization MLP Example - FUSED VERSION (ALL CONFIGURATIONS - ALL DATASETS)")
    print("=" * 70)
    
    # Base paths relative to this code file
    code_dir = os.path.dirname(os.path.abspath(__file__))
    # Results directory in the same repo (no hardcoded absolute path)
    base_results_dir = os.path.join(code_dir, "Results_SQ")
    os.makedirs(base_results_dir, exist_ok=True)

    # Dataset directory: all CSVs under CodeFiles/DataSet
    data_dir = os.path.join(code_dir, "DataSet")
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    csv_files = sorted(
        f for f in os.listdir(data_dir)
        if f.lower().endswith(".csv")
    )
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in dataset directory: {data_dir}")
    
    # Generate all (q, M) combinations for all possible bits from q=2-8, M=1-4
    # Calculate all possible bits from these combinations
    q_values = [2, 3]#, 4, 5, 6, 7, 8]
    M_values = [1, 2]#, 3, 4]
    all_possible_bits = set()
    for q in q_values:
        for M in M_values:
            bits = compute_bits(q, M)
            all_possible_bits.add(bits)
    target_bits = sorted(list(all_possible_bits))
    combinations = get_q_m_combinations(target_bits)
    
    print(f"\nFound {len(combinations)} (q, M) combinations:")
    for bits, q, M, bits_actual in combinations:
        print(f"  bits={bits_actual:.15g} (floor={bits}), q={q}, M={M}")
    print("=" * 70)
    
    # Run experiments for each CSV discovered from DataSet using generic prepare_dataloaders
    for csv_name in csv_files:
        csv_path = os.path.join(data_dir, csv_name)
        dataset_name, train_loader, val_loader, test_loader = prepare_dataloaders(csv_path, batch_size=64, data_fraction=0.2)
        print(f"\n{'#'*70}")
        print(f"# DATASET: {dataset_name}")
        print(f"{'#'*70}")
        
        # Create dataset-specific results directory
        results_dir = base_results_dir
        os.makedirs(results_dir, exist_ok=True)
        print(f"Results will be saved to: {results_dir}")
        
        # At this point train_loader/val_loader/test_loader are already prepared
        print(f"\nLoading {dataset_name} dataset from CSV: {csv_name}")
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        print(f"Test samples: {len(test_loader.dataset)}")
        
        torch.cuda.empty_cache()
        
        # Store results for all configurations for this dataset
        all_results = []
        
        # Run experiments for each (q, M) combination
        for idx, (bits, q, M, bits_actual) in enumerate(combinations, 1):
            print(f"\n{'='*70}")
            print(f"Dataset: {dataset_name} | Experiment {idx}/{len(combinations)}: bits={bits_actual:.15g} (floor={bits}), q={q}, M={M}")
            print(f"{'='*70}")
            
            try:
                # Reset seeds before each experiment for reproducibility
                torch.manual_seed(42)
                np.random.seed(42)
                random.seed(42)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(42)
                
                # Create model with current q and M (with warmup)
                model = QuantizedMLP(
                    input_size=768,  # BERT embedding dimension
                    hidden_size=512,
                    output_size=1,
                    q=q,
                    M=M,
                    use_warmup=False,
                    warmup_epochs=0
                )
                
                # Print model info
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(f"Total parameters: {total_params:,}")
                print(f"Trainable parameters: {trainable_params:,}")
                print(f"Input padding for 8-D blocks: {model.input_padding}")
                print(f"Effective bits: {bits} (q={q}, M={M})")
                
                # Train model
                results = train_model(model, train_loader, val_loader, test_loader, epochs=2, lr=0.001, warmup_epochs=0)
                
                print(f"\nTraining completed for {dataset_name} - bits={bits}, q={q}, M={M}!")
                
                # Scalar quantization doesn't have overload concept (each element quantized independently)
                with torch.no_grad():
                    fc1 = model.fc1
                    fc1.quantization_enabled = True
                    
                    # Run quantizer once to fill _last_W_aff_q_blocks
                    _ = fc1._quantize_weights()
                    
                    Wq_blocks = fc1._last_W_aff_q_blocks.to(device)   # [N_blocks, 8]
                    total_blocks = Wq_blocks.size(0)
                    
                    # Scalar quantization doesn't have overload (each element quantized independently)
                    over_count = 0
                    overload_pct = 0.0
                    print(f"[Scalar Quantization] No overload concept - each element quantized independently")
                    print(f"Total blocks: {total_blocks}")
                
                # Store results
                all_results.append({
                    'bits': bits,  # Integer bits for quantization
                    'bits_actual': bits_actual,  # Actual computed bits value
                    'q': q,
                    'M': M,
                    'train_metrics': results['train_metrics'],
                    'val_metrics': results['val_metrics'],
                    'test_metrics': results['test_metrics'],
                    'best_val_f1': results['best_val_f1'],
                    'best_epoch': results['best_epoch'],
                    'overload_count': over_count,
                    'overload_pct': overload_pct,
                    'total_blocks': total_blocks,
                })
                
                # Clean up model to free memory
                del model
                torch.cuda.empty_cache()
                gc.collect()
                
            except Exception as e:
                print(f"ERROR in {dataset_name} experiment bits={bits}, q={q}, M={M}: {e}")
                import traceback
                traceback.print_exc()
                # Continue with next experiment
                continue
        
        # Print summary of all results for this dataset
        print(f"\n{'='*70}")
        print(f"SUMMARY OF ALL EXPERIMENTS - {dataset_name}")
        print(f"{'='*70}")
        print(f"{'Bits':<8} {'q':<4} {'M':<4} {'Test Acc':<10} {'Test F1':<10} {'Val F1':<10}")
        print("-" * 70)
        for res in all_results:
            bits_display = res.get('bits_actual', res['bits'])
            if isinstance(bits_display, float):
                bits_str = f"{bits_display:.15g}"
            else:
                bits_str = str(bits_display)
            print(f"{bits_str:<8} {res['q']:<4} {res['M']:<4} "
                  f"{res['test_metrics']['acc']:<10.2f} {res['test_metrics']['f1']:<10.4f} "
                  f"{res['best_val_f1']:<10.4f}")
        print("=" * 70)
        
        # Save all metrics to CSV for this dataset
        if all_results:
            csv_path = os.path.join(results_dir, f"experiment_metrics_SQ_WU0_{dataset_name.lower()}.csv")
            save_metrics_to_csv(all_results, save_path=csv_path)
        
        # Final memory cleanup after each dataset
        torch.cuda.empty_cache()
        gc.collect()
        
        print(f"\n{'#'*70}")
        print(f"# COMPLETED: {dataset_name}")
        print(f"{'#'*70}\n")
    
    print(f"\n{'='*70}")
    print("ALL DATASETS COMPLETED!")
    print(f"Results saved to: {base_results_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()



