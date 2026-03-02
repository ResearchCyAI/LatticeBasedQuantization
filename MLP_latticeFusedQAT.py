"""
Simple MLP for binary classification using Coset core APIs with FUSED quantization.
This version uses e8_quantize_fused instead of encode-decode approach.
"""
'''
Instead of using the theoretical Δ₀ formula, just use a fixed scale proportional to the 8-dimensional tile size (like 8.0). 
Each row is tiled into 8-D E8 blocks (512→64 tiles). 
Scale those blocks by 8 for quantization — simpler and numerically more stable.
'''
import os, sys
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
    from coset.core.base import LatticeConfig, Lattice
    from coset.core.layers import HNLQLinearQAT, ste_quantize
    from coset.core.e8 import E8Lattice
    from coset.core.e8.codecs import e8_quantize_fused
    from coset.core.e8.codecs import e8_encode  # the exact hierarchical encoder
    print("✓ Successfully imported coset core modules (FUSED version)")
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

def set_random_seed(seed=42):
    """Set random seed for reproducibility across all random number generators."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set initial seed
set_random_seed(42)




# --- Adapter so HNLQLinearQAT can call fused E8 quantizer like quantize_fn(x, q=...) ---
weight_clip_value=2.0
E8_LATTICE = E8Lattice(device=device)

def compute_bits(q: int, M: int) -> float:
    """Compute effective bits from q and M: bits = M * log2(q) (can be non-integer)"""
    return M * math.log2(q)

def compute_bits_int(q: int, M: int) -> int:
    """Compute integer bits (floor): bits = floor(M * log2(q))"""
    return int(math.floor(M * math.log2(q)))

def format_bits_for_filename(bits: float) -> str:
    """Format bits value for use in filenames (handles non-integer bits)."""
    # Replace dot with underscore for non-integer bits
    # e.g., 2.5 -> "2_5", 3.0 -> "3"
    if bits == int(bits):
        return str(int(bits))
    else:
        return str(bits).replace('.', '_')


def get_all_q_m_combinations(q_min: int = 2, q_max: int = 8, M_min: int = 1, M_max: int = 4) -> list:
    """
    Generate all (q, M) combinations for given ranges.
    Args:
        q_min: Minimum q value (inclusive)
        q_max: Maximum q value (inclusive)
        M_min: Minimum M value (inclusive)
        M_max: Maximum M value (inclusive)
    Returns:
        List of tuples (bits, q, M) where bits is the actual computed bits (may be non-integer)
        Sorted by bits, then q, then M
    """
    combinations = []
    for q in range(q_min, q_max + 1):
        for M in range(M_min, M_max + 1):
            bits = compute_bits(q, M)  # Actual bits (may be non-integer)
            combinations.append((bits, q, M))
    # Sort by bits, then q, then M
    combinations.append((4.0, 16, 1))
    combinations = sorted(combinations, key=lambda x: (x[0], x[1], x[2]))
    return combinations

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


def make_fused_eq8_quantize_fn(M: int, tol: float = 1e-6):
    """
    Returns a quantize_fn(x, q) that implements Eq. (8) in fused form:

        x_hat = Q_L(x) - q^M * r,

    where r = Q_L(g_M) and

        g_0 = x,
        g_{m+1} = Q_L(g_m) / q   for m = 0..M-1.

    This matches the hierarchical definition (Eq. (4)) and Eq. (8),
    but stays fully fused (no explicit digits/encode/decode).
    """
    lattice = E8_LATTICE

    def _adapter(x: torch.Tensor, q: int) -> torch.Tensor:
        # x: [N, 8] (already tiled into E8 blocks by the layer)

        # 1) Base nearest-point projection Q_L(x), keep projection_babai for using babai projection
        QL_x = lattice.projection(x)              # [N, 8]

        # 2) Hierarchical recursion g_{m+1} = Q_L(g_m) / q
        g_tilde = x
        for _ in range(M):
            g_bar = lattice.projection(g_tilde)   # Q_L(g_m), keep projection_babai for using babai projection
            g_tilde = g_bar / q                   # g_{m+1}

        # 3) Residual lattice point r = Q_L(g_M)
        r = lattice.projection(g_tilde)           # [N, 8], keep projection_babai for using babai projection

        # 4) Tail term Q^{∘M}(x) = q^M * r
        qM = float(q ** M)
        tail = qM * r

        # 5) Eq. (8):  x_hat = Q_L(x) - q^M * r
        x_hat = QL_x - tail                       # [N, 8]

        return x_hat

    return _adapter
    
class QuantizedMLP(nn.Module):
    def __init__(
        self,
        input_size=768,
        hidden_size=512,
        output_size=1,
        q=4, M=2,
        warmup_epochs=0,                  # warmup epochs before enabling quantization (no warmup)
        # warmup_epochs=5,                  # warmup epochs before enabling quantization
        use_warmup=False,
        # use_warmup=True,                  # enable warmup
        tiling="row",                    # or "block"
        block_size=8,                    # matches E8
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
        
        self.fc1 = GlobalScaledHNLQLinearQAT(
            in_features=input_size,
            out_features=hidden_size,
            lattice=E8_LATTICE,
            quantize_fn=make_fused_eq8_quantize_fn(M),#make_e8_quantize_fn_adapter_1(Delta0_layer, q, M, E8_LATTICE),  # or your fused E8 adapter
            q=q, M=M,  # Use the parameters passed to QuantizedMLP, not global variables
            tiling=tiling,
            block_size=8,
            warmup_epochs=0,
            # warmup_epochs=5,  # with warmup
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
        3. Quantization: W_q = QE8(W_r)
        4. Save quantized blocks for overload checking
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
        
        Delta0_base = 1.5
        qM = float(self.q ** self.M)
        Y_max = Delta0_base * (qM - 1) / 2.0
        
        # Compute β_r = Y_max / (C_b * sigma_r) where C_b = 1
        C_b = 5.0
        beta_r = Y_max / (C_b * sigma_r)  # [out_features, 1]
        
        # Apply row-wise scaling
        W_r = beta_r * W_s  # [out_features, in_features]

        # ---- 3) Quantization: W_q = QE8(W_r) ----
        # Tile into 8-D blocks
        W_blocks = W_r.view(self.out_features, self.blocks_per_row, self.block_size)
        W_blocks_flat = W_blocks.reshape(-1, self.block_size)  # [N_blocks, 8]

        # Quantize using STE (Straight-Through Estimator)
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
    Expects: DataSet/embeddings/ with
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
    f1 = f1_score(all_targets, all_preds, zero_division=0) * 100.0  # Convert to percentage

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
    warmup_epochs=0,   # ← epochs to ignore for early-stopping decisions (no warmup)
    # warmup_epochs=5,   # ← epochs to ignore for early-stopping decisions (with warmup)
    bits=None,         # ← target bits for filename
    output_dir="."     # ← output directory for saving models
):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    print(f"\nTraining quantized MLP (FUSED) for up to {epochs} epochs...")
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
        # Per-epoch metrics
        'epoch_time': [],           # Wall clock time per epoch
        'overload_count': [],        # Overload count per epoch
        'overload_pct': [],          # Overload percentage per epoch
    }

    for epoch in range(epochs):
        epoch_start_time = time.perf_counter()  # Track total epoch time
        # ---- update QAT warmup state ----
        # If warmup disabled → enable quantization from epoch 0
        if hasattr(model, "fc1") and hasattr(model.fc1, "update_epoch"):
            if model.use_warmup:
                model.fc1.update_epoch(epoch)
            else:
                model.fc1.quantization_enabled = True    # <-- warmup bypass

        # ===== Train =====
        model.train()
        start_time = time.perf_counter()
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

        train_time = time.perf_counter() - start_time
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
        
        # ===== Track per-epoch overload count =====
        epoch_time = time.perf_counter() - epoch_start_time
        history['epoch_time'].append(epoch_time)
        
        # Compute overload count for this epoch (only if quantization is enabled)
        overload_count = 0
        overload_pct = 0.0
        if hasattr(model, 'fc1') and model.fc1.quantization_enabled:
            with torch.no_grad():
                fc1 = model.fc1
                _ = fc1._quantize_weights()
                
                if hasattr(fc1, '_last_W_aff_q_blocks'):
                    Wq_blocks = fc1._last_W_aff_q_blocks.to(device)  # [N_blocks, 8]
                    total_blocks = Wq_blocks.size(0)
                    
                    # Get q and M from the model
                    q_val = getattr(fc1, 'q', 4)
                    M_val = getattr(fc1, 'M', 2)
                    
                    cfg = LatticeConfig(
                        lattice_type="E8",
                        q=q_val,
                        M=M_val,
                        beta=1.0,
                        alpha=1.0,
                        max_scaling_iterations=0,
                        with_dither=False,
                        disable_overload_protection=True,
                    )
                    
                    # Hierarchical overload (residual)
                    _, overload_flags, _ = e8_encode(Wq_blocks, cfg, lattice=E8_LATTICE)
                    overload_count = int(overload_flags.sum().item())
                    overload_pct = 100.0 * overload_count / max(1, total_blocks)
        
        history['overload_count'].append(overload_count)
        history['overload_pct'].append(overload_pct)

        # ===== Track best (in memory only) =====
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

        print(f"Epoch {epoch+1}/{epochs} in {epoch_time:.2f}s (train: {train_time:.2f}s)")
        print(f"  Train: loss={train_avg_loss:.4f} acc={train_acc:.2f}%")
        print(
            f"  Val  : loss={val_loss:.4f} acc={val_acc:.2f}% f1={val_f1:.2f}% "
            f"{'⬅︎ best' if improved else ''}"
        )
        if model.fc1.quantization_enabled:
            print(f"  Overload: {overload_count} blocks ({overload_pct:.2f}%) | σ_r mean={history['sigma_mean'][-1]:.4f}")
        # if (epoch + 1) > warmup_epochs:
        #     print(f"  EarlyStop patience: {epochs_no_improve}/{patience}")
        print("-" * 60)

        # Early stopping logic commented out - model will run for all epochs
        # if (epoch + 1) > warmup_epochs and epochs_no_improve >= patience:
        #     print(f" Early stopping triggered at epoch {epoch+1}. Best at epoch {best_epoch} (F1={best_val_f1:.2f}%).")
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
            

    # ===== Final evaluation with the best model (restore from in-memory state) =====
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
        f'f1={train_metrics["f1"]:.2f}%'
    )
    print("Validation performance: ")
    print(
        f'loss={valid_metrics["loss"]:.4f}, '
        f'acc={valid_metrics["acc"]:.2f}%, '
        f'prec={valid_metrics["precision"]:.3f}, '
        f'rec={valid_metrics["recall"]:.3f}, '
        f'f1={valid_metrics["f1"]:.2f}%'
    )
    print("Test performance: ")
    print(
        f'Best Val F1={best_val_f1:.2f}% (epoch {best_epoch}) | Test: '
        f'loss={test_metrics["loss"]:.4f}, '
        f'acc={test_metrics["acc"]:.2f}%, '
        f'prec={test_metrics["precision"]:.3f}, '
        f'rec={test_metrics["recall"]:.3f}, '
        f'f1={test_metrics["f1"]:.2f}%'
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

def print_quantized_weight_analysis(model, q, M, bits, sample_size=100):
    """
    Print and analyze the quantized weight matrix to verify quantization.
    
    Args:
        model: The trained model
        q: Quantization parameter
        M: Number of hierarchical levels
        bits: Target bits
        sample_size: Number of elements to sample for detailed analysis
    """
    print(f"\n{'='*70}")
    print(f"QUANTIZED WEIGHT ANALYSIS (bits={bits}, q={q}, M={M})")
    print(f"{'='*70}")
    
    if not hasattr(model, 'fc1'):
        print("Model does not have fc1 layer")
        return
    
    fc1 = model.fc1
    fc1.eval()
    
    # Enable quantization if not already enabled
    with torch.no_grad():
        fc1.quantization_enabled = True
        
        # Run quantizer to get quantized weights
        _ = fc1._quantize_weights()
        
        # Get the quantized weight matrix (before undoing scaling)
        if hasattr(fc1, '_last_W_aff_q'):
            W_quantized = fc1._last_W_aff_q  # [out_features, in_features]
            
            print(f"\nQuantized Weight Matrix Shape: {W_quantized.shape}")
            print(f"Total elements: {W_quantized.numel():,}")
            
            # Sample a portion of the matrix for detailed view
            out_feat, in_feat = W_quantized.shape
            sample_rows = min(5, out_feat)
            sample_cols = min(20, in_feat)
            
            print(f"\nSample of Quantized Weight Matrix (first {sample_rows} rows, first {sample_cols} columns):")
            print(W_quantized[:sample_rows, :sample_cols].cpu().numpy())
            
            # Statistics
            W_flat = W_quantized.flatten().cpu()
            print(f"\nQuantized Weight Statistics:")
            print(f"  Min: {W_flat.min().item():.6f}")
            print(f"  Max: {W_flat.max().item():.6f}")
            print(f"  Mean: {W_flat.mean().item():.6f}")
            print(f"  Std: {W_flat.std().item():.6f}")
            print(f"  Median: {W_flat.median().item():.6f}")
            
            # Count unique values (to see quantization levels)
            # For E8 quantization, values should be from lattice points
            W_unique = torch.unique(W_flat)
            print(f"\nUnique Values in Quantized Matrix: {len(W_unique)}")
            bits_str = f"{bits:.4f}" if isinstance(bits, float) else str(bits)
            print(f"  (For {bits_str}-bit quantization, expect limited distinct values)")
            
            # Show first 20 unique values
            n_show = min(20, len(W_unique))
            print(f"\nFirst {n_show} unique values:")
            for i, val in enumerate(W_unique[:n_show]):
                count = (W_flat == val).sum().item()
                pct = 100.0 * count / len(W_flat)
                print(f"  {val.item():8.6f}: {count:8,} occurrences ({pct:5.2f}%)")
            
            # Analyze quantization blocks
            if hasattr(fc1, '_last_W_aff_q_blocks'):
                W_blocks = fc1._last_W_aff_q_blocks  # [N_blocks, 8]
                print(f"\nQuantized Block Analysis:")
                print(f"  Total blocks: {W_blocks.shape[0]:,}")
                print(f"  Block size: {W_blocks.shape[1]}")
                
                # Show first few blocks
                print(f"\nFirst 3 quantized blocks (8-D vectors):")
                for i in range(min(3, W_blocks.shape[0])):
                    print(f"  Block {i}: {W_blocks[i].cpu().numpy()}")
            
            # Compute effective bits
            effective_bits = int(math.floor(M * math.log2(q)))
            actual_bits = M * math.log2(q)
            print(f"\nQuantization Configuration:")
            print(f"  q = {q}, M = {M}")
            print(f"  Actual bits = M * log2(q) = {M} * log2({q}) = {actual_bits:.4f}")
            print(f"  Integer bits (floor) = {effective_bits}")
            bits_str = f"{bits:.4f}" if isinstance(bits, float) else str(bits)
            print(f"  Target bits = {bits_str}")
            
            if effective_bits == bits:
                print(f"  ✓ Effective bits match target bits!")
            else:
                print(f"  ⚠ Effective bits ({effective_bits}) do not match target bits ({bits})")
            
        else:
            print("Quantized weights not available. Make sure quantization is enabled.")
    
    print(f"{'='*70}\n")


def save_metrics_to_csv(all_results, save_path="experiment_metrics.csv"):
    """
    Save all experiment metrics (train, validation, test) to a CSV file.
    
    Args:
        all_results: List of dictionaries containing results for each experiment
        save_path: Path to save the CSV file
    """
    if not all_results:
        print("No results to save to CSV.")
        return
    
    # Prepare data for CSV
    rows = []
    for res in all_results:
        row = {
            'bits': res['bits'],
            'q': res['q'],
            'M': res['M'],
            # Train metrics
            'train_loss': res['train_metrics']['loss'],
            'train_acc': res['train_metrics']['acc'],
            'train_precision': res['train_metrics']['precision'],
            'train_recall': res['train_metrics']['recall'],
            'train_f1': res['train_metrics']['f1'],
            # Validation metrics
            'val_loss': res['val_metrics']['loss'],
            'val_acc': res['val_metrics']['acc'],
            'val_precision': res['val_metrics']['precision'],
            'val_recall': res['val_metrics']['recall'],
            'val_f1': res['val_metrics']['f1'],
            # Test metrics
            'test_loss': res['test_metrics']['loss'],
            'test_acc': res['test_metrics']['acc'],
            'test_precision': res['test_metrics']['precision'],
            'test_recall': res['test_metrics']['recall'],
            'test_f1': res['test_metrics']['f1'],
            # Additional info
            'best_val_f1': res['best_val_f1'],
            'best_epoch': res['best_epoch'],
            'overload_count': res.get('overload_count', 0),
            'overload_pct': res.get('overload_pct', 0.0),
        }
        rows.append(row)
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)
    print(f"✓ All experiment metrics saved to: {save_path}")


def main():
    """Main function to run the MLP example with FUSED quantization for all q,M combinations across all datasets."""
    print("Coset Core APIs Quantized MLP Example - FUSED VERSION (ALL CONFIGURATIONS)")
    print("=" * 70)

    code_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(code_dir, "DataSet")
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
    csv_files = sorted(
        f for f in os.listdir(data_dir)
        if f.lower().endswith(".csv")
    )
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in dataset directory: {data_dir}")

    # Generate all (q, M) combinations for q=2 to 8 and M=1 to 4
    combinations = get_all_q_m_combinations(q_min=2, q_max=4, M_min=1, M_max=2)

    print(f"\nFound {len(combinations)} (q, M) combinations:")
    print(f"{'Bits':<10} {'q':<6} {'M':<6} {'Bits (int)':<12}")
    print("-" * 40)
    for bits, q, M in combinations:
        bits_int = compute_bits_int(q, M)
        print(f"  {bits:<10.4f} {q:<6} {M:<6} {bits_int:<12}")
    print("=" * 70)

    # Run experiments for each CSV in DataSet (embeddings from DataSet/embeddings_<csv_stem>/)
    for dataset_idx, csv_name in enumerate(csv_files, 1):
        csv_path = os.path.join(data_dir, csv_name)
        dataset_name, train_loader, val_loader, test_loader = prepare_dataloaders(
            csv_path, batch_size=64, data_fraction=0.2
        )
        output_dir = os.path.join(code_dir, "AllResults_E8")

        print(f"\n{'#'*70}")
        print(f"# DATASET {dataset_idx}/{len(csv_files)}: {dataset_name}")
        print(f"{'#'*70}")

        os.makedirs(output_dir, exist_ok=True)
        print(f"\nOutput directory: {output_dir}")
        print(f"All results, models, and plots will be saved to: {output_dir}")

        print(f"\nLoading {dataset_name} dataset from CSV: {csv_name}")
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        print(f"Test samples: {len(test_loader.dataset)}")
        torch.cuda.empty_cache()
        
        # Store results for all configurations for this dataset
        all_results = []
        
        # Run experiments for each (q, M) combination
        for idx, (bits, q, M) in enumerate(combinations, 1):
            print(f"\n{'='*70}")
            print(f"Dataset: {dataset_name} | Experiment {idx}/{len(combinations)}: bits={bits:.4f} (int: {compute_bits_int(q, M)}), q={q}, M={M}")
            print(f"{'='*70}")
            
            # Reset random seed before each configuration for reproducibility
            set_random_seed(42)
            print(f"Random seed reset to 42 for this configuration")
            
            try:
                # Create model with current q and M
                model = QuantizedMLP(
                    input_size=768,  # BERT embedding dimension
                    hidden_size=512,
                    output_size=1,
                    q=q,
                    M=M,
                    use_warmup=False,
                    # use_warmup=True,  # with warmup
                    warmup_epochs=0,
                    # warmup_epochs=5   # with warmup
                )
                
                # Print model info
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(f"Total parameters: {total_params:,}")
                print(f"Trainable parameters: {trainable_params:,}")
                print(f"Input padding for E8: {model.input_padding}")
                
                # Train model with bits parameter for filename
                results = train_model(model, train_loader, val_loader, test_loader, epochs=2, lr=0.001, bits=bits, output_dir=output_dir)
                
                print(f"\nTraining completed for bits={bits:.4f} (int: {compute_bits_int(q, M)}), q={q}, M={M}!")
                
                best_model = results.get('best_model', model)
                
                # Overload analysis and quantized weight analysis
                # Use the best model from results
                with torch.no_grad():
                    fc1 = best_model.fc1
                    fc1.quantization_enabled = True
                    
                    # Run quantizer once to fill _last_W_aff_q_blocks
                    _ = fc1._quantize_weights()
                    
                    Wq_blocks = fc1._last_W_aff_q_blocks.to(device)   # [N_blocks, 8]
                    total_blocks = Wq_blocks.size(0)
                    
                    cfg = LatticeConfig(
                        lattice_type="E8",
                        q=q,
                        M=M,
                        beta=1.0,
                        alpha=1.0,
                        max_scaling_iterations=0,
                        with_dither=False,
                        disable_overload_protection=True,
                    )
                    
                    # Hierarchical overload (residual)
                    _, overload_flags, _ = e8_encode(Wq_blocks, cfg, lattice=E8_LATTICE)
                    
                    over_count = int(overload_flags.sum().item())
                    overload_pct = 100.0 * over_count / max(1, total_blocks)
                    print(f"[Overload|encode_residual] {over_count}/{total_blocks} blocks ({overload_pct:.2f}%)")
                
                # Print quantized weight analysis to verify 2-bit quantization
                #print_quantized_weight_analysis(best_model, q=q, M=M, bits=bits)
                
                # Store results
                all_results.append({
                    'bits': bits,
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
                print(f"ERROR in experiment bits={bits}, q={q}, M={M}: {e}")
                import traceback
                traceback.print_exc()
                # Continue with next experiment
                continue
        
        # Print summary of all results for this dataset
        print(f"\n{'='*70}")
        print(f"SUMMARY OF ALL EXPERIMENTS - {dataset_name}")
        print(f"{'='*70}")
        print(f"{'Bits':<10} {'q':<4} {'M':<4} {'Test Acc':<10} {'Test F1':<11} {'Val F1':<11} {'Overload %':<12}")
        print("-" * 70)
        for res in all_results:
            bits_val = res['bits']
            bits_str = f"{bits_val:.4f}" if isinstance(bits_val, float) else str(bits_val)
            print(f"{bits_str:<10} {res['q']:<4} {res['M']:<4} "
                  f"{res['test_metrics']['acc']:<10.2f} {res['test_metrics']['f1']:<10.2f}% "
                  f"{res['best_val_f1']:<10.2f}% {res['overload_pct']:<12.2f}")
        print("=" * 70)
        
        # Save all metrics to CSV for this dataset
        if all_results:
            csv_path = os.path.join(output_dir, "experiment_metrics_spam.csv")
            save_metrics_to_csv(all_results, save_path=csv_path)
        
        # Memory cleanup after each dataset
        torch.cuda.empty_cache()
        gc.collect()
        
        print(f"\n{'#'*70}")
        print(f"# COMPLETED DATASET: {dataset_name}")
        print(f"{'#'*70}\n")
    
    # Final summary across all datasets
    print(f"\n{'='*70}")
    print("ALL DATASETS COMPLETED!")
    print(f"{'='*70}")
    print("Results saved in:")
    for csv_name in csv_files:
        dataset_name = os.path.splitext(csv_name)[0]
        output_dir = os.path.join(code_dir, "AllResults_E8", dataset_name)
        print(f"  - {dataset_name}: {output_dir}")
    print("=" * 70)
    
    # Final memory cleanup
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()



