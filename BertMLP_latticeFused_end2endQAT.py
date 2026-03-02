import os
import sys
import warnings

# Suppress Python warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*oneDNN.*")
warnings.filterwarnings("ignore", message=".*cuFFT.*")
warnings.filterwarnings("ignore", message=".*cuDNN.*")
warnings.filterwarnings("ignore", message=".*cuBLAS.*")
warnings.filterwarnings("ignore", message=".*Unable to register.*")

# Suppress TensorFlow logging
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)

# Add coset path
#sys.path.insert(0, "coset")

import pandas as pd
import numpy as np
import torch
import math
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import gc
from torch.utils.data import DataLoader, TensorDataset, Subset

os.environ["TRANSFORMERS_NO_FLASH_ATTENTION"] = "1"
# Suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 0=all, 1=info, 2=warnings, 3=errors
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import coset
# Import coset core modules
try:
    from coset.core.base import LatticeConfig
    from coset.core.layers import HNLQLinearQAT, ste_quantize
    from coset.core.e8 import E8Lattice
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
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Clear CUDA cache to avoid "device busy" errors
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    # Synchronize to ensure all previous operations are complete
    torch.cuda.synchronize()
    # Reset CUDA device to clear any error states
    try:
        torch.cuda.reset_peak_memory_stats()
    except:
        pass

weight_clip_value=2.0

E8_LATTICE = E8Lattice(device=device)
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
class GlobalScaledHNLQLinearQAT(HNLQLinearQAT):
    
    def __init__(self, *args, a_init: float = 1.0, b_init: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer('beta_global', torch.tensor(1.0))
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
        
        Delta0_base = 1.5#0.3#1.5
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

def make_fused_e8_quantize_fn(M: int, tol: float = 1e-6):
    lattice = E8_LATTICE
    def _adapter(x: torch.Tensor, q: int) -> torch.Tensor:
        # x: [N, 8] (already tiled into E8 blocks by the layer)

        # 1) Base nearest-point projection Q_L(x)
        #lattice.projection_babai(x) if babai version is to be used.
        QL_x = lattice.projection(x)              # [N, 8]

        # 2) Hierarchical recursion g_{m+1} = Q_L(g_m) / q
        g_tilde = x
        for _ in range(M):
            #lattice.projection_babai(x) if babai version is to be used.
            g_bar = lattice.projection(g_tilde)   # Q_L(g_m)
            g_tilde = g_bar / q                   # g_{m+1}

        # 3) Residual lattice point r = Q_L(g_M)
        #lattice.projection_babai(x) if babai version is to be used.
        r = lattice.projection(g_tilde)           # [N, 8]
        #print("r check", r)
        # 4) Tail term Q^{∘M}(x) = q^M * r
        qM = float(q ** M)
        tail = qM * r

        # 5) Eq. (8):  x_hat = Q_L(x) - q^M * r
        x_hat = QL_x - tail                       # [N, 8]

        return x_hat
        #return QL_x
    return _adapter

def make_e8_linear_from_linear(
    linear: nn.Linear,
    q: int = 4,
    M: int = 2,
    block_size: int = 8,
    warmup_epochs: int = 0,
    enable_diagnostics: bool = False,
    act_bit_width: int = 8,
    Delta0: float = 1.5,
    a_init: float = 1.0,
    b_init: float = 0.0,
) -> GlobalScaledHNLQLinearQAT:
    in_features  = linear.in_features
    out_features = linear.out_features
    assert in_features % block_size == 0, \
        f"in_features={in_features} not divisible by block_size={block_size}"

    e8_linear = GlobalScaledHNLQLinearQAT(
        in_features=in_features,
        out_features=out_features,
        lattice=E8_LATTICE,
        quantize_fn=make_fused_e8_quantize_fn(M),
        q=q,
        M=M,
        tiling="row",
        block_size=block_size,
        warmup_epochs=warmup_epochs,
        enable_diagnostics=enable_diagnostics,
        weight_clip_value=weight_clip_value,
        quantize_activations=False,
        act_bit_width=act_bit_width,
        Delta0=Delta0,
        a_init=a_init,
        b_init=b_init,
        init_method='normal',
        init_kwargs={'mean': 0.0, 'std': 1.0},
    )

    with torch.no_grad():
        e8_linear.weight.copy_(linear.weight)
        if linear.bias is not None and e8_linear.bias is not None:
            e8_linear.bias.copy_(linear.bias)

    return e8_linear


def replace_linears_with_e8(module: nn.Module, q: int = 4, M: int = 2, warmup_epochs: int = 0):
    """
    Recursively replace nn.Linear with GlobalScaledHNLQLinearQAT in BERT.
    We DO NOT touch QuantizedMLP here (it already uses GlobalScaledHNLQLinearQAT).
    """
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            e8_linear = make_e8_linear_from_linear(child, q=q, M=M, warmup_epochs=warmup_epochs)
            setattr(module, name, e8_linear)
        else:
            replace_linears_with_e8(child, q=q, M=M, warmup_epochs=warmup_epochs)

class QuantizedMLP(nn.Module):
    def __init__(
        self,
        input_size=768,
        hidden_size=512,
        output_size=1,
        q=4, M=2,
        warmup_epochs=0,                  # warmup epochs before enabling quantization
        use_warmup=False,
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
            quantize_fn=make_fused_e8_quantize_fn(M),  # or your fused E8 adapter
            q=q, M=M,  # Use the parameters passed to QuantizedMLP, not global variables
            tiling=tiling,
            block_size=8,
            warmup_epochs=0,
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
class BertWithQuantizedMLP(nn.Module):
    """
    BERT encoder (E8-quantized linears) + your existing QuantizedMLP head.
    - BERT outputs CLS embedding [B, 768]
    - QuantizedMLP: 768 → 512 (E8 HNLQ) → 1 (logit)
    - Loss: BCEWithLogitsLoss with float labels {0.0, 1.0}
    """
    def __init__(self, bert_model: AutoModel, mlp_head: QuantizedMLP):
        super().__init__()
        self.bert = bert_model
        self.mlp = mlp_head

    def forward(self, input_ids, attention_mask=None, labels=None):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]      # [B, 768]
        logits = self.mlp(cls).view(-1)           # [B] (because MLP outputs [B, 1])

        loss = None
        if labels is not None:
            labels = labels.float().view(-1)      # BCELogits expects float
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(logits, labels)

        return {"loss": loss, "logits": logits}


def build_model(q=2, M=2):
    bert_model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(bert_model_name)

    # 1) Load BERT encoder (no classification head)
    bert_encoder = AutoModel.from_pretrained(bert_model_name)

    # 2) Replace ALL nn.Linear in BERT with E8QAT layers
    replace_linears_with_e8(bert_encoder, q=q, M=M, warmup_epochs=0)

    # 3) Build your existing QuantizedMLP head (exact same config)
    mlp_head = QuantizedMLP(
        input_size=768,
        hidden_size=512,
        output_size=1,
        q=q,
        M=M,
        warmup_epochs=0,
        use_warmup=False,
        tiling="row",
        block_size=8,
        quantize_activations=False,
        act_bit_width=8,
        weight_clip_value=weight_clip_value,
        enable_diagnostics=False,
        Delta0=1.5,
    )

    # 4) Combine BERT + MLP
    model = BertWithQuantizedMLP(bert_encoder, mlp_head)
    model.to(device)

    # 5) Enable quantization for all HNLQ layers (BERT + MLP)
    for m in model.modules():
        if isinstance(m, GlobalScaledHNLQLinearQAT):
            m.quantization_enabled = True   # you can add warmup logic if needed

    return model, tokenizer
@torch.no_grad()
def evaluate(model, data_loader, device, threshold=0.5, compute_loss=False):
    model.eval()
    all_logits = []
    all_labels = []
    total_loss = 0.0
    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for batch in data_loader:
            input_ids, attention_mask, labels = [t.to(device) for t in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=None)
            logits = outputs["logits"]
            
            if compute_loss:
                # Compute loss for validation
                labels_float = labels.float()
                loss = criterion(logits.view(-1), labels_float.view(-1))
                total_loss += loss.item()
            
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels).numpy()

    probs = torch.sigmoid(all_logits).numpy()
    preds = (probs >= threshold).astype(np.float32)

    acc  = accuracy_score(all_labels, preds) * 100.0
    prec = precision_score(all_labels, preds, zero_division=0)
    rec  = recall_score(all_labels, preds, zero_division=0)
    f1   = f1_score(all_labels, preds, zero_division=0) * 100.0

    result = {
        "acc": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
    }
    
    if compute_loss:
        result["loss"] = total_loss / max(len(data_loader), 1)
    
    return result
def prepare_dataloaders(csv_path, tokenizer, max_len=128, batch_size=32):
    # Load CSV with encoding fallback
    encodings_to_try = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-8-sig']
    df = None
    for encoding in encodings_to_try:
        try:
            df = pd.read_csv(csv_path, encoding=encoding)
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
        except Exception:
            continue
    
    if df is None:
        raise ValueError(f"Could not read CSV file {csv_path} with any of the tried encodings: {encodings_to_try}")
    
    # Handle different CSV column names and label formats
    # Check for text column (could be 'text', 'review', 'statement', 'v2', etc.)
    text_col = None
    for col in ['v2', 'text', 'review', 'statement', 'sentence', 'comment']:
        if col in df.columns:
            text_col = col
            break
    
    if text_col is None:
        raise ValueError(f"Could not find text column in CSV. Available columns: {df.columns.tolist()}")
    
    # Check for label column (could be 'label', 'sentiment', 'class', 'v1', etc.)
    label_col = None
    for col in ['v1', 'label', 'sentiment', 'class', 'target']:
        if col in df.columns:
            label_col = col
            break
    
    if label_col is None:
        raise ValueError(f"Could not find label column in CSV. Available columns: {df.columns.tolist()}")
    
    # Extract text and labels
    df["text"] = df[text_col].fillna("").astype(str)
    
    # Convert labels: handle string labels (positive/negative, spam/ham -> 1/0) or numeric labels
    labels_raw = df[label_col]
    if labels_raw.dtype == 'object' or labels_raw.dtype.name == 'string':
        # Convert string labels: positive/spam -> 1, negative/ham -> 0
        label_mapping = {
            'positive': 1, 'negative': 0, 
            'pos': 1, 'neg': 0,
            'spam': 1, 'ham': 0,
            '1': 1, '0': 0
        }
        df["label"] = labels_raw.str.lower().str.strip().map(label_mapping).fillna(-1)
        # Check if any labels couldn't be mapped
        if (df["label"] == -1).any():
            unique_labels = labels_raw.unique()
            raise ValueError(f"Unknown label values found: {unique_labels}. Expected: positive/negative, spam/ham, or 1/0")
    else:
        # Already numeric, just convert to int
        df["label"] = labels_raw.astype(int)
    
    # Filter out any invalid labels if needed
    df = df[df["label"].isin([0, 1])]

    texts = df["text"].tolist()
    labels = df["label"].tolist()

    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_len,
        return_tensors="pt"
    )

    input_ids = encodings["input_ids"]
    attention_mask = encodings["attention_mask"]
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    dataset = TensorDataset(input_ids, attention_mask, labels_tensor)

    indices = np.arange(len(dataset))
    train_idx, temp_idx, train_labels, temp_labels = train_test_split(
        indices,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )
    val_idx, test_idx, _, _ = train_test_split(
        temp_idx,
        temp_labels,
        test_size=0.5,
        random_state=42,
        stratify=temp_labels,
    )

    train_dataset = Subset(dataset, train_idx)
    val_dataset   = Subset(dataset, val_idx)
    test_dataset  = Subset(dataset, test_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

 

def check_overload_per_epoch(model, q, M, max_bert_layers=5):
    """
    Check overload/residual counts for MLP and BERT layers efficiently.
    Designed to be called per epoch during training.
    
    Args:
        model: The model to check
        q: Quantization parameter q
        M: Quantization parameter M
        max_bert_layers: Maximum number of BERT layers to check (for efficiency)
    
    Returns:
        Dictionary with overload statistics
    """
    with torch.no_grad():
        mlp_overload_count = 0
        mlp_total_blocks = 0
        bert_overload_count = 0
        bert_total_blocks = 0
        
        # Check MLP head (fc1 layer)
        if hasattr(model, 'mlp') and hasattr(model.mlp, 'fc1'):
            fc1 = model.mlp.fc1
            fc1.quantization_enabled = True
            _ = fc1._quantize_weights()
            
            if hasattr(fc1, '_last_W_aff_q_blocks'):
                Wq_blocks = fc1._last_W_aff_q_blocks.to(device)
                mlp_total_blocks = Wq_blocks.size(0)
                
                q_val = getattr(fc1, 'q', q)
                M_val = getattr(fc1, 'M', M)
                
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
                
                _, overload_flags, _ = e8_encode(Wq_blocks, cfg, lattice=E8_LATTICE)
                mlp_overload_count = int(overload_flags.sum().item())
        
        # Check BERT encoder layers (sample first max_bert_layers for efficiency)
        if hasattr(model, 'bert'):
            bert_quantized_layers = []
            for module in model.bert.modules():
                if isinstance(module, GlobalScaledHNLQLinearQAT):
                    bert_quantized_layers.append(module)
            
            for layer in bert_quantized_layers[:max_bert_layers]:
                layer.quantization_enabled = True
                _ = layer._quantize_weights()
                
                if hasattr(layer, '_last_W_aff_q_blocks'):
                    Wq_blocks = layer._last_W_aff_q_blocks.to(device)
                    layer_blocks = Wq_blocks.size(0)
                    bert_total_blocks += layer_blocks
                    
                    q_val = getattr(layer, 'q', q)
                    M_val = getattr(layer, 'M', M)
                    
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
                    
                    _, overload_flags, _ = e8_encode(Wq_blocks, cfg, lattice=E8_LATTICE)
                    bert_overload_count += int(overload_flags.sum().item())
        
        # Calculate percentages
        mlp_overload_pct = 100.0 * mlp_overload_count / max(1, mlp_total_blocks) if mlp_total_blocks > 0 else 0.0
        bert_overload_pct = 100.0 * bert_overload_count / max(1, bert_total_blocks) if bert_total_blocks > 0 else 0.0
        total_blocks = mlp_total_blocks + bert_total_blocks
        total_overload = mlp_overload_count + bert_overload_count
        total_overload_pct = 100.0 * total_overload / max(1, total_blocks) if total_blocks > 0 else 0.0
        
        return {
            'mlp_overload_count': mlp_overload_count,
            'mlp_total_blocks': mlp_total_blocks,
            'mlp_overload_pct': mlp_overload_pct,
            'bert_overload_count': bert_overload_count,
            'bert_total_blocks': bert_total_blocks,
            'bert_overload_pct': bert_overload_pct,
            'total_overload_count': total_overload,
            'total_blocks': total_blocks,
            'total_overload_pct': total_overload_pct,
        }

def train_end_to_end(csv_path, q=2, M=2, epochs=10, lr=2e-5, batch_size=32):
    # Build model and data
    model, tokenizer = build_model(q=q, M=M)
    train_loader, val_loader, test_loader = prepare_dataloaders(
        csv_path=csv_path,
        tokenizer=tokenizer,
        max_len=128,
        batch_size=batch_size,
    )

    optimizer = optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            input_ids, attention_mask, labels = [t.to(device) for t in batch]

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs["loss"]
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(len(train_loader), 1)
        val_metrics = evaluate(model, val_loader, device, compute_loss=True)

        # Evaluate on train set per epoch
        train_metrics = evaluate(model, train_loader, device)
        
        # Check overload counts for this epoch
        overload_stats = check_overload_per_epoch(model, q=q, M=M, max_bert_layers=5)
        
        # Evaluate on test set per epoch
        test_metrics = evaluate(model, test_loader, device)

        # Print train F1, test F1, and overload percentage
        '''print(f"Epoch {epoch+1}/{epochs} | Train F1={train_metrics['f1']:.2f}% | Test F1={test_metrics['f1']:.2f}% | "
              f"Overload={overload_stats['total_overload_pct']:.2f}%")'''
        print(
		    f"Epoch {epoch+1}/{epochs} | "
		    f"Train F1={train_metrics['f1']:.2f}% | "
		    f"Test F1={test_metrics['f1']:.2f}% | "
		    f"Overload: Total={overload_stats['total_overload_pct']:.2f}% "
		    f"(MLP={overload_stats['mlp_overload_pct']:.2f}%, "
		    f"BERT={overload_stats['bert_overload_pct']:.2f}%) "
		    f"[blocks: MLP {overload_stats['mlp_overload_count']}/{overload_stats['mlp_total_blocks']}, "
		    f"BERT {overload_stats['bert_overload_count']}/{overload_stats['bert_total_blocks']}]")	
    # Final summary: Train F1, Test F1, and separate overload % for BERT and MLP
    final_train_metrics = evaluate(model, train_loader, device)
    final_test_metrics = evaluate(model, test_loader, device)
    final_overload_stats = check_overload_per_epoch(model, q=q, M=M, max_bert_layers=5)
    
    print(f"\n{'='*70}")
    print(f"FINAL RESULTS (q={q}, M={M}, Delta0_base=1.5)")
    print(f"{'='*70}")
    print(f"Train F1: {final_train_metrics['f1']:.2f}%")
    print(f"Test F1: {final_test_metrics['f1']:.2f}%")
    print(f"MLP Overload: {final_overload_stats['mlp_overload_pct']:.2f}%")
    print(f"BERT Overload: {final_overload_stats['bert_overload_pct']:.2f}%")
    print(f"Total Overload: {final_overload_stats['total_overload_pct']:.2f}%")
    print(f"{'='*70}\n")

    # Return results for summary table
    results_summary = {
        'q': q,
        'M': M,
        'train_f1': final_train_metrics['f1'],
        'test_f1': final_test_metrics['f1'],
        'mlp_overload_pct': final_overload_stats['mlp_overload_pct'],
        'bert_overload_pct': final_overload_stats['bert_overload_pct'],
        'total_overload_pct': final_overload_stats['total_overload_pct']
    }

    # Final test evaluation
    # test_metrics = evaluate(model, test_loader, device)
    # print("\n=== Final Test Metrics ===")
    # print(f"acc={test_metrics['acc']:.2f}% | "
    #       f"prec={test_metrics['precision']:.3f} | "
    #       f"rec={test_metrics['recall']:.3f} | "
    #       f"f1={test_metrics['f1']:.2f}%")
    
    return model, results_summary

if __name__ == "__main__":
    from pathlib import Path
    dataset_dir = Path(__file__).resolve().parent / "DataSet"

    # Auto-detect a CSV in the DataSet folder
    csv_files = sorted(dataset_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {dataset_dir}")
    if len(csv_files) > 1:
        print(f"Found multiple CSV files in {dataset_dir}. Using the first one:")
        for f in csv_files:
            print(f"  - {f.name}")
    csv_path = csv_files[0]
    
    # Extract dataset name from path
    dataset_name = csv_path.stem  # e.g., IMDB for IMDB.csv
    
    # Run for multiple configurations
    configurations = [
        (8, 1),   # q=8, M=1
    ]
    
    print(f"\n{'='*70}")
    print(f"RUNNING {len(configurations)} CONFIGURATIONS FOR {dataset_name} DATASET (Delta0_base=1.5)")
    print(f"{'='*70}\n")
    
    all_results = []
    
    for idx, (q, M) in enumerate(configurations, 1):
        bits = int(M * math.log2(q))
        print(f"\n{'='*70}")
        print(f"CONFIGURATION {idx}/{len(configurations)}: q={q}, M={M}, bits={bits}, Delta0_base=1.5")
        print(f"{'='*70}\n")
        
        model, results_summary = train_end_to_end(csv_path, q=q, M=M, epochs=2, lr=2e-5, batch_size=32)
        
        # Add dataset name to results
        results_summary['dataset'] = dataset_name
        all_results.append(results_summary)
        
        print(f"\n{'='*70}")
        print(f"COMPLETED CONFIGURATION {idx}/{len(configurations)}: q={q}, M={M}, bits={bits}")
        print(f"{'='*70}\n")
        
        # Clear GPU cache between runs
        if torch.cuda.is_available():
            del model
            torch.cuda.empty_cache()
            gc.collect()
    
    # Print summary table with all configurations
    print(f"\n{'='*70}")
    print(f"SUMMARY OF ALL CONFIGURATIONS - {dataset_name} DATASET (Delta0_base=1.5)")
    print(f"{'='*70}")
    print(f"{'Config':<12} {'Dataset':<10} {'Train F1':<12} {'Test F1':<12} {'MLP Overload':<15} {'BERT Overload':<15} {'Total Overload':<15}")
    print(f"{'-'*70}")
    for result in all_results:
        config_str = f"q={result['q']}, M={result['M']}"
        print(f"{config_str:<12} {result['dataset']:<10} "
              f"{result['train_f1']:>10.2f}% {result['test_f1']:>10.2f}% "
              f"{result['mlp_overload_pct']:>13.2f}% {result['bert_overload_pct']:>13.2f}% "
              f"{result['total_overload_pct']:>13.2f}%")
    print(f"{'='*70}\n")
    
    print(f"\n{'='*70}")
    print(f"ALL {len(configurations)} CONFIGURATIONS COMPLETED")
    print(f"{'='*70}\n")
