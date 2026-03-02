"""
Approach 1: ACIQ for Activations + VQ (E8) for Weights
- E8 Vector Quantization for weights (unchanged)
- ACIQ (Analytical Clipping + Uniform Quantization) for activations
"""
import os
#import sys
import warnings

# Suppress TensorFlow warnings and errors (must be set before any TensorFlow imports)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TRANSFORMERS_NO_FLASH_ATTENTION"] = "1"

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
import copy
import math
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import time
import datetime
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import gc
from torch.utils.data import DataLoader, TensorDataset, Subset

os.environ["TRANSFORMERS_NO_FLASH_ATTENTION"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from typing import Optional, Tuple
import coset
# Import coset core modules
try:
    from coset.core.base import LatticeConfig, Lattice
    from coset.core.layers import HNLQLinearQAT, ste_quantize
    from coset.core.e8 import E8Lattice
    from coset.core.e8.codecs import e8_quantize_fused
    from coset.core.e8.codecs import e8_encode
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

# Clear CUDA cache
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    try:
        torch.cuda.reset_peak_memory_stats()
    except:
        pass

weight_clip_value = 2.0
E8_LATTICE = E8Lattice(device=device)

# ============================================================================
# ACIQ Activation Quantizer
# ============================================================================
class ACIQActivationQuantizer(nn.Module):
    """
    ACIQ for activations: analytical clipping + uniform quantization.
    Based on ACIQ paper: https://openreview.net/pdf?id=B1x33sC9KQ
    """
    
    def __init__(self, bit_width=8, clip_type='laplace', use_stats=True):
        super().__init__()
        self.bit_width = bit_width
        self.clip_type = clip_type  # 'laplace', 'gaus', or 'mix'
        self.use_stats = use_stats
        
        # ACIQ optimal alpha coefficients from paper
        # Extended to support up to 16 bits (extrapolated for >8 bits)
        self.alpha_gaus = {2: 1.71, 3: 2.15, 4: 2.55, 5: 2.93, 6: 3.28, 7: 3.61, 8: 3.92,
                           9: 4.22, 10: 4.51, 11: 4.79, 12: 5.06, 13: 5.33, 14: 5.59, 15: 5.84, 16: 6.09}
        self.alpha_laplace = {2: 2.83, 3: 3.89, 4: 5.03, 5: 6.2, 6: 7.41, 7: 8.64, 8: 9.89,
                              9: 11.16, 10: 12.45, 11: 13.76, 12: 15.09, 13: 16.44, 14: 17.81, 15: 19.20, 16: 20.61}
        
        # Statistics buffers
        self.register_buffer('running_mean', None)
        self.register_buffer('running_std', None)
        self.register_buffer('running_b', None)  # Laplace scale parameter
        self.register_buffer('running_min', None)
        self.register_buffer('running_max', None)
        self.momentum = 0.1
    
    def update_stats(self, x):
        """Update running statistics during calibration/training."""
        with torch.no_grad():
            if self.running_mean is None:
                self.running_mean = x.mean()
                self.running_std = x.std()
                self.running_b = torch.mean(torch.abs(x - x.mean()))
                self.running_min = x.min()
                self.running_max = x.max()
            else:
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * x.mean()
                self.running_std = (1 - self.momentum) * self.running_std + self.momentum * x.std()
                b = torch.mean(torch.abs(x - x.mean()))
                self.running_b = (1 - self.momentum) * self.running_b + self.momentum * b
                self.running_min = torch.min(self.running_min, x.min())
                self.running_max = torch.max(self.running_max, x.max())
    
    def get_alpha_laplace(self, x):
        """Compute optimal alpha for Laplace distribution."""
        # ACIQ supports 2-16 bits
        assert self.bit_width >= 2, f"ACIQ does not support {self.bit_width}-bit quantization. Minimum is 2-bit."
        
        if self.running_b is not None and self.use_stats:
            b = self.running_b.item()
        else:
            b = torch.mean(torch.abs(x - x.mean())).item()
        # For >16 bits, extrapolate using linear approximation
        if self.bit_width > 16:
            # Extrapolate: alpha_laplace[16] = 20.61, approximate slope ~1.4 per bit
            alpha_coef = 20.61 + (self.bit_width - 16) * 1.4
        else:
            alpha_coef = self.alpha_laplace.get(self.bit_width, 5.03)  # default to 4-bit
        return alpha_coef * b
    
    def get_alpha_gaus(self, x):
        """Compute optimal alpha for Gaussian distribution."""
        # ACIQ supports 2-16 bits
        assert self.bit_width >= 2, f"ACIQ does not support {self.bit_width}-bit quantization. Minimum is 2-bit."
        
        if self.running_std is not None and self.use_stats:
            std = self.running_std.item()
        else:
            std = x.std().item()
        
        # Estimate std from range if needed (ACIQ formula)
        if self.running_min is not None and self.running_max is not None:
            range_val = (self.running_max - self.running_min).item()
            N = x.numel()
            gaussian_const = (0.5 * 0.35) * (1 + (math.pi * math.log(4)) ** 0.5)
            std_est = (range_val * gaussian_const) / ((2 * math.log(max(N, 1))) ** 0.5)
            std = std_est if std < 1e-6 else std
        
        # For >16 bits, extrapolate using linear approximation
        if self.bit_width > 16:
            # Extrapolate: alpha_gaus[16] = 6.09, approximate slope ~0.25 per bit
            alpha_coef = 6.09 + (self.bit_width - 16) * 0.25
        else:
            alpha_coef = self.alpha_gaus.get(self.bit_width, 2.55)  # default to 4-bit
        return alpha_coef * std
    
    def get_optimal_alpha(self, x):
        """Get optimal clipping alpha based on clip_type."""
        if self.clip_type == 'laplace':
            return self.get_alpha_laplace(x)
        elif self.clip_type == 'gaus':
            return self.get_alpha_gaus(x)
        elif self.clip_type == 'mix':
            # Choose between Laplace and Gaussian based on MSE estimate
            alpha_laplace = self.get_alpha_laplace(x)
            alpha_gaus = self.get_alpha_gaus(x)
            b = self.running_b.item() if self.running_b is not None else torch.mean(torch.abs(x - x.mean())).item()
            std = self.running_std.item() if self.running_std is not None else x.std().item()
            
            mse_laplace = 2 * (b ** 2) * np.exp(-alpha_laplace / b) + ((alpha_laplace ** 2) / (3 * 2 ** (2 * self.bit_width)))
            mse_gaus = self._mse_gaus(std, alpha_gaus, num_bits=self.bit_width)
            return alpha_laplace if mse_laplace < mse_gaus else alpha_gaus
        else:
            # No clipping - use full range
            if self.running_min is not None and self.running_max is not None:
                return (self.running_max - self.running_min).item() / 2.0
            else:
                return (x.max() - x.min()).item() / 2.0
    
    @staticmethod
    def _mse_gaus(sigma, alpha, num_bits=8):
        """Estimate MSE for Gaussian distribution."""
        clipping_err = (sigma ** 2 + (alpha ** 2)) * (1 - math.erf(alpha / (sigma * np.sqrt(2.0)))) - \
                       np.sqrt(2.0 / math.pi) * alpha * sigma * (np.e ** ((-1) * (0.5 * (alpha ** 2)) / sigma ** 2))
        quant_err = (alpha ** 2) / (3 * (2 ** (2 * num_bits)))
        return clipping_err + quant_err
    
    def forward(self, x):
        """Quantize activations using ACIQ."""
        # ACIQ supports 2-16 bits (and beyond with extrapolation)
        assert self.bit_width >= 2, f"ACIQ does not support {self.bit_width}-bit quantization. Minimum is 2-bit."
        
        if self.training:
            self.update_stats(x)
        
        alpha = self.get_optimal_alpha(x)
        x_clipped = torch.clamp(x, -alpha, alpha)
        
        # Uniform quantization (ACIQ supports 2-8 bits)
        qmax = 2 ** (self.bit_width - 1) - 1  # e.g., 127 for 8-bit
        scale = alpha / qmax
        x_scaled = x_clipped / scale
        x_rounded = torch.round(x_scaled) * scale
        x_fake = x_clipped + (x_rounded - x_clipped).detach()  # STE
        return x_fake


# ============================================================================
# Mean Center
# ============================================================================
class MeanCenter(nn.Module):
    """Subtracts the batch mean per feature during training."""
    def __init__(self, momentum: float = 0.1, eps: float = 1e-5):
        super().__init__()
        self.momentum = momentum
        self.eps = eps
        self.register_buffer("running_mean", None)
        self.last_mu = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.running_mean is None:
            with torch.no_grad():
                rm = x.mean(dim=0)
            self.running_mean = rm.detach().clone()

        if self.training:
            mu = x.mean(dim=0, keepdim=True)
            with torch.no_grad():
                self.running_mean.mul_(1 - self.momentum).add_(self.momentum * mu.squeeze(0))
            self.last_mu = mu.detach()
            return x - mu
        else:
            mu = self.running_mean.view(1, -1)
            return x - mu

# ============================================================================
# E8 VQ Linear Layer with ACIQ Activations
# ============================================================================
class GlobalScaledHNLQLinearQAT_ACIQ(HNLQLinearQAT):
    """
    Approach 1: E8 VQ for weights + ACIQ for activations
    """
    
    def __init__(self, *args, a_init: float = 1.0, b_init: float = 0.0, 
                 use_aciq_activations=True, act_clip_type='laplace', act_bit_width=8, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer('beta_global', torch.tensor(1.0))
        self.register_buffer('alpha_global', torch.tensor(0.0))
        
        # ACIQ for activations (replace LSQ if enabled)
        if use_aciq_activations and self.quantize_activations:
            self.actq = None  # Remove LSQ
            self.aciq_act = ACIQActivationQuantizer(
                bit_width=act_bit_width,
                clip_type=act_clip_type,
                use_stats=True
            )
            # Print activation quantization info (only once per layer type, to avoid spam)
            if not hasattr(GlobalScaledHNLQLinearQAT_ACIQ, '_printed_act_info'):
                print(f"  [Layer Info] Activation quantization: {act_bit_width} bits (ACIQ, {act_clip_type})")
                GlobalScaledHNLQLinearQAT_ACIQ._printed_act_info = True
        else:
            self.aciq_act = None

    def _quantize_weights(self):
        """
        E8 VQ quantization for weights (unchanged from original).
        """
        W = self.weight

        if not self.quantization_enabled:
            if self.enable_diagnostics:
                self._weight_history.append(W.detach().clone())
            return W

        # Global affine scaling
        beta_global = self.beta_global
        alpha_g = self.alpha_global
        W_s = beta_global * W + alpha_g

        # Row-wise scaling
        sigma_r = W_s.std(dim=1, keepdim=True) + 1e-8
        Delta0_base = 0.3#1.5
        qM = float(self.q ** self.M)
        Y_max = Delta0_base * (qM - 1) / 2.0
        C_b = 5.0
        beta_r = Y_max / (C_b * sigma_r)
        W_r = beta_r * W_s

        # E8 quantization
        W_blocks = W_r.view(self.out_features, self.blocks_per_row, self.block_size)
        W_blocks_flat = W_blocks.reshape(-1, self.block_size)
        W_quantized_blocks = ste_quantize(W_blocks_flat, self.quantize_fn, self.q)
        W_quantized_blocks_reshaped = W_quantized_blocks.reshape(
            self.out_features, self.blocks_per_row, self.block_size
        )
        W_quantized = W_quantized_blocks_reshaped.reshape(
            self.out_features, self.in_features
        )

        # Save for overload checking
        self._last_W_aff_q_blocks = W_quantized_blocks.detach()
        self._last_W_aff_q = W_quantized.detach()

        # Undo scalings
        W_quant_undo_row = W_quantized / beta_r
        W_out = (W_quant_undo_row - alpha_g) / (beta_global + 1e-8)

        # Fixed weight clipping
        W_out = torch.clamp(W_out, -self.weight_clip_value, self.weight_clip_value)

        if self.enable_diagnostics:
            self._weight_history.append(W.detach().clone())
            self._quantization_errors.append(torch.norm(W - W_out).item())

        return W_out
    
    def forward(self, x):
        """Forward: E8 weights + ACIQ activations."""
        W_q = self._quantize_weights()
        x = F.linear(x, W_q, self.bias)
        
        # ACIQ for activations
        if self.aciq_act is not None:
            x = self.aciq_act(x)
        elif self.quantize_activations and self.actq is not None:
            x = self.actq(x)  # Fallback to LSQ
        
        return x

# ============================================================================
# E8 Quantization Functions
# ============================================================================
def make_fused_e8_quantize_fn(M: int, tol: float = 1e-6):
    lattice = E8_LATTICE
    def _adapter(x: torch.Tensor, q: int) -> torch.Tensor:
        QL_x = lattice.projection_babai(x)
        g_tilde = x
        for _ in range(M):
            g_bar = lattice.projection_babai(g_tilde)
            g_tilde = g_bar / q
        r = lattice.projection_babai(g_tilde)
        qM = float(q ** M)
        tail = qM * r
        x_hat = QL_x - tail
        return x_hat
    return _adapter

def make_e8_linear_from_linear(
    linear: nn.Linear,
    q: int = 4,
    M: int = 2,
    block_size: int = 8,
    warmup_epochs: int = 5,
    enable_diagnostics: bool = False,
    act_bit_width: int = 8,
    Delta0: float = 1.5,
    a_init: float = 1.0,
    b_init: float = 0.0,
    use_aciq_activations: bool = True,
    act_clip_type: str = 'laplace',
) -> GlobalScaledHNLQLinearQAT_ACIQ:
    in_features = linear.in_features
    out_features = linear.out_features
    assert in_features % block_size == 0, \
        f"in_features={in_features} not divisible by block_size={block_size}"

    e8_linear = GlobalScaledHNLQLinearQAT_ACIQ(
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
        quantize_activations=True,  # Set to True to enable ACIQ activations
        act_bit_width=act_bit_width,
        Delta0=Delta0,
        a_init=a_init,
        b_init=b_init,
        init_method='normal',
        init_kwargs={'mean': 0.0, 'std': 1.0},
        use_aciq_activations=use_aciq_activations,
        act_clip_type=act_clip_type,
    )

    with torch.no_grad():
        e8_linear.weight.copy_(linear.weight)
        if linear.bias is not None and e8_linear.bias is not None:
            e8_linear.bias.copy_(linear.bias)

    return e8_linear

def replace_linears_with_e8(module: nn.Module, q: int = 4, M: int = 2, 
                            use_aciq_activations: bool = True, act_clip_type: str = 'laplace', act_bit_width: int = 8):
    """Recursively replace nn.Linear with GlobalScaledHNLQLinearQAT_ACIQ in BERT."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            e8_linear = make_e8_linear_from_linear(
                child, q=q, M=M, use_aciq_activations=use_aciq_activations, 
                act_clip_type=act_clip_type, act_bit_width=act_bit_width
            )
            setattr(module, name, e8_linear)
        else:
            replace_linears_with_e8(child, q=q, M=M, use_aciq_activations=use_aciq_activations, 
                                   act_clip_type=act_clip_type, act_bit_width=act_bit_width)

# ============================================================================
# Quantized MLP
# ============================================================================
class QuantizedMLP(nn.Module):
    def __init__(
        self,
        input_size=768,
        hidden_size=512,
        output_size=1,
        q=4, M=2,
        warmup_epochs=5,
        use_warmup=True,
        tiling="row",
        block_size=8,
        quantize_activations=True,
        act_bit_width=8,
        weight_clip_value=weight_clip_value,
        enable_diagnostics=False,
        Delta0=1.5,
        use_aciq_activations=True,
        act_clip_type='laplace',
    ):
        super().__init__()
        self.use_warmup = use_warmup
        self.input_padding = (8 - (input_size % 8)) % 8
        
        self.center = MeanCenter(momentum=0.1)
        
        self.fc1 = GlobalScaledHNLQLinearQAT_ACIQ(
            in_features=input_size,
            out_features=hidden_size,
            lattice=E8_LATTICE,
            quantize_fn=make_fused_e8_quantize_fn(M),
            q=q, M=M,
            tiling=tiling,
            block_size=8,
            warmup_epochs=5,
            enable_diagnostics=enable_diagnostics,
            weight_clip_value=weight_clip_value,
            quantize_activations=quantize_activations,
            act_bit_width=act_bit_width,
            Delta0=1.5,
            a_init=1.0,
            b_init=0.0,
            init_method='normal',
            init_kwargs={'mean': 0.0, 'std': 1.0},
            use_aciq_activations=use_aciq_activations,
            act_clip_type=act_clip_type,
        )

        self.fc3 = nn.Linear(hidden_size, output_size)
        nn.init.normal_(self.fc3.weight, mean=0.0, std=1.0)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        if self.input_padding > 0:
            pad = torch.zeros(x.size(0), self.input_padding, device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=1)

        x = self.center(x)
        x = self.relu(self.fc1(x))
        x = self.fc3(x)
        return x

# ============================================================================
# BERT Model
# ============================================================================
class BertWithQuantizedMLP(nn.Module):
    """BERT encoder (E8-quantized) + QuantizedMLP head with ACIQ activations."""
    def __init__(self, bert_model: AutoModel, mlp_head: QuantizedMLP):
        super().__init__()
        self.bert = bert_model
        self.mlp = mlp_head

    def forward(self, input_ids, attention_mask=None, labels=None):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        logits = self.mlp(cls).view(-1)

        loss = None
        if labels is not None:
            labels = labels.float().view(-1)
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(logits, labels)

        return {"loss": loss, "logits": logits}

def build_model(q=2, M=2, use_aciq_activations=True, act_clip_type='laplace', act_bit_width=None):
    bert_model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(bert_model_name)

    # Calculate effective weight bits
    weight_bits = M * math.log2(q)
    
    # Calculate activation bits to match weight bits (rounded to nearest integer, clamped to [2, 16])
    # Note: ACIQ supports 2-16 bits (and beyond with extrapolation)
    if act_bit_width is None:
        act_bit_width = int(round(weight_bits))
        act_bit_width = max(2, min(16, act_bit_width))  # Clamp to valid ACIQ range [2, 16]
    
    print(f"\n{'='*70}")
    print(f"BUILDING MODEL WITH QUANTIZATION CONFIGURATION:")
    print(f"{'='*70}")
    print(f"  Weight Quantization (E8 VQ):")
    print(f"    - q = {q}, M = {M}")
    print(f"    - Effective bits = M * log2(q) = {M} * log2({q}) = {weight_bits:.3f} bits")
    print(f"  Activation Quantization (ACIQ):")
    print(f"    - Enabled: {use_aciq_activations}")
    if use_aciq_activations:
        print(f"    - Bit width: {act_bit_width} bits (matched to weight bits: {weight_bits:.3f} -> {act_bit_width})")
        print(f"    - Clip type: {act_clip_type}")
    else:
        print(f"    - Not enabled")
    print(f"{'='*70}\n")

    bert_encoder = AutoModel.from_pretrained(bert_model_name)
    replace_linears_with_e8(bert_encoder, q=q, M=M, use_aciq_activations=use_aciq_activations, 
                            act_clip_type=act_clip_type, act_bit_width=act_bit_width)

    mlp_head = QuantizedMLP(
        input_size=768,
        hidden_size=512,
        output_size=1,
        q=q, M=M,
        warmup_epochs=5,
        use_warmup=True,
        tiling="row",
        block_size=8,
        quantize_activations=True,  # Set to True to enable ACIQ activations
        act_bit_width=act_bit_width,
        weight_clip_value=weight_clip_value,
        enable_diagnostics=False,
        Delta0=1.5,
        use_aciq_activations=use_aciq_activations,
        act_clip_type=act_clip_type,
    )

    model = BertWithQuantizedMLP(bert_encoder, mlp_head)
    model.to(device)

    # Enable quantization for all HNLQ layers
    quantized_layer_count = 0
    for m in model.modules():
        if isinstance(m, GlobalScaledHNLQLinearQAT_ACIQ):
            m.quantization_enabled = True
            quantized_layer_count += 1
    
    print(f"  Total quantized layers (BERT + MLP): {quantized_layer_count}")
    print(f"{'='*70}\n")

    return model, tokenizer

# ============================================================================
# Evaluation and Training Functions (same as reference)
# ============================================================================
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
                labels_float = labels.float()
                loss = criterion(logits.view(-1), labels_float.view(-1))
                total_loss += loss.item()
            
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels).numpy()

    probs = torch.sigmoid(all_logits).numpy()
    # Ensure probs is an array (handle scalar case)
    probs = np.atleast_1d(probs)
    preds = (probs >= threshold).astype(np.float32)
    # Ensure preds is an array (handle scalar case)
    preds = np.atleast_1d(preds)

    # Flatten arrays to 1D for sklearn metrics (handles both (N,) and (N,1) shapes)
    all_labels = all_labels.flatten()
    preds = preds.flatten()
    
    # Remove any NaN or inf values and ensure valid binary classification format
    all_labels = np.nan_to_num(all_labels, nan=0.0, posinf=1.0, neginf=0.0)
    preds = np.nan_to_num(preds, nan=0.0, posinf=1.0, neginf=0.0)
    
    # Ensure labels are in valid range [0, 1] and convert to binary integers
    # Round to nearest integer to handle any floating point precision issues
    all_labels = np.round(np.clip(all_labels, 0, 1)).astype(np.int32)
    preds = np.round(np.clip(preds, 0, 1)).astype(np.int32)
    
    # Final validation: ensure arrays contain only 0 or 1
    assert np.all((all_labels == 0) | (all_labels == 1)), "Labels must be binary (0 or 1)"
    assert np.all((preds == 0) | (preds == 1)), "Predictions must be binary (0 or 1)"

    acc = accuracy_score(all_labels, preds) * 100.0
    prec = precision_score(all_labels, preds, zero_division=0)
    rec = recall_score(all_labels, preds, zero_division=0)
    f1 = f1_score(all_labels, preds, zero_division=0) * 100.0

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
        raise ValueError(f"Could not read CSV file {csv_path}")
    
    text_col = None
    for col in ['v2', 'text', 'review', 'statement', 'sentence', 'comment']:
        if col in df.columns:
            text_col = col
            break
    
    if text_col is None:
        raise ValueError(f"Could not find text column in CSV. Available columns: {df.columns.tolist()}")
    
    label_col = None
    for col in ['v1', 'label', 'sentiment', 'class', 'target']:
        if col in df.columns:
            label_col = col
            break
    
    if label_col is None:
        raise ValueError(f"Could not find label column in CSV. Available columns: {df.columns.tolist()}")
    
    df["text"] = df[text_col].fillna("").astype(str)
    
    labels_raw = df[label_col]
    if labels_raw.dtype == 'object' or labels_raw.dtype.name == 'string':
        label_mapping = {
            'positive': 1, 'negative': 0, 
            'pos': 1, 'neg': 0,
            'spam': 1, 'ham': 0,
            '1': 1, '0': 0
        }
        df["label"] = labels_raw.str.lower().str.strip().map(label_mapping).fillna(-1)
        if (df["label"] == -1).any():
            unique_labels = labels_raw.unique()
            raise ValueError(f"Unknown label values found: {unique_labels}")
    else:
        df["label"] = labels_raw.astype(int)
    
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
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def check_overload_per_epoch(model, q, M, max_bert_layers=5):
    """Check overload/residual counts for MLP and BERT layers."""
    with torch.no_grad():
        mlp_overload_count = 0
        mlp_total_blocks = 0
        bert_overload_count = 0
        bert_total_blocks = 0
        
        # Check MLP head
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
        
        # Check BERT encoder layers
        if hasattr(model, 'bert'):
            bert_quantized_layers = []
            for module in model.bert.modules():
                if isinstance(module, GlobalScaledHNLQLinearQAT_ACIQ):
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

def train_end_to_end(csv_path, q=2, M=2, epochs=10, lr=2e-5, batch_size=32, 
                     use_aciq_activations=True, act_clip_type='laplace', act_bit_width=None):
    total_start_time = time.time()
    
    # Calculate activation bits to match weight bits if not specified
    # Note: ACIQ supports 2-16 bits (and beyond with extrapolation)
    weight_bits = M * math.log2(q)
    if act_bit_width is None:
        act_bit_width = int(round(weight_bits))
        act_bit_width = max(2, min(16, act_bit_width))  # Clamp to valid ACIQ range [2, 16]
    
    model, tokenizer = build_model(q=q, M=M, use_aciq_activations=use_aciq_activations, 
                                   act_clip_type=act_clip_type, act_bit_width=act_bit_width)
    train_loader, val_loader, test_loader = prepare_dataloaders(
        csv_path=csv_path,
        tokenizer=tokenizer,
        max_len=128,
        batch_size=batch_size,
    )

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    train_losses = []
    val_losses = []
    epoch_overload_stats = []

    for epoch in range(epochs):
        epoch_start_time = time.time()
        
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
        
        train_losses.append(avg_loss)
        val_losses.append(val_metrics['loss'])
        
        train_metrics = evaluate(model, train_loader, device)
        
        overload_stats = check_overload_per_epoch(model, q=q, M=M, max_bert_layers=5)
        overload_stats['epoch'] = epoch + 1
        epoch_overload_stats.append(overload_stats)
        
        test_metrics = evaluate(model, test_loader, device)
        
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        
        # Detailed epoch printout with overload stats and block counts
        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train F1={train_metrics['f1']:.2f}% | "
            f"Test F1={test_metrics['f1']:.2f}% | "
            f"Overload: Total={overload_stats['total_overload_pct']:.2f}% "
            f"(MLP={overload_stats['mlp_overload_pct']:.2f}%, "
            f"BERT={overload_stats['bert_overload_pct']:.2f}%) "
            f"[blocks: MLP {overload_stats['mlp_overload_count']}/{overload_stats['mlp_total_blocks']}, "
            f"BERT {overload_stats['bert_overload_count']}/{overload_stats['bert_total_blocks']}]"
        )

    final_train_metrics = evaluate(model, train_loader, device)
    final_test_metrics = evaluate(model, test_loader, device)
    final_overload_stats = check_overload_per_epoch(model, q=q, M=M, max_bert_layers=5)
    
    weight_bits = M * math.log2(q)
    print(f"\n{'='*70}")
    print(f"FINAL RESULTS (q={q}, M={M}, Delta0_base=1.5)")
    print(f"{'='*70}")
    print(f"Train F1: {final_train_metrics['f1']:.2f}%")
    print(f"Test F1: {final_test_metrics['f1']:.2f}%")
    print(f"MLP Overload: {final_overload_stats['mlp_overload_pct']:.2f}%")
    print(f"BERT Overload: {final_overload_stats['bert_overload_pct']:.2f}%")
    print(f"Total Overload: {final_overload_stats['total_overload_pct']:.2f}%")
    print(f"{'='*70}\n")

    results_summary = {
        'q': q,
        'M': M,
        'train_f1': final_train_metrics['f1'],
        'test_f1': final_test_metrics['f1'],
        'mlp_overload_pct': final_overload_stats['mlp_overload_pct'],
        'bert_overload_pct': final_overload_stats['bert_overload_pct'],
        'total_overload_pct': final_overload_stats['total_overload_pct']
    }

    return model, results_summary

if __name__ == "__main__":
    # Locate dataset directory: strictly use 'DataSet' under the current code directory
    code_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(code_dir, "DataSet")
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    # Collect all CSV files in the dataset directory
    csv_files = sorted(
        f for f in os.listdir(data_dir)
        if f.lower().endswith(".csv")
    )
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in dataset directory: {data_dir}")

    # Define configurations: (weight_bits, q, M, activation_bits_list)
    # Example: Weight bits = 2 -> (q=4, M=1), activations = [2]
    configurations = [
        (2, 4, 1, [2]),
        # Add more configurations as needed, e.g.:
        # (3, 8, 1, [2, 3, 4, 8]),
    ]

    # Activation quantization configuration
    act_clip_type = 'mix'  # 'laplace', 'gaus', or 'mix'

    for csv_filename in csv_files:
        csv_path = os.path.join(data_dir, csv_filename)
        dataset_name = os.path.splitext(csv_filename)[0]

        total_configs = sum(len(acts) for _, _, _, acts in configurations)

        print(f"\n{'='*70}")
        print(f"RUNNING {total_configs} CONFIGURATIONS FOR {dataset_name} DATASET (Delta0_base=1.5)")
        print(f"{'='*70}\n")

        all_results = []
        config_idx = 0

        for target_weight_bits, q, M, act_bit_widths in configurations:
            weight_bits = M * math.log2(q)

            print(f"\n{'='*70}")
            print(f"WEIGHT BITS = {target_weight_bits} (q={q}, M={M}, effective={weight_bits:.3f} bits)")
            print(f"Testing with activation bits: {act_bit_widths}")
            print(f"{'='*70}\n")

            for act_bit_width in act_bit_widths:
                # Reset seeds for each configuration to ensure reproducibility
                torch.manual_seed(42)
                np.random.seed(42)
                random.seed(42)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(42)

                config_idx += 1
                print(f"\n{'='*70}")
                print(f"CONFIGURATION {config_idx}/{total_configs}: q={q}, M={M}, bits={target_weight_bits}, Delta0_base=1.5")
                print(f"{'='*70}\n")

                model, results_summary = train_end_to_end(
                    csv_path, q=q, M=M, epochs=2, lr=2e-5, batch_size=32,
                    use_aciq_activations=True, act_clip_type=act_clip_type, act_bit_width=act_bit_width
                )

                # Add additional metadata to results
                results_summary['dataset'] = dataset_name
                results_summary['weight_bits'] = weight_bits
                results_summary['act_bit_width'] = act_bit_width
                results_summary['act_clip_type'] = act_clip_type
                all_results.append(results_summary)

                print(f"\n{'='*70}")
                print(f"COMPLETED CONFIGURATION {config_idx}/{total_configs}: q={q}, M={M}, bits={target_weight_bits}")
                print(f"{'='*70}\n")

                if torch.cuda.is_available():
                    del model
                    torch.cuda.empty_cache()
                    gc.collect()

        print(f"\n{'='*70}")
        print(f"SUMMARY OF ALL CONFIGURATIONS - {dataset_name} DATASET (Delta0_base=1.5)")
        print(f"{'='*70}")
        print(f"{'Config':<15} {'Dataset':<10} {'Train F1':<12} {'Test F1':<12} {'MLP Overload':<15} {'BERT Overload':<15} {'Total Overload':<15}")
        print(f"{'-'*110}")
        for result in all_results:
            weight_bits = result.get('weight_bits', result['M'] * math.log2(result['q']))
            act_bit_width = result.get('act_bit_width', int(round(weight_bits)))
            config_str = f"q={result['q']}, M={result['M']}"
            print(f"{config_str:<15} {result['dataset']:<10} "
                  f"{result['train_f1']:>10.2f}% {result['test_f1']:>10.2f}% "
                  f"{result['mlp_overload_pct']:>13.2f}% {result['bert_overload_pct']:>13.2f}% "
                  f"{result['total_overload_pct']:>13.2f}%")
        print(f"{'='*110}\n")
        print(f"{'='*70}")
        print(f"ALL {total_configs} CONFIGURATIONS COMPLETED")
        print(f"{'='*70}\n")
    
    # Save results to CSV
    if all_results:
        # Create DataFrame with all results
        df_results = pd.DataFrame(all_results)
        
        # Reorder columns for better readability
        column_order = [
            'q', 'M', 'weight_bits', 'act_bit_width', 'act_clip_type',
            'dataset', 'train_f1', 'test_f1',
            'mlp_overload_pct', 'bert_overload_pct', 'total_overload_pct'
        ]
        
        # Ensure all columns exist
        for col in column_order:
            if col not in df_results.columns:
                if col == 'weight_bits':
                    df_results['weight_bits'] = df_results['M'] * np.log2(df_results['q'])
                elif col == 'act_bit_width':
                    # act_bit_width is already set in results_summary, no need to recalculate
                    pass
                elif col == 'act_clip_type':
                    df_results['act_clip_type'] = act_clip_type
        
        # Reorder columns
        existing_cols = [col for col in column_order if col in df_results.columns]
        remaining_cols = [col for col in df_results.columns if col not in existing_cols]
        df_results = df_results[existing_cols + remaining_cols]
        
        # Generate CSV filename with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_output_path = f"Bert_ACIQ_Approach1_mix_results_{dataset_name}_{timestamp}_all_activations.csv"
        
        # Save to CSV
        df_results.to_csv(csv_output_path, index=False)
        print(f"\n{'='*70}")
        print(f"RESULTS SAVED TO CSV:")
        print(f"  File: {csv_output_path}")
        print(f"  Total configurations: {len(df_results)}")
        print(f"  Columns: {', '.join(df_results.columns.tolist())}")
        print(f"{'='*70}\n")

