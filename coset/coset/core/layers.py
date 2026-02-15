"""
Vector Quantization Layers with QAT Cold Start

This module provides vector quantization layers with hierarchical nested lattice quantization (HNLQ)
and learnable scale quantization (LSQ) for quantization-aware training with cold start support.

Features:
- Cold start (warmup) training: Train without quantization for specified epochs
- Weight diagnostics: Quantile statistics, quantization error analysis
- Epoch tracking: Automatic quantization enable/disable based on epoch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Dict, Any, List, Tuple
from dataclasses import dataclass
import numpy as np

def ste_round(x):
    """Straight-through estimator for rounding."""
    return (x - x.detach()) + torch.round(x).detach()

def ste_clip(x, min_val, max_val):
    """Straight-through estimator for clipping."""
    return (x - x.detach()) + torch.clamp(x, min_val, max_val).detach()


def get_generators(lattice) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Get generator matrices from a lattice instance.
    
    This is a generic helper that works with any Lattice instance.
    For specific lattices, import from their modules:
    - from coset.core.e8 import E8Lattice
    - from coset.core.d4 import D4Lattice (when available)
    
    Args:
        lattice: A Lattice instance (e.g., E8Lattice, D4Lattice)
        
    Returns:
        Tuple of (G, G_inv) where:
        - G: Generator matrix of shape [d, d]
        - G_inv: Inverse of G of shape [d, d]
        
    Example:
        >>> from coset.core.e8 import E8Lattice
        >>> from coset.core.layers import get_generators
        >>> lattice = E8Lattice()
        >>> G, Ginv = get_generators(lattice)
    """
    return lattice.get_generators()


# Note: Lattice-specific quantization functions are in their respective modules:
# - from coset.core.e8 import e8_quantize
# - from coset.core.d4 import d4_quantize (when available)


# ---------- Straight-Through Estimator (STE) Functions ----------

class StraightThroughQuantize(torch.autograd.Function):
    """
    Straight-through estimator for quantization.
    Forward pass: quantizes the input
    Backward pass: passes gradients through unchanged
    """
    @staticmethod
    def forward(ctx, x, quantize_fn, q):
        """
        Forward pass: quantize the input.
        
        Args:
            x: Input tensor to quantize
            quantize_fn: Lattice-specific quantization function
            q: Quantization parameter
        """
        # Store for backward pass
        ctx.quantize_fn = quantize_fn
        ctx.q = q
        ctx.save_for_backward(x)
        
        # Quantize
        return quantize_fn(x, q=q)
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: pass gradients through unchanged (straight-through).
        """
        # Get original input
        x, = ctx.saved_tensors
        
        # Straight-through: pass gradients through unchanged
        return grad_output, None, None


def ste_quantize(x, quantize_fn, q):
    """
    Straight-through quantization wrapper.
    
    Args:
        x: Input tensor
        quantize_fn: Lattice-specific quantization function
        q: Quantization parameter
        
    Returns:
        Quantized tensor with STE gradients
    """
    return StraightThroughQuantize.apply(x, quantize_fn, q)




# ---------- Activation Quantizer (LSQ-A style, per-tensor) ----------

class LSQActivation(nn.Module):
    """
    Learnable per-tensor activation quantizer with STE.
    
    Uses learnable clip (alpha) for symmetric uniform quantization in [-alpha, +alpha].
    """
    
    def __init__(self, bit_width: int = 8, init_alpha: float = 1.0):
        super().__init__()
        self.bit_width = bit_width
        self.qmax = 2 ** (bit_width - 1) - 1  # e.g., 127 for 8-bit
        self.alpha = nn.Parameter(torch.tensor(init_alpha))
    
    def forward(self, x):
        # symmetric uniform quantization in [-alpha, +alpha]
        alpha = torch.relu(self.alpha) + 1e-8
        x_clipped = ste_clip(x, -alpha, alpha)
        scale = alpha / self.qmax
        y = x_clipped / scale
        yq = ste_round(y)
        return yq * scale  # quantized activation (STE)

# ---------- HNLQ Linear with flexible tiling & bias (not quantized) ----------

@dataclass
class HNLQConfig:
    """Configuration for HNLQ Linear layer."""
    q: int = 4
    M: int = 2
    Delta0: float = 1.5
    eta: float = 0.1
    k: int = 1
    tiling: str = 'row'  # 'row' or 'block'
    block_size: int = 8
    quantize_activations: bool = False
    act_bit_width: int = 8
    act_init_alpha: float = 1.0
    init_method: str = 'normal'
    init_kwargs: Dict[str, Any] = None

    def __post_init__(self):
        if self.init_kwargs is None:
            self.init_kwargs = {}
        if self.tiling not in ['row', 'block']:
            raise ValueError(f"tiling must be 'row' or 'block', got {self.tiling}")
        if self.block_size <= 0:
            raise ValueError(f"block_size must be positive, got {self.block_size}")


class HNLQLinearQAT(nn.Module):
    """
    Hierarchical Nested Lattice Quantization Linear Layer with QAT Cold Start.
    
    This layer implements quantization-aware training with hierarchical nested lattice quantization
    for the weights and optional activation quantization using LSQ, with cold start support.
    
    The layer supports:
    - Cold start (warmup) training: Train without quantization for specified epochs
    - Learnable scaling factors (beta) for weights
    - Flexible tiling (row-level or block-level scaling)
    - Optional activation quantization
    - Multiple lattice types (E8, D4, etc.)
    - Various weight initialization methods
    - Weight diagnostics and quantization analysis
    
    Args:
        in_features: Input dimension
        out_features: Output dimension
        lattice: Lattice object (e.g., E8Lattice, D4Lattice) with geometric properties
        quantize_fn: Lattice-specific quantization function (e.g., e8_quantize)
        q: Quantization parameter (alphabet size)
        M: Number of hierarchical levels
        Delta0: Base quantization step size (optional, computed from lattice if None)
        eta: Learning rate for scaling factors
        tiling: Tiling strategy ('row' or 'block')
        block_size: Size of quantization blocks
        quantize_activations: Whether to quantize activations
        act_bit_width: Bit width for activation quantization
        act_init_alpha: Initial alpha value for activation quantization
        init_method: Weight initialization method
        init_kwargs: Additional arguments for weight initialization
        bias: Whether to use bias
        warmup_epochs: Number of epochs to train without quantization (cold start)
        enable_diagnostics: Whether to enable weight diagnostics
        weight_clip_value: Maximum absolute value for weight clipping
        theta_trainable: Whether theta_beta is a learnable parameter (True) or fixed buffer (False)
        theta_init_value: Initial value for theta_beta (used when not trainable)
        rho: Scaling factor for Delta0 computation (default: 0.95)
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        lattice,  # Lattice object instead of G, Ginv
        quantize_fn,
        q: int = 4,
        M: int = 2,
        Delta0: Optional[float] = None,  # Optional, will be computed from lattice if None
        eta: float = 0.1,
        tiling: str = 'row',
        block_size: int = 8,
        quantize_activations: bool = False,
        act_bit_width: int = 8,
        act_init_alpha: float = 1.0,
        init_method: str = 'normal',
        init_kwargs: Optional[Dict[str, Any]] = None,
        bias: bool = True,
        warmup_epochs: int = 0,
        enable_diagnostics: bool = False,
        weight_clip_value: float = 2.0,
        theta_trainable: bool = True,
        theta_init_value: float = 0.0,
        rho: float = 0.95  # Scaling factor for Delta0 computation
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.quantize_fn = quantize_fn
        self.lattice = lattice
        self.lattice_type = lattice.name
        self.q = q
        self.M = M
        self.eta = eta
        self.tiling = tiling
        self.block_size = block_size
        self.quantize_activations = quantize_activations
        self.init_method = init_method
        self.init_kwargs = init_kwargs or {}
        self.theta_trainable = theta_trainable
        self.theta_init_value = theta_init_value
        self.rho = rho
        
        # Compute Delta0 from lattice geometry if not provided
        if Delta0 is None:
            self.Delta0 = lattice.compute_delta0(q, M, rho)
        else:
            self.Delta0 = Delta0
        #print("Delta is", self.Delta0)
        # Store lattice matrices
        G, Ginv = lattice.get_generators()
        self.register_buffer('G', G)
        self.register_buffer('Ginv', Ginv)
        
        # Compute dimensions
        self.blocks_per_row = in_features // block_size
        if in_features % block_size != 0:
            raise ValueError(f"Input dimension {in_features} must be divisible by block_size {block_size}")
        
        # Initialize weights
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self._initialize_weights()
        
        # Initialize bias
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)
        
        # Tiling configuration - theta_beta is single scalar per tile
        tiles = out_features if tiling == 'row' else out_features * (in_features // block_size)
        
        # Initialize theta_beta as parameter or buffer based on trainable flag
        if theta_trainable:
            self.theta_beta = nn.Parameter(torch.full((tiles,), theta_init_value))
        else:
            self.register_buffer('theta_beta', torch.full((tiles,), theta_init_value))
        
        # EMA buffers for adaptive scaling
        self.register_buffer('sigma_ema', torch.ones(tiles))
        self.register_buffer('xmax_ema', torch.ones(tiles))
        self.ema_momentum = 0.99
        self.stat_update_interval = 256
        self.register_buffer('step_count', torch.tensor(0, dtype=torch.long))
        
        # Compute gamma_inv from lattice
        self.register_buffer('gamma_inv', (Ginv.abs().sum(dim=1)).max())
        
        # Initialize activation quantizer if requested
        if quantize_activations:
            self.actq = LSQActivation(bit_width=act_bit_width, init_alpha=act_init_alpha)
        else:
            self.actq = None
        
        # Cold start (QAT warmup) parameters
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0
        self.quantization_enabled = warmup_epochs == 0  # Enable quantization immediately if no warmup
        self.enable_diagnostics = enable_diagnostics
        self.weight_clip_value = weight_clip_value
        
        # Diagnostic storage
        self._weight_history = [] if enable_diagnostics else None
        self._quantization_errors = [] if enable_diagnostics else None
    
    def _initialize_weights(self):
        """Initialize weights using the specified method."""
        if self.init_method == 'normal':
            #print(f"[DEBUG] Initializing weights with normal(mean={self.init_kwargs.get('mean', 0)}, std={self.init_kwargs.get('std', 1)})")
            nn.init.normal_(self.weight, **self.init_kwargs)
        elif self.init_method == 'xavier':
            nn.init.xavier_normal_(self.weight, **self.init_kwargs)
        elif self.init_method == 'kaiming':
            nn.init.kaiming_normal_(self.weight, **self.init_kwargs)
        elif self.init_method == 'zeros':
            nn.init.zeros_(self.weight)
        elif self.init_method == 'ones':
            nn.init.ones_(self.weight)
        elif self.init_method == 'load':
            # Weights will be loaded externally
            pass
        else:
            raise ValueError(f"Unknown init_method: {self.init_method}")
    
    def load_weights(self, weight: torch.Tensor, bias: Optional[torch.Tensor] = None):
        """
        Load weights and bias from tensors.
        
        Args:
            weight: Weight tensor of shape [out_features, in_features]
            bias: Optional bias tensor of shape [out_features]
        """
        if weight.shape != (self.out_features, self.in_features):
            raise ValueError(f"Weight shape {weight.shape} doesn't match expected {(self.out_features, self.in_features)}")
        
        self.weight.data = weight.clone()
        
        if bias is not None:
            if self.bias is None:
                raise ValueError("Cannot load bias: layer was initialized with bias=False")
            if bias.shape != (self.out_features,):
                raise ValueError(f"Bias shape {bias.shape} doesn't match expected {(self.out_features,)}")
            self.bias.data = bias.clone()
    
    def load_from_linear(self, linear_layer: nn.Linear):
        """
        Load weights and bias from an existing nn.Linear layer.
        
        Args:
            linear_layer: nn.Linear layer to copy weights from
        """
        if linear_layer.in_features != self.in_features:
            raise ValueError(f"Input features mismatch: {linear_layer.in_features} vs {self.in_features}")
        if linear_layer.out_features != self.out_features:
            raise ValueError(f"Output features mismatch: {linear_layer.out_features} vs {self.out_features}")
        
        self.load_weights(linear_layer.weight.data, linear_layer.bias.data if linear_layer.bias is not None else None)
    
    def _gather_stats(self, W_blocks):
        """Gather EMA statistics from weight blocks."""
        with torch.no_grad():
            m = self.ema_momentum
            if self.tiling == 'row':
                sigma = W_blocks.std(dim=(1,2)) + 1e-8
                xmax = W_blocks.abs().amax(dim=(1,2)) + 1e-8
                self.sigma_ema[:self.out_features].mul_(m).add_((1-m)*sigma)
                self.xmax_ema[:self.out_features].mul_(m).add_((1-m)*xmax)
            else:  # block
                sigma = W_blocks.std(dim=2).reshape(-1) + 1e-8
                xmax = W_blocks.abs().amax(dim=2).reshape(-1) + 1e-8
                self.sigma_ema.mul_(m).add_((1-m)*sigma)
                self.xmax_ema.mul_(m).add_((1-m)*xmax)
    
    def _bounds(self):
        """Compute adaptive beta bounds from EMA statistics."""
        qM = float(self.q ** self.M)
        ginv = float(self.gamma_inv)
        
        # Minimum bound: ensure quantization range covers eta*sigma
        beta_min = (self.Delta0 / qM) / (self.eta * ginv * self.sigma_ema)
        
        # Maximum bounds: deterministic (xmax) and probabilistic (eta*sigma)
        beta_max_det = (self.Delta0 * (qM - 1)) / (2 * ginv * self.xmax_ema)
        beta_max_prob = (self.Delta0 * (qM - 1)) / (2 * self.eta * ginv * self.sigma_ema)
        
        beta_max = torch.minimum(beta_max_det, beta_max_prob)
        beta_min = torch.minimum(beta_min, beta_max * 0.9)
        
        return beta_min, beta_max
    
    def _quantize_weights(self):
        """Quantize weights using hierarchical nested lattice quantization."""
        #print("Qunatizing")
        W = self.weight  # [out_dim, in_dim]
        
        # Cold start: return original weights if quantization is disabled
        if not self.quantization_enabled:
            if self.enable_diagnostics:
                self._weight_history.append(W.detach().clone())
            return W
        
        # Update statistics periodically
        self.step_count += 1
        if (self.step_count % self.stat_update_interval) == 0:
            # Reshape to blocks temporarily for statistics gathering
            W_blocks_temp = W.view(self.out_features, self.blocks_per_row, self.block_size)
            self._gather_stats(W_blocks_temp.detach())
        
        # Compute adaptive beta bounds
        beta_min, beta_max = self._bounds()
        #print("beta_min", beta_min, "beta_max", beta_max)
        # Compute beta from learnable theta using sigmoid interpolation
        if self.tiling == 'row':
            theta = self.theta_beta[:self.out_features]
            beta_row = beta_min[:self.out_features] + torch.sigmoid(theta) * (beta_max[:self.out_features] - beta_min[:self.out_features])
            beta = beta_row.view(-1, 1)  # [out_features, 1] - one scaling factor per row
        else:  # block
            theta = self.theta_beta
            beta = beta_min + torch.sigmoid(theta) * (beta_max - beta_min)
            beta = beta.view(self.out_features, self.blocks_per_row, 1)
        
        # Apply scaling BEFORE tiling
        if self.tiling == 'row':
            # Scale each row by its corresponding beta
            W_scaled = W * beta  # [out_features, in_features] * [out_features, 1] -> [out_features, in_features]
        else:  # block
            # For block tiling, we need to reshape and apply per-block scaling
            W_blocks = W.view(self.out_features, self.blocks_per_row, self.block_size)
            W_scaled_blocks = W_blocks * beta  # [out_features, blocks_per_row, block_size] * [out_features, blocks_per_row, 1]
            W_scaled = W_scaled_blocks.reshape(self.out_features, self.in_features)
        
        # Now tile the scaled weights for quantization
        W_blocks = W_scaled.view(self.out_features, self.blocks_per_row, self.block_size)
        W_blocks_flat = W_blocks.reshape(-1, self.block_size)  # [out_dim * blocks_per_row, block_size]
        
        # Quantize using STE (Straight-Through Estimator)
        # This allows gradients to flow through quantization during training
        W_quantized_blocks = ste_quantize(W_blocks_flat, self.quantize_fn, self.q)
        
        # Reshape back to block shape (no rescaling - keep in scaled space)
        W_quantized_blocks_reshaped = W_quantized_blocks.reshape(self.out_features, self.blocks_per_row, self.block_size)
        
        # Reshape to original weight shape
        W_quantized = W_quantized_blocks_reshaped.reshape(self.out_features, self.in_features)
        
        # Rescale the weights to the original scale
        if self.tiling == 'row':
            # For row tiling, beta is already [out_features, 1]
            W_quantized = W_quantized / beta
        else:  # block
            # For block tiling, beta is [out_features, blocks_per_row, 1]
            # Reshape beta to [out_features, in_features] for division
            beta_reshaped = beta.reshape(self.out_features, self.in_features)
            W_quantized = W_quantized / beta_reshaped
        
        # Apply weight clipping to stabilize training
        W_quantized = torch.clamp(W_quantized, -self.weight_clip_value, self.weight_clip_value)
        
        # Store diagnostics if enabled
        if self.enable_diagnostics:
            self._weight_history.append(W.detach().clone())
            self._quantization_errors.append(torch.norm(W - W_quantized).item())
        
        return W_quantized
    
    def forward(self, x):
        """Forward pass with optional activation quantization."""
        # Quantize weights
        W_q = self._quantize_weights()
        
        # Linear transformation
        x = F.linear(x, W_q, self.bias)
        
        # Quantize activations if requested
        if self.quantize_activations and self.actq is not None:
            x = self.actq(x)
        
        return x
    
    # Cold start control methods
    # def update_epoch(self, epoch: int):
    #     """Update current epoch and enable quantization if warmup is complete."""
    #     self.current_epoch = epoch
    #     if epoch >= self.warmup_epochs and not self.quantization_enabled:
    #         self.quantization_enabled = True
    #         print(f"Epoch {epoch}: Enabling quantization (warmup complete)")
    def update_epoch(self, epoch: int):
        self.current_epoch = epoch
        if (epoch + 1) >= self.warmup_epochs and not self.quantization_enabled:
            self.quantization_enabled = True
            print(f"Epoch {epoch+1}: Enabling quantization (warmup complete)")

    def enable_quantization(self):
        """Manually enable quantization."""
        self.quantization_enabled = True
        print("Quantization manually enabled")
    
    def disable_quantization(self):
        """Manually disable quantization."""
        self.quantization_enabled = False
        print("Quantization manually disabled")
    
    def is_quantization_enabled(self) -> bool:
        """Check if quantization is currently enabled."""
        return self.quantization_enabled
    
    # Weight diagnostic methods
    def get_weight_quantiles(self, quantiles: List[float] = None) -> torch.Tensor:
        """Get quantiles of the current weight matrix."""
        if quantiles is None:
            quantiles = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        
        W = self.weight.detach().flatten()
        return torch.quantile(W, torch.tensor(quantiles, device=W.device))
    
    def get_weight_statistics(self) -> Dict[str, float]:
        """Get comprehensive statistics of the current weight matrix."""
        W = self.weight.detach()
        return {
            'mean': W.mean().item(),
            'std': W.std().item(),
            'min': W.min().item(),
            'max': W.max().item(),
            'median': W.median().item(),
            'l2_norm': torch.norm(W).item(),
            'l1_norm': torch.norm(W, p=1).item(),
        }
    
    def get_quantization_error(self) -> float:
        """Get L2 quantization error between original and quantized weights."""
        if not self.quantization_enabled:
            return 0.0
        
        W_orig = self.weight.detach()
        W_quant = self._quantize_weights().detach()
        return torch.norm(W_orig - W_quant).item()
    
    def get_scaling_factors(self) -> torch.Tensor:
        """Get current scaling factors (theta_beta)."""
        return self.theta_beta.detach()
    
    def get_quantization_histogram(self, bins: int = 50) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get histogram of quantized weight values."""
        if not self.quantization_enabled:
            W = self.weight.detach()
        else:
            W = self._quantize_weights().detach()
        
        W_flat = W.flatten()
        hist, bin_edges = torch.histogram(W_flat, bins=bins)
        return hist, bin_edges
    
    def get_sparsity_ratio(self, threshold: float = 1e-6) -> float:
        """Get ratio of near-zero weights."""
        W = self.weight.detach()
        near_zero = torch.abs(W) < threshold
        return near_zero.float().mean().item()
    
    def get_effective_bits(self) -> float:
        """Estimate effective bits used after quantization."""
        if not self.quantization_enabled:
            return 32.0  # Assume float32
        
        W_quant = self._quantize_weights().detach()
        unique_values = torch.unique(W_quant)
        return torch.log2(torch.tensor(len(unique_values), dtype=torch.float32)).item()
    
    def compare_weights(self) -> Dict[str, float]:
        """Compare original vs quantized weights."""
        if not self.quantization_enabled:
            return {'error': 0.0, 'relative_error': 0.0, 'cosine_similarity': 1.0}
        
        W_orig = self.weight.detach()
        W_quant = self._quantize_weights().detach()
        
        error = torch.norm(W_orig - W_quant).item()
        relative_error = error / torch.norm(W_orig).item()
        cosine_sim = F.cosine_similarity(W_orig.flatten(), W_quant.flatten(), dim=0).item()
        
        return {
            'error': error,
            'relative_error': relative_error,
            'cosine_similarity': cosine_sim
        }
    
    def get_weight_clipping_stats(self) -> Dict[str, float]:
        """Get statistics about weight clipping."""
        W = self.weight.detach()
        clipped_count = torch.sum((W.abs() > self.weight_clip_value).float()).item()
        total_count = W.numel()
        clipping_ratio = clipped_count / total_count
        
        return {
            'clipping_ratio': clipping_ratio,
            'clipped_count': clipped_count,
            'total_count': total_count,
            'clip_value': self.weight_clip_value,
            'max_weight': W.abs().max().item()
        }
    
    def get_diagnostic_summary(self) -> Dict[str, Any]:
        """Get comprehensive diagnostic summary."""
        summary = {
            'epoch': self.current_epoch,
            'quantization_enabled': self.quantization_enabled,
            'warmup_epochs': self.warmup_epochs,
            'weight_stats': self.get_weight_statistics(),
            'quantization_error': self.get_quantization_error(),
            'effective_bits': self.get_effective_bits(),
            'sparsity_ratio': self.get_sparsity_ratio(),
            'weight_clipping': self.get_weight_clipping_stats(),
        }
        
        if self.quantization_enabled:
            summary['weight_comparison'] = self.compare_weights()
        
        return summary
    
    def export_quantized(self):
        """
        Export quantized weights and metadata for inference.
        
        Returns:
            Dictionary containing quantized weights and configuration
        """
        with torch.no_grad():
            W_q = self._quantize_weights()
            
            metadata = {
                'lattice_type': self.lattice_type,
                'q': self.q,
                'M': self.M,
                'Delta0': self.Delta0,
                'tiling': self.tiling,
                'block_size': self.block_size,
                'quantize_activations': self.quantize_activations,
            }
            
            if self.quantize_activations and self.actq is not None:
                metadata.update({
                    'act_bit_width': self.actq.bit_width,
                    'act_alpha': self.actq.alpha.item(),
                })
            
            return {
                'weight': W_q,
                'bias': self.bias,
                'metadata': metadata
            }
    
    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}, ' \
               f'lattice_type={self.lattice_type}, q={self.q}, M={self.M}, ' \
               f'Delta0={self.Delta0:.4f}, tiling={self.tiling}, block_size={self.block_size}, ' \
               f'quantize_activations={self.quantize_activations}, bias={self.bias is not None}, ' \
               f'theta_trainable={self.theta_trainable}, theta_init_value={self.theta_init_value}, ' \
               f'rho={self.rho}'


# Backward compatibility alias
HNLQLinear = HNLQLinearQAT
