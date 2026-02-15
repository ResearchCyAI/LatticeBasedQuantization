"""
Scalar Quantized Linear Layers

This module provides scalar quantized linear layers with straight-through
estimation for quantization-aware training.
"""

import torch
import torch.nn as nn
from typing import Optional
# config.py was removed
# quantizers.py was removed
from .codecs import lsq_scalar_quantize


# ScalarStraightThroughQuantize and ScalarQLinear classes removed due to missing config.py and quantizers.py

def create_lsq_scalar_linear(
    in_dim: int,
    out_dim: int,
    q: int = 4,
    M: int = 2,
    tiling: str = 'row',
    block_size: Optional[int] = None,
    device: Optional[torch.device] = None,
    **kwargs
) -> nn.Module:
    """
    Create HNLQLinearQAT with LSQ scalar quantization.
    
    This convenience function creates an HNLQLinearQAT layer configured for LSQ scalar
    quantization, which provides learnable per-row or per-tile scaling parameters.
        
        Args:
        in_dim: Input dimension
        out_dim: Output dimension
        q: Base quantization levels (alphabet size)
        M: Hierarchical levels (effective bits = floor(M * log2(q)))
        tiling: Tiling strategy ('row' for per-row scaling, 'block' for per-tile scaling)
        block_size: Size of quantization blocks (defaults to 1 for scalar quantization)
        device: Device to place the layer on (defaults to cuda if available, else CPU)
        **kwargs: Additional arguments passed to HNLQLinearQAT
        
    Returns:
        HNLQLinearQAT instance configured for LSQ scalar quantization
        
    Example:
        >>> # Create 4-bit LSQ scalar linear layer with per-row scaling
        >>> layer = create_lsq_scalar_linear(512, 128, q=4, M=2, tiling='row')
        >>> 
        >>> # Create 8-bit LSQ scalar linear layer with per-tile scaling
        >>> layer = create_lsq_scalar_linear(512, 128, q=2, M=8, tiling='block', block_size=8)
    """
    # Import here to avoid circular imports
    from ..layers import HNLQLinearQAT
    
    # Default to cuda if available and device not specified
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set default block_size for scalar quantization
    if block_size is None:
        block_size = 1  # Scalar quantization uses block_size=1
    
    # Validate tiling mode
    if tiling not in ['row', 'block']:
        raise ValueError(f"tiling must be 'row' or 'block', got {tiling}")
    
    # Validate block_size for block tiling
    if tiling == 'block' and in_dim % block_size != 0:
        raise ValueError(f"Input dimension {in_dim} must be divisible by block_size {block_size} for block tiling")
    
    # Set default LSQ scalar parameters
    defaults = {
        'G': torch.eye(1, device=device),  # Dummy - not used for scalar quantization
        'Ginv': torch.eye(1, device=device),  # Dummy - not used for scalar quantization
        'quantize_fn': lsq_scalar_quantize,  # Pass LSQ scalar quantizer
        'lattice_type': 'LSQ_Scalar',
        'q': q,
        'M': M,
        'tiling': tiling,
        'block_size': block_size,
        'warmup_epochs': 0,  # Default: no warmup
        'enable_diagnostics': False,  # Default: no diagnostics
        'weight_clip_value': 2.0,  # Default clipping value
        'theta_trainable': True,  # Default: learnable scale parameters
        'theta_init_value': 0.0,  # Default: start at midpoint of bounds
    }
    
    # Update with provided kwargs
    defaults.update(kwargs)
    
    return HNLQLinearQAT(in_dim, out_dim, **defaults)
