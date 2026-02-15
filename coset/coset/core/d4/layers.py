"""
D4-specific layer utilities and convenience functions.

This module provides D4-specific helper functions and convenience wrappers
for the generic vector quantization layers.
"""

import torch
import torch.nn as nn
from typing import Optional
from .lattice import D4Lattice


def create_d4_hnlq_linear(in_dim, out_dim, device=None, **kwargs):
    """
    Convenience function to create HNLQLinearQAT with D4 lattice.
    
    Args:
        in_dim: Input dimension (must be divisible by 4)
        out_dim: Output dimension
        device: Device to place the layer on (defaults to cuda if available, else CPU)
        **kwargs: Additional arguments passed to HNLQLinearQAT
                 (Delta0 will be computed automatically from lattice geometry if not provided)
        
    Returns:
        HNLQLinearQAT instance configured for D4 lattice
        
    Example:
        >>> layer = create_d4_hnlq_linear(512, 256, q=4, M=2)  # Delta0 computed automatically
        >>> layer = create_d4_hnlq_linear(512, 256, q=4, M=2, Delta0=1.5)  # Custom Delta0
    """
    # Import here to avoid circular imports
    from ..layers import HNLQLinearQAT
    
    # Ensure input dimension is divisible by 4
    if in_dim % 4 != 0:
        raise ValueError(f"Input dimension {in_dim} must be divisible by 4 for D4 lattice")
    
    # Default to cuda if available and device not specified
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create D4 lattice
    lattice = D4Lattice(device=device)
    
    # Create D4 quantization function
    def d4_quantize(x, q):
        return lattice.projection(x)
    
    # Set default D4 parameters with QAT defaults
    defaults = {
        'lattice': lattice,
        'quantize_fn': d4_quantize,  # Pass D4-specific quantizer
        'block_size': 4,
        'q': 4,
        'M': 2,
        'Delta0': None,  # Will be computed automatically from lattice geometry
        'warmup_epochs': 0,  # Default: no warmup
        'enable_diagnostics': False,  # Default: no diagnostics
        'weight_clip_value': 2.0,  # Default clipping value
        'theta_trainable': True,  # Default: learnable scale parameters
        'theta_init_value': 0.0,  # Default: start at midpoint of bounds
        'rho': 0.95,  # Scaling factor for Delta0 computation
    }
    
    # Update with provided kwargs
    defaults.update(kwargs)
    
    return HNLQLinearQAT(in_dim, out_dim, **defaults)
