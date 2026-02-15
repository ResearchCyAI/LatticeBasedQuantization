"""
E8-specific layer utilities and convenience functions.

This module provides E8-specific helper functions and convenience wrappers
for the generic vector quantization layers.
"""

import torch
import torch.nn as nn
from typing import Optional
from .lattice import E8Lattice
from .codecs import e8_encode, e8_decode


def create_e8_hnlq_linear(in_dim, out_dim, device=None, **kwargs):
    """
    Convenience function to create HNLQLinearQAT with E8 lattice.
    
    Args:
        in_dim: Input dimension (must be divisible by 8)
        out_dim: Output dimension
        device: Device to place the layer on (defaults to cuda if available, else CPU)
        **kwargs: Additional arguments passed to HNLQLinearQAT
                 (Delta0 will be computed automatically from lattice geometry if not provided)
        
    Returns:
        HNLQLinearQAT instance configured for E8 lattice
        
    Example:
        >>> layer = create_e8_hnlq_linear(512, 256, q=4, M=2)  # Delta0 computed automatically
        >>> layer = create_e8_hnlq_linear(512, 256, q=4, M=2, Delta0=1.5)  # Custom Delta0
    """
    # Import here to avoid circular imports
    from ..layers import HNLQLinearQAT
    from .codecs import e8_quantize
    
    # Ensure input dimension is divisible by 8
    if in_dim % 8 != 0:
        raise ValueError(f"Input dimension {in_dim} must be divisible by 8 for E8 lattice")
    
    # Default to cuda if available and device not specified
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create E8 lattice
    lattice = E8Lattice(device=device)
    
    # Set default E8 parameters with QAT defaults
    defaults = {
        'lattice': lattice,
        'quantize_fn': e8_quantize,  # Pass E8-specific quantizer
        'block_size': 8,
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