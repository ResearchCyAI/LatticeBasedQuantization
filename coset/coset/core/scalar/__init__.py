"""
Scalar Quantization Optimized Module

This module provides optimized implementations for scalar quantization,
including configuration, quantizers, layers, and CUDA acceleration.
"""

# config.py was removed
# quantizers.py was removed
from .layers import create_lsq_scalar_linear
from .codecs import (
    lsq_scalar_quantize,
    lsq_scalar_encode,
    lsq_scalar_decode,
    lsq_scalar_quantize_batch,
    get_lsq_scalar_bounds,
    get_lsq_scalar_effective_bits,
    get_lsq_scalar_quantization_levels,
)
# cuda.py was removed

__all__ = [
    # Configuration (removed)
    
    # Quantizers (removed)
    
    # Layers
    'create_lsq_scalar_linear',
    
    # LSQ Codecs
    'lsq_scalar_quantize',
    'lsq_scalar_encode',
    'lsq_scalar_decode',
    'lsq_scalar_quantize_batch',
    'get_lsq_scalar_bounds',
    'get_lsq_scalar_effective_bits',
    'get_lsq_scalar_quantization_levels',
    
    # CUDA (removed)
]
