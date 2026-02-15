"""
E8 Lattice Optimized Module

This module provides optimized implementations specifically for E8 lattice quantization,
including configuration, codecs, layers, lookup tables, and CUDA acceleration.
"""

from .lattice import E8Lattice
from .codecs import (
    e8_encode,
    e8_decode,
    e8_quantize,
    E8Decoder,
    DecodingMethod
)
from .layers import create_e8_hnlq_linear
# LUT classes are experimental and not supported
# from .lut import E8OneSidedLUT, E8TwoSidedLUT, E8LUTManager

__all__ = [
    # Lattice
    'E8Lattice',
    
    # Codecs
    'e8_encode',
    'e8_decode',
    'e8_quantize',
    'E8Decoder',
    'DecodingMethod',
    
    # E8-specific convenience functions
    'create_e8_hnlq_linear',
    
    # Lookup Tables (experimental - not supported)
    # 'E8OneSidedLUT',
    # 'E8TwoSidedLUT', 
    # 'E8LUTManager',
]
