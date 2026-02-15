"""
Optimized Lattice Modules

This package provides lattice-specific optimized implementations for different
lattice types (E8, D4, etc.) and scalar quantization with specialized codecs, 
layers, and CUDA acceleration.
"""

# E8 optimized module
from .e8 import (
    e8_encode,
    e8_decode,
    e8_quantize,
    create_e8_hnlq_linear,
    # LUT classes are experimental and not supported
    # E8OneSidedLUT,
    # E8TwoSidedLUT,
    # E8LUTManager,
)

# Vector quantization layers
from .layers import HNLQLinear, HNLQLinearQAT, LSQActivation, get_generators, ste_quantize

# Scalar quantization layers (removed - use core.scalar module instead)

# Note: Scalar configs and utilities available in core.scalar submodule

# TODO: Add D4 optimized module when implemented
# from .d4 import (
#     D4Config,
#     get_d4_config,
#     d4_quantize,
#     batch_d4_quantize,
#     d4_encode,
#     d4_decode,
#     batch_d4_encode,
#     batch_d4_decode,
#     D4QLinear,
#     D4StraightThroughQuantize,
#     D4OneSidedLUT,
#     D4TwoSidedLUT,
#     D4LUTManager,
# )

__all__ = [
    # E8 exports
    'e8_encode',
    'e8_decode',
    'e8_quantize',
    'create_e8_hnlq_linear',
    
    # Vector quantization layers
    'HNLQLinear',
    'HNLQLinearQAT',
    'LSQActivation',
    'get_generators',
    'ste_quantize',
    
    # Scalar quantization layers (removed - use core.scalar module instead)
]
