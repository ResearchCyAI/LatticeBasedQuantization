"""
Quantization module for hierarchical nested-lattice quantization.

This module provides the core quantization functionality including:
- Configuration management
- Encoding and decoding functions
- Lookup table operations
- Overload handling
"""

from .params import QuantizationConfig
from .functional import encode, decode, quantize, batch_quantize, mac_modq, accumulate_modq
from .vlut import vLUTManager, build_vlut, vlut_mac_operation, vlut_accumulate_operation
from .elut import eLUTManager, build_elut, elut_mac_operation, elut_accumulate_operation
from .xac import mac, aac, mac_with_dither, mac_with_scaling, adaptive_mac, batch_mac, validate_operations

__all__ = [
    "QuantizationConfig",
    "encode", 
    "decode",
    "quantize",
    "batch_quantize",
    "mac_modq",
    "accumulate_modq",
    "vLUTManager",
    "build_vlut",
    "vlut_mac_operation",
    "vlut_accumulate_operation",
    "eLUTManager",
    "build_elut",
    "elut_mac_operation",
    "elut_accumulate_operation",
    "mac",
    "aac",
    "mac_with_dither",
    "mac_with_scaling",
    "adaptive_mac",
    "batch_mac",
    "validate_operations",
]
