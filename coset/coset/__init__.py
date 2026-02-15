"""
Coset: Hierarchical Nested-Lattice Quantization for PyTorch

A high-performance PyTorch library implementing Hierarchical Nested-Lattice Quantization (HNLQ)
for quantization-aware training (QAT) and distributed training optimization.

Key Features:
- Multi-lattice support (Z², D₄, E₈)
- Hierarchical encoding/decoding with M levels
- CUDA acceleration
- QAT integration
- Distributed training optimization
- Lookup table-based inner products
"""

__version__ = "0.1.0"
__author__ = "Coset Development Team"

# Core imports - Essential user-facing API
from .core import (
    # Main layers
    HNLQLinear,
    # E8 convenience
    create_e8_hnlq_linear,
    # STE utilities
    ste_quantize,
)

# Legacy imports (deprecated - use core modules instead)
import warnings
warnings.warn(
    "Legacy modules (lattices, nn, quant) are deprecated. Use core modules instead.",
    DeprecationWarning,
    stacklevel=2
)
from .legacy.lattices import Lattice, Z2Lattice, D4Lattice, E8Lattice
from .legacy.quant import (
    QuantizationConfig,
    encode as legacy_encode,
    decode as legacy_decode,
    quantize as legacy_quantize
)
from .legacy.nn import QLinear

# Re-export legacy functions with their original names for backward compatibility
# Note: Core 'quantize' takes precedence
encode = legacy_encode
decode = legacy_decode

__all__ = [
    # Core functionality (recommended)
    "HNLQLinear",
    "create_e8_hnlq_linear",
    "ste_quantize",
    
    # Legacy functionality (deprecated)
    "Lattice",
    "Z2Lattice", 
    "D4Lattice",
    "E8Lattice",
    "QuantizationConfig",
    "encode",
    "decode",
    "QLinear",
]
