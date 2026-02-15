"""
D4 lattice implementation for vector quantization.

This module provides the D4 lattice implementation with known geometric properties.
"""

from .lattice import D4Lattice
from .layers import create_d4_hnlq_linear

__all__ = ['D4Lattice', 'create_d4_hnlq_linear']
