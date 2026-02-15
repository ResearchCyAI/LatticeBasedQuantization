"""
Lattice implementations for hierarchical nested-lattice quantization.

This module provides implementations of various lattices including:
- Z²: 2D integer lattice (baseline)
- D₄: 4D checkerboard lattice (recommended)
- E₈: 8D optimal lattice (high precision)
"""

from .base import Lattice
from .z2 import Z2Lattice
from .d4 import D4Lattice
from .e8 import E8Lattice

__all__ = ["Lattice", "Z2Lattice", "D4Lattice", "E8Lattice"]
