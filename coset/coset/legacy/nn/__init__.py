"""
Neural network modules for quantization-aware training.

This module provides PyTorch modules that integrate hierarchical nested-lattice
quantization with neural network training.
"""

from .qlinear import QLinear

__all__ = ["QLinear"]
