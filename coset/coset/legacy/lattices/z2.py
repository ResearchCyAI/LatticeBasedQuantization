"""
Z² lattice implementation.

The Z² lattice is the standard integer lattice in 2D, providing a baseline
implementation for hierarchical nested-lattice quantization.
"""

import torch
from typing import Optional
from .base import Lattice


class Z2Lattice(Lattice):
    """
    Z² lattice implementation.
    
    The Z² lattice consists of all integer points in 2D space.
    This is the simplest lattice and serves as a baseline for comparison.
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        """
        Initialize the Z² lattice with identity generator matrix.
        
        Args:
            device: Device to place the generator matrix on
        """
        G = torch.eye(2, dtype=torch.float32)
        super().__init__(G, "Z2", device=device)
    
    def Q(self, x: torch.Tensor) -> torch.Tensor:
        """
        Nearest-neighbor quantization to Z² lattice.
        
        For Z², this is simply rounding to the nearest integer.
        
        Args:
            x: Input tensor to quantize
            
        Returns:
            Quantized tensor (nearest integer point)
        """
        return self.custom_round(x)
