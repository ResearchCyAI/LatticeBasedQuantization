"""
D₄ lattice implementation.

The D₄ lattice is a 4D lattice where the sum of coordinates must be even.
It has excellent packing properties in 4D and is commonly used in lattice quantization.
"""

import torch
from typing import Optional
from .base import Lattice


class D4Lattice(Lattice):
    """
    D₄ lattice implementation.
    
    The D₄ lattice consists of all integer points where the sum of coordinates
    is even. It has excellent packing properties in 4D and is optimal for
    4D sphere packing.
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        """
        Initialize the D₄ lattice with its generator matrix.
        
        Args:
            device: Device to place the generator matrix on
        """
        G = torch.tensor([
            [-1, -1, 0, 0],
            [1, -1, 0, 0], 
            [0, 1, -1, 0],
            [0, 0, 1, -1]
        ], dtype=torch.float32).T
        super().__init__(G, "D4", device=device)
    
    def g_x(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute g(x) by rounding the vector x to the nearest integers,
        but flip the rounding for the coordinate farthest from an integer.
        
        This is a helper function for the D₄ lattice closest point algorithm.
        
        Args:
            x: Input vector to be processed
            
        Returns:
            Modified rounded vector with one coordinate flipped to ensure
            the sum of components has the desired parity
        """
        f_x = self.custom_round(x)
        delta = torch.abs(x - f_x)
        k = torch.argmax(delta)
        g_x_result = f_x.clone()
        
        x_k = x[k]
        f_x_k = f_x[k]
        
        if x_k >= 0:
            g_x_result[k] = f_x_k + 1 if f_x_k < x_k else f_x_k - 1
        else:
            g_x_result[k] = f_x_k + 1 if f_x_k <= x_k else f_x_k - 1
        
        return g_x_result
    
    def Q(self, x: torch.Tensor) -> torch.Tensor:
        """
        Nearest-neighbor quantization to D₄ lattice.
        
        The D₄ lattice algorithm:
        1. Round x to the nearest integer vector
        2. If the sum is even, this is the closest point
        3. If the sum is odd, flip the rounding for the coordinate farthest
           from an integer to get a valid D₄ lattice point
        
        Args:
            x: Input tensor to quantize
            
        Returns:
            Quantized tensor (nearest D₄ lattice point)
        """
        f_x = self.custom_round(x)
        g_x_result = self.g_x(x)
        
        # Check if sum is even
        if torch.sum(f_x) % 2 == 0:
            return f_x
        else:
            return g_x_result
