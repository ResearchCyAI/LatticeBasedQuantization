"""
E₈ lattice implementation.

The E₈ lattice is an 8D lattice with exceptional properties.
It is optimal for 8D sphere packing and has many applications in coding theory and quantization.
"""

import torch
from typing import Optional
from .base import Lattice


class E8Lattice(Lattice):
    """
    E₈ lattice implementation.
    
    The E₈ lattice is an 8-dimensional lattice that can be constructed
    from the D₈ lattice and a coset. It has the highest known packing
    density in 8D.
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        """
        Initialize the E₈ lattice with its generator matrix.
        
        Args:
            device: Device to place the generator matrix on
        """
        G = torch.tensor([
            [2, 0, 0, 0, 0, 0, 0, 0],
            [-1, 1, 0, 0, 0, 0, 0, 0],
            [0, -1, 1, 0, 0, 0, 0, 0],
            [0, 0, -1, 1, 0, 0, 0, 0],
            [0, 0, 0, -1, 1, 0, 0, 0],
            [0, 0, 0, 0, -1, 1, 0, 0],
            [0, 0, 0, 0, 0, -1, 1, 0],
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        ], dtype=torch.float32).T
        super().__init__(G, "E8", device=device)
    
    def g_x(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute g(x) by rounding the vector x to the nearest integers,
        but flip the rounding for the coordinate farthest from an integer.
        
        This is a helper function for the E₈ lattice closest point algorithm.
        
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
        Nearest-neighbor quantization to E₈ lattice.
        
        The E₈ lattice is constructed as the union of D₈ and D₈ + (0.5)⁸.
        The algorithm:
        1. Finds the closest point in D₈ to x
        2. Finds the closest point in D₈ + (0.5)⁸ to x
        3. Returns the closer of the two points
        
        Args:
            x: Input tensor to quantize
            
        Returns:
            Quantized tensor (nearest E₈ lattice point)
        """
        # Find closest point in D₈
        f_x = self.custom_round(x)
        y_0 = f_x if torch.sum(f_x) % 2 == 0 else self.g_x(x)
        
        # Find closest point in D₈ + (0.5)⁸
        f_x_shifted = self.custom_round(x - 0.5)
        g_x_shifted = self.g_x(x - 0.5)
        
        y_1 = f_x_shifted + 0.5 if torch.sum(f_x_shifted) % 2 == 0 else g_x_shifted + 0.5
        
        # Return the closer point
        dist_0 = torch.norm(x - y_0)
        dist_1 = torch.norm(x - y_1)
        
        if dist_0 < dist_1:
            return y_0
        else:
            return y_1
