"""
D₄ lattice implementation.

The D₄ lattice is a 4D lattice where the sum of coordinates must be even.
It has excellent packing properties in 4D and is commonly used in lattice quantization.
"""

import torch
import numpy as np
from typing import Optional
from ..base import Lattice, LatticeConfig


class D4Lattice(Lattice):
    """
    D₄ lattice implementation.
    
    The D₄ lattice consists of all integer points where the sum of coordinates
    is even. It has excellent packing properties in 4D and is optimal for
    4D sphere packing.
    """
    
    def __init__(self, device: Optional[torch.device] = None, config: Optional[LatticeConfig] = None):
        """
        Initialize the D₄ lattice with its generator matrix.
        
        Args:
            device: Device to place the generator matrix on
            config: Optional lattice configuration
        """
        G = torch.tensor([
            [-1, -1, 0, 0],
            [1, -1, 0, 0], 
            [0, 1, -1, 0],
            [0, 0, 1, -1]
        ], dtype=torch.float32, device=device).T
        super().__init__(G, "D4", device=device, config=config)
    
    def _compute_packing_radius(self) -> float:
        """
        Compute the packing radius of the D4 lattice.
        
        The D4 lattice has a known packing radius of 1/sqrt(2) ≈ 0.707.
        This is the optimal packing radius for 4D lattices.
        
        Returns:
            Packing radius as a float
        """
        return 1.0 / np.sqrt(2)
    
    def _compute_covering_radius(self) -> float:
        """
        Compute the covering radius of the D4 lattice.
        
        The D4 lattice has a known covering radius of 1.
        
        Returns:
            Covering radius as a float
        """
        return 1.0
    
    def g_x(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute g(x) by rounding the vector x to the nearest integers,
        but flip the rounding for the coordinate farthest from an integer.
        
        This is a helper function for the D₄ lattice closest point algorithm.
        Handles both single vectors and batch inputs.
        
        Args:
            x: Input vector to be processed (shape [d] or [batch_size, d])
            
        Returns:
            Modified rounded vector with one coordinate flipped to ensure
            the sum of components has the desired parity
        """
        # Handle both single vector and batch cases
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        f_x = self.custom_round(x)
        delta = torch.abs(x - f_x)
        k = torch.argmax(delta, dim=-1)  # Find max along last dimension
        g_x_result = f_x.clone()
        
        # Use advanced indexing for batch processing
        batch_indices = torch.arange(x.shape[0], device=x.device)
        x_k = x[batch_indices, k]
        f_x_k = f_x[batch_indices, k]
        
        # Vectorized conditional update
        positive_mask = x_k >= 0
        flip_positive = positive_mask & (f_x_k < x_k)
        flip_negative = ~positive_mask & (f_x_k <= x_k)
        flip_mask = flip_positive | flip_negative
        
        # Apply flips
        g_x_result[batch_indices[flip_positive], k[flip_positive]] += 1
        g_x_result[batch_indices[~flip_positive], k[~flip_positive]] -= 1
        
        if squeeze_output:
            g_x_result = g_x_result.squeeze(0)
        
        return g_x_result
    
    def projection(self, x: torch.Tensor) -> torch.Tensor:
        """
        Nearest-neighbor quantization to D₄ lattice.
        
        The D₄ lattice algorithm:
        1. Round x to the nearest integer vector
        2. If the sum is even, this is the closest point
        3. If the sum is odd, flip the rounding for the coordinate farthest
           from an integer to get a valid D₄ lattice point
        
        Args:
            x: Input tensor to quantize (shape [d] or [batch_size, d])
            
        Returns:
            Quantized tensor (nearest D₄ lattice point)
        """
        # Handle both single vector and batch cases
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        f_x = self.custom_round(x)
        sum_parity = torch.sum(f_x, dim=-1) % 2  # [batch_size] or scalar
        
        # If sum is even, f_x is the closest point
        # If sum is odd, use g_x to get a valid D₄ point
        result = torch.where(
            (sum_parity == 0).unsqueeze(-1) if x.dim() > 1 else (sum_parity == 0),
            f_x,
            self.g_x(x)
        )
        
        if squeeze_output:
            result = result.squeeze(0)
        
        return result
