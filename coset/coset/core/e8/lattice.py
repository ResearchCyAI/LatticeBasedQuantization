"""
E₈ lattice implementation.

The E₈ lattice is an 8D lattice with exceptional properties.
It is optimal for 8D sphere packing and has many applications in coding theory and quantization.
"""

import torch
import numpy as np
from typing import Optional
from ..base import Lattice, LatticeConfig


class E8Lattice(Lattice):
    """
    E₈ lattice implementation.
    
    The E₈ lattice is an 8-dimensional lattice that can be constructed
    from the D₈ lattice and a coset. It has the highest known packing
    density in 8D.
    """
    
    def __init__(self, device: Optional[torch.device] = None, config: Optional["LatticeConfig"] = None):
        """
        Initialize the E₈ lattice with its generator matrix.
        
        Args:
            device: Device to place the generator matrix on
            config: Optional lattice configuration
        """
        # Create G tensor directly on target device to avoid CPU-GPU transfer
        G = torch.tensor([
            [2, 0, 0, 0, 0, 0, 0, 0],
            [-1, 1, 0, 0, 0, 0, 0, 0],
            [0, -1, 1, 0, 0, 0, 0, 0],
            [0, 0, -1, 1, 0, 0, 0, 0],
            [0, 0, 0, -1, 1, 0, 0, 0],
            [0, 0, 0, 0, -1, 1, 0, 0],
            [0, 0, 0, 0, 0, -1, 1, 0],
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        ], dtype=torch.float32, device=device).T
        super().__init__(G, "E8", device=device, config=config)
        # Precompute for Babai:
        self.G_inv = torch.linalg.inv(self.G)

    def _compute_packing_radius(self) -> float:
        """
        Compute the packing radius of the E8 lattice.
        
        The E8 lattice has a known packing radius of 1/2 = 0.5.
        This is the optimal packing radius for 8D lattices.
        
        Returns:
            Packing radius as a float (0.5)
        """
        return 0.5
    
    def _compute_covering_radius(self) -> float:
        """
        Compute the covering radius of the E8 lattice.
        
        The E8 lattice has a known covering radius of sqrt(2)/2 ≈ 0.707.
        
        Returns:
            Covering radius as a float
        """
        return np.sqrt(2) / 2
    
    def g_x(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute g(x) by rounding the vector x to the nearest integers,
        but flip the rounding for the coordinate farthest from an integer.
        
        This is a helper function for the E₈ lattice closest point algorithm.
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
        Nearest-neighbor quantization to E₈ lattice.
        
        The E₈ lattice is constructed as the union of D₈ and D₈ + (0.5)⁸.
        The algorithm:
        1. Finds the closest point in D₈ to x
        2. Finds the closest point in D₈ + (0.5)⁸ to x
        3. Returns the closer of the two points
        
        Args:
            x: Input tensor to quantize (shape [d] or [batch_size, d])
            
        Returns:
            Quantized tensor (nearest E₈ lattice point)
        """
        # Handle both single vector and batch cases
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # Find closest point in D₈
        f_x = self.custom_round(x)
        sum_parity = torch.sum(f_x, dim=-1) % 2  # [batch_size] or scalar
        y_0 = torch.where(
            (sum_parity == 0).unsqueeze(-1) if x.dim() > 1 else (sum_parity == 0),
            f_x, 
            self.g_x(x)
        )
        
        # Find closest point in D₈ + (0.5)⁸
        # Ensure 0.5 constant is on the same device as input
        half = torch.tensor(0.5, device=x.device, dtype=x.dtype)
        f_x_shifted = self.custom_round(x - half)
        g_x_shifted = self.g_x(x - half)
        sum_parity_shifted = torch.sum(f_x_shifted, dim=-1) % 2
        
        y_1 = torch.where(
            (sum_parity_shifted == 0).unsqueeze(-1) if x.dim() > 1 else (sum_parity_shifted == 0),
            f_x_shifted + half,
            g_x_shifted + half
        )
        
        # Return the closer point
        dist_0 = torch.norm(x - y_0, dim=-1)  # [batch_size] or scalar
        dist_1 = torch.norm(x - y_1, dim=-1)
        
        closer_mask = (dist_0 < dist_1).unsqueeze(-1) if x.dim() > 1 else (dist_0 < dist_1)
        result = torch.where(closer_mask, y_0, y_1)
        
        if squeeze_output:
            result = result.squeeze(0)
        
        return result

    def projection_babai(self, u: torch.Tensor) -> torch.Tensor:
        """
        Babai approximation: Q̃(u) = G * round(G^{-1} u)
        u: [B,8] or [8]
        """
        
        squeeze = False
        if u.dim() == 1:
            u = u.unsqueeze(0)
            squeeze = True
        #Q(u) = G round(G^−1 u).
        z = torch.round(u @ self.G_inv.T)   # [B,8]
        y = z @ self.G.T                    # [B,8]
        return y.squeeze(0) if squeeze else y
