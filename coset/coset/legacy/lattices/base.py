"""
Base lattice class for hierarchical nested-lattice quantization.

This module provides the abstract base class for all lattice implementations,
defining the common interface and shared functionality.
"""

from abc import ABC, abstractmethod
from typing import Optional
import torch
import numpy as np


class Lattice(ABC):
    """
    Abstract base class for lattice implementations.
    
    All lattice implementations must inherit from this class and implement
    the nearest-neighbor quantization method Q().
    
    Attributes:
        G: Generator matrix for the lattice
        G_inv: Inverse of the generator matrix
        name: Name of the lattice
        d: Dimension of the lattice
    """
    
    def __init__(self, G: torch.Tensor, name: str, device: Optional[torch.device] = None):
        """
        Initialize the lattice.
        
        Args:
            G: Generator matrix for the lattice
            name: Name of the lattice
            device: Device to place G and G_inv on (defaults to G's device)
        """
        if G.shape[0] != G.shape[1]:
            raise ValueError("Generator matrix G must be square")
        
        # Move G to specified device if given
        if device is not None:
            G = G.to(device)
        
        self.G = G
        self.G_inv = torch.linalg.inv(G)
        self.name = name
        self.d = G.shape[0]
        self.device = G.device
    
    @abstractmethod
    def Q(self, x: torch.Tensor) -> torch.Tensor:
        """
        Nearest-neighbor quantization to the lattice.
        
        Args:
            x: Input tensor to quantize
            
        Returns:
            Quantized tensor (nearest lattice point)
        """
        pass
    
    def encode_coords(self, x: torch.Tensor, q: int) -> torch.Tensor:
        """
        Convert lattice point to encoding coordinates.
        
        Args:
            x: Lattice point
            q: Quantization parameter
            
        Returns:
            Encoding coordinates in [0, q-1]^d
        """
        # Ensure G_inv is on the same device as x
        G_inv = self.G_inv.to(x.device)
        return torch.round(torch.matmul(G_inv, x)) % q
    
    def decode_coords(self, b: torch.Tensor, q: int) -> torch.Tensor:
        """
        Convert encoding coordinates to lattice point.
        
        Args:
            b: Encoding coordinates in [0, q-1]^d
            q: Quantization parameter
            
        Returns:
            Lattice point
        """
        # Ensure G is on the same device as b
        G = self.G.to(b.device)
        return torch.matmul(G, b.float())
    
    def custom_round(self, x: torch.Tensor, tiny: Optional[float] = None) -> torch.Tensor:
        """
        Custom rounding function that handles edge cases for lattice quantization.
        
        This function implements a rounding scheme that ensures consistent behavior
        at boundary points (0.5) for lattice quantization algorithms by nudging
        toward zero so exact .5 falls to the nearer-integer toward zero.
        
        Args:
            x: Input tensor to round
            tiny: A microscopic nudge relative to dtype. If None, uses machine epsilon.
            
        Returns:
            Rounded tensor
        """
        if tiny is None:
            # Choose a microscopic nudge relative to dtype
            tiny = torch.finfo(x.dtype if x.dtype.is_floating_point else torch.float32).eps
        
        # Nudge toward zero so exact .5 falls to the nearer-integer toward zero
        y = x - torch.sign(x) * tiny
        
        # Round-to-nearest via floor(x+0.5) works for all signs after the nudge
        return torch.floor(y + 0.5)
    
    def generate_tie_dither(self, beta: float = 1.0, Rin: float = 0.5, magnitude: str = "auto") -> torch.Tensor:
        """
        Generate a constant, sample-independent dither to break ties.
        
        Args:
            beta: Scaling parameter
            Rin: Input radius parameter
            magnitude: Magnitude of dither ("auto" or specific value)
            
        Returns:
            Tie-breaking dither vector
        """
        # Direction with irrational components -> avoids alignment with faces
        primes = [2, 3, 5, 7, 11, 13, 17, 19][:self.d]
        irr = torch.tensor([np.sqrt(p) for p in primes], dtype=torch.float32)
        u = (irr - torch.floor(irr)) - 0.5
        u = u / torch.norm(u)
        
        if magnitude == "auto":
            # Very small relative to scale & lattice packing radius
            eta = 2.0**-40  # use 2**-20 if float32 end-to-end
            delta = eta * beta * Rin
        else:
            delta = float(magnitude)
        
        return delta * u  # add to x before Q_L(x)
    
    def __repr__(self) -> str:
        """String representation of the lattice."""
        return f"{self.name}Lattice(d={self.d})"
    
    def __str__(self) -> str:
        """String representation of the lattice."""
        return self.__repr__()
