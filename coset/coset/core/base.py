"""
Base lattice class for hierarchical nested-lattice quantization.

This module provides the abstract base class for all lattice implementations,
defining the common interface and shared functionality.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import torch
import numpy as np

from pydantic import BaseModel

class LatticeConfig(BaseModel):
    lattice_type: str
    q: int
    M: int
    beta: float = 1.0
    alpha: float = 1.0
    decoding: str = "full"
    check_overload: bool = True
    disable_overload_protection: bool = True
    disable_scaling: bool = True
    with_tie_dither: bool = True
    with_dither: bool = False
    max_scaling_iterations: int = 10

    def __post_init__(self):
        if self.decoding not in ["full", "coarse_to_fine", "progressive"]:
            raise ValueError(f"Unknown decoding method: {self.decoding}")
        if not isinstance(self.check_overload, bool):
            raise ValueError("check_overload must be a Boolean")
        if not isinstance(self.disable_overload_protection, bool):
            raise ValueError("disable_overload_protection must be a Boolean")
        if not isinstance(self.disable_scaling, bool):
            raise ValueError("disable_scaling must be a Boolean")
        if not isinstance(self.with_tie_dither, bool):
            raise ValueError("with_tie_dither must be a Boolean")
        if not isinstance(self.with_dither, bool):
            raise ValueError("with_dither must be a Boolean")
        if self.max_scaling_iterations <= 0:
            raise ValueError("max_scaling_iterations must be positive")
        if self.beta <= 0:
            raise ValueError("beta must be positive")
        if self.alpha <= 0:
            raise ValueError("alpha must be positive")
        if self.q <= 0:
            raise ValueError("q must be positive")
        if self.M <= 0:
            raise ValueError("M must be positive")
        if self.lattice_type not in ["D4", "E8", "A2", "Z2", "Z3"]:
            raise ValueError(f"Unknown lattice type: {self.lattice_type}")  

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "LatticeConfig":
        """Create configuration from dictionary."""
        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.model_dump()

class Lattice(ABC):
    """
    Abstract base class for lattice implementations.
    
    All lattice implementations must inherit from this class and implement
    the nearest-neighbor quantization method projection().
    
    Attributes:
        G: Generator matrix for the lattice
        G_inv: Inverse of the generator matrix
        name: Name of the lattice
        d: Dimension of the lattice
        config: Optional lattice configuration
        r_pack: Packing radius of the lattice
        r_cov: Covering radius of the lattice
    """
    
    def __init__(self, G: torch.Tensor, name: str, device: Optional[torch.device] = None, config: Optional[LatticeConfig] = None):
        """
        Initialize the lattice.
        
        Args:
            G: Generator matrix for the lattice
            name: Name of the lattice
            device: Device to place G and G_inv on (defaults to G's device)
            config: Optional lattice configuration
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
        self.config = config
        
        # Compute lattice geometric properties
        self.r_pack = self._compute_packing_radius()
        self.r_cov = self._compute_covering_radius()
    
    def get_generators(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get the generator matrix G and its inverse G_inv.
        
        Returns:
            Tuple of (G, G_inv) where:
            - G: Generator matrix of shape [d, d]
            - G_inv: Inverse of G of shape [d, d]
        """
        return self.G.clone().detach(), self.G_inv.clone().detach()
    
    @abstractmethod
    def projection(self, x: torch.Tensor) -> torch.Tensor:
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
            x: Lattice point (shape [d] or [batch_size, d])
            q: Quantization parameter
            
        Returns:
            Encoding coordinates in [0, q-1]^d
        """
        # Ensure G_inv is on the same device as x
        G_inv = self.G_inv.to(x.device)
        
        # Handle both single vector and batch cases
        if x.dim() == 1:
            # Single vector: G_inv @ x
            return torch.round(torch.matmul(G_inv, x)) % q
        else:
            # Batch: G_inv @ x.T then transpose back
            return torch.round(torch.matmul(G_inv, x.T)).T % q
    
    def decode_coords(self, b: torch.Tensor, q: int) -> torch.Tensor:
        """
        Convert encoding coordinates to lattice point.
        
        Args:
            b: Encoding coordinates in [0, q-1]^d (shape [d] or [batch_size, d])
            q: Quantization parameter
            
        Returns:
            Lattice point
        """
        # Ensure G is on the same device as b
        G = self.G.to(b.device)
        
        # Handle both single vector and batch cases
        if b.dim() == 1:
            # Single vector: G @ b
            return torch.matmul(G, b.float())
        else:
            # Batch: G @ b.T then transpose back
            return torch.matmul(G, b.float().T).T
    
    
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
    
    def _compute_packing_radius(self) -> float:
        """
        Compute the packing radius of the lattice.
        
        The packing radius is half the length of the shortest non-zero vector
        in the lattice. For most lattices, this can be computed from the
        generator matrix.
        
        Returns:
            Packing radius as a float
        """
        # Compute the Gram matrix G^T * G
        gram_matrix = torch.mm(self.G.T, self.G)
        
        # Find the minimum non-zero eigenvalue (squared length of shortest vector)
        eigenvals = torch.linalg.eigvals(gram_matrix)
        min_eigenval = torch.min(torch.real(eigenvals[eigenvals.real > 1e-10]))
        
        # Packing radius is half the length of the shortest vector
        return 0.5 * torch.sqrt(min_eigenval).item()
    
    def _compute_covering_radius(self) -> float:
        """
        Compute the covering radius of the lattice.
        
        The covering radius is the radius of the largest sphere that can be
        inscribed within the Voronoi cell of the lattice. This is more complex
        to compute and may require numerical methods for general lattices.
        
        For now, we provide a default implementation that can be overridden
        by specific lattice implementations.
        
        Returns:
            Covering radius as a float
        """
        # Default implementation: use packing radius * sqrt(dimension)
        # This is a rough approximation that can be overridden
        return self.r_pack * np.sqrt(self.d)
    
    def compute_delta0(self, q: int, M: int, rho: float = 0.95) -> float:
        """
        Compute Delta0 parameter for the lattice quantizer.
        
        Delta0 = 2 * rho * r_pack / (q^M - 1)
        
        Args:
            q: Quantization parameter (alphabet size)
            M: Number of hierarchical levels
            rho: Scaling factor (default: 0.95)
            
        Returns:
            Delta0 parameter as a float
        """
        return (2.0 * rho * self.r_pack) / (q**M - 1)
    
    def __repr__(self) -> str:
        """String representation of the lattice."""
        return f"{self.name}Lattice(d={self.d})"
    
    def __str__(self) -> str:
        """String representation of the lattice."""
        return self.__repr__()
