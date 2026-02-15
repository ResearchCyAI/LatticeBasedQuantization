"""
E8 Lattice Lookup Tables Module

This module provides optimized lookup table implementations specifically for E8 lattice
quantization, including one-sided and two-sided LUTs for efficient inner product computation.
"""

import torch
import numpy as np
from typing import Optional, Dict, Tuple
from ...lattices import E8Lattice


class E8OneSidedLUT:
    """
    EXPERIMENTAL: One-sided lookup table for E8 lattice operations.
    
    ⚠️  WARNING: This class is experimental and not supported.
    Use HNLQLinear from coset.optim.layers for production use.
    
    Stores precomputed inner products between lattice points and a fixed vector,
    enabling fast lookup-based matrix multiplication.
    """
    
    def __init__(self, q: int = 4, M: int = 2, beta: float = 0.3, lattice: Optional[E8Lattice] = None):
        """
        Initialize E8 one-sided LUT.
        
        Args:
            q: Quantization parameter (alphabet size)
            M: Number of hierarchical levels
            beta: Scaling parameter for quantization
            lattice: E8Lattice instance (defaults to E8Lattice)
        """
        import warnings
        warnings.warn(
            "E8OneSidedLUT is experimental and not supported. Use HNLQLinear from coset.optim.layers for production use.",
            UserWarning,
            stacklevel=2
        )
        
        self.q = q
        self.M = M
        self.beta = beta
        self.lattice = lattice if lattice is not None else E8Lattice()
        self.d = 8  # E8 dimension
        
        # Cache for LUTs
        self._lut_cache: Dict[Tuple, torch.Tensor] = {}
    
    def build_lut(self, vector: torch.Tensor, device: torch.device = None) -> torch.Tensor:
        """
        Build one-sided LUT for a given vector.
        
        Args:
            vector: Input vector of shape [8] or [batch_size, 8]
            device: Device to store the LUT on
            
        Returns:
            LUT tensor containing inner products
        """
        if device is None:
            device = vector.device
        
        vector = vector.to(device)
        original_shape = vector.shape
        
        # Handle batch case
        if vector.dim() == 2:
            batch_size = vector.shape[0]
            vector_flat = vector.view(-1, 8)
        else:
            batch_size = 1
            vector_flat = vector.view(1, 8)
        
        # Generate all possible E8 lattice points in the quantization space
        lattice_points = self._generate_lattice_points(device)
        
        # Compute inner products
        # lattice_points: [q^8, 8], vector_flat: [batch_size, 8]
        inner_products = torch.matmul(vector_flat, lattice_points.T)  # [batch_size, q^8]
        
        # Reshape to original batch shape
        if batch_size == 1:
            inner_products = inner_products.squeeze(0)
        
        return inner_products
    
    def _generate_lattice_points(self, device: torch.device) -> torch.Tensor:
        """
        Generate all possible E8 lattice points in the quantization space.
        
        Args:
            device: Device to generate points on
            
        Returns:
            Tensor of shape [q^8, 8] containing all lattice points
        """
        # For E8, we generate points in the range [0, q-1] for each dimension
        # This is a simplified approach - in practice, we'd use the actual E8 lattice structure
        
        # Generate all combinations of coordinates
        coords = torch.arange(self.q, device=device)
        points = torch.cartesian_prod(*[coords] * self.d)  # [q^8, 8]
        
        return points.float()
    
    def lookup(self, vector: torch.Tensor, indices: torch.Tensor, device: torch.device = None) -> torch.Tensor:
        """
        Lookup inner products for given indices.
        
        Args:
            vector: Input vector of shape [8] or [batch_size, 8]
            indices: Indices to lookup of shape [batch_size, num_indices]
            device: Device to perform lookup on
            
        Returns:
            Inner products of shape [batch_size, num_indices]
        """
        if device is None:
            device = vector.device
        
        # Build LUT if not cached
        cache_key = (tuple(vector.shape), device)
        if cache_key not in self._lut_cache:
            self._lut_cache[cache_key] = self.build_lut(vector, device)
        
        lut = self._lut_cache[cache_key]
        
        # Perform lookup
        if vector.dim() == 1:
            # Single vector case
            return lut[indices]
        else:
            # Batch case
            batch_size = vector.shape[0]
            batch_indices = torch.arange(batch_size, device=device).unsqueeze(1)
            return lut[batch_indices, indices]


class E8TwoSidedLUT:
    """
    EXPERIMENTAL: Two-sided lookup table for E8 lattice operations.
    
    ⚠️  WARNING: This class is experimental and not supported.
    Use HNLQLinear from coset.optim.layers for production use.
    
    Stores precomputed inner products between all pairs of lattice points,
    enabling fast lookup-based operations.
    """
    
    def __init__(self, q: int = 4, M: int = 2, beta: float = 0.3, lattice: Optional[E8Lattice] = None):
        """
        Initialize E8 two-sided LUT.
        
        Args:
            q: Quantization parameter (alphabet size)
            M: Number of hierarchical levels
            beta: Scaling parameter for quantization
            lattice: E8Lattice instance (defaults to E8Lattice)
        """
        import warnings
        warnings.warn(
            "E8TwoSidedLUT is experimental and not supported. Use HNLQLinear from coset.optim.layers for production use.",
            UserWarning,
            stacklevel=2
        )
        
        self.q = q
        self.M = M
        self.beta = beta
        self.lattice = lattice if lattice is not None else E8Lattice()
        self.d = 8  # E8 dimension
        
        # Cache for LUT
        self._lut: Optional[torch.Tensor] = None
        self._device: Optional[torch.device] = None
    
    def build_lut(self, device: torch.device = None) -> torch.Tensor:
        """
        Build two-sided LUT containing all inner products.
        
        LUT[i, j] = ⟨lattice_point_i, lattice_point_j⟩
        
        Args:
            device: Device to store the LUT on
            
        Returns:
            LUT tensor of shape (q^8, q^8) containing inner products
        """
        if device is None:
            device = torch.device('cpu')
        
        # Return cached LUT if available
        if self._lut is not None and self._device == device:
            return self._lut
        
        # Generate all possible E8 lattice points
        lattice_points = self._generate_lattice_points(device)
        
        # Compute all pairwise inner products
        # lattice_points: [q^8, 8]
        inner_products = torch.matmul(lattice_points, lattice_points.T)  # [q^8, q^8]
        
        # Cache the result
        self._lut = inner_products
        self._device = device
        
        return inner_products
    
    def _generate_lattice_points(self, device: torch.device) -> torch.Tensor:
        """
        Generate all possible E8 lattice points in the quantization space.
        
        Args:
            device: Device to generate points on
            
        Returns:
            Tensor of shape [q^8, 8] containing all lattice points
        """
        # For E8, we generate points in the range [0, q-1] for each dimension
        # This is a simplified approach - in practice, we'd use the actual E8 lattice structure
        
        # Generate all combinations of coordinates
        coords = torch.arange(self.q, device=device)
        points = torch.cartesian_prod(*[coords] * self.d)  # [q^8, 8]
        
        return points.float()
    
    def lookup(self, indices_i: torch.Tensor, indices_j: torch.Tensor, device: torch.device = None) -> torch.Tensor:
        """
        Lookup inner products for given index pairs.
        
        Args:
            indices_i: First indices of shape [batch_size, num_pairs]
            indices_j: Second indices of shape [batch_size, num_pairs]
            device: Device to perform lookup on
            
        Returns:
            Inner products of shape [batch_size, num_pairs]
        """
        if device is None:
            device = indices_i.device
        
        # Build LUT if not cached
        if self._lut is None or self._device != device:
            self.build_lut(device)
        
        # Perform lookup
        return self._lut[indices_i, indices_j]
    
    def get_memory_usage(self) -> int:
        """
        Get memory usage of the LUT in bytes.
        
        Returns:
            Memory usage in bytes
        """
        if self._lut is None:
            return 0
        
        return self._lut.numel() * self._lut.element_size()
    
    def clear_cache(self):
        """Clear the LUT cache to free memory."""
        self._lut = None
        self._device = None


class E8LUTManager:
    """
    EXPERIMENTAL: Manager for E8 lattice lookup tables.
    
    ⚠️  WARNING: This class is experimental and not supported.
    Use HNLQLinear from coset.optim.layers for production use.
    
    Provides a unified interface for managing both one-sided and two-sided LUTs
    with automatic caching and memory management.
    """
    
    def __init__(self, q: int = 4, M: int = 2, beta: float = 0.3, lattice: Optional[E8Lattice] = None):
        """
        Initialize E8 LUT manager.
        
        Args:
            q: Quantization parameter (alphabet size)
            M: Number of hierarchical levels
            beta: Scaling parameter for quantization
            lattice: E8Lattice instance (defaults to E8Lattice)
        """
        import warnings
        warnings.warn(
            "E8LUTManager is experimental and not supported. Use HNLQLinear from coset.optim.layers for production use.",
            UserWarning,
            stacklevel=2
        )
        
        self.q = q
        self.M = M
        self.beta = beta
        self.lattice = lattice if lattice is not None else E8Lattice()
        self.one_sided_lut = E8OneSidedLUT(q, M, beta, lattice)
        self.two_sided_lut = E8TwoSidedLUT(q, M, beta, lattice)
    
    def get_one_sided_lut(self, vector: torch.Tensor, device: torch.device = None) -> torch.Tensor:
        """
        Get one-sided LUT for a vector.
        
        Args:
            vector: Input vector
            device: Device to use
            
        Returns:
            One-sided LUT
        """
        return self.one_sided_lut.build_lut(vector, device)
    
    def get_two_sided_lut(self, device: torch.device = None) -> torch.Tensor:
        """
        Get two-sided LUT.
        
        Args:
            device: Device to use
            
        Returns:
            Two-sided LUT
        """
        return self.two_sided_lut.build_lut(device)
    
    def clear_all_caches(self):
        """Clear all LUT caches."""
        self.one_sided_lut._lut_cache.clear()
        self.two_sided_lut.clear_cache()
    
    def get_total_memory_usage(self) -> int:
        """
        Get total memory usage of all LUTs.
        
        Returns:
            Total memory usage in bytes
        """
        one_sided_memory = sum(lut.numel() * lut.element_size() for lut in self.one_sided_lut._lut_cache.values())
        two_sided_memory = self.two_sided_lut.get_memory_usage()
        return one_sided_memory + two_sided_memory
