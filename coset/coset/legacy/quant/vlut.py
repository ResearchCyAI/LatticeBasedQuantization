"""
Value Lookup Table (vLUT) management for modulo arithmetic operations.

This module provides efficient vLUT operations for MAC and A&A operations
in the HNLQ encoding space. vLUTs store actual scalar values of inner products,
providing fast lookup-based operations with PyTorch GPU acceleration.
"""

import torch
import numpy as np
from typing import List, Tuple, Optional, Dict
from ..lattices.base import Lattice
from .params import QuantizationConfig


class vLUTManager:
    """
    Manages value LUTs (vLUTs) for modulo arithmetic operations.
    
    vLUTs store actual scalar values of inner products, providing
    fast lookup-based operations with PyTorch GPU acceleration.
    """
    
    def __init__(self, lattice: Lattice, config: QuantizationConfig):
        self.lattice = lattice
        self.config = config
        self.q = config.q
        self.d = lattice.d
        self.M = config.M
        
        # Cache for vLUTs
        self._two_sided_vlut: Optional[torch.Tensor] = None
        self._one_sided_vluts: Dict[str, torch.Tensor] = {}
        
    def build_two_sided_vlut(self, device: torch.device = None) -> torch.Tensor:
        """
        Build two-sided vLUT containing scalar values of inner products.
        
        vLUT[i, j] = ⟨lattice_point_i, lattice_point_j⟩
        
        Args:
            device: Device to store the vLUT on
            
        Returns:
            vLUT tensor of shape (q^d, q^d) containing scalar values
        """
        if device is None:
            device = torch.device('cpu')
            
        if self._two_sided_vlut is not None and self._two_sided_vlut.device == device:
            return self._two_sided_vlut
            
        # Generate all possible encodings
        all_encodings = self._generate_all_encodings()
        
        # Decode to actual lattice points and move to device
        lattice_points = self._decode_encodings_to_lattice_points(all_encodings).to(device)
        
        # Build vLUT: vLUT[i,j] = ⟨lattice_point_i, lattice_point_j⟩
        lut_size = self.q ** self.d
        vlut = torch.zeros((lut_size, lut_size), dtype=torch.float32, device=device)
        
        for i in range(lut_size):
            for j in range(lut_size):
                # Compute inner product
                inner_product = torch.dot(lattice_points[i], lattice_points[j])
                vlut[i, j] = inner_product
                
        self._two_sided_vlut = vlut
        return vlut
    
    def build_one_sided_vlut(self, query_vector: torch.Tensor, device: torch.device = None) -> torch.Tensor:
        """
        Build one-sided vLUT containing scalar values of inner products with query.
        
        For HNLQ residual-based quantization:
        vLUT[i] = ⟨query_vector, residual_i⟩
        where residual_i = (b_i @ G) - q·Q((b_i @ G)/q)
        
        Args:
            query_vector: Query vector of shape (d,)
            device: Device to store the vLUT on
            
        Returns:
            vLUT tensor of shape (q^d,) containing scalar values
        """
        if device is None:
            device = torch.device('cpu')
            
        # Create cache key
        query_key = f"{query_vector.sum().item():.6f}_{query_vector.shape}_{device}"
        if query_key in self._one_sided_vluts and self._one_sided_vluts[query_key].device == device:
            return self._one_sided_vluts[query_key]
            
        # Generate all possible encodings
        all_encodings = self._generate_all_encodings()
        
        # Decode to actual lattice points and move to device
        lattice_points = self._decode_encodings_to_lattice_points(all_encodings).to(device)
        query_vector = query_vector.to(device)
        
        # Build vLUT with residuals: vLUT[i] = ⟨query_vector, residual_i⟩
        # residual_i = lattice_point_i - q * Q(lattice_point_i / q)
        lut_size = self.q ** self.d
        vlut = torch.zeros(lut_size, dtype=torch.float32, device=device)
        
        for i in range(lut_size):
            # Compute residual: x_i = Gb_i - q * Q(Gb_i / q)
            Gb_i = lattice_points[i]
            residual_i = Gb_i - self.q * self.lattice.Q(Gb_i / self.q)
            
            # Compute inner product with residual
            inner_product = torch.dot(query_vector, residual_i)
            vlut[i] = inner_product
                
        self._one_sided_vluts[query_key] = vlut
        return vlut
    
    def _generate_all_encodings(self) -> torch.Tensor:
        """
        Generate all possible encodings using Kronecker product approach.
        
        For each dimension, we have q possible values [0, 1, ..., q-1].
        The Kronecker product gives us all combinations.
        
        Returns:
            Tensor of shape (q^d, d) containing all possible encodings
        """
        # Create base vectors for each dimension
        base_vectors = [torch.arange(self.q) for _ in range(self.d)]
        
        # Use cartesian product to generate all combinations
        all_encodings = torch.cartesian_prod(*base_vectors)
        
        return all_encodings
    
    def _decode_encodings_to_lattice_points(self, encodings: torch.Tensor) -> torch.Tensor:
        """
        Decode encodings to actual lattice points using the generator matrix.
        
        For each encoding, we decode it to get the corresponding lattice point.
        This respects the lattice structure (D4, E8, etc.).
        
        Args:
            encodings: All possible encodings (q^d, d)
            
        Returns:
            Tensor of shape (q^d, d) containing actual lattice points
        """
        # Convert encodings to lattice points using generator matrix
        # lattice_point = G @ encoding (matching decode_coords implementation)
        # For batch: lattice_points[i] = G @ encodings[i] => (encodings @ G.T).T = G @ encodings.T
        # But we want shape (q^d, d), so we use: (G @ encodings.T).T = encodings @ G.T
        lattice_points = encodings.float() @ self.lattice.G.T
        
        return lattice_points
    
    def get_vlut_size(self) -> int:
        """Get the size of the vLUT (q^d)."""
        return self.q ** self.d
    
    def build_sum_lut(self, device: torch.device = None) -> torch.Tensor:
        """
        Build sum LUT containing sums of lattice points.
        
        sum_LUT[i, j] = lattice_point_i + lattice_point_j
        
        Args:
            device: Device to store the sum LUT on
            
        Returns:
            Sum LUT tensor of shape (q^d, q^d) containing lattice point sums
        """
        if device is None:
            device = torch.device('cpu')
            
        # Generate all possible encodings
        all_encodings = self._generate_all_encodings()
        
        # Decode to actual lattice points
        lattice_points = self._decode_encodings_to_lattice_points(all_encodings)
        
        # Build sum LUT: sum_LUT[i,j] = lattice_point_i + lattice_point_j
        lut_size = self.q ** self.d
        sum_lut = torch.zeros((lut_size, lut_size, self.d), dtype=torch.float32, device=device)
        
        for i in range(lut_size):
            for j in range(lut_size):
                # Compute lattice point sum
                point_sum = lattice_points[i] + lattice_points[j]
                sum_lut[i, j] = point_sum
                
        return sum_lut
    
    def clear_cache(self):
        """Clear the vLUT cache."""
        self._two_sided_vlut = None
        self._one_sided_vluts.clear()


def build_vlut(lattice: Lattice, config: QuantizationConfig, device: torch.device = None) -> torch.Tensor:
    """
    Convenience function to build a two-sided vLUT.
    
    Args:
        lattice: Lattice instance
        config: Quantization configuration
        device: Device to store the vLUT on
        
    Returns:
        Two-sided vLUT tensor
    """
    vlut_manager = vLUTManager(lattice, config)
    return vlut_manager.build_two_sided_vlut(device)


def _encoding_to_index(encoding: torch.Tensor, q: int) -> torch.Tensor:
    """
    Convert encoding to vLUT index (vectorized).
    
    Args:
        encoding: Encoding tensor [batch_size, d]
        q: Quantization parameter
        
    Returns:
        Index tensor [batch_size]
    """
    batch_size, d = encoding.shape
    device = encoding.device
    
    # Vectorized computation: idx = Σⱼ encoding[:, j] * q^(d-1-j)
    # Create powers of q: [q^(d-1), q^(d-2), ..., q^1, q^0]
    powers = torch.pow(q, torch.arange(d-1, -1, -1, dtype=torch.long, device=device))
    
    # Vectorized index computation
    indices = torch.sum(encoding * powers, dim=1)
    
    return indices.long()


def vlut_mac_operation(encodings_x: List[torch.Tensor], encodings_y: List[torch.Tensor], 
                      vlut: torch.Tensor) -> torch.Tensor:
    """
    MAC operation using vLUT that returns scalar results (fully vectorized).
    
    Implements: ⟨x̂, ŷ⟩ = Σᵢ vLUT[bᵢ(x), bᵢ(y)]
    The vLUT already contains the scaled inner products.
    
    Args:
        encodings_x: List of M encoding tensors for x
        encodings_y: List of M encoding tensors for y
        vlut: Two-sided vLUT of shape (q^d, q^d)
        
    Returns:
        Scalar tensor containing MAC result
    """
    batch_size = encodings_x[0].shape[0]
    device = vlut.device
    M = len(encodings_x)  # Number of layers
    
    # Derive q from vLUT shape: vlut.shape[0] = q^d
    # For D4 lattice (d=4), q^4 = vlut.shape[0], so q = (vlut.shape[0])^(1/4)
    d = encodings_x[0].shape[1]  # Lattice dimension
    q = int(round(vlut.shape[0] ** (1.0 / d)))
    
    # Convert all encodings to indices (vectorized)
    # Shape: [M, batch_size]
    idx_x_all = torch.stack([_encoding_to_index(encodings_x[i], q) for i in range(M)])
    idx_y_all = torch.stack([_encoding_to_index(encodings_y[i], q) for i in range(M)])
    
    # Vectorized vLUT lookup: vLUT[idx_x_all[i], idx_y_all[i]] for all i
    # Shape: [M, batch_size]
    lut_values = vlut[idx_x_all, idx_y_all]
    
    # MAC operation: sum across M dimension
    # Shape: [batch_size]
    result = torch.sum(lut_values, dim=0)
    
    return result


def vlut_accumulate_operation(encodings: List[torch.Tensor]) -> List[torch.Tensor]:
    """
    A&A operation using sum LUT (vectorized).
    
    Args:
        encodings: List of M encoding tensors
        
    Returns:
        List of M encoding tensors representing accumulated result
    """
    # This is a compatibility API - vLUT accumulation is just tensor sum
    # since we're working with actual encodings, not indices
    batch_size = encodings[0].shape[0]
    d = encodings[0].shape[1]
    device = encodings[0].device
    M = len(encodings[0])  # Number of layers per encoding
    
    # Simply sum all encodings across the list (vectorized)
    result = []
    for i in range(M):
        # Stack all i-th layer encodings and sum
        layer_encodings = torch.stack([enc[i] for enc in encodings], dim=0)  # (num_encodings, batch_size, d)
        layer_sum = torch.sum(layer_encodings, dim=0)  # (batch_size, d)
        result.append(layer_sum)
    
    return result


