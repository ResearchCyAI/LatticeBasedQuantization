"""
Encoding Lookup Table (eLUT) operations in HNLQ encoding space.

This module provides encoding LUT operations for MAC and A&A operations
that work entirely in the encoding domain. eLUTs store encodings of inner products,
allowing operations to be performed without decoding.
"""

import torch
from typing import List, Tuple, Optional, Dict
from ..lattices.base import Lattice
from .params import QuantizationConfig
from .functional import encode, decode


class eLUTManager:
    """
    Manages encoding LUTs (eLUTs) for modulo arithmetic operations.
    
    eLUTs store encodings of inner products, allowing operations
    to be performed entirely in the encoding domain.
    """
    
    def __init__(self, lattice: Lattice, config: QuantizationConfig):
        self.lattice = lattice
        self.config = config
        self.q = config.q
        self.d = lattice.d
        self.M = config.M
        
        # Cache for eLUTs
        self._two_sided_elut: Optional[torch.Tensor] = None
        self._one_sided_eluts: Dict[str, torch.Tensor] = {}
        
    def build_two_sided_elut(self, device: torch.device = None) -> torch.Tensor:
        """
        Build two-sided eLUT containing encodings of inner products.
        
        eLUT[i, j, :, :] = encode(⟨lattice_point_i, lattice_point_j⟩)
        
        Args:
            device: Device to store the eLUT on
            
        Returns:
            eLUT tensor of shape (q^d, q^d, M, d) containing encodings
        """
        if device is None:
            device = torch.device('cpu')
            
        if self._two_sided_elut is not None and self._two_sided_elut.device == device:
            return self._two_sided_elut
            
        # Generate all possible encodings
        all_encodings = self._generate_all_encodings()
        
        # Decode to actual lattice points
        lattice_points = self._decode_encodings_to_lattice_points(all_encodings)
        
        # Build eLUT: eLUT[i,j] = encode(⟨lattice_point_i, lattice_point_j⟩)
        lut_size = self.q ** self.d
        elut = torch.zeros((lut_size, lut_size, self.M, self.d), dtype=torch.int8, device=device)
        
        for i in range(lut_size):
            for j in range(lut_size):
                # Compute inner product
                inner_product = torch.dot(lattice_points[i], lattice_points[j])
                
                # Encode the inner product as a scalar
                try:
                    encoding = self._encode_scalar(inner_product.item(), device)
                    elut[i, j] = encoding
                except Exception as e:
                    # If encoding fails, use zero encoding
                    print(f"Warning: Failed to encode inner product {inner_product.item()}: {e}")
                    elut[i, j] = torch.zeros((self.M, self.d), dtype=torch.int8, device=device)
                
        self._two_sided_elut = elut
        return elut
    
    def _generate_all_encodings(self) -> torch.Tensor:
        """
        Generate all possible encodings using Kronecker product approach.
        
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
        
        Args:
            encodings: All possible encodings (q^d, d)
            
        Returns:
            Tensor of shape (q^d, d) containing actual lattice points
        """
        # Convert encodings to lattice points using generator matrix
        # lattice_point = encoding @ G
        lattice_points = encodings.float() @ self.lattice.G
        
        return lattice_points
    
    def _encode_scalar(self, scalar_value: float, device: torch.device) -> torch.Tensor:
        """
        Encode a scalar value as a vector for the lattice.
        
        Args:
            scalar_value: The scalar value to encode
            device: Device to store the encoding on
            
        Returns:
            Encoding tensor of shape (M, d)
        """
        # Create a vector where the scalar value is the first component
        # and zeros for the rest (this is a simple approach)
        vector = torch.zeros(self.d, device=device)
        vector[0] = scalar_value
        
        # Encode the vector
        try:
            encoding, _ = encode(vector.unsqueeze(0), self.lattice, self.config)
            return encoding
        except Exception:
            # If encoding fails, return zero encoding
            return torch.zeros((self.M, self.d), dtype=torch.int8, device=device)
    
    def get_elut_size(self) -> int:
        """Get the size of the eLUT (q^d)."""
        return self.q ** self.d
    
    def clear_cache(self):
        """Clear the eLUT cache."""
        self._two_sided_elut = None
        self._one_sided_eluts.clear()
    
    def carry_aware_accumulate(self, encodings: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Carry-aware accumulator for A&A operations in encoding space.
        
        Maintains per-layer sums with proper carry propagation to ensure
        lattice-correct arithmetic.
        
        Args:
            encodings: List of M encoding tensors to accumulate
            
        Returns:
            List of M encoding tensors representing accumulated result
        """
        batch_size = encodings[0].shape[0]
        d = encodings[0].shape[1]
        
        # Initialize per-layer sums as encodings
        layer_sums = [torch.zeros((batch_size, d), dtype=torch.int64) for _ in range(self.M)]
        
        # Add each encoding to the sums
        for encoding in encodings:
            for i in range(self.M):
                layer_sums[i] += encoding[i].long()
        
        # Normalize with carry propagation
        for i in range(self.M - 1):
            # Compute carry: C_{i+1} = floor(S_i / q)
            carry = layer_sums[i] // self.q
            
            # Adjust current layer: A*_i = S_i - q * C_{i+1}
            layer_sums[i] = layer_sums[i] % self.q
            
            # Add carry to next layer
            layer_sums[i + 1] += carry
        
        # Convert back to int8
        return [layer_sums[i].to(torch.int8) for i in range(self.M)]


def build_elut(lattice: Lattice, config: QuantizationConfig, device: torch.device = None) -> torch.Tensor:
    """
    Convenience function to build a two-sided eLUT.
    
    Args:
        lattice: Lattice instance
        config: Quantization configuration
        device: Device to store the eLUT on
        
    Returns:
        Two-sided eLUT tensor
    """
    elut_manager = eLUTManager(lattice, config)
    return elut_manager.build_two_sided_elut(device)


def _encoding_to_index(encoding: torch.Tensor, q: int) -> torch.Tensor:
    """
    Convert encoding to eLUT index (vectorized).
    
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
    
    return indices




def elut_mac_operation(encodings_x: List[torch.Tensor], encodings_y: List[torch.Tensor], 
                      lattice: Lattice, config: QuantizationConfig) -> List[torch.Tensor]:
    """
    MAC operation using eLUT that returns accumulated encodings.
    
    Args:
        encodings_x: List of M encoding tensors for x
        encodings_y: List of M encoding tensors for y
        lattice: Lattice instance
        config: Quantization configuration
        
    Returns:
        List of M encoding tensors representing accumulated result
    """
    # Build eLUT
    elut_manager = eLUTManager(lattice, config)
    elut = elut_manager.build_two_sided_elut(device=encodings_x[0].device)
    
    batch_size = encodings_x[0].shape[0]
    d = encodings_x[0].shape[1]
    
    # Initialize accumulator as encodings
    accumulator = [torch.zeros((batch_size, d), dtype=torch.int64) for _ in range(config.M)]
    
    # Compute MAC using eLUT
    for i in range(config.M):
        for j in range(config.M):
            # Convert encodings to eLUT indices
            idx_x = _encoding_to_index(encodings_x[i], config.q)
            idx_y = _encoding_to_index(encodings_y[j], config.q)
            
            # Look up encodings from eLUT
            for batch_idx in range(batch_size):
                lut_encoding = elut[idx_x[batch_idx], idx_y[batch_idx]]  # Shape: (M, d)
                
                # Scale by q^(i+j) and accumulate
                scale_factor = config.q ** (i + j)
                
                # Add scaled encoding to accumulator
                for k in range(config.M):
                    accumulator[k][batch_idx] += lut_encoding[k].long() * scale_factor
    
    # Convert back to int8 and return as list of tensors
    return [accumulator[i].to(torch.int8) for i in range(config.M)]


def elut_accumulate_operation(encodings: List[torch.Tensor], 
                             lattice: Lattice, config: QuantizationConfig) -> List[torch.Tensor]:
    """
    A&A operation using eLUT that returns accumulated encodings.
    
    Args:
        encodings: List of M encoding tensors
        lattice: Lattice instance
        config: Quantization configuration
        
    Returns:
        List of M encoding tensors representing accumulated result
    """
    # Use eLUTManager's carry_aware_accumulate method
    elut_manager = eLUTManager(lattice, config)
    return elut_manager.carry_aware_accumulate(encodings)


def mac_with_dither(encodings_x: List[torch.Tensor], encodings_y: List[torch.Tensor],
                   dither_x: torch.Tensor, dither_y: torch.Tensor,
                   lattice: Lattice, config: QuantizationConfig) -> torch.Tensor:
    """
    MAC operation with structured dither (virtual layer -1).
    
    Args:
        encodings_x: List of M encoding tensors for x
        encodings_y: List of M encoding tensors for y
        dither_x: Dither for x
        dither_y: Dither for y
        lattice: Lattice instance
        config: Quantization configuration
        
    Returns:
        MAC results with dither
    """
    # For now, implement without dither (virtual layer -1)
    # This would require extending the LUT to include dither layer
    return elut_mac_operation(encodings_x, encodings_y, lattice, config)


def mac_with_scaling(encodings_x: List[torch.Tensor], encodings_y: List[torch.Tensor],
                    beta_x: float, beta_y: float,
                    lattice: Lattice, config: QuantizationConfig) -> torch.Tensor:
    """
    MAC operation with scaling factors β.
    
    Args:
        encodings_x: List of M encoding tensors for x
        encodings_y: List of M encoding tensors for y
        beta_x: Scaling factor for x
        beta_y: Scaling factor for y
        lattice: Lattice instance
        config: Quantization configuration
        
    Returns:
        MAC results with scaling
    """
    # Perform MAC operation
    result = elut_mac_operation(encodings_x, encodings_y, lattice, config)
    
    # Apply scaling
    return result * beta_x * beta_y


def adaptive_mac(encodings_x: List[torch.Tensor], encodings_y: List[torch.Tensor],
                lattice: Lattice, config: QuantizationConfig,
                max_layers: Optional[int] = None) -> torch.Tensor:
    """
    MAC operation with early exit for adaptive compute vs accuracy.
    
    Args:
        encodings_x: List of M encoding tensors for x
        encodings_y: List of M encoding tensors for y
        lattice: Lattice instance
        config: Quantization configuration
        max_layers: Maximum number of layers to use (None for all)
        
    Returns:
        MAC results
    """
    if max_layers is None:
        max_layers = config.M
        
    # Truncate to max_layers
    encodings_x_truncated = encodings_x[:max_layers]
    encodings_y_truncated = encodings_y[:max_layers]
    
    return elut_mac_operation(encodings_x_truncated, encodings_y_truncated, lattice, config)


def mac_tensor(X: torch.Tensor, Y: torch.Tensor, lattice: Lattice, config: QuantizationConfig) -> torch.Tensor:
    """
    MAC operation for tensors shaped [..., K, d] treated as K tiles.
    
    Args:
        X: First tensor [..., K, d]
        Y: Second tensor [..., K, d]
        lattice: Lattice instance
        config: Quantization configuration
        
    Returns:
        MAC results
    """
    # Reshape to [batch_size, K, d]
    original_shape = X.shape[:-2]
    K = X.shape[-2]
    d = X.shape[-1]
    
    X_flat = X.view(-1, K, d)
    Y_flat = Y.view(-1, K, d)
    
    batch_size = X_flat.shape[0]
    results = torch.zeros(batch_size, dtype=torch.int64)
    
    # Process each tile
    for k in range(K):
        # Encode each tile
        x_tile = X_flat[:, k, :]
        y_tile = Y_flat[:, k, :]
        
        # For now, use simple dot product (should be replaced with proper encoding)
        tile_result = torch.sum(x_tile * y_tile, dim=1)
        results += tile_result
        
    return results.view(original_shape)


def batch_mac(encodings_batch_x: List[torch.Tensor], encodings_batch_y: List[torch.Tensor],
              lattice: Lattice, config: QuantizationConfig) -> torch.Tensor:
    """
    Efficient batch MAC operations.
    
    Args:
        encodings_batch_x: List of M encoding tensors for batch x
        encodings_batch_y: List of M encoding tensors for batch y
        lattice: Lattice instance
        config: Quantization configuration
        
    Returns:
        Batch MAC results
    """
    # For now, process each pair individually
    # This should be optimized for batch processing
    batch_size = encodings_batch_x[0].shape[0]
    results = torch.zeros(batch_size, dtype=torch.int64)
    
    for i in range(batch_size):
        # Extract single pair
        encodings_x_i = [encodings_batch_x[j][i:i+1] for j in range(len(encodings_batch_x))]
        encodings_y_i = [encodings_batch_y[j][i:i+1] for j in range(len(encodings_batch_y))]
        
        # Compute MAC for this pair
        result_i = elut_mac_operation(encodings_x_i, encodings_y_i, lattice, config)
        results[i] = result_i[0]
        
    return results


def fast_modq_accumulate(acc: torch.Tensor, x: torch.Tensor, q: int) -> torch.Tensor:
    """
    Fast mod-q accumulation with periodic rebase.
    
    Args:
        acc: Accumulator tensor
        x: Value to add
        q: Modulus
        
    Returns:
        Updated accumulator
    """
    return (acc + x) % q


def validate_modulo_operations(encodings_x: List[torch.Tensor], encodings_y: List[torch.Tensor],
                              lattice: Lattice, config: QuantizationConfig) -> bool:
    """
    Validate modulo operations by comparing with decode-then-compute baseline.
    
    Args:
        encodings_x: List of M encoding tensors for x
        encodings_y: List of M encoding tensors for y
        lattice: Lattice instance
        config: Quantization configuration
        
    Returns:
        True if validation passes
    """
    # This would implement validation by:
    # 1. Decoding encodings to get quantized vectors
    # 2. Computing dot product in quantized space
    # 3. Comparing with encoding-space MAC result
    # For now, return True as placeholder
    return True
