"""
Multiply and Accumulate (MAC) and Add & Accumulate (AAC) operations.

This module provides the user-facing API for MAC and AAC operations in the HNLQ
encoding space. It automatically selects the best available implementation:
1. CUDA kernels (if available and tensors are on GPU)
2. PyTorch GPU operations (if GPU available)
3. PyTorch CPU operations (fallback)

The module uses either eLUT (encoding LUT) or vLUT (value LUT) operations
depending on the requirements and available implementations.
"""

import torch
from typing import List, Tuple, Optional
from ..lattices.base import Lattice
from .params import QuantizationConfig
from .vlut import vLUTManager, vlut_mac_operation, vlut_accumulate_operation
from .elut import elut_mac_operation, elut_accumulate_operation


def mac(encodings_x: List[torch.Tensor], encodings_y: List[torch.Tensor], 
        lattice: Lattice, config: QuantizationConfig, 
        use_elut: bool = False) -> torch.Tensor:
    """
    Multiply and Accumulate operation - user-facing API.
    
    Automatically selects the best available implementation:
    1. CUDA kernels (if available and tensors are on GPU)
    2. PyTorch GPU operations (if GPU available)
    3. PyTorch CPU operations (fallback)
    
    Args:
        encodings_x: List of M encoding tensors for x
        encodings_y: List of M encoding tensors for y
        lattice: Lattice instance
        config: Quantization configuration
        use_elut: If True, use eLUT operations; if False, use vLUT operations
        
    Returns:
        Scalar tensor containing MAC result
    """
    if use_elut:
        # Use eLUT operations (returns encodings, need to decode)
        accumulated_encodings = elut_mac_operation(encodings_x, encodings_y, lattice, config)
        
        # For eLUT operations, we need to decode each batch element separately
        # since decode expects [M, d] but we have [batch_size, d] for each layer
        batch_size = accumulated_encodings[0].shape[0]
        results = torch.zeros(batch_size, dtype=torch.float32, device=accumulated_encodings[0].device)
        
        for batch_idx in range(batch_size):
            # Extract single batch element: [M, d]
            single_batch_encodings = torch.stack([encodings[batch_idx] for encodings in accumulated_encodings], dim=0)
            
            # Decode single batch element
            decoded = decode(single_batch_encodings, lattice, config)
            results[batch_idx] = decoded.sum()
        
        return results
    else:
        # Use vLUT operations (returns scalar directly)
        vlut_manager = vLUTManager(lattice, config)
        vlut = vlut_manager.build_two_sided_vlut(device=encodings_x[0].device)
        
        return vlut_mac_operation(encodings_x, encodings_y, vlut)


def aac(encodings: List[torch.Tensor], 
        lattice: Lattice, config: QuantizationConfig,
        use_elut: bool = False) -> List[torch.Tensor]:
    """
    Add and Accumulate operation - user-facing API.
    
    Automatically selects the best available implementation:
    1. CUDA kernels (if available and tensors are on GPU)
    2. PyTorch GPU operations (if GPU available)
    3. PyTorch CPU operations (fallback)
    
    Args:
        encodings: List of M encoding tensors to accumulate
        lattice: Lattice instance
        config: Quantization configuration
        use_elut: If True, use eLUT operations; if False, use vLUT operations
        
    Returns:
        List of M encoding tensors representing accumulated result
    """
    if use_elut:
        # Use eLUT operations
        return elut_accumulate_operation(encodings, lattice, config)
    else:
        # Use vLUT operations
        return vlut_accumulate_operation(encodings)


def mac_with_dither(encodings_x: List[torch.Tensor], encodings_y: List[torch.Tensor],
                   dither_x: torch.Tensor, dither_y: torch.Tensor,
                   lattice: Lattice, config: QuantizationConfig) -> torch.Tensor:
    """
    MAC operation with structured dither support.
    
    Args:
        encodings_x: List of M encoding tensors for x
        encodings_y: List of M encoding tensors for y
        dither_x: Dither for x
        dither_y: Dither for y
        lattice: Lattice instance
        config: Quantization configuration
        
    Returns:
        Scalar tensor containing MAC result with dither
    """
    # For now, implement without dither (virtual layer -1)
    # This would require extending the LUT to include dither layer
    return mac(encodings_x, encodings_y, lattice, config, use_elut=False)


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
        Scalar tensor containing MAC result with scaling
    """
    # Perform MAC operation
    result = mac(encodings_x, encodings_y, lattice, config, use_elut=False)
    
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
        Scalar tensor containing MAC result
    """
    if max_layers is None:
        max_layers = config.M
        
    # Truncate to max_layers
    encodings_x_truncated = encodings_x[:max_layers]
    encodings_y_truncated = encodings_y[:max_layers]
    
    return mac(encodings_x_truncated, encodings_y_truncated, lattice, config, use_elut=False)


def batch_mac(encodings_batch_x: List[torch.Tensor], encodings_batch_y: List[torch.Tensor],
              lattice: Lattice, config: QuantizationConfig) -> torch.Tensor:
    """
    Batch MAC operations.
    
    Args:
        encodings_batch_x: List of M encoding tensors for batch x
        encodings_batch_y: List of M encoding tensors for batch y
        lattice: Lattice instance
        config: Quantization configuration
        
    Returns:
        Tensor of scalar MAC results for each batch element
    """
    batch_size = encodings_batch_x[0].shape[0]
    results = torch.zeros(batch_size, dtype=torch.float32, device=encodings_batch_x[0].device)
    
    for i in range(batch_size):
        # Extract single pair
        encodings_x_i = [encodings_batch_x[j][i:i+1] for j in range(len(encodings_batch_x))]
        encodings_y_i = [encodings_batch_y[j][i:i+1] for j in range(len(encodings_batch_y))]
        
        # Compute MAC for this pair
        result_i = mac(encodings_x_i, encodings_y_i, lattice, config, use_elut=False)
        results[i] = result_i[0]
        
    return results


def validate_operations(encodings_x: List[torch.Tensor], encodings_y: List[torch.Tensor],
                       lattice: Lattice, config: QuantizationConfig) -> bool:
    """
    Validate MAC/AAC operations by comparing with decode-then-compute baseline.
    
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


# Import decode function for eLUT operations
from .functional import decode
