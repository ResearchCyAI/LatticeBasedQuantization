"""
LSQ Scalar Quantization Codecs Module

This module provides LSQ (Learned Step Size Quantization) scalar quantization functions
that work with the existing HNLQLinearQAT layer infrastructure.

The quantization formula is: quantized = clip(round(x), -q_n, q_p) where q_n = q_p = 2^{bits-1}-1
and bits = floor(M * log2(q)). The scaling is handled by HNLQLinearQAT's learnable beta parameters.
"""

import torch
import math
from typing import Optional


def lsq_scalar_quantize(x: torch.Tensor, q: int, M: int = 2) -> torch.Tensor:
    """
    LSQ scalar quantization function.
    
    This function applies symmetric scalar quantization to input tensors that have already
    been scaled by the learned beta parameters from HNLQLinearQAT.
    
    Args:
        x: Input tensor that has been pre-scaled by learned beta parameters
        q: Base quantization levels (alphabet size)
        M: Hierarchical levels (effective bits = floor(M * log2(q)))
        
    Returns:
        Quantized tensor in the same scaled space
        
    Note:
        The input tensor x is expected to be already scaled by the learned scaling factors
        from HNLQLinearQAT. This function only applies the quantization step.
    """
    # Calculate effective bits: bits = floor(M * log2(q))
    bits = int(math.floor(M * math.log2(q)))
    
    # Calculate quantization bounds: q_n = q_p = 2^{bits-1}-1
    q_max = (2 ** (bits - 1)) - 1
    q_min = -q_max
    
    # Apply symmetric quantization: clip(round(x), q_min, q_max)
    x_quantized = torch.clamp(torch.round(x), q_min, q_max)
    
    return x_quantized


def lsq_scalar_encode(x: torch.Tensor, q: int, M: int = 2) -> torch.Tensor:
    """
    LSQ scalar encoding (same as quantization for scalar case).
    
    This is provided for API consistency with other codec modules.
    For scalar quantization, encoding and quantization are the same operation.
    
    Args:
        x: Input tensor that has been pre-scaled by learned beta parameters
        q: Base quantization levels (alphabet size)
        M: Hierarchical levels (effective bits = floor(M * log2(q)))
        
    Returns:
        Encoded/quantized tensor
    """
    return lsq_scalar_quantize(x, q, M)


def lsq_scalar_decode(x_encoded: torch.Tensor, q: int, M: int = 2) -> torch.Tensor:
    """
    LSQ scalar decoding (identity for scalar case).
    
    For scalar quantization, the encoded values are already in the correct format
    for use in computations, so decoding is a no-op.
    
    Args:
        x_encoded: Encoded/quantized tensor
        q: Base quantization levels (alphabet size) - unused but kept for API consistency
        M: Hierarchical levels - unused but kept for API consistency
        
    Returns:
        Decoded tensor (same as input)
    """
    return x_encoded


def lsq_scalar_quantize_batch(x: torch.Tensor, q: int, M: int = 2) -> torch.Tensor:
    """
    LSQ scalar quantization for batch processing.
    
    This function applies LSQ scalar quantization to a batch of tensors.
    For scalar quantization, this is the same as the regular quantize function.
    
    Args:
        x: Input tensor of shape [batch_size, ...] that has been pre-scaled
        q: Base quantization levels (alphabet size)
        M: Hierarchical levels (effective bits = floor(M * log2(q)))
        
    Returns:
        Quantized tensor with same shape as input
    """
    return lsq_scalar_quantize(x, q, M)


def get_lsq_scalar_bounds(q: int, M: int = 2) -> tuple[int, int]:
    """
    Get quantization bounds for LSQ scalar quantization.
    
    Args:
        q: Base quantization levels (alphabet size)
        M: Hierarchical levels (effective bits = floor(M * log2(q)))
        
    Returns:
        Tuple of (q_min, q_max) bounds
    """
    bits = int(math.floor(M * math.log2(q)))
    q_max = (2 ** (bits - 1)) - 1
    q_min = -q_max
    return q_min, q_max


def get_lsq_scalar_effective_bits(q: int, M: int = 2) -> int:
    """
    Get effective number of bits for LSQ scalar quantization.
    
    Args:
        q: Base quantization levels (alphabet size)
        M: Hierarchical levels (effective bits = floor(M * log2(q)))
        
    Returns:
        Effective number of bits
    """
    return int(math.floor(M * math.log2(q)))


def get_lsq_scalar_quantization_levels(q: int, M: int = 2) -> int:
    """
    Get total number of quantization levels for LSQ scalar quantization.
    
    Args:
        q: Base quantization levels (alphabet size)
        M: Hierarchical levels (effective bits = floor(M * log2(q)))
        
    Returns:
        Total number of quantization levels
    """
    bits = get_lsq_scalar_effective_bits(q, M)
    return 2 ** bits
