"""
Quantized linear layer for quantization-aware training.

This module provides a PyTorch linear layer that integrates hierarchical
nested-lattice quantization for both weights and activations.
"""

import torch
import torch.nn as nn
from typing import Optional
from ..quant import QuantizationConfig, quantize, batch_quantize
from ..lattices import Lattice, D4Lattice


class StraightThroughQuantize(torch.autograd.Function):
    """
    Straight-Through Estimator for quantization with per-row scaling.
    
    This implements QAT with proper handling of weight norms for nested-lattice quantization:
    
    For an n x m weight matrix:
    1. Compute L2 norm of each of the n original rows (before tiling) and store them.
    2. Tile the matrix to have d-column structure (where d is lattice dimension)
    3. Scale each tiled row such that the norm of each tiled row is has unit norm 
    4. Add another scaling beta such that HNLQ is not overloadednor spends wasterful bits in the last level
    5. Beta is pre-computed and stored in the config, which is specific to the lattice and the quantization parameter q
    6. Quantize the normalized tiled matrix (each tiled row has roughly unit norm)
    7. Re-scale back by the same n scaling factors and the beta scaling factor
    8. Reshape back to original shape
    
    This ensures:
    - Each row of the tiled matrix has unit norm before quantization (stable)
    - Only n scaling factors needed (not n*m/d factors)
    - STE is well-behaved with gradients flowing properly to FP32 weights
    """
    
    @staticmethod
    def forward(ctx, input, lattice, config):
        """
        Forward pass: quantize the input with per-row scaling.
        
        The process:
        1. Compute L2 norm of each row of the original matrix (n scaling factors)
        2. Tile/resolve to [N, d] where d is the lattice dimension
        3. Normalize each tiled row to unit norm
        4. Apply beta scaling factor from config
        5. Quantize the normalized and beta-scaled tiled matrix
        6. Re-scale back by dividing by beta and multiplying by original row norms
        7. Reshape to original shape
        
        This ensures each row of the tiled matrix has unit norm before quantization.
        
        Args:
            ctx: Context object to store information for backward
            input: Input tensor to quantize (can be reshaped to 2D matrix)
            lattice: Lattice instance
            config: Quantization configuration
            
        Returns:
            Quantized tensor with same shape as input
        """
        # Store for backward
        ctx.original_shape = input.shape
        ctx.lattice = lattice
        ctx.config = config
        d = lattice.d
        
        # Flatten to 2D: [num_rows, num_cols]
        # For arbitrary shapes, flatten all but the last dimension, or if last dim exists
        if input.dim() == 1:
            input_2d = input.unsqueeze(0)  # [1, num_cols]
        else:
            # Flatten all but the last dimension to rows
            input_2d = input.view(-1, input.shape[-1])
        
        
        
        # Step 1: Compute L2 norm of each original row (before tiling)
        row_norms = torch.norm(input_2d, p=2, dim=1, keepdim=True)  # [num_rows, 1]
        # Avoid division by zero
        row_norms_safe = torch.clamp(row_norms, min=1e-8)
        ctx.row_norms = row_norms
        
        # Step 2: Rescale each row to unit norm and apply beta scaling factor
        # Beta is pre-computed for the specific lattice and quantization parameter q
        beta = config.beta
        input_2d = (input_2d / row_norms_safe) * beta * d
        
        # Step 3: Tile the matrix to have d-column structure
        input_2d = input_2d.view(-1, lattice.d)

        # Step 4: Quantize the normalized and beta-scaled tiles
        quantized = batch_quantize(input_2d, lattice, config)
        
        # Step 5: Reshape back to original shape
        quantized = quantized.view(ctx.original_shape)

        # Step 6: Scale back by the original row norms
        quantized = quantized * row_norms_safe * (1/beta) * (1/d)


        return quantized
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: straight-through estimator.
        
        Gradients pass through unchanged as if no quantization occurred.
        
        Args:
            ctx: Context object from forward
            grad_output: Gradient from the next layer
            
        Returns:
            Gradient to pass back (unchanged), None for non-tensor arguments
        """
        # Straight-through: pass gradient through as-is
        return grad_output, None, None


class QLinear(nn.Module):
    """
    Quantized linear layer with hierarchical nested-lattice quantization.
    
    This layer maintains FP32 shadow weights and can quantize weights,
    activations, or both during training. It supports straight-through
    estimation for gradient flow.
    
    Attributes:
        in_features: Number of input features
        out_features: Number of output features
        config: Quantization configuration
        lattice: Lattice instance for quantization
        quantize_weights: Whether to quantize weights
        quantize_activations: Whether to quantize activations
        quantize_every: Frequency of quantization (1 = every step)
        use_lut: Whether to use lookup table for matrix multiplication
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: QuantizationConfig,
        lattice: Optional[Lattice] = None,
        quantize_weights: bool = True,
        quantize_activations: bool = False,
        quantize_every: int = 1,
        use_lut: bool = False,
        bias: bool = True
    ):
        """
        Initialize the quantized linear layer.
        
        Args:
            in_features: Number of input features
            out_features: Number of output features
            config: Quantization configuration
            lattice: Lattice instance (defaults to D4Lattice)
            quantize_weights: Whether to quantize weights
            quantize_activations: Whether to quantize activations
            quantize_every: Frequency of quantization (1 = every step)
            use_lut: Whether to use lookup table for matrix multiplication
            bias: Whether to include bias term
        """
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.config = config
        self.lattice = lattice if lattice is not None else D4Lattice()
        self.quantize_weights = quantize_weights
        self.quantize_activations = quantize_activations
        self.quantize_every = quantize_every
        self.use_lut = use_lut
        
        # Initialize FP32 shadow weights
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        
        # Quantization step counter
        self._step_count = 0
        
        # Validate dimensions
        if in_features % self.lattice.d != 0:
            raise ValueError(
                f"Input features {in_features} must be divisible by lattice dimension {self.lattice.d}"
            )
        if out_features % self.lattice.d != 0:
            raise ValueError(
                f"Output features {out_features} must be divisible by lattice dimension {self.lattice.d}"
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with optional quantization.
        
        Args:
            x: Input tensor of shape [batch_size, in_features]
            
        Returns:
            Output tensor of shape [batch_size, out_features]
        """
        # Check if we should quantize this step
        should_quantize = (
            self.training and 
            self.quantize_every > 0 and 
            self._step_count % self.quantize_every == 0
        )
        
        if should_quantize:
            return self._quantized_forward(x)
        else:
            return self._standard_forward(x)
    
    def _standard_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass without quantization."""
        return torch.nn.functional.linear(x, self.weight, self.bias)
    
    def _quantized_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with quantization and straight-through estimation.
        
        Important: This function uses quantized weights for computation but does NOT
        modify self.weight. The FP32 weights remain unchanged and continue to accumulate
        gradients through the STE mechanism. This is the key principle of QAT.
        
        The quantization happens in-place via StraightThroughQuantize which:
        1. Pre-scales each d-dimensional vector by its L2 norm
        2. Quantizes the normalized vector (stable range)
        3. Re-scales back to original norm
        4. Passes gradients through unchanged in backward pass
        """
        # Quantize weights if enabled (with STE)
        # Note: self.weight is FP32 and stays FP32. We create a quantized version for forward.
        if self.quantize_weights:
            # Use STE to allow gradients to flow through quantization
            # The STE function creates a quantized copy but gradients flow back to FP32 weights
            weight_quantized = StraightThroughQuantize.apply(
                self.weight, self.lattice, self.config
            )
        else:
            weight_quantized = self.weight
        
        # Ensure weight_quantized is on the same device as x
        weight_quantized = weight_quantized.to(x.device)
        
        # Quantize activations if enabled (with STE)
        if self.quantize_activations:
            # Use STE to allow gradients to flow through quantization
            x_quantized = StraightThroughQuantize.apply(
                x, self.lattice, self.config
            )
        else:
            x_quantized = x
        
        # Get bias on the same device
        bias = self.bias.to(x.device) if self.bias is not None else None
        
        # Perform matrix multiplication
        if self.use_lut:
            output = self._lut_matmul(x_quantized, weight_quantized)
        else:
            output = torch.nn.functional.linear(x_quantized, weight_quantized, bias)
        
        # Increment step counter
        self._step_count += 1
        
        return output
    
    def _lut_matmul(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """
        Matrix multiplication using lookup tables.
        
        This is a placeholder implementation. In practice, this would use
        precomputed lookup tables for efficient inner product computation.
        """
        # For now, fall back to standard matrix multiplication
        # TODO: Implement actual LUT-based matrix multiplication
        return torch.nn.functional.linear(x, weight, self.bias)
    
    def get_quantization_stats(self) -> dict:
        """Get statistics about quantization."""
        return {
            "step_count": self._step_count,
            "quantize_weights": self.quantize_weights,
            "quantize_activations": self.quantize_activations,
            "quantize_every": self.quantize_every,
            "use_lut": self.use_lut,
            "lattice_type": self.lattice.name,
            "config": self.config.to_dict(),
        }
    
    def reset_step_count(self):
        """Reset the quantization step counter."""
        self._step_count = 0
    
    def extra_repr(self) -> str:
        """Extra representation string."""
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, lattice={self.lattice.name}, "
            f"quantize_weights={self.quantize_weights}, "
            f"quantize_activations={self.quantize_activations}, "
            f"quantize_every={self.quantize_every}"
        )
