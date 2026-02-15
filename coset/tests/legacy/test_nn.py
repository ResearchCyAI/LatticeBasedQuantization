"""
Tests for neural network modules.

This module tests the quantized linear layer and its integration
with PyTorch's autograd system.
"""

import pytest
import torch
import torch.nn as nn
from coset.legacy.nn import QLinear
from coset.legacy.quant import QuantizationConfig
from coset.legacy.lattices import Z2Lattice, D4Lattice


class TestQLinear:
    """Test cases for quantized linear layer."""
    
    def test_initialization(self):
        """Test layer initialization."""
        config = QuantizationConfig(lattice_type="Z2", q=4, M=2)
        layer = QLinear(4, 8, config)
        
        assert layer.in_features == 4
        assert layer.out_features == 8
        assert layer.weight.shape == (8, 4)
        assert layer.bias.shape == (8,)
        assert layer.lattice.name == "Z2"
    
    def test_forward_pass(self):
        """Test forward pass without quantization."""
        config = QuantizationConfig(lattice_type="Z2", q=4, M=2)
        layer = QLinear(4, 8, config, quantize_every=0)  # Disable quantization
        
        x = torch.randn(2, 4)
        output = layer(x)
        
        assert output.shape == (2, 8)
        
        # Should match standard linear layer
        standard_layer = nn.Linear(4, 8)
        standard_layer.weight.data = layer.weight.data
        standard_layer.bias.data = layer.bias.data
        expected_output = standard_layer(x)
        
        assert torch.allclose(output, expected_output)
    
    def test_quantized_forward_pass(self):
        """Test forward pass with quantization."""
        config = QuantizationConfig(lattice_type="Z2", q=4, M=2)
        layer = QLinear(4, 8, config, quantize_weights=True, quantize_every=1)
        
        x = torch.randn(2, 4)
        output = layer(x)
        
        assert output.shape == (2, 8)
        assert layer._step_count == 1
    
    def test_weight_quantization(self):
        """Test weight quantization."""
        config = QuantizationConfig(lattice_type="Z2", q=4, M=2)
        layer = QLinear(4, 8, config, quantize_weights=True, quantize_every=1)
        
        # Set specific weights
        layer.weight.data = torch.randn(8, 4) * 2.0
        
        x = torch.randn(2, 4)
        output = layer(x)
        
        # Check that weights were quantized
        assert layer._step_count == 1
    
    def test_activation_quantization(self):
        """Test activation quantization."""
        config = QuantizationConfig(lattice_type="Z2", q=4, M=2)
        layer = QLinear(4, 8, config, quantize_activations=True, quantize_every=1)
        
        x = torch.randn(2, 4)
        output = layer(x)
        
        assert output.shape == (2, 8)
        assert layer._step_count == 1
    
    def test_gradient_flow(self):
        """Test gradient flow through quantized layer."""
        config = QuantizationConfig(lattice_type="Z2", q=4, M=2)
        layer = QLinear(4, 8, config, quantize_weights=True, quantize_every=1)
        
        x = torch.randn(2, 4, requires_grad=True)
        output = layer(x)
        loss = output.sum()
        loss.backward()
        
        # Check that gradients exist
        assert x.grad is not None
        assert layer.weight.grad is not None
        assert layer.bias.grad is not None
    
    def test_dimension_validation(self):
        """Test dimension validation."""
        config = QuantizationConfig(lattice_type="D4", q=4, M=2)
        
        # Should work with divisible dimensions
        layer = QLinear(8, 12, config)  # 8 and 12 are divisible by 4
        assert layer.in_features == 8
        assert layer.out_features == 12
        
        # Should fail with non-divisible dimensions
        with pytest.raises(ValueError):
            QLinear(7, 12, config)  # 7 is not divisible by 4
        
        with pytest.raises(ValueError):
            QLinear(8, 13, config)  # 13 is not divisible by 4
    
    def test_quantization_stats(self):
        """Test quantization statistics."""
        config = QuantizationConfig(lattice_type="Z2", q=4, M=2)
        layer = QLinear(4, 8, config)
        
        stats = layer.get_quantization_stats()
        
        assert "step_count" in stats
        assert "quantize_weights" in stats
        assert "quantize_activations" in stats
        assert "lattice_type" in stats
        assert "config" in stats
        
        assert stats["step_count"] == 0
        assert stats["lattice_type"] == "Z2"
    
    def test_step_count_reset(self):
        """Test step count reset."""
        config = QuantizationConfig(lattice_type="Z2", q=4, M=2)
        layer = QLinear(4, 8, config, quantize_every=1)
        
        x = torch.randn(2, 4)
        layer(x)
        assert layer._step_count == 1
        
        layer.reset_step_count()
        assert layer._step_count == 0
    
    def test_different_lattices(self):
        """Test with different lattice types."""
        # Test Z2 lattice
        config_z2 = QuantizationConfig(lattice_type="Z2", q=4, M=2)
        layer_z2 = QLinear(4, 8, config_z2)
        assert layer_z2.lattice.name == "Z2"
        
        # Test D4 lattice
        config_d4 = QuantizationConfig(lattice_type="D4", q=4, M=2)
        layer_d4 = QLinear(8, 12, config_d4)  # Use dimensions divisible by 4
        assert layer_d4.lattice.name == "D4"
    
    def test_custom_lattice(self):
        """Test with custom lattice instance."""
        lattice = Z2Lattice()
        config = QuantizationConfig(lattice_type="Z2", q=4, M=2)
        layer = QLinear(4, 8, config, lattice=lattice)
        
        assert layer.lattice is lattice
        assert layer.lattice.name == "Z2"


if __name__ == "__main__":
    pytest.main([__file__])
