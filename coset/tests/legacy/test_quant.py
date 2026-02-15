"""
Tests for quantization functions.

This module tests the correctness of encoding, decoding, and quantization
algorithms.
"""

import pytest
import torch
import numpy as np
from coset.legacy.quant import QuantizationConfig, encode, decode, quantize, mac_modq, accumulate_modq
from coset.legacy.lattices import Z2Lattice, D4Lattice


class TestQuantizationConfig:
    """Test cases for quantization configuration."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = QuantizationConfig()
        assert config.lattice_type == "D4"
        assert config.q == 4
        assert config.M == 2
        assert config.beta == 1.0
        assert config.alpha == 1.0
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = QuantizationConfig(
            lattice_type="Z2",
            q=8,
            M=3,
            beta=2.0,
            alpha=1.5
        )
        assert config.lattice_type == "Z2"
        assert config.q == 8
        assert config.M == 3
        assert config.beta == 2.0
        assert config.alpha == 1.5
    
    def test_config_validation(self):
        """Test configuration validation."""
        with pytest.raises(ValueError):
            QuantizationConfig(lattice_type="INVALID")
        
        with pytest.raises(ValueError):
            QuantizationConfig(q=0)
        
        with pytest.raises(ValueError):
            QuantizationConfig(beta=0)
    
    def test_config_serialization(self):
        """Test configuration serialization."""
        config = QuantizationConfig(lattice_type="Z2", q=8, M=3)
        
        # Test to_dict
        config_dict = config.to_dict()
        assert config_dict["lattice_type"] == "Z2"
        assert config_dict["q"] == 8
        assert config_dict["M"] == 3
        
        # Test from_dict
        config2 = QuantizationConfig.from_dict(config_dict)
        assert config2.lattice_type == config.lattice_type
        assert config2.q == config.q
        assert config2.M == config.M


class TestQuantizationFunctions:
    """Test cases for quantization functions."""
    
    def test_encode_decode_roundtrip(self):
        """Test encode-decode round-trip for Z² lattice."""
        lattice = Z2Lattice()
        config = QuantizationConfig(lattice_type="Z2", q=4, M=2)
        
        # Test with simple input
        x = torch.tensor([1.0, 2.0])
        b, T = encode(x, lattice, config)
        
        # Check output shapes
        assert b.shape == (2, 2)  # M=2, d=2
        assert isinstance(T, int)
        assert T >= 0
        
        # Decode and check reconstruction
        x_reconstructed = decode(b, lattice, config, T)
        assert x_reconstructed.shape == x.shape
        
        # Should be close to original (within quantization error)
        assert torch.norm(x_reconstructed - x) < 1.0
    
    def test_quantize_function(self):
        """Test the quantize convenience function."""
        lattice = Z2Lattice()
        config = QuantizationConfig(lattice_type="Z2", q=4, M=2)
        
        x = torch.tensor([1.0, 2.0])
        x_quantized = quantize(x, lattice, config)
        
        assert x_quantized.shape == x.shape
        assert torch.norm(x_quantized - x) < 1.0
    
    def test_d4_quantization(self):
        """Test quantization with D₄ lattice."""
        lattice = D4Lattice()
        config = QuantizationConfig(lattice_type="D4", q=4, M=2)
        
        x = torch.randn(4) * 2.0
        b, T = encode(x, lattice, config)
        
        assert b.shape == (2, 4)  # M=2, d=4
        assert isinstance(T, int)
        
        x_reconstructed = decode(b, lattice, config, T)
        assert x_reconstructed.shape == x.shape
    
    def test_modq_arithmetic(self):
        """Test modular arithmetic functions."""
        x = torch.tensor([1, 2, 3])
        y = torch.tensor([2, 3, 4])
        q = 5
        
        # Test MAC
        result = mac_modq(x, y, q)
        expected = (1*2 + 2*3 + 3*4) % 5
        assert result == expected
        
        # Test accumulation
        acc = torch.tensor(0)
        for val in x:
            acc = accumulate_modq(acc, val, q)
        expected = torch.sum(x) % q
        assert acc == expected
    
    def test_batch_operations(self):
        """Test batch encoding and decoding."""
        from coset.quant import batch_encode, batch_decode
        
        lattice = Z2Lattice()
        config = QuantizationConfig(lattice_type="Z2", q=4, M=2)
        
        # Test batch encoding
        X = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        encoded_vectors, scaling_counts = batch_encode(X, lattice, config)
        
        assert encoded_vectors.shape == (3, 2, 2)  # batch_size, M, d
        assert scaling_counts.shape == (3,)
        
        # Test batch decoding
        decoded_vectors = batch_decode(encoded_vectors, scaling_counts, lattice, config)
        assert decoded_vectors.shape == X.shape
        
        # Check reconstruction quality
        for i in range(X.shape[0]):
            assert torch.norm(decoded_vectors[i] - X[i]) < 1.0


class TestOverloadHandling:
    """Test cases for overload handling."""
    
    def test_overload_detection(self):
        """Test overload detection and handling."""
        lattice = Z2Lattice()
        config = QuantizationConfig(
            lattice_type="Z2", 
            q=2,  # Small q to increase overload probability
            M=3,  # Large M to increase overload probability
            beta=0.1,  # Small beta to increase overload probability
            max_scaling_iterations=5
        )
        
        # Use large input to trigger overload
        x = torch.tensor([10.0, 20.0])
        b, T = encode(x, lattice, config)
        
        # Should have some scaling iterations
        assert T >= 0
        assert T <= config.max_scaling_iterations
    
    def test_disable_overload_protection(self):
        """Test disabling overload protection."""
        lattice = Z2Lattice()
        config = QuantizationConfig(
            lattice_type="Z2",
            q=2,
            M=3,
            disable_overload_protection=True
        )
        
        x = torch.tensor([10.0, 20.0])
        b, T = encode(x, lattice, config)
        
        # Should not perform scaling iterations
        assert T == 0


if __name__ == "__main__":
    pytest.main([__file__])
