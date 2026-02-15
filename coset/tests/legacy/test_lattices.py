"""
Tests for lattice implementations.

This module tests the correctness of lattice nearest-neighbor algorithms
and basic lattice operations.
"""

import pytest
import torch
import numpy as np
from coset.legacy.lattices import Z2Lattice, D4Lattice, E8Lattice


class TestZ2Lattice:
    """Test cases for Z² lattice."""
    
    def test_initialization(self):
        """Test lattice initialization."""
        lattice = Z2Lattice()
        assert lattice.name == "Z2"
        assert lattice.d == 2
        assert torch.allclose(lattice.G, torch.eye(2))
    
    def test_quantization(self):
        """Test nearest-neighbor quantization."""
        lattice = Z2Lattice()
        
        # Test simple cases
        x = torch.tensor([0.3, 0.7])
        result = lattice.Q(x)
        expected = torch.tensor([0.0, 1.0])
        assert torch.allclose(result, expected)
        
        # Test negative values
        x = torch.tensor([-0.3, -0.7])
        result = lattice.Q(x)
        expected = torch.tensor([0.0, -1.0])
        assert torch.allclose(result, expected)
    
    def test_encode_decode_coords(self):
        """Test encoding and decoding coordinates."""
        lattice = Z2Lattice()
        
        # Test round-trip
        x = torch.tensor([2.0, 3.0])
        b = lattice.encode_coords(x, q=4)
        x_reconstructed = lattice.decode_coords(b, q=4)
        assert torch.allclose(x, x_reconstructed)


class TestD4Lattice:
    """Test cases for D₄ lattice."""
    
    def test_initialization(self):
        """Test lattice initialization."""
        lattice = D4Lattice()
        assert lattice.name == "D4"
        assert lattice.d == 4
        assert lattice.G.shape == (4, 4)
    
    def test_quantization(self):
        """Test nearest-neighbor quantization."""
        lattice = D4Lattice()
        
        # Test simple case
        x = torch.tensor([0.3, 0.7, 0.2, 0.8])
        result = lattice.Q(x)
        
        # Check that sum is even (D₄ constraint)
        assert torch.sum(result) % 2 == 0
        
        # Test that result is close to input
        assert torch.norm(result - x) < torch.norm(x)
    
    def test_g_x_function(self):
        """Test the g_x helper function."""
        lattice = D4Lattice()
        
        # Test with sum that needs adjustment
        x = torch.tensor([0.1, 0.1, 0.1, 0.1])
        result = lattice.g_x(x)
        
        # Should be different from simple rounding
        simple_round = lattice.custom_round(x)
        assert not torch.allclose(result, simple_round)


class TestE8Lattice:
    """Test cases for E₈ lattice."""
    
    def test_initialization(self):
        """Test lattice initialization."""
        lattice = E8Lattice()
        assert lattice.name == "E8"
        assert lattice.d == 8
        assert lattice.G.shape == (8, 8)
    
    def test_quantization(self):
        """Test nearest-neighbor quantization."""
        lattice = E8Lattice()
        
        # Test simple case
        x = torch.randn(8) * 0.5
        result = lattice.Q(x)
        
        # Test that result is close to input
        assert torch.norm(result - x) < torch.norm(x)


class TestLatticeBase:
    """Test cases for base lattice functionality."""
    
    def test_custom_round(self):
        """Test custom rounding function."""
        lattice = Z2Lattice()
        
        # Test edge case at 0.5
        x = torch.tensor([0.5, -0.5])
        result = lattice.custom_round(x)
        
        # Should round toward zero
        expected = torch.tensor([0.0, 0.0])
        assert torch.allclose(result, expected)
    
    def test_tie_dither_generation(self):
        """Test tie dither generation."""
        lattice = Z2Lattice()
        
        dither = lattice.generate_tie_dither()
        assert dither.shape == (2,)
        assert torch.norm(dither) > 0
        
        # Should be deterministic
        dither2 = lattice.generate_tie_dither()
        assert torch.allclose(dither, dither2)


if __name__ == "__main__":
    pytest.main([__file__])
