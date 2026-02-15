"""
Unit tests for E8 GPU-accelerated quantization operations.

Tests cover:
- Correctness: GPU results match CPU reference
- Device handling: Proper tensor device management
- Batch operations: Vectorized batch processing
- Edge cases: Special values and boundary conditions
"""

import pytest
import torch
import numpy as np
from coset.legacy.lattices import E8Lattice
from coset.legacy.quant import (
    QuantizationConfig,
    batch_e8_quantize,
    batch_encode_e8,
    batch_decode_e8,
    batch_quantize_e8
)
from coset.legacy.quant.functional import encode, decode, quantize


@pytest.fixture
def config():
    """Standard quantization config for testing."""
    return QuantizationConfig(
        lattice_type="E8",
        q=4,
        M=2,
        beta=1.0,
        alpha=1.0,
        disable_overload_protection=True
    )


@pytest.fixture
def small_batch():
    """Small batch for basic tests."""
    return torch.randn(10, 8)


@pytest.fixture
def large_batch():
    """Large batch for performance tests."""
    return torch.randn(1000, 8)


class TestE8Quantization:
    """Tests for E8 quantization operations."""
    
    def test_batch_e8_quantize_single(self, config):
        """Test batch quantization with single vector."""
        x = torch.randn(1, 8)
        result = batch_e8_quantize(x)
        
        assert result.shape == (1, 8)
        assert torch.all(torch.isfinite(result))
    
    def test_batch_e8_quantize_multiple(self, small_batch, config):
        """Test batch quantization with multiple vectors."""
        result = batch_e8_quantize(small_batch)
        
        assert result.shape == small_batch.shape
        assert torch.all(torch.isfinite(result))
    
    def test_batch_e8_quantize_device(self, config):
        """Test batch quantization on specific device."""
        device = torch.device('cpu')
        x = torch.randn(5, 8)
        
        result = batch_e8_quantize(x, device=device)
        assert result.device == device
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_batch_e8_quantize_gpu(self, config):
        """Test batch quantization on GPU."""
        device = torch.device('cuda')
        x = torch.randn(10, 8, device=device)
        
        result = batch_e8_quantize(x, device=device)
        assert result.device == device
    
    def test_batch_e8_quantize_idempotence(self, small_batch, config):
        """Test that quantization is idempotent (Q(Q(x)) = Q(x))."""
        q1 = batch_e8_quantize(small_batch)
        q2 = batch_e8_quantize(q1)
        
        # Quantized vectors should be close (allowing for floating point errors)
        assert torch.allclose(q1, q2, atol=1e-6)
    
    def test_batch_e8_quantize_near_origin(self, config):
        """Test quantization near origin."""
        x = torch.randn(10, 8) * 0.1  # Small values near origin
        result = batch_e8_quantize(x)
        
        assert torch.all(torch.isfinite(result))
    
    def test_batch_e8_quantize_large_values(self, config):
        """Test quantization with large values."""
        x = torch.randn(10, 8) * 100  # Large values
        result = batch_e8_quantize(x)
        
        assert torch.all(torch.isfinite(result))


class TestBatchEncoding:
    """Tests for batch encoding operations."""
    
    def test_batch_encode_shape(self, small_batch, config):
        """Test that encoding produces correct shape."""
        lattice = E8Lattice()
        encodings, T_values = batch_encode_e8(small_batch, lattice, config)
        
        batch_size = small_batch.shape[0]
        assert encodings.shape == (batch_size, config.M, 8)
        assert T_values.shape == (batch_size,)
    
    def test_batch_encode_device(self, small_batch, config):
        """Test encoding on specific device."""
        device = torch.device('cpu')
        lattice = E8Lattice(device=device)
        
        encodings, T_values = batch_encode_e8(small_batch, lattice, config, device=device)
        assert encodings.device == device
        assert T_values.device == device
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_batch_encode_gpu(self, small_batch, config):
        """Test encoding on GPU."""
        device = torch.device('cuda')
        lattice = E8Lattice(device=device)
        x = small_batch.to(device)
        
        encodings, T_values = batch_encode_e8(x, lattice, config, device=device)
        assert encodings.device == device
        assert T_values.device == device
    
    def test_batch_encode_correctness(self, config):
        """Test that batch encoding matches single vector encoding."""
        x = torch.randn(1, 8)
        x_single = x[0]
        
        lattice = E8Lattice()
        enc_single, T_single = encode(x_single, lattice, config)
        
        enc_batch, T_batch = batch_encode_e8(x, lattice, config)
        
        assert torch.allclose(enc_batch[0], enc_single)
        assert T_batch[0] == T_single


class TestBatchDecoding:
    """Tests for batch decoding operations."""
    
    def test_batch_decode_shape(self, small_batch, config):
        """Test that decoding produces correct shape."""
        lattice = E8Lattice()
        encodings, T_values = batch_encode_e8(small_batch, lattice, config)
        
        decoded = batch_decode_e8(encodings, T_values, lattice, config)
        assert decoded.shape == small_batch.shape
    
    def test_batch_decode_device(self, small_batch, config):
        """Test decoding on specific device."""
        device = torch.device('cpu')
        lattice = E8Lattice(device=device)
        
        encodings, T_values = batch_encode_e8(small_batch, lattice, config, device=device)
        decoded = batch_decode_e8(encodings, T_values, lattice, config, device=device)
        
        assert decoded.device == device
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_batch_decode_gpu(self, small_batch, config):
        """Test decoding on GPU."""
        device = torch.device('cuda')
        lattice = E8Lattice(device=device)
        x = small_batch.to(device)
        
        encodings, T_values = batch_encode_e8(x, lattice, config, device=device)
        decoded = batch_decode_e8(encodings, T_values, lattice, config, device=device)
        
        assert decoded.device == device
    
    def test_batch_decode_correctness(self, config):
        """Test that batch decode matches single vector decode."""
        x = torch.randn(1, 8)
        x_single = x[0]
        
        lattice = E8Lattice()
        enc_single, T_single = encode(x_single, lattice, config)
        decoded_single = decode(enc_single, lattice, config, T_single)
        
        enc_batch, T_batch = batch_encode_e8(x, lattice, config)
        decoded_batch = batch_decode_e8(enc_batch, T_batch, lattice, config)
        
        assert torch.allclose(decoded_batch[0], decoded_single, atol=1e-5)


class TestBatchQuantization:
    """Tests for complete batch quantization (encode + decode)."""
    
    def test_batch_quantize_roundtrip(self, small_batch, config):
        """Test that batch quantization produces stable results."""
        lattice = E8Lattice()
        
        # Quantize
        quantized = batch_quantize_e8(small_batch, lattice, config)
        
        # Quantize again - should be stable
        quantized2 = batch_quantize_e8(quantized, lattice, config)
        
        # Should be very close (idempotence)
        assert torch.allclose(quantized, quantized2, atol=1e-5)
    
    def test_batch_quantize_device(self, small_batch, config):
        """Test quantization on specific device."""
        device = torch.device('cpu')
        lattice = E8Lattice(device=device)
        
        quantized = batch_quantize_e8(small_batch, lattice, config, device=device)
        assert quantized.device == device
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_batch_quantize_gpu(self, small_batch, config):
        """Test quantization on GPU."""
        device = torch.device('cuda')
        lattice = E8Lattice(device=device)
        x = small_batch.to(device)
        
        quantized = batch_quantize_e8(x, lattice, config, device=device)
        assert quantized.device == device
    
    def test_batch_quantize_correctness(self, config):
        """Test that batch quantization matches single vector quantization."""
        x = torch.randn(1, 8)
        x_single = x[0]
        
        lattice = E8Lattice()
        quantized_single = quantize(x_single, lattice, config)
        quantized_batch = batch_quantize_e8(x, lattice, config)
        
        assert torch.allclose(quantized_batch[0], quantized_single, atol=1e-5)


class TestPerformance:
    """Performance and scalability tests."""
    
    def test_batch_e8_quantize_scaling(self, config):
        """Test that batch quantization scales with batch size."""
        for batch_size in [1, 10, 100]:
            x = torch.randn(batch_size, 8)
            result = batch_e8_quantize(x)
            assert result.shape == (batch_size, 8)
    
    def test_batch_encode_scaling(self, config):
        """Test that batch encoding scales with batch size."""
        lattice = E8Lattice()
        
        for batch_size in [1, 10, 100, 1000]:
            x = torch.randn(batch_size, 8)
            encodings, T_values = batch_encode_e8(x, lattice, config)
            assert encodings.shape[0] == batch_size
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_speedup(self, config):
        """Compare CPU vs GPU performance."""
        import time
        
        batch_size = 1000
        x_cpu = torch.randn(batch_size, 8)
        x_gpu = x_cpu.cuda()
        
        lattice_cpu = E8Lattice(device=torch.device('cpu'))
        lattice_gpu = E8Lattice(device=torch.device('cuda'))
        
        # CPU timing
        start = time.perf_counter()
        _ = batch_quantize_e8(x_cpu, lattice_cpu, config, device=torch.device('cpu'))
        cpu_time = time.perf_counter() - start
        
        # GPU timing (with warmup)
        _ = batch_quantize_e8(x_gpu, lattice_gpu, config, device=torch.device('cuda'))
        torch.cuda.synchronize()
        
        start = time.perf_counter()
        _ = batch_quantize_e8(x_gpu, lattice_gpu, config, device=torch.device('cuda'))
        torch.cuda.synchronize()
        gpu_time = time.perf_counter() - start
        
        speedup = cpu_time / gpu_time
        print(f"\nGPU speedup: {speedup:.2f}x (CPU: {cpu_time:.4f}s, GPU: {gpu_time:.4f}s)")
        
        # GPU should be faster for large batches
        assert speedup > 1.0, "GPU should be faster than CPU"


class TestEdgeCases:
    """Tests for edge cases and special values."""
    
    def test_batch_e8_quantize_zeros(self, config):
        """Test quantization of zero vectors."""
        x = torch.zeros(10, 8)
        result = batch_e8_quantize(x)
        
        assert torch.allclose(result, torch.zeros_like(result))
    
    def test_batch_e8_quantize_ones(self, config):
        """Test quantization of ones."""
        x = torch.ones(10, 8)
        result = batch_e8_quantize(x)
        
        assert torch.all(torch.isfinite(result))
    
    def test_batch_e8_quantize_extreme_values(self, config):
        """Test quantization with extreme values."""
        x = torch.randn(10, 8) * 1e6
        result = batch_e8_quantize(x)
        
        assert torch.all(torch.isfinite(result))
    
    def test_batch_encode_edge_cases(self, config):
        """Test encoding with edge case values."""
        lattice = E8Lattice()
        
        # Test with very small values
        x_small = torch.randn(5, 8) * 1e-6
        encodings, T_values = batch_encode_e8(x_small, lattice, config)
        assert encodings.shape[0] == 5
        
        # Test with very large values
        x_large = torch.randn(5, 8) * 1e6
        encodings, T_values = batch_encode_e8(x_large, lattice, config)
        assert encodings.shape[0] == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
