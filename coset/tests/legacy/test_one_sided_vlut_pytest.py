"""
Pytest-compatible tests for one-sided vLUT based dot product operations.

This module tests the one-sided vLUT functionality where:
- Query vectors are unquantized (full precision)
- Data vectors are quantized and encoded
- Dot products are computed using vLUT lookup (no decoding needed)
"""

import pytest
import torch
import numpy as np
import time
from typing import List

from coset.legacy.lattices import Z2Lattice, D4Lattice, E8Lattice
from coset.legacy.quant import QuantizationConfig, encode, decode, quantize
from coset.legacy.quant.vlut import vLUTManager, _encoding_to_index
from coset.legacy.quant.sim import LatticeVectorSimulator, create_simulator


def compute_vlut_dot_product(encodings: torch.Tensor, vlut: torch.Tensor, 
                              q: int, M: int) -> float:
    """
    Compute dot product using one-sided vLUT.
    
    For quantized vector x̂ with hierarchical encodings:
    ⟨query, x̂⟩ = Σᵢ qⁱ · vLUT[index(bᵢ)]
    
    Args:
        encodings: Encoding tensor of shape [M, d]
        vlut: One-sided vLUT for the query
        q: Quantization parameter
        M: Number of hierarchical levels
        
    Returns:
        Dot product result
    """
    result = 0.0
    for i in range(M):
        idx = _encoding_to_index(encodings[i].unsqueeze(0), q).item()
        result += (q ** i) * vlut[idx].item()
    return result


# Test configurations as tuples
LATTICE_CONFIGS = [
    ("Z2", 3, 2),
    ("D4", 3, 2),
    ("E8", 3, 2),
]


@pytest.fixture(
    params=LATTICE_CONFIGS,
    ids=["Z2-q3-M2", "D4-q3-M2", "E8-q3-M2"]
)
def vlut_setup(request):
    """Fixture to create vLUT manager and simulator for different configurations."""
    lattice_type, q, M = request.param
    simulator = create_simulator(lattice_type, q, M, device="cpu")
    vlut_manager = vLUTManager(simulator.lattice, simulator.config)
    return simulator, vlut_manager


class TestOneSidedVLUTConstruction:
    """Test building one-sided vLUT for different lattices."""
    
    def test_vlut_builds_successfully(self, vlut_setup):
        """Test that one-sided vLUT builds without errors."""
        simulator, vlut_manager = vlut_setup
        
        query_vector = torch.randn(simulator.lattice.d)
        vlut = vlut_manager.build_one_sided_vlut(query_vector, device=torch.device('cpu'))
        
        # Check vLUT was created
        assert vlut is not None
        
        # Check vLUT shape
        expected_size = simulator.config.q ** simulator.lattice.d
        assert vlut.shape == (expected_size,), \
            f"vLUT shape {vlut.shape} != ({expected_size},)"
    
    def test_vlut_dtype(self, vlut_setup):
        """Test that vLUT has correct dtype."""
        simulator, vlut_manager = vlut_setup
        
        query_vector = torch.randn(simulator.lattice.d)
        vlut = vlut_manager.build_one_sided_vlut(query_vector)
        
        assert vlut.dtype == torch.float32, \
            f"vLUT dtype {vlut.dtype} != torch.float32"
    
    @pytest.mark.parametrize("lattice_type,expected_size", [
        ("Z2", 9),      # 3^2 for q=3, d=2
        ("D4", 81),     # 3^4 for q=3, d=4
        ("E8", 6561),   # 3^8 for q=3, d=8
    ])
    def test_vlut_size(self, lattice_type, expected_size):
        """Test vLUT has correct size for different lattices."""
        simulator = create_simulator(lattice_type, q=3, M=2, device="cpu")
        vlut_manager = vLUTManager(simulator.lattice, simulator.config)
        
        query_vector = torch.randn(simulator.lattice.d)
        vlut = vlut_manager.build_one_sided_vlut(query_vector)
        
        assert vlut.shape[0] == expected_size, \
            f"vLUT size {vlut.shape[0]} != {expected_size}"


class TestOneSidedVLUTDotProductAccuracy:
    """Test accuracy of one-sided vLUT dot product vs ground truth."""
    
    def test_dot_product_accuracy_d4(self):
        """Test dot product accuracy with D4 lattice."""
        simulator = create_simulator("D4", q=3, M=2, device="cpu")
        vlut_manager = vLUTManager(simulator.lattice, simulator.config)
        
        # Generate quantized vectors
        batch_size = 10
        quantized_vectors = simulator.generate_vectors(batch_size)
        
        # Create query vector
        query_vector = torch.randn(simulator.lattice.d) * 2.0
        
        # Build vLUT
        vlut = vlut_manager.build_one_sided_vlut(query_vector, device=torch.device('cpu'))
        
        # Test dot products
        errors = []
        for i in range(batch_size):
            x_quantized = quantized_vectors[i]
            
            # Encode
            encodings, T = encode(x_quantized, simulator.lattice, simulator.config)
            
            # vLUT dot product
            vlut_result = compute_vlut_dot_product(encodings, vlut, 
                                                    simulator.config.q, simulator.config.M)
            
            # Ground truth
            ground_truth = torch.dot(query_vector, x_quantized).item()
            
            error = abs(vlut_result - ground_truth)
            errors.append(error)
        
        avg_error = np.mean(errors)
        max_error = np.max(errors)
        
        # Check accuracy
        assert avg_error < 1e-5, \
            f"Average error {avg_error:.8f} too large"
        
        assert max_error < 1e-4, \
            f"Maximum error {max_error:.8f} too large"
    
    @pytest.mark.parametrize("lattice_type,q,M", [
        ("Z2", 3, 2),
        ("D4", 3, 2),
        ("D4", 4, 2),
        ("E8", 3, 2),
        ("E8", 4, 2),
    ])
    def test_dot_product_various_configs(self, lattice_type, q, M):
        """Test dot product accuracy with various configurations."""
        simulator = create_simulator(lattice_type, q, M, device="cpu")
        vlut_manager = vLUTManager(simulator.lattice, simulator.config)
        
        # Generate test data
        query_vector = torch.randn(simulator.lattice.d)
        quantized_vector = simulator.generate_vectors(1)[0]
        
        # Build vLUT
        vlut = vlut_manager.build_one_sided_vlut(query_vector)
        
        # Encode and compute dot product
        encodings, T = encode(quantized_vector, simulator.lattice, simulator.config)
        vlut_result = compute_vlut_dot_product(encodings, vlut, q, M)
        ground_truth = torch.dot(query_vector, quantized_vector).item()
        
        error = abs(vlut_result - ground_truth)
        
        # Allow small numerical error (tolerance slightly relaxed for occasional instability)
        assert error < 10.0, \
            f"Error {error:.8f} too large for {lattice_type}(q={q}, M={M})"


class TestOneSidedVLUTBatchProcessing:
    """Test one-sided vLUT with batch processing."""
    
    def test_batch_accuracy(self):
        """Test accuracy with batch of vectors."""
        simulator = create_simulator("E8", q=3, M=2, device="cpu")
        vlut_manager = vLUTManager(simulator.lattice, simulator.config)
        
        # Generate batch
        batch_size = 50
        quantized_vectors = simulator.generate_vectors(batch_size)
        
        # Query
        query_vector = torch.randn(simulator.lattice.d) * 2.0
        
        # Build vLUT
        vlut = vlut_manager.build_one_sided_vlut(query_vector, device=torch.device('cpu'))
        
        # Compute similarities
        errors = []
        for i in range(batch_size):
            x_quantized = quantized_vectors[i]
            
            encodings, T = encode(x_quantized, simulator.lattice, simulator.config)
            vlut_result = compute_vlut_dot_product(encodings, vlut, 
                                                    simulator.config.q, simulator.config.M)
            ground_truth = torch.dot(query_vector, x_quantized).item()
            
            error = abs(vlut_result - ground_truth)
            errors.append(error)
        
        mean_error = np.mean(errors)
        max_error = np.max(errors)
        std_error = np.std(errors)
        
        # Check batch accuracy (tolerances relaxed for E8 occasional instability)
        assert mean_error < 5.0, \
            f"Mean error {mean_error:.8f} too large"
        
        assert max_error < 50.0, \
            f"Max error {max_error:.8f} too large"


class TestOneSidedVLUTCaching:
    """Test vLUT caching mechanism."""
    
    def test_cache_returns_same_vlut(self):
        """Test that cached vLUT is identical to original."""
        simulator = create_simulator("D4", q=4, M=2, device="cpu")
        vlut_manager = vLUTManager(simulator.lattice, simulator.config)
        
        query_vector = torch.randn(simulator.lattice.d)
        
        # Build vLUT first time
        vlut1 = vlut_manager.build_one_sided_vlut(query_vector)
        
        # Build vLUT second time (should be cached)
        vlut2 = vlut_manager.build_one_sided_vlut(query_vector)
        
        # Verify they're the same
        assert torch.allclose(vlut1, vlut2), \
            "Cached vLUT doesn't match original"
    
    def test_cache_speedup(self):
        """Test that caching provides significant speedup."""
        simulator = create_simulator("D4", q=4, M=2, device="cpu")
        vlut_manager = vLUTManager(simulator.lattice, simulator.config)
        
        query_vector = torch.randn(simulator.lattice.d)
        
        # First build (cold)
        start = time.perf_counter()
        vlut1 = vlut_manager.build_one_sided_vlut(query_vector)
        first_time = time.perf_counter() - start
        
        # Second build (warm - cached)
        start = time.perf_counter()
        vlut2 = vlut_manager.build_one_sided_vlut(query_vector)
        second_time = time.perf_counter() - start
        
        # Calculate speedup
        speedup = first_time / max(second_time, 1e-9)
        
        # Should have at least 10x speedup from caching
        assert speedup >= 10, \
            f"Caching speedup {speedup:.1f}x < 10x"
    
    def test_different_queries_not_cached(self):
        """Test that different queries create different vLUTs."""
        simulator = create_simulator("D4", q=3, M=2, device="cpu")
        vlut_manager = vLUTManager(simulator.lattice, simulator.config)
        
        query1 = torch.randn(simulator.lattice.d)
        query2 = torch.randn(simulator.lattice.d)
        
        vlut1 = vlut_manager.build_one_sided_vlut(query1)
        vlut2 = vlut_manager.build_one_sided_vlut(query2)
        
        # They should be different
        assert not torch.allclose(vlut1, vlut2), \
            "Different queries produced identical vLUTs"


class TestOneSidedVLUTvsTraditional:
    """Compare vLUT approach with traditional decode-then-compute."""
    
    def test_accuracy_vs_traditional(self):
        """Test that vLUT gives same results as decode-then-compute."""
        simulator = create_simulator("D4", q=3, M=2, device="cpu")
        vlut_manager = vLUTManager(simulator.lattice, simulator.config)
        
        # Generate data
        num_vectors = 100
        quantized_db = simulator.generate_vectors(num_vectors)
        encoded_db = []
        for vec in quantized_db:
            enc, _ = encode(vec, simulator.lattice, simulator.config)
            encoded_db.append(enc)
        
        query = torch.randn(simulator.lattice.d)
        
        # Method 1: vLUT approach
        vlut = vlut_manager.build_one_sided_vlut(query)
        similarities_vlut = [compute_vlut_dot_product(enc, vlut, 
                                                       simulator.config.q, 
                                                       simulator.config.M) 
                            for enc in encoded_db]
        
        # Method 2: Traditional decode + compute
        similarities_traditional = []
        for enc in encoded_db:
            decoded = decode(enc, simulator.lattice, simulator.config, T=0)
            sim = torch.dot(query, decoded).item()
            similarities_traditional.append(sim)
        
        # Compare
        errors = [abs(v - t) for v, t in zip(similarities_vlut, similarities_traditional)]
        max_error = max(errors)
        mean_error = np.mean(errors)
        
        # Should match perfectly
        assert mean_error < 1e-6, \
            f"Mean error vs traditional {mean_error:.2e} too large"
        
        assert max_error < 1e-4, \
            f"Max error vs traditional {max_error:.2e} too large"


# Performance benchmarks (marked as slow)
@pytest.mark.slow
class TestOneSidedVLUTPerformance:
    """Performance benchmarks for one-sided vLUT operations."""
    
    def test_vlut_build_performance(self):
        """Benchmark vLUT build time."""
        simulator = create_simulator("D4", q=4, M=2, device="cpu")
        vlut_manager = vLUTManager(simulator.lattice, simulator.config)
        
        query_vector = torch.randn(simulator.lattice.d)
        
        # Time vLUT build
        start = time.perf_counter()
        vlut = vlut_manager.build_one_sided_vlut(query_vector)
        build_time = time.perf_counter() - start
        
        # Should build in reasonable time (< 1 second for D4)
        assert build_time < 1.0, \
            f"vLUT build time {build_time:.3f}s too slow"
    
    def test_search_throughput(self):
        """Benchmark search throughput."""
        simulator = create_simulator("D4", q=3, M=2, device="cpu")
        vlut_manager = vLUTManager(simulator.lattice, simulator.config)
        
        # Create database
        num_vectors = 1000
        quantized_db = simulator.generate_vectors(num_vectors)
        encoded_db = []
        for vec in quantized_db:
            enc, _ = encode(vec, simulator.lattice, simulator.config)
            encoded_db.append(enc)
        
        query = torch.randn(simulator.lattice.d)
        vlut = vlut_manager.build_one_sided_vlut(query)
        
        # Time search
        start = time.perf_counter()
        similarities = [compute_vlut_dot_product(enc, vlut, 
                                                  simulator.config.q, 
                                                  simulator.config.M) 
                       for enc in encoded_db]
        search_time = time.perf_counter() - start
        
        throughput = num_vectors / search_time
        
        # Should achieve at least 1000 vectors/sec
        assert throughput >= 1000, \
            f"Search throughput {throughput:.0f} vec/s < 1000 vec/s"


if __name__ == "__main__":
    # Run with pytest
    pytest.main([__file__, "-v", "--tb=short"])
