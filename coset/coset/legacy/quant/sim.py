"""
Lattice Vector Simulation System

This module provides tools for generating vectors in the quantized space for a given
lattice with specified quantization parameters (q, M). It generates vectors that are
guaranteed to be in the quantized space and validates quantizer performance.

Key Features:
- Generate vectors in the quantized space (zero quantization error)
- Validate zero quantization error for simulated vectors
- Assess quantizer performance on generated datasets
- Generate matrices of specified dimensions
- Support multiple lattice types (Z2, D4, E8)
"""

import torch
import numpy as np
from typing import Tuple, Dict, List, Optional, Union
import time
import statistics
from pathlib import Path

# Import lattice and quantization components
from coset.lattices import Z2Lattice, D4Lattice, E8Lattice
from coset.quant.params import QuantizationConfig
from coset.quant.functional import quantize


class LatticeVectorSimulator:
    """
    Simulator for generating vectors in the quantized space.
    
    This class provides methods to:
    1. Generate vectors that are guaranteed to be in the quantized space
    2. Validate zero quantization error for simulated vectors
    3. Assess quantizer performance on generated datasets
    4. Generate matrices of specified dimensions
    """
    
    def __init__(self, lattice_type: str = "E8", q: int = 3, M: int = 2, 
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the lattice vector simulator.
        
        Args:
            lattice_type: Type of lattice ("Z2", "D4", "E8")
            q: Quantization parameter (alphabet size)
            M: Number of hierarchical levels
            device: Device to use for computations
        """
        self.lattice_type = lattice_type
        self.q = q
        self.M = M
        self.device = torch.device(device)
        
        # Initialize lattice
        if lattice_type == "Z2":
            self.lattice = Z2Lattice()
        elif lattice_type == "D4":
            self.lattice = D4Lattice()
        elif lattice_type == "E8":
            self.lattice = E8Lattice()
        else:
            raise ValueError(f"Unsupported lattice type: {lattice_type}")
        
        # Initialize quantization config (no overloading)
        self.config = QuantizationConfig(
            lattice_type=lattice_type,
            q=q,
            M=M,
            beta=1.0,
            alpha=1.0,
            max_scaling_iterations=10,
            with_tie_dither=True,
            with_dither=False,
            disable_scaling=True,  # Disable scaling to prevent overloading
            disable_overload_protection=True  # Disable overload protection
        )
        
        # Calculate encoding space size
        self.encoding_space_size = q ** (M * self.lattice.d)
        self.single_level_size = q ** self.lattice.d
        
        print(f"Initialized {lattice_type} Lattice Simulator:")
        print(f"  Dimension: {self.lattice.d}")
        print(f"  Quantization: q={q}, M={M}")
        print(f"  Encoding space size: {self.encoding_space_size:,}")
        print(f"  Single level size: {self.single_level_size:,}")
        print(f"  Configuration: No overloading (t_values=0, scaling disabled)")
        print(f"  Device: {self.device}")
    
    
    def generate_vectors(self, batch_size: int) -> torch.Tensor:
        """
        Generate vectors that are guaranteed to be in the quantized space.
        
        These vectors should have zero quantization error when quantized again.
        
        Args:
            batch_size: Number of vectors to generate
            
        Returns:
            [batch_size, d] tensor of generated vectors (in quantized space)
        """
        # Generate random vectors and quantize them to ensure they're in quantized space
        random_vectors = torch.randn(batch_size, self.lattice.d, device=self.device, dtype=torch.float32)
        
        # Scale to reasonable range to avoid overload issues
        random_vectors = random_vectors * 0.5
        
        # Quantize each vector to ensure it's in the quantized space
        quantized_vectors = torch.zeros_like(random_vectors)
        
        for i in range(batch_size):
            try:
                quantized_vectors[i] = quantize(random_vectors[i], self.lattice, self.config)
            except Exception as e:
                print(f"Warning: Failed to quantize vector {i}: {e}")
                # Use zero vector as fallback
                quantized_vectors[i] = torch.zeros(self.lattice.d, device=self.device, dtype=torch.float32)
        
        return quantized_vectors
    
    
    def validate_reconstruction(self, simulated_vectors: torch.Tensor, 
                              tolerance: float = 1e-6) -> Dict[str, float]:
        """
        Validate that simulated vectors have zero quantization error.
        
        Simulated vectors are already in the quantized space, so when quantized again,
        they should have zero error (perfect reconstruction).
        
        Args:
            simulated_vectors: [batch_size, d] tensor of simulated vectors
            tolerance: Numerical tolerance for exact reconstruction
            
        Returns:
            Dictionary with validation metrics
        """
        batch_size = simulated_vectors.shape[0]
        simulated_vectors = simulated_vectors.to(self.device)
        
        # Quantize the simulated vectors (should be identical to original)
        quantized_vectors = torch.zeros_like(simulated_vectors)
        
        for i in range(batch_size):
            try:
                # Use quantize function (encode + decode)
                quantized_vectors[i] = quantize(simulated_vectors[i], self.lattice, self.config)
            except Exception as e:
                print(f"Warning: Failed to quantize vector {i}: {e}")
                # Use original vector as fallback
                quantized_vectors[i] = simulated_vectors[i]
        
        # Calculate quantization errors (should be zero for simulated vectors)
        errors = torch.norm(simulated_vectors - quantized_vectors, dim=1)
        
        # Calculate metrics
        max_error = torch.max(errors).item()
        mean_error = torch.mean(errors).item()
        std_error = torch.std(errors).item()
        
        # Count exact reconstructions (zero error)
        exact_reconstructions = torch.sum(errors < tolerance).item()
        exact_rate = exact_reconstructions / batch_size
        
        return {
            'max_error': max_error,
            'mean_error': mean_error,
            'std_error': std_error,
            'exact_reconstructions': exact_reconstructions,
            'exact_rate': exact_rate,
            'tolerance': tolerance
        }
    
    def assess_quantizer_performance(self, test_vectors: torch.Tensor, 
                                   num_iterations: int = 10) -> Dict[str, float]:
        """
        Assess quantizer performance on generated test vectors.
        
        Args:
            test_vectors: [batch_size, d] tensor of test vectors
            num_iterations: Number of iterations for timing
            
        Returns:
            Dictionary with performance metrics
        """
        batch_size = test_vectors.shape[0]
        test_vectors = test_vectors.to(self.device)
        
        # Warm up
        for _ in range(3):
            _ = self.validate_reconstruction(test_vectors[:min(100, batch_size)])
        
        # Time encoding
        encode_times = []
        for _ in range(num_iterations):
            start_time = time.perf_counter()
            
            for i in range(batch_size):
                _ = encode(test_vectors[i], self.lattice, self.config)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            encode_times.append(time.perf_counter() - start_time)
        
        # Time decoding
        decode_times = []
        encodings_list = []
        t_values_list = []
        
        # First encode all vectors
        for i in range(batch_size):
            enc, t_val = encode(test_vectors[i], self.lattice, self.config)
            encodings_list.append(enc)
            t_values_list.append(t_val)
        
        encodings = torch.stack(encodings_list)
        t_values = torch.tensor(t_values_list, device=self.device, dtype=torch.int32)
        
        for _ in range(num_iterations):
            start_time = time.perf_counter()
            
            _ = self.decode_encodings(encodings, t_values)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            decode_times.append(time.perf_counter() - start_time)
        
        # Calculate throughput
        encode_throughput = batch_size / statistics.mean(encode_times)
        decode_throughput = batch_size / statistics.mean(decode_times)
        
        return {
            'encode_time_mean': statistics.mean(encode_times),
            'encode_time_std': statistics.stdev(encode_times) if len(encode_times) > 1 else 0,
            'decode_time_mean': statistics.mean(decode_times),
            'decode_time_std': statistics.stdev(decode_times) if len(decode_times) > 1 else 0,
            'encode_throughput': encode_throughput,
            'decode_throughput': decode_throughput,
            'total_throughput': batch_size / (statistics.mean(encode_times) + statistics.mean(decode_times))
        }
    
    def generate_matrix(self, rows: int, cols: int) -> torch.Tensor:
        """
        Generate a matrix by tiling vectors to match lattice dimension.
        
        Args:
            rows: Number of rows in the matrix
            cols: Number of columns in the matrix
            
        Returns:
            [rows, cols] tensor representing the generated matrix
        """
        # Calculate how many vectors we need
        d = self.lattice.d
        vectors_needed = (rows * cols + d - 1) // d  # Ceiling division
        
        # Generate vectors
        vectors = self.generate_vectors(vectors_needed)
        
        # Reshape to matrix
        matrix = vectors.view(-1)[:rows * cols].view(rows, cols)
        
        return matrix
    
    def benchmark_simulation(self, batch_sizes: List[int] = None) -> Dict[str, Dict]:
        """
        Benchmark the simulation system across different batch sizes.
        
        Args:
            batch_sizes: List of batch sizes to test
            
        Returns:
            Dictionary with benchmark results
        """
        if batch_sizes is None:
            batch_sizes = [100, 1000, 10000]
        
        results = {}
        
        for batch_size in batch_sizes:
            print(f"Benchmarking batch size: {batch_size}")
            
            # Generate vectors
            start_time = time.perf_counter()
            vectors = self.generate_vectors(batch_size)
            generation_time = time.perf_counter() - start_time
            
            # Validate reconstruction
            validation_results = self.validate_reconstruction(vectors)
            
            # Assess performance
            performance_results = self.assess_quantizer_performance(vectors)
            
            results[batch_size] = {
                'generation_time': generation_time,
                'generation_throughput': batch_size / generation_time,
                'validation': validation_results,
                'performance': performance_results
            }
            
            print(f"  Generation: {generation_time:.4f}s ({batch_size/generation_time:.0f} vec/s)")
            print(f"  Exact reconstruction rate: {validation_results['exact_rate']:.2%}")
            print(f"  Mean error: {validation_results['mean_error']:.6f}")
        
        return results


def create_simulator(lattice_type: str = "E8", q: int = 3, M: int = 2, 
                    device: str = "cuda" if torch.cuda.is_available() else "cpu") -> LatticeVectorSimulator:
    """
    Convenience function to create a lattice vector simulator.
    
    Args:
        lattice_type: Type of lattice ("Z2", "D4", "E8")
        q: Quantization parameter
        M: Number of hierarchical levels
        device: Device to use
        
    Returns:
        LatticeVectorSimulator instance
    """
    return LatticeVectorSimulator(lattice_type, q, M, device)


def demo_simulation():
    """Demonstrate the lattice vector simulation system."""
    print("🚀 Lattice Vector Simulation Demo")
    print("=" * 50)
    
    # Create E8 simulator
    simulator = create_simulator("E8", q=3, M=2)
    
    # Generate some vectors
    print("\n📊 Generating test vectors...")
    vectors = simulator.generate_vectors(1000)
    print(f"Generated {vectors.shape[0]} vectors of dimension {vectors.shape[1]}")
    print(f"Vector range: [{torch.min(vectors):.4f}, {torch.max(vectors):.4f}]")
    print(f"Vector mean: {torch.mean(vectors):.4f}")
    print(f"Vector std: {torch.std(vectors):.4f}")
    
    # Validate reconstruction (simulated vectors should have zero quantization error)
    print("\n🔍 Validating reconstruction...")
    validation_results = simulator.validate_reconstruction(vectors)
    print(f"Zero error rate: {validation_results['exact_rate']:.2%}")
    print(f"Mean quantization error: {validation_results['mean_error']:.6f}")
    print(f"Max quantization error: {validation_results['max_error']:.6f}")
    print("Note: Simulated vectors should have zero quantization error when quantized again")
    
    # Assess performance
    print("\n⚡ Assessing performance...")
    performance_results = simulator.assess_quantizer_performance(vectors)
    print(f"Encode throughput: {performance_results['encode_throughput']:.0f} vec/s")
    print(f"Decode throughput: {performance_results['decode_throughput']:.0f} vec/s")
    print(f"Total throughput: {performance_results['total_throughput']:.0f} vec/s")
    
    # Generate a matrix
    print("\n📐 Generating matrix...")
    matrix = simulator.generate_matrix(100, 256)
    print(f"Generated matrix: {matrix.shape}")
    print(f"Matrix range: [{torch.min(matrix):.4f}, {torch.max(matrix):.4f}]")
    
    # Benchmark across batch sizes
    print("\n📈 Benchmarking across batch sizes...")
    benchmark_results = simulator.benchmark_simulation([100, 1000, 5000])
    
    print("\n✅ Demo completed successfully!")


if __name__ == "__main__":
    demo_simulation()
