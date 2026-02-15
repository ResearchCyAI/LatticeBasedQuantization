"""
E8 Lattice CUDA Module

This module provides CUDA-accelerated operations for E8 lattice quantization,
including JIT-compiled kernels and automatic fallback to PyTorch GPU implementations.
"""

import torch
from torch.utils.cpp_extension import load_inline
import os


# Define CUDA kernel code
cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cmath> // For roundf, fabsf, sqrtf, copysignf, floorf, powf

// Device helper functions
__device__ __forceinline__ float custom_round(float x) {
    float eps = 1e-7f;
    return floorf(x - copysignf(eps, x) + 0.5f);
}

__device__ __forceinline__ float compute_distance(const float* x, const float* y) {
    float dist = 0.0f;
    for (int i = 0; i < 8; i++) {
        float diff = x[i] - y[i];
        dist += diff * diff;
    }
    return sqrtf(dist);
}

// E8 Quantization Kernel
__global__ void e8_quantize_kernel(
    const float* x,
    float* y,
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;
    
    const float* x_vec = x + idx * 8;
    float* y_vec = y + idx * 8;
    
    // Candidate 0: D8 quantization
    float f_x[8], g_x[8];
    for (int i = 0; i < 8; i++) {
        f_x[i] = custom_round(x_vec[i]);
    }
    
    float sum = 0.0f;
    for (int i = 0; i < 8; i++) {
        sum += f_x[i];
    }
    int sum_parity = ((int)sum) % 2;
    
    float max_delta = 0.0f;
    int k = 0;
    for (int i = 0; i < 8; i++) {
        float delta = fabsf(x_vec[i] - f_x[i]);
        if (delta > max_delta) {
            max_delta = delta;
            k = i;
        }
    }
    
    for (int i = 0; i < 8; i++) {
        g_x[i] = f_x[i];
    }
    
    if (x_vec[k] >= 0.0f) {
        if (f_x[k] < x_vec[k]) {
            g_x[k] += 1.0f;
        } else {
            g_x[k] -= 1.0f;
        }
    } else {
        if (f_x[k] <= x_vec[k]) {
            g_x[k] += 1.0f;
        } else {
            g_x[k] -= 1.0f;
        }
    }
    
    float* y_0 = (sum_parity != 0) ? g_x : f_x;
    
    // Candidate 1: D8 + (0.5)^8
    float x_shifted[8], f_x_shifted[8], g_x_shifted[8];
    for (int i = 0; i < 8; i++) {
        x_shifted[i] = x_vec[i] - 0.5f;
        f_x_shifted[i] = custom_round(x_shifted[i]);
    }
    
    float sum_shifted = 0.0f;
    for (int i = 0; i < 8; i++) {
        sum_shifted += f_x_shifted[i];
    }
    int sum_parity_shifted = ((int)sum_shifted) % 2;
    
    float max_delta_shifted = 0.0f;
    int k_shifted = 0;
    for (int i = 0; i < 8; i++) {
        float delta = fabsf(x_shifted[i] - f_x_shifted[i]);
        if (delta > max_delta_shifted) {
            max_delta_shifted = delta;
            k_shifted = i;
        }
    }
    
    for (int i = 0; i < 8; i++) {
        g_x_shifted[i] = f_x_shifted[i];
    }
    
    if (x_shifted[k_shifted] >= 0.0f) {
        if (f_x_shifted[k_shifted] < x_shifted[k_shifted]) {
            g_x_shifted[k_shifted] += 1.0f;
        } else {
            g_x_shifted[k_shifted] -= 1.0f;
        }
    } else {
        if (f_x_shifted[k_shifted] <= x_shifted[k_shifted]) {
            g_x_shifted[k_shifted] += 1.0f;
        } else {
            g_x_shifted[k_shifted] -= 1.0f;
        }
    }
    
    float* y_1_mid = (sum_parity_shifted != 0) ? g_x_shifted : f_x_shifted;
    float y_1[8];
    for (int i = 0; i < 8; i++) {
        y_1[i] = y_1_mid[i] + 0.5f;
    }
    
    float dist_0 = compute_distance(x_vec, y_0);
    float dist_1 = compute_distance(x_vec, y_1);
    
    if (dist_0 < dist_1) {
        for (int i = 0; i < 8; i++) {
            y_vec[i] = y_0[i];
        }
    } else {
        for (int i = 0; i < 8; i++) {
            y_vec[i] = y_1[i];
        }
    }
}
"""

cpp_source = """
#include <torch/extension.h>
#include <vector>

// Forward declaration of the CUDA kernel
void e8_quantize_kernel(const float* x, float* y, int batch_size);

torch::Tensor e8_quantize_cuda_jit(torch::Tensor x, c10::optional<torch::Device> device) {
    TORCH_CHECK(x.dim() == 2 && x.size(1) == 8, "Input tensor must be of shape [batch_size, 8]");
    
    if (!device.has_value()) {
        device = x.device();
    }
    TORCH_CHECK(device->is_cuda(), "Input tensor must be on CUDA device for CUDA kernel.");

    torch::Tensor y = torch::empty_like(x);
    int batch_size = x.size(0);

    // Determine grid and block dimensions
    const int threadsPerBlock = 256;
    const int blocks = (batch_size + threadsPerBlock - 1) / threadsPerBlock;

    e8_quantize_kernel<<<blocks, threadsPerBlock>>>(
        x.contiguous().data_ptr<float>(),
        y.contiguous().data_ptr<float>(),
        batch_size
    );
    AT_CUDA_CHECK(cudaGetLastError());
    return y;
}

bool is_cuda_extension_available() {
    return true; // If this module loads, CUDA extension is available
}
"""

try:
    e8_cuda_kernels = load_inline(
        name="e8_cuda_kernels",
        cuda_sources=[cuda_source],
        cpp_sources=[cpp_source],
        functions=["e8_quantize_cuda_jit", "is_cuda_extension_available"],
        verbose=True,
        extra_cuda_cflags=["-O3", "--use_fast_math", "--ptxas-options=-v"],
        extra_cflags=["-O3"],
    )
    e8_quantize_cuda_jit = e8_cuda_kernels.e8_quantize_cuda_jit
    e8_cuda_available = e8_cuda_kernels.is_cuda_extension_available
except Exception as e:
    print(f"Failed to load E8 CUDA JIT kernels: {e}")
    e8_quantize_cuda_jit = None
    e8_cuda_available = lambda: False


def e8_quantize_cuda_wrapper(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Wrapper function for E8 CUDA quantization with automatic fallback.
    
    Args:
        x: Input tensor of shape [batch_size, 8]
        device: CUDA device to use
        
    Returns:
        Quantized tensor of shape [batch_size, 8]
    """
    if not device.type == 'cuda':
        raise ValueError("CUDA wrapper requires CUDA device")
    
    if e8_cuda_available():
        try:
            return e8_quantize_cuda_jit(x, device=device)
        except Exception as e:
            print(f"CUDA kernel failed, falling back to PyTorch GPU: {e}")
            # Fallback to PyTorch GPU implementation
            from .codecs import batch_e8_quantize
            return batch_e8_quantize(x, device=device)
    else:
        # Fallback to PyTorch GPU implementation
        from .codecs import batch_e8_quantize
        return batch_e8_quantize(x, device=device)
