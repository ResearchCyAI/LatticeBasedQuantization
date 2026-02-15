# 🧩 Product Requirement Document (PRD): Lattice-Based Quantization-Aware Training (QAT)

## 1. 🎯 Overview

This PRD describes the implementation of **Quantization-Aware Training (QAT)** using **lattice-based quantization** techniques, integrated as a **Linear Layer module** in deep neural networks. The key principle is to encode floating-point vectors into quantized representations via lattice encodings and use these representations to perform **dot products and matrix multiplications** efficiently through **lookup tables (LUTs)** or **CUDA kernels**. 

This design enables neural networks to train with quantization effects simulated during forward propagation, while retaining full-precision gradient updates during backpropagation.

---

## 2. 🧠 Core Concept

### 2.1 Encoding & Decoding

- **Encoding**: Maps a floating-point input vector `x ∈ ℝ^D` to an **index** in the quantized lattice codebook.
- **Decoding**: Maps the **index** back to its corresponding quantized vector in the **codebook**.
- **Quantization Process**:  
  Full-precision vector → (encode → decode) → quantized vector.

Mathematically:
\[
\hat{x} = \text{decode}(\text{encode}(x))
\]
where `𝑥̂` is the quantized approximation of `x`.

---

## 3. ⚙️ Forward and Backward Pass Mechanics

### 3.1 Forward Pass (Quantized Arithmetic)

- Take full-precision weights `W` and input `X`.
- Encode both (or only one, in one-sided LUT design) using the lattice quantizer.
- Compute **matrix multiplication** in the **encoding space** using LUT-based dot products:
  \[
  Y_q = f(\text{decode}(\text{encode}(X)) \cdot \text{decode}(\text{encode}(W)))
  \]
- Apply activation functions (e.g., ReLU, GELU) on quantized outputs.

**Variants:**
- **CPU/Torch implementation:** Encoding and decoding implemented in Python/Torch.
- **GPU implementation:** CUDA kernels for encoding, decoding, and LUT-based dot products.

### 3.2 Backward Pass (Straight-Through Estimator, STE)

- Gradients are propagated **as if no quantization occurred**.
- Use **Straight-Through Estimator (STE)**:
  \[
  \frac{\partial L}{\partial x} \approx \frac{\partial L}{\partial \hat{x}}
  \]
- Full-precision weights `W_fp` are updated using standard backpropagation.

---

## 4. 🧮 Quantization Lattice Models

### 4.1 Baseline: Scalar Quantization
- **Lattice**: Euclidean lattice `ℤ` (dimension = 1).
- Reduces to **memoryless scalar quantizer**.
- Used for initial validation and benchmarking.

### 4.2 Higher-Dimensional Lattices
| Lattice | Dim | Description | Notes |
|----------|-----|--------------|-------|
| **D4** | 4 | Dense 4D lattice | Efficient LUT size |
| **E8** | 8 | Optimal packing lattice in 8D | Default high-performance option |
| **A2/A3** | 2/3 | Simple hexagonal lattices | Visualization & teaching |
| **Nested Lattice (Λ₁ ⊂ Λ₂)** | M×D | Multi-level quantizer | Enables multi-bit precision levels |

---

## 5. 🔩 Implementation Components

### 5.1 Lattice-Based Linear Layer

**Class:** `LatticeLinear(nn.Module)`  
**Responsibilities:**
- Implements forward (quantized) and backward (STE) operations.
- Supports lattice configuration (E8, D4, scalar, etc.).
- Configurable LUT backend: Torch or CUDA.

**Forward Path:**
1. Encode input and/or weights → index space.
2. Perform matrix multiplication via LUT-based dot product.
3. Decode results to floating point.
4. Apply activation (if any).

**Backward Path:**
- Bypass quantization (use STE).
- Update full-precision weights.

### 5.2 MLP Integration

**Class:** `LatticeMLP(nn.Module)`  
**Composition:**
- Stack of `LatticeLinear` layers.
- Interleaved with activations and optional normalization layers.

**Example:**
```python
class LatticeMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, lattice="E8"):
        super().__init__()
        self.fc1 = LatticeLinear(in_dim, hidden_dim, lattice=lattice)
        self.act = nn.ReLU()
        self.fc2 = LatticeLinear(hidden_dim, out_dim, lattice=lattice)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))