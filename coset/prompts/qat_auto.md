# 🧩 Software Requirements Document (SRD)
## Quantization-Aware Training (QAT-Lite) Linear Layer with Hierarchical Nested Lattice Quantization (HNLQ)

---

### 1️⃣ Overview

This SRD describes a **Quantization-Aware Training (QAT-Lite)** framework for linear layers using **Hierarchical Nested Lattice Quantization (HNLQ)** with optional **E₈** or **D₄** base lattices.

The design supports:
- **Row-wise** or **block-wise** scaling (β-tiling).
- **Straight-Through Estimator (STE)** for rounding.
- **Learnable quantized activations (LSQ-A style)**.
- **Float biases** (biases are not quantized).
- **Export helpers** to fold scales, serialize quantization parameters, and build deployable inference models.

This framework balances **quantization fidelity** with **computational simplicity** — ideal for low-overhead QAT pipelines and hardware-friendly deployment.

---

### 2️⃣ Design Goals

| Goal | Description |
|------|--------------|
| **Granular control** | Support both row-wise and block-wise β computation and quantization. |
| **Stability** | Use EMA (Exponential Moving Average) statistics for smooth β updates. |
| **Simplicity** | Minimal overhead per batch, no clustering or per-block tying. |
| **Compatibility** | Plug-and-play replacement for `nn.Linear`. |
| **Exportability** | Produce portable quantized weights + metadata. |
| **Lattice flexibility** | Works with `E8`, `D4`, or other lattices (via user-supplied `G`, `G⁻¹`). |

---

### 3️⃣ Key Concepts

#### 3.1 Preconditioning
- Optionally standardize `W` once (zero mean, unit variance).
- Keeps global range bounded so per-row β doesn’t saturate.

#### 3.2 Tiling Granularity
Two modes (set at initialization):
- **Row-wise**: one β per output channel (lightweight).
- **Block-wise**: one β per contiguous block (e.g., 8 for `E8`, 4 for `D4`).

All per-tile statistics and β computations respect this granularity.

#### 3.3 EMA (Exponential Moving Average)
Used to estimate per-tile statistics smoothly:
\[
\text{EMA}_t = m \cdot \text{EMA}_{t-1} + (1-m) \cdot x_t
\]
- \( m \) = momentum (e.g., 0.99)
- Maintains long-term stability of σ (std) and Xₘₐₓ (∞-norm).

#### 3.4 β Bounds

For each tile \( g \):
\[
\beta_{\min,g} = \frac{\Delta_0 q^{-M}}{\eta \gamma_{\mathrm{inv}} \sigma_g}, \quad
\beta_{\max,g}^{\mathrm{det}} = \frac{\Delta_0 q^{M}}{2 \gamma_{\mathrm{inv}} X_{\max,g}}, \quad
\beta_{\max,g}^{\mathrm{prob}} = \frac{\Delta_0 q^{M}}{2k \gamma_{\mathrm{inv}} \sigma_g}
\]
with:
- \( \eta \) — utilization factor (e.g. 0.2),
- \( k \) — probabilistic safety margin (e.g. 5),
- \( \gamma_{\mathrm{inv}} = \|G^{-1}\|_{\infty\to\infty} \).

β is learned via sigmoid interpolation:
\[
\beta_g = \beta_{\min,g} + \sigma(\theta_g)(\beta_{\max,g} - \beta_{\min,g})
\]

---

### 4️⃣ System Architecture

#### 4.1 Components

| Component | Description |
|------------|-------------|
| **Activation Quantizer (LSQ-A)** | Learnable symmetric quantizer with STE; per-tensor α (clip). |
| **Weight Quantizer (HNLQ)** | Transform–Round–Inverse pipeline with base lattice G, depth M, radix q. |
| **β Estimator** | Per-row or per-block scaling parameter computed from EMA stats. |
| **Bias** | Kept in float precision (not quantized). |
| **Export Engine** | Folds quantized weights, serializes metadata, and builds inference-ready Linear module. |

#### 4.2 Computation Flow