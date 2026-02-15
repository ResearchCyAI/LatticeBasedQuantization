# Multiply-and-Accumulate (MAC) and Add-and-Accumulate (A&A) in HNLQ Domain

We want to operate **entirely in the HNLQ (Hierarchical Nested-Lattice Quantization) domain** without decoding to quantized space, which would be wasteful. Below is the high-level design for MAC and A&A directly on encodings.

---

## High-Level Goals

1. **No decode in hot paths** – only encodings and LUTs are used.  
2. **Layerwise algebra** – operate per layer \(i \in \{0, \dots, M-1\}\) (and \(-1\) if dither is used).  
3. **Shape discipline** – apply across arbitrary leading dims; last dim is lattice dimension \(d\).  
4. **Bounded memory** – LUT sizes must fit L1/L2 cache (prefer \(d\in\{4,8\}\), \(M \le 4\)).

---

## A. Multiply-and-Accumulate (MAC)

**Use case:** dot products / matmul / contractions along the last dim.

### A.1 Primitive: LUT inner product
Given encodings \(\{b_i(x)\}_{i=0}^{M-1}\), \(\{b_j(y)\}_{j=0}^{M-1}\):

\[
\langle \hat x, \hat y\rangle
\;=\;
\sum_{i=0}^{M-1}\sum_{j=0}^{M-1} q^{i+j}\; L(b_i(x), b_j(y))
\]

where \(L\) is a two-sided LUT over \(A_q \times A_q\).  

If one side is in full precision:  

\[
\langle y, \hat x\rangle
= \sum_{i=0}^{M-1} q^i L_y(b_i(x)).
\]

**Directives:**
- Build LUT once per \((L,q,d)\).  
- Store entries as integers if lattice supports (D4/E8).  
- Use integer multiplies (bit-shift if q is power of 2).  
- Extend indices to \(-1\) for dither.

### A.2 MAC across many pairs
For \(\sum_k \langle \hat x^{(k)}, \hat y^{(k)}\rangle\):  

- Accumulate LUT outputs directly in integer accumulators.  
- Use 64-bit accumulators (128-bit if necessary).  
- Integer additions are deterministic and reproducible.  
- Can truncate high layers (early exit) for adaptive compute vs accuracy.

---

## B. Add-and-Accumulate (A&A)

**Goal:** compute \(\sum_k \hat x^{(k)}\) without decoding.

### B.1 Carry-aware layerwise accumulation
Maintain per-layer sums \(S_i\).

1. **Update** each new vector’s representative:  
   \(S_i \leftarrow S_i + X_i,\; X_i \in A_q\).
2. **Normalize with carry**:  
   - Carry: \(C_{i+1} = Q_L(S_i/q)\)  
   - Adjusted: \(A^*_i = S_i - q C_{i+1}\)  
   - Update next layer: \(S_{i+1} \leftarrow S_{i+1} + C_{i+1}\), set \(S_i = A^*_i\).  

This is base-q addition in the lattice.

**Guidance:**
- Normalize every T additions (e.g., 16–32).  
- Keep accumulators bounded.  
- Output normalized encodings when needed.

### B.2 Fast path: pure mod-q addition
- Accumulate encodings mod q per layer.  
- Not exact — ignores carries.  
- Rebase periodically with carry normalization to fix drift.

---

## C. Tensors & Tiles

- Tensors shaped `[..., K, d]` treated as K tiles.  
- MAC: chunk contracting dim into tiles, accumulate LUT outputs.  
- A&A: per-layer accumulators with periodic carry normalization.

---

## D. Dither & Scaling

- **Structured dither**: treat as virtual layer (-1). Extend LUT and accumulators accordingly.  
- **Scaling (β)**: track scale outside accumulators. Combine at the end.

---

## E. Safeguards

- Use 64-bit accumulators (or wider if needed).  
- Keep LUT sizes within L1/L2 cache.  
- Ensure deterministic reduction order.  
- Monitor overload rates, accumulator growth, normalization cadence.  

---

## F. When to Decode

- Only for validation or debug.  
- Decode small samples to compare against compressed-domain results.  
- Test vectors near Voronoi borders explicitly.

---

## G. Implementation Priority

1. Two-sided LUT MAC with integer accumulators.  
2. Carry-aware A&A accumulator.  
3. Fast-path mod-q A&A with periodic rebase.  
4. One-sided LUT for query–database workloads.  
5. Instrumentation (counters, guards, error checks).

---

**Rule:** Always prefer lattice-correct arithmetic with occasional batched \(Q_L\) calls over per-vector decode.
