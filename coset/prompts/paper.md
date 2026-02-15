# Algorithms 1 & 2 — Hierarchical Nested-Lattice Encoder/Decoder

This summary is based on Kaplan & Ordentlich (2025), *High-Rate Nested-Lattice Quantized Matrix Multiplication with Small Lookup Tables* [arXiv:2505.13164v1].

---

## Background
The algorithms implement a **rate-\(R\)**, **\(M\)-layer** hierarchical nested-lattice quantizer over a base lattice \(L \subset \mathbb{R}^d\) with generating matrix \(G\).  
- The **encoder** outputs \(M\) encoding vectors \(b_m \in [q]^d\).  
- The **decoder** reconstructs a lattice point \(\hat x \in L\).  
- If no overload occurs, then \(\hat x = Q_L(x)\), the nearest-neighbor quantization of \(x\).

---

## Notation
- **Nearest-neighbor quantizer:**  
  \( Q_L(x) = \arg\min_{\lambda \in L} \|x - \lambda\| \).  
- **Nested scaling:**  
  \( Q_{qL}(x) = q \, Q_L(x/q) \).  
- **Coset alphabet:**  
  \( A_q = L \cap (qV) \cong (\mathbb{Z}/q\mathbb{Z})^d \).  

---

## Algorithm 1 — Encoder

**Inputs:**  
- \(x \in \mathbb{R}^d\)  
- Lattice \(L\) with generator \(G\)  
- Nesting ratio \(q \in \mathbb{N}\)  
- Depth \(M \in \mathbb{N}\)

**Outputs:**  
- Encoding vectors \(b_0, b_1, \dots, b_{M-1} \in [q]^d\)  
- Binary flag `OverloadError`

**Procedure:**  
1. Set \(\tilde g \leftarrow x\).  
2. For \(m = 0, \dots, M-1\):  
   - \(\tilde g \leftarrow Q_L(\tilde g)\)  
   - \(b_m \leftarrow [G^{-1}\tilde g] \bmod q\)  
   - \(\tilde g \leftarrow \tilde g / q\)  
3. `OverloadError` \(\leftarrow 1\{ Q_L(\tilde g) \neq 0 \}\).  
4. Return \(b_0, \dots, b_{M-1}\), `OverloadError`.

**Rate:** \(R = M d \log_2 q\) bits.

---

## Algorithm 2 — Decoder

**Inputs:**  
- Encoding vectors \(b_0, \dots, b_{M-1}\)  
- Lattice \(L\), generator \(G\), nesting ratio \(q\), depth \(M\)

**Output:**  
- Reconstructed vector \(\hat x \in L\)

**Procedure:**  
1. Initialize \(\hat x \leftarrow 0\).  
2. For \(m = 0, \dots, M-1\):  
   - \(x_m \leftarrow G b_m - q \, Q_L((G b_m)/q)\)  
   - \(\hat x \leftarrow \hat x + q^m x_m\)  
3. Return \(\hat x\).

---

## Correctness

By construction:
\[
\hat x = \sum_{m=0}^{M-1} q^m \big(G b_m - q Q_L((G b_m)/q)\big).
\]

**Lemma.**  
\(\hat x = Q_L(x)\) if and only if \(Q^{\circ}_M(x) = 0\).  

This means the decoder reconstructs the exact nearest-neighbor quantization whenever no overload occurs. Overload events are detected via the encoder’s flag.

---

## Notes
- **Scaling & Dithering:** To reduce distortion, inputs may be scaled by \(\beta\) and shifted by a dither \(z \in V\).  
- **Overload Avoidance:** Increase \(\beta\) geometrically until overload clears. Transmit the counter \(T\) with entropy coding (\(\approx H(T)\) bits).  
- **Successive Refinement:** If decoding only the last \(t\) layers, one gets a coarse reconstruction, refined as more layers are decoded.  
- **Complexity:**  
  - Encoder: \(M\) quantizations + modular reductions  
  - Decoder: \(M\) quantizations + scaled additions  

---
