# High-Level Design Objectives & Build Plan (Architect’s Guidance)

Below is the plan I want you to follow. Treat this as the source of truth while you implement. Keep it crisp, modular, and testable end-to-end at every phase.

---

## 1) Mission & Non-Negotiables

**Mission:** Deliver a PyTorch stack that:
- Quantizes along the **last dimension** of any tensor (vectors/rows/tiles) using **hierarchical nested-lattice** quantization (Algorithms 1 & 2 from the paper).
- Supports **Quantization-Aware Training (QAT)** for linear layers (weights and/or activations).
- Provides **compressed-domain compute** where it makes sense (e.g., LUT-based inner products), with clean fallback to dequantized ops.
- **Compresses gradients** in distributed training via a **DDP communication hook**, reducing network bandwidth without breaking convergence.

**Non-negotiables:**
- Vectorization across all leading dimensions; only the last dimension is lattice-dimension `d`.
- Encoders/decoders must be **deterministic**, numerically stable, and **idempotent** in no-overload regimes.
- Modular boundaries: lattice math, quantization, LUTs, CUDA kernels, PyTorch layers, and distributed hooks must remain decoupled.

---

## 2) Core Concepts You Must Align To

- **Quantization = Encode + Decode.** Encoding produces per-layer base-q labels; decoding reconstructs lattice points. Together they implement quantization.
- **Layered (M-deep) construction.** Smaller LUTs, predictable compute, cache-friendly. Limit M to practical values (≤4).
- **A_q alphabet & LUTs.** The alphabet is induced by the base lattice and q; LUTs precompute inner products on A_q×A_q (two-sided) and per-query (one-sided).
- **Compressed-domain arithmetic.** Mod-q addition on encodings is exact in A_q but does **not** guarantee exact equality post-decode to real-domain sums; document this and expose validations.
- **Overload control.** Scaling (β) and optional structured dithers; overload avoidance with geometric β ramp.

---

## 3) Architectural Layout (what lives where, conceptually)

- **Lattice layer (math):** Nearest-neighbor quantizer Q_L, generator matrices, helpers for A_q mapping. Start with Z^d (baseline), then D4 (primary), E8 later.
- **Quantization layer:** Vectorized, last-dim apply. Implements Algorithms 1 (encode) and 2 (decode), plus convenience quantize(). No training logic here.
- **LUT subsystem:** Builders for two-sided and one-sided LUTs; indexing utilities from base-q labels; memory footprint controls.
- **GPU acceleration:** CUDA kernels for encode, decode, LUT inner products, mod-q arithmetic. Fused and batched. Respect memory hierarchy (L1/L2/shared).
- **QAT Linear:** Drop-in Linear replacement with FP32 shadow weights; periodic quantization of weights/activations; optional LUT-based matmul path; STE variants.
- **Distributed hooks:** DDP comm hook for gradient compression; clear policy on when to decode (pre/post all-reduce) with a conservative path first.
- **Test & bench suite:** Unit tests for math correctness; integration tests for QAT; benchmarks for throughput, cache effects, bandwidth savings.

---

## 4) Data Model & Shapes (don’t deviate)

- Inputs can be N-D tensors; **only last dimension** equals lattice dimension `d`.
- Encodings are per-layer, per-coordinate base-q labels; store as compact integer tensors with shape “…, M, d”.
- Keep optional bit-packing as a second pass; do the simple, correct thing first.

---

## 5) Performance & Quality Targets

- **Encode/Decode throughput:** Linear in batch size; amortize the M loop; no Python loops over batch dims.
- **LUT size discipline:** With D4, q=4, M=2, the two-sided LUT is 4^(2*4)=65,536 entries—L1/L2 friendly. Keep it that way.
- **Numerical behavior:** With sufficiently large β and no overload, decode(encode(x)) must match Q_L(x). Telescoping identity is the invariant—test it.
- **DDP bandwidth:** Demonstrable reduction vs FP32 gradients on realistic models without tanking convergence.

---

## 6) Implementation Phases (do not skip; test at each gate)

### Phase A — Foundations (Correctness First)
- Lattice math: Z^d, then D4. Validate Q_L properties (idempotence, nearest-point).
- Vectorized encode/decode on CPU/GPU (no CUDA yet). Apply strictly on last dim; handle arbitrary leading dims.
- Overload detection and basic β control (no dithers initially).
- Tests: exactness in no-overload cases; telescoping identity; random stress on shapes.

**Exit criteria:** encode/decode stable, fast enough on GPU in pure PyTorch, and correct.

---

### Phase B — LUT & Compressed-Domain Ops
- Two-sided and one-sided LUT builders; shape and indexing finalized.
- Inner-product reconstruction using LUTs; parity with explicit decode-then-dot.
- Mod-q elementwise addition on encodings; document limits and provide post-decode validation utility.
- Tests: parity within floating-point tolerance; assert LUT memory fits target cache budgets.

**Exit criteria:** LUT IP path matches dequantized matmul accuracy; measurable speed benefit in micro-benchmarks.

---

### Phase C — CUDA Acceleration
- Encode kernel: fuse M-loop, vectorize across batch; shared memory for small d; coalesce reads/writes; avoid warp divergence in tie-breaking.
- Decode kernel: avoid repeated Q_L by staging tiny decode aids; still correct for D4.
- LUT IP kernel: aggressive parallelization across tiles; unroll small M; consider texture cache for LUT.
- Mod-q arithmetic kernel: elementwise ops and optional packed variants.
- Tests: bit-for-bit (or ulp-level) agreement with reference; perf smoke on large batches.

**Exit criteria:** Clear GPU wins over pure Torch; no correctness regressions.

---

### Phase D — QAT Linear (Training Integration)
- FP32 shadow weights; periodic quantization policy (`quantize_every` steps) with cache of encodings or reconstructions.
- Activation quantization as a switch; ability to quantize weights only.
- Option to compute with LUT IP or standard matmul; configurable per layer.
- STE policy abstraction (straight-through vs softened rounding). Start conservative; expose the knob.
- Tests: shape and dtype parity; MNIST/CIFAR smoke; QAT curves sane vs baseline.

**Exit criteria:** Small model converges; training stable; layer usable as a drop-in.

---

### Phase E — Distributed Gradient Compression
- DDP comm hook: quantize gradient buckets along last dim, then conservative path = **decode before all-reduce**. All-reduce in compressed domain is future work.
- Padding and tiling for buckets where last dim ≠ d; track and strip padding.
- Instrumentation: bytes sent vs baseline; step time; convergence tracking.
- Tests: single-node multi-GPU correctness; convergence on a modest model.

**Exit criteria:** Demonstrated bandwidth reduction without harming convergence; clean logs and metrics.

---

## 7) Design Policies & Constraints (follow them)

- **Determinism:** Tie-breaking in Q_L must be fixed and documented. No data-dependent randomness in production paths.
- **Parameter menus:** Start with D4, d=4, q=4, M=2 as defaults. Only expand when benchmarks justify it.
- **Fallbacks:** Always keep a dequantized fallback for debugging and regression isolation.
- **Error accounting:** Log overload rates, β ramps, and any decode mismatches in validation modes.

---

## 8) Parallelism & Memory Strategy

- Batch over all leading dims. Only the M loop is per-vector; keep it inside fused kernels.
- Use small shared memory tiles for d∈{4,8}; avoid spills.
- LUT residency: keep two-sided LUT resident per device; pin per-query LUTs when using one-sided mode.
- Be conscious of kernel launch overhead; prefer fused kernels to long Python graphs.

---

## 9) Testing & Benchmarking (be ruthless)

- **Unit correctness:** lattice Q_L, encode/decode invariants, LUT parity, mod-q semantics.
- **Integration:** QLinear forward parity vs nn.Linear (no overload), QAT loss curves.
- **Performance:** encode/decode throughput, LUT IP vs matmul, end-to-end tokens/sec, DDP bandwidth/time.
- **Stress:** large batch sizes, random shapes, boundary values near Voronoi region edges.

If a test fails, fix the **lowest** layer responsible; do not patch at higher layers.

---

## 10) Risk Register & Mitigations

- **Overload spikes:** Implement β ramp with cap; expose metrics; allow per-layer β policies.
- **Compressed-domain sum mismatch:** Provide validation mode; route to dequantized path when exactness is required.
- **Kernel divergence / perf cliffs:** Keep a clear CPU/Torch reference path; bisect with micro-benches.
- **Convergence regressions:** Start with weights-only quantization; add activations later; keep STE conservative.

---

## 11) Documentation & Developer Experience

- Single, authoritative README that explains: shapes, last-dim rule, parameters (d,q,M), overload handling, and trade-offs between LUT vs dequantized paths.
- Minimal runnable examples: encode/decode demo; QLinear on MNIST; DDP hook on a toy model with metric logging.
- Clear “when to use what” guidance: e.g., one-sided LUT for retrieval-style workloads; two-sided for symmetric matmul proxies.

---

## 12) Acceptance Criteria (what “done” means)

- Stable encode/decode with invariants passing on CPU and GPU.
- LUT IP matches dequantized reference within tolerance; provides a performance win for chosen defaults.
- QLinear trains a small model with sensible accuracy and stable loss.
- DDP hook shows bandwidth reduction with maintained convergence.
- Tests and benchmarks are automated; docs explain all critical trade-offs.

---

Execute in order. Don’t jump ahead. Keep PRs scoped per phase with benchmarks and tests attached. If you hit ambiguity, prefer correctness and observability over premature optimization.
