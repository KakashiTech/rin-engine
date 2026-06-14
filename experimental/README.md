# Experimental Components

These components are **research prototypes**. They demonstrate novel techniques for energy-efficient neural computation, but should not be considered production-ready without further validation.

---

## LIF Engine (`rin_lif_engine.h`)

Multiplication-free Leaky Integrate-and-Fire neurons using fixed-point Q15 arithmetic and bit shifts. Replaces floating-point `v = v * decay + input` with `v = (v >> decay_shift) + (input >> input_shift)`.

**Status:** Implemented and tested. Works on any CPU without FPU. Claim of "zero multiplication" is algorithmically true — but real-world accuracy vs floating-point LIF depends on bit-width and scaling factors.

---

## Power-of-Two Softmax (`rin_ptsoftmax.h`)

Softmax using a precomputed 256-entry LUT for `2^x` instead of `expf()`. Eliminates the most expensive function in Transformer inference.

**Status:** Implemented. 2550/2550 tests pass. Error vs true softmax depends on Q8 sampling resolution. No independent accuracy evaluation on real NLP tasks.

---

## BSPN (`rin_bspn.h`)

Bit Shifting PowerNorm — replaces LayerNorm with L1-based normalization using only bit shifts. Claims <0.5% error vs LayerNorm.

**Status:** Implemented. Shift-based norm is mathematically sound. Error measurement is synthetic (random vectors) — no evaluation on actual trained models.

---

## Phase Gating (`rin_phase_gating.h`)

Dynamic sparsity mechanism where each connection has a "phase" angle. Signals propagate only when input phase aligns with weight phase within a configurable threshold.

**Claim:** 90% of network can be "off" without losing information.

**Status:** **Unverified.** The mechanism is implemented and runs. The 90% sparsity claim requires:
- Training a model with phase parameters from scratch
- Measuring accuracy before/after gating at various thresholds
- Third-party reproduction

---

## Betti Calculator (`rin_betti_calculator.h`)

Computes Betti numbers (β₀, β₁, β₂) for Topological Data Analysis on neural representations. Intended to verify structural preservation during model distillation.

**Status:** **Research prototype.** Betti number computation works on synthetic data. No validation that preserved Betti numbers correlate with preserved accuracy in distilled models.

---

## Mechanistic Distillation (`rin_mechanistic_distill.h`)

Distillation technique that tries to replicate internal activation patterns (not just output logits) of a teacher model.

**Status:** **Research prototype.** Framework is implemented. Requires end-to-end training experiments to validate.

---

## DCT Engine (`rin_dct_engine.h`)

Fixed-point DCT-II 8-point using Arai-Agui-Nakajima algorithm — 5 multiplications vs 64 for naive. Q15 fixed-point, high-frequency pruning.

**Status:** Implemented. Fixed-point error measured at ~7.3% vs floating-point DCT. Use case (spectral compression of embeddings) needs validation.

---

## Distributed Inference (`rin_distributed.h`)

Multi-node inference protocol with resonance synchronization. Up to 16 peers, heartbeat mechanism. Designed for LoRa/WiFi HaLow.

**Status:** **Wireframe.** Communication primitives exist. No actual distributed inference has been performed.

---

## RIN-X: AVX-512 Kernels (`experimental/rin-x/`)

AVX-512 VNNI kernels, block-sparse 4×4 GEMM, kernel fusion, cache blocking, SOA layout. Requires x86 hardware with AVX-512.

**Status:** Code compiles on AVX-512 hardware. Claims of 133× speedup vs PyTorch training and 8× energy efficiency need:
- Hardware with AVX-512 (Xeon, AMD Zen 4)
- Controlled benchmark with locked frequency and isolated cores
- Third-party reproduction

---

## How to contribute

Pick an experimental component and:
1. Run its test suite
2. Validate claims on your hardware
3. Open an issue with your results

The goal is to move components from `experimental/` to production as claims are independently verified.
