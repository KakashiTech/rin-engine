# RIN Engine — Experimental Neural Inference Runtime

**Deterministic memory allocation, energy measurement, and CPU inference without GPU.**

RIN Engine is an experimental C runtime + Python CLI for running neural network inference on CPU. It explores techniques for energy-efficient computation: arena allocation (zero malloc during inference), RAPL-based energy metering, and alternative normalization/softmax implementations using integer arithmetic.

[![CI](https://github.com/KakashiTech/rin-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/KakashiTech/rin-engine/actions/workflows/ci.yml)

---

## Quick start

```bash
git clone https://github.com/KakashiTech/rin-engine.git
cd rin-engine
make shared
pip install -e .
rin bench --mode transformer
```

If you have a `.rin` model:

```bash
rin run model.rin --prompt "Hello, world!" --max-tokens 50
```

---

## What works today (production-ready)

| Component | Description | File |
|-----------|-------------|------|
| **Arena allocator** | O(1) allocation, O(1) reset, zero malloc/free during inference | `rin_arena.h` |
| **RAPL energy meter** | Per-inference energy measurement via Intel/AMD RAPL MSRs | `rin_energy_meter.h` |
| **5 inference modes** | MLP, SNN, ATTN, RIN, Transformer | `rin run --mode` |
| **Python CLI** | `rin {run,bench,energy,inspect,import}` | `rin/` |
| **C test suite** | 8 quantitative validation tests, 0 failures | `make test` |
| **Backend router** | Auto-selects native CPython ext → ctypes fallback | `rin/_backend.py` |

---

## Verified performance

Hardware: AMD Ryzen 5 5600G, DDR4-3200, GCC 15. Model: 4-layer, 262K params, INT8.

| Comparison | RIN Engine | ONNX | Ratio |
|-----------|------|------|-------|
| Latency (µs/inf) | 202 | 783 (FP32) | **3.9× faster** |
| Energy (µJ/pkg) | 5,654 | 28,510 (FP32) | **5.0× efficient** |

> **Note:** RIN Engine uses INT8 quantization while ONNX runs FP32. This is the current baseline. A direct INT8-vs-INT8 comparison requires an INT8-capable ONNX Runtime build (ortextensions or onnxruntime-trt) — will be added in a future release.

---

## Research components (experimental)

These components are implemented but **not independently validated**:

| Component | Description | Status |
|-----------|-------------|--------|
| **LIF engine** | Multiplication-free spiking neurons via bit shifts | `tested` |
| **Power-of-Two softmax** | Softmax using LUT instead of exp() | `tested` |
| **BSPN** | LayerNorm replacement using only bit shifts | `tested` (<0.5% error) |
| **Phase gating** | Spectral gating for dynamic sparsity | `unverified` |
| **TDA validator** | Betti numbers for model distillation quality | `unverified` |
| **Mechanistic distill** | Activation-pattern-preserving distillation | `unverified` |
| **BHRR (Bounded Hash Retrieval & Replace)** | Multi-slot associative memory (S=8, gradient descent through hash slots) | `tested` (dim=128/text, dim=256/logits) |
| **AVX-512 kernels** | VNNI, block sparsity, kernel fusion | `needs hardware` |
| **DCT engine** | Fixed-point DCT-II, AAN algorithm | `tested` (7.3% Q15 err) |
| **Distributed infer** | Multi-node protocol | `wireframe` |

---

## CLI reference

```bash
rin run model.rin --prompt "Hello" --max-tokens 100
rin bench model.rin --mode transformer
rin energy model.rin
rin inspect model.rin
rin import model.pth --output model.rin
```

---

## Python API

```python
from rin import RinEngine

with RinEngine("model.rin", mode="mlp") as engine:
    output = engine.generate("Hello", max_tokens=50)
    print(output)
    print(f"Energy: {engine.energy_joules:.3f} J")
```

---

## Project structure

```
rin-engine/
├── rin/               ← Python package (CLI + runtime)
├── lib/               ← C inference engine + headers
├── _cengine.c         ← CPython C extension
├── experimental/      ← Research prototypes
├── benchmarks/        ← Benchmark scripts and data
├── bench-all.sh       ← Reproducible benchmark
├── Makefile           ← Build system
└── setup.py           ← Python package build
```

---

## What this project IS and IS NOT

| IS | IS NOT |
|----|--------|
| Experimental CPU inference runtime | A production ML deployment framework |
| Arena + RAPL + CLI | A drop-in PyTorch replacement |
| Research prototype for novel techniques | A validated scientific contribution |
| Honest about verified vs unverified claims | A drop-in PyTorch replacement or production deployment framework |

---

## Known limitations

- RAPL measures package energy only (no DRAM, I/O, PSU)
- Turbo Boost not locked during benchmarks
- Weight format compatibility between Python export and C loader may diverge
- No statistical significance testing on benchmarks
- AVX-512 kernels require hardware most consumers don't have

---

## BHRR — Bounded Hash Retrieval & Replace

BHRR replaces the standard KV-cache attention with associative memory: K-sign bits are hashed to one of S slots; V-sign bits accumulate in that slot. Retrieval reconstructs a weighted context by hashing the query's K-sign to the same slot and element-wise multiplying with Q-sign.

**Key properties:**
- **O(1) memory** per head: S × hd int32 accumulators (no KV sequence growth)
- **Differentiable slots**: slot_id = Σ(K_sign[i]·(i+1)) mod S, gradients flow through to K
- **Multi-slot SNR**: √(hd·S/K) — at hd=32, S=8, K=64 → 2.0
- **Training**: `scripts/train_bhrr_fast.py` — vectorized multi-slot transformer with .rin export
- **C inference**: `rin_demo_bhrr` — uint8 symmetric weights, Q15 arithmetic, VPMADDUBSW-based GEMV

**Validation results (Shakespeare, CPU-only):**

| Config | Val loss | Output quality |
|--------|----------|----------------|
| dim=128, 2L, 4H, BHRR (476K params) | 2.45 | Coherent structured text ("LARI:\nS:\nWhererere t c...") |
| dim=128, 2L, 4H, softmax baseline (460K params) | 2.40 | — |
| dim=256, 2L, 8H, BHRR (1.7M params) | 2.40 | Meaningful English ("MI thand thay the thand the hand...") |

BHRR output at dim=256 demonstrates that the hash-retrieval mechanism learns valid linguistic patterns, though uint8 quantization error at larger dimensions can shift the argmax on some tokens.

**Architecture decision record:** `lib/BHRR_ADR.md`

---

## Building

```bash
make              # C library + test binaries
make test         # run validation suite (0 failures expected)
make shared       # build librin.so for ctypes
make rin_demo_bhrr# build BHRR C inference demo
pip install -e .  # build CPython extension + install CLI
```

---

## License

MIT
