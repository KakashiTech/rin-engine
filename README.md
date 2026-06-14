# RIN Engine — Experimental Neural Inference Runtime

**Deterministic memory allocation, energy measurement, and CPU inference without GPU.**

RIN Engine is an experimental C runtime + Python CLI for running neural network inference on CPU. It explores techniques for energy-efficient computation: arena allocation (zero malloc during inference), RAPL-based energy metering, and alternative normalization/softmax implementations using integer arithmetic.

[![CI](https://github.com/thor-ai/rinin/actions/workflows/ci.yml/badge.svg)](https://github.com/thor-ai/rinin/actions/workflows/ci.yml)

---

## Quick start

```bash
git clone https://github.com/thor-ai/rinin.git
cd thor
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
| **5 inference modes** | MLP, SNN, ATTN, THOR, Transformer | `rin run --mode` |
| **Python CLI** | `thor {run,bench,energy,inspect,import}` | `rin/` |
| **C test suite** | 8 quantitative validation tests, 0 failures | `make test` |
| **Backend router** | Auto-selects native CPython ext → ctypes fallback | `thor/_backend.py` |

---

## Verified performance

Hardware: AMD Ryzen 5 5600G, DDR4-3200, GCC 15. Model: 4-layer, 262K params, INT8.

| Comparison | THOR | ONNX | Ratio |
|-----------|------|------|-------|
| Latency (µs/inf) | 202 | 783 (FP32) | **3.9× faster** |
| Energy (µJ/pkg) | 5,654 | 28,510 (FP32) | **5.0× efficient** |

> **Note:** THOR uses INT8 quantization while ONNX runs FP32. This is the current baseline. A direct INT8-vs-INT8 comparison requires an INT8-capable ONNX Runtime build (ortextensions or onnxruntime-trt) — this will be added in a future release. Methodology details and limitations in `CLAIMS_HONEST.md`.

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
from rin import ThorEngine

with ThorEngine("model.rin", mode="mlp") as engine:
    output = engine.generate("Hello", max_tokens=50)
    print(output)
    print(f"Energy: {engine.energy_joules:.3f} J")
```

---

## Project structure

```
thor/
├── thor/              ← Python package (CLI + runtime)
├── rin_core.c/.h      ← C inference engine (1,334 lines)
├── thor_api.c/.h      ← Public C API (33 functions)
├── _cengine.c         ← CPython C extension
├── rin_arena.h        ← Arena allocator
├── rin_energy_meter.h ← RAPL energy metering
├── rin_test_suite.h   ← Validation test framework
├── experimental/      ← Research prototypes
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
| Honest about verified vs unverified claims | Revolutionary or hyperbolic |

---

## Known limitations

- RAPL measures package energy only (no DRAM, I/O, PSU)
- Turbo Boost not locked during benchmarks
- Weight format compatibility between Python export and C loader may diverge
- No statistical significance testing on benchmarks
- AVX-512 kernels require hardware most consumers don't have

---

## Building

```bash
make              # C library + test binaries
make test         # run validation suite (0 failures expected)
make shared       # build librin.so for ctypes
pip install -e .  # build CPython extension + install CLI
```

---

## License

MIT
