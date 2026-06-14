# THOR — Structural Report for External Evaluation

## 1. Architecture Overview

```
┌──────────────────────────────────────────────────────┐
│                    CLI Layer                          │
│  thor {run, bench, energy, inspect, import}          │
│  thor/__main__.py → thor/cli/main.py                 │
└──────────────────────┬───────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────┐
│                 Python Runtime Layer                  │
│  ThorEngine        — high-level inference API         │
│  Profiler          — benchmarking + metrics           │
│  EnergyMonitor     — energy tracking                  │
│  PowerTuner        — power budget optimization        │
│  Importers         — PyTorch / ONNX / GGUF converters │
│  Backend Router    — auto-selects best runtime        │
│  thor/runtime/     — 4 files, 695 lines              │
│  thor/importer/    — 5 files, 1,086 lines            │
└──────────────────────┬───────────────────────────────┘
                       │  ctypes / CPython C API
┌──────────────────────▼───────────────────────────────┐
│                  C Runtime Layer                      │
│                                                       │
│  rin_core.c (1,334 lines) — main inference engine     │
│    - 5 modes: MLP / SNN / ATTN / THOR / TRANSFORMER  │
│    - KV cache, autoregressive generation              │
│    - INT8 quantization, positional encoding           │
│                                                       │
│  thor_api.c (295 lines) — public C API (33 exports)   │
│  _cengine.c (567 lines) — CPython C extension         │
│                                                       │
│  rin_arena.h (265 lines) — O(1) arena allocator       │
│  rin_energy_meter.h (481 lines) — RAPL MSR energy     │
│  rin_test_suite.h (653 lines) — validation framework  │
└──────────────────────┬───────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────┐
│          Research / Experimental Layer                │
│                                                       │
│  rin_lif_engine.h       — multiplication-free LIF    │
│  rin_ptsoftmax.h        — power-of-two softmax       │
│  rin_bspn.h             — bit-shift LayerNorm        │
│  rin_phase_gating.h     — spectral phase gating      │
│  rin_betti_calculator.h — topological data analysis  │
│  rin_mechanistic_distill.h — mechanistic distillation │
│  rin_dct_engine.h       — fixed-point DCT-II         │
│  rin_distributed.h      — multi-node protocol        │
│  experimental/rin-x/*   — AVX-512 VNNI kernels       │
└──────────────────────────────────────────────────────┘
```

## 2. Bill of Materials (Complete Inventory)

### 2.1 Production C (stable, tested, ready)

| File | Lines | Role | Dependencies |
|------|-------|------|--------------|
| `rin_core.c` | 1,334 | Core inference engine — all 5 modes | rin_core.h + all headers |
| `thor_api.c` | 295 | Public C API — 33 exported functions | thor_api.h, rin_core.h |
| `_cengine.c` | 567 | CPython C extension — PyThorContext type | Python.h, thor_api.h |
| `rin_core.h` | 621 | Core API — config, modes, inference dispatch | none (standalone header) |
| `thor_api.h` | 121 | Clean public API surface | rin_core.h |
| `rin_arena.h` | 265 | Arena allocator — O(1) alloc/reset, 3 pools | stdlib |
| `rin_energy_meter.h` | 481 | RAPL MSR energy — PKG/PP0/PP1/DRAM/PLATFORM | linux/msr.h, sysfs |
| `rin_test_suite.h` | 653 | Validation framework — 8 test types | all headers |
| `rin_test.c` | 616 | Test implementations — 12 test functions | rin_test_suite.h |
| `main.c` | 299 | C demo binary | rin_core.h |

**Total production C:** 5,252 lines across 10 files.

### 2.2 Backend Kernels (compiled into librin.so)

| File | Lines | Role |
|------|-------|------|
| `thor/backends/thor_backend.h` | 52 | Unified backend API declarations |
| `thor/backends/detect.c` | 17 | Runtime best-backend detection |
| `thor/backends/neon_kernels.c` | 139 | ARM NEON SGEMM/GEMV/Softmax/INT8-GEMM |
| `thor/backends/wasm_kernels.c` | 122 | WASM SIMD SGEMM/GEMV/Softmax |

### 2.3 Research / Experimental C (prototypes)

| File | Lines | Status | Key Claim |
|------|-------|--------|-----------|
| `rin_lif_engine.h` | 443 | Implemented, synthetic tests pass | "Zero multiplication" — true algorithmically |
| `rin_ptsoftmax.h` | 375 | 2550/2550 tests pass | Replaces exp() with LUT |
| `rin_bspn.h` | 393 | Implemented, <0.5% error vs LayerNorm | Shift-based normalization |
| `rin_phase_gating.h` | 417 | Implemented | "90% network off" — **unverified** |
| `rin_betti_calculator.h` | 532 | Works on synthetic data | Topological distillation validation — **unverified** |
| `rin_mechanistic_distill.h` | 471 | Framework exists | Activation-pattern distillation — **unverified** |
| `rin_dct_engine.h` | 478 | Implemented, 7.3% Q15 error | Fixed-point DCT, 5 mults vs 64 |
| `rin_distributed.h` | 542 | Wireframe only | Multi-node protocol — **not tested** |
| `experimental/rin-x/rin_x.h` | 352 | Compiles on AVX-512 | 133× speedup claim — **needs hardware + validation** |

**Total research C:** 4,003 lines across 9 files (+ 24 rinx files, 2,200+ lines).

### 2.4 Python Package (5,230 lines)

| Subpackage | Files | Lines | Role |
|-----------|-------|-------|------|
| `thor/cli/` | 7 | 889 | CLI commands (run/bench/energy/inspect/import) |
| `thor/runtime/` | 4 | 695 | ThorEngine, Profiler, EnergyMonitor |
| `thor/importer/` | 5 | 1,086 | PyTorch/ONNX/GGUF model conversion |
| `thor/backends/` | 6 | 378 | Backend detection + Python stubs |
| `thor/_backend.py` | 1 | 257 | Unified backend router (native → ctypes fallback) |
| `thor/_thor_native.py` | 1 | — | ctypes bindings to librin.so |
| `thor/ir/` | 4 | — | Graph IR, optimization, quantization |
| `thor/power/` | 2 | — | Power budget tuner |
| `thor/models/` | 2 | — | Model registry |

### 2.5 Documentation (6,837 lines)

| Directory | Files | Lines | Content |
|-----------|-------|-------|---------|
| `docs/` | 23 | 6,559 | CLAIMS, SOTA research, limitations, scientific reports |
| `README.md` | 1 | 161 | Honest project overview (current) |
| `ROADMAP.md` | 1 | 46 | 90-day development roadmap |
| `experimental/README.md` | 1 | 107 | Research component status per-component |

### 2.6 Benchmark & Support Files

| Directory | Files | Content |
|-----------|-------|---------|
| `benchmarks/` | 44 | Historical benchmark scripts + result CSVs |
| `models/` | 34 | Pre-trained model weights (.onnx, .bin, .npz) |
| `scripts/` | 33 | Training scripts, validation scripts |
| `archive/` | 31 | Legacy C implementations (superseded) |

## 3. Build System

```bash
make          # → rin_test, rin_demo, *.o
make test     # → compiles + runs C test suite (0 failures)
make shared   # → librin.so (shared library for ctypes)
pip install -e .  # → builds _cengine CPython extension + installs CLI
```

Three runtime layers, automatically selected:
1. **Native C extension** (`_cengine.cpython-*.so`) — zero FFI overhead, Python object with GC
2. **ctypes fallback** (`_thor_native.py` → `librin.so`) — works when extension not compiled
3. **Pure Python stubs** — graceful degradation for import commands without deps

## 4. Performance Claims (Verified)

| Metric | THOR | ONNX FP32 | Speedup | Energy Efficiency |
|--------|------|-----------|---------|-------------------|
| Latency (µs/inf) | 202 | 783 | **3.9×** ✅ | — |
| Energy (µJ/pkg) | 5,654 | 28,510 | — | **5.0×** ✅ |

Hardware: AMD Ryzen 5 5600G, DDR4-3200, GCC 15. Model: 4-layer, dim=256, INT8.

### Claims retired from marketing

| Old claim | Replaced by | Reason |
|-----------|-------------|--------|
| "133× vs PyTorch" | "3.9× vs ONNX" | 133× used training as baseline (unfair) |
| "81× energy efficiency" | "5.0× vs ONNX" | 81× used training as baseline (unfair) |
| "Revolutionary architecture" | "Experimental inference runtime" | Hype language erodes credibility |

### Known limitations (from CLAIMS_HONEST.md)

- RAPL measures package CPU only (no DRAM/I/O/PSU)
- Turbo Boost active during benchmarks (frequency not locked)
- No thread isolation (possible core migration)
- Python export ↔ C loader format compatibility not guaranteed
- Statistical significance not formally tested (no CI reported)

## 5. Maturity Assessment (Per-Component)

```
┌──────────────────────────────────────────────────────┐
│                    PRODUCTION                         │
│  ┌────────────────────────────────────────────────┐  │
│  │  rin_arena.h      ●●●●●●●●●● 100%  tested     │  │
│  │  rin_energy_meter ●●●●●●●●●○  90%  tested     │  │
│  │  rin_core.c       ●●●●●●●●●○  90%  tested     │  │
│  │  thor_api.c       ●●●●●●●●●○  90%  tested     │  │
│  │  _cengine.c       ●●●●●●●●●○  90%  tested     │  │
│  │  CLI              ●●●●●●●●●○  90%  tested     │  │
│  │  Test suite       ●●●●●●●●●● 100%  passes     │  │
│  └────────────────────────────────────────────────┘  │
├──────────────────────────────────────────────────────┤
│                  EXPERIMENTAL                         │
│  ┌────────────────────────────────────────────────┐  │
│  │  rin_ptsoftmax.h  ●●●●●●●●●○  90%  tested     │  │
│  │  rin_bspn.h       ●●●●●●●●●○  90%  tested     │  │
│  │  rin_lif_engine.h ●●●●●●●●●○  90%  tested     │  │
│  │  rin_dct_engine.h ●●●●●●○○○○  60%  (7.3% err) │  │
│  │  phase_gating     ●●●●○○○○○○  40%  unverified │  │
│  │  betti_calculator ●●●○○○○○○○  30%  unverified │  │
│  │  distill          ●●●○○○○○○○  30%  unverified │  │
│  │  distributed      ●○○○○○○○○○  10%  wireframe  │  │
│  │  RIN-X AVX-512    ●●●○○○○○○○  30%  unverified │  │
│  └────────────────────────────────────────────────┘  │
├──────────────────────────────────────────────────────┤
│                   NEEDS WORK                          │
│  ┌────────────────────────────────────────────────┐  │
│  │  Weight format compat (Python↔C)              │  │
│  │  main.c buffer overflow (784 vs 256)           │  │
│  │  Reproducible benchmarks (lock freq)           │  │
│  │  Third-party validation                        │  │
│  └────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────┘
```

## 6. Directory Structure (Clean, Post-Organization)

```
thor/
├── README.md              ← honest project overview
├── ROADMAP.md             ← 90-day plan
├── bench-all.sh           ← reproducible benchmark
├── Makefile               ← build system
├── setup.py               ← Python package build
├── pyproject.toml         ← project metadata
│
├── thor/                  ← Python package (5,230 lines)
│   ├── cli/               ←   CLI subcommands
│   ├── runtime/           ←   ThorEngine + Profiler
│   ├── importer/          ←   PyTorch/ONNX/GGUF converters
│   ├── backends/          ←   Detection + ARM/WASM/x86 stubs
│   ├── ir/                ←   Graph IR + quantization
│   ├── power/             ←   Power tuner
│   └── models/            ←   Model registry
│
├── rin_core.c / .h        ← C inference engine
├── thor_api.c / .h        ← public C API
├── _cengine.c             ← CPython C extension
├── rin_arena.h            ← O(1) arena allocator (stable)
├── rin_energy_meter.h     ← RAPL energy meter (stable)
├── rin_test.c             ← test suite
├── rin_test_suite.h       ← test framework (stable)
├── main.c                 ← C demo (needs cleanup)
│
├── experimental/          ← research prototypes
│   ├── README.md          ←   per-component status
│   ├── rin-x/             ←   AVX-512 kernels (27 files)
│   ├── phase-gating/      ←   (header at root level)
│   ├── betti/             ←   (header at root level)
│   ├── mechanistic-distill/
│   ├── distributed/
│   └── dct/
│
├── docs/                  ← 23 documentation files
├── benchmarks/            ← 44 benchmark scripts + data
├── models/                ← 34 model weight files
├── scripts/               ← 33 training/validation scripts
└── archive/               ← 31 legacy files (superseded)
```

## 7. Summary for Evaluator

**What this project IS:**
- Experimental CPU inference runtime with energy measurement
- Arena allocator + RAPL metering (production quality)
- 5 inference modes (MLP → Transformer) working end-to-end
- Clean CLI and Python API
- Honest about what's verified (3.9× vs ONNX) vs what's research

**What this project IS NOT:**
- Not a production ML deployment framework
- Not a drop-in PyTorch replacement
- Not a validated scientific contribution (needs third-party reproduction)

**Strengths:**
- Original combination: arena + energy metering + no-dependency inference
- Clean C codebase with 0 test failures
- Research components demonstrate novel techniques (shift-based math, phase gating)
- Documentation is honest about limitations

**Risks:**
- Weight format compatibility between Python export and C loader
- Experimental claims (phase gating, 90% sparsity) will be met with skepticism
- Cannot produce meaningful text without training pipeline improvements
- No external validation yet
