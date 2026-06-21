# Roadmap

## Sprint 1 — Reproducibility (Days 1-30)

- [x] Reproducible benchmark script (`bench-all.sh`)
- [x] Clean repo organization (stable/experimental/docs)
- [x] Honest README with verified claims only
- [ ] CI/CD with GitHub Actions publishing benchmark results
- [ ] Docker image with RIN pre-compiled
- [ ] Third-party: invite 3 engineers to run `bench-all.sh`

## Sprint 2 — Real-world demo (Days 31-60)

- [x] BHRR multi-slot associative memory (S=8) — differentiable hash slots, O(1) memory per head
- [x] BHRR C engine with uint8 quantization, Q15 arithmetic, VPMADDUBSW GEMV
- [x] BHRR validation: dim=128 (2.45 val_loss, coherent text), dim=256 (2.40, meaningful English)
- [x] Architecture Decision Record (`lib/BHRR_ADR.md`)
- [ ] BHRR vs softmax ablation at equivalent param count and layer depth
- [ ] Quantization-aware training for BHRR to close argmax gap at larger dimensions
- [ ] Train tiny-gpt-124M on real data (The Pile sample)
- [ ] Export to .rin + publish on HuggingFace
- [ ] Meaningful text generation demo
- [ ] Raspberry Pi 4 benchmark and demo
- [ ] `rin run` end-to-end working with downloaded model

## Sprint 3 — External validation (Days 61-90)

- [ ] Publish third-party benchmark results
- [ ] Fix Python/C weight format compatibility
- [ ] Lock CPU frequency for reproducible benchmarks
- [ ] Add `--json` flag to `rin bench` for CI
- [ ] Open issues for community reproduction
- [ ] Move 1-2 experimental components to stable based on validation

## Future

- [ ] WebAssembly demo page with `rin_wasm_*` kernels
- [ ] `rin chat` interactive mode
- [ ] ARM NEON runtime detection and kernel selection
- [ ] Power budget controller integration
- [ ] Model hub (`rin pull model-name`)
