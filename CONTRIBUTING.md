# Contributing to RIN Engine

## Build

```bash
make              # build C library + test binary
make test         # run C tests
make shared       # build librin.so
pip install -e .  # install Python CLI
```

## Run tests

```bash
make test         # C tests
python -m pytest tests/   # Python tests
```

## Code style

- C: GCC-compatible, -Wall -Wextra clean, AVX2 available
- Python: type hints everywhere, follow PEP 8, ruff-compatible
- Comments: English only

## Pull request process

1. Ensure `make clean && make && make test` passes
2. Ensure `pip install -e .` works
3. Add tests for new functionality
4. Keep commits focused and messages descriptive

## Project structure

```
rin-engine/
├── lib/              # C inference engine (headers in include/, source in src/)
├── rin/              # Python package
├── scripts/          # Training and validation scripts
├── tests/            # Python tests
├── Makefile          # Build system
└── setup.py          # Python package build
```
