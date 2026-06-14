#!/usr/bin/env python3
"""THORIN bench — benchmark inference speed.

Usage:
    thor bench <model> [--mode <MODE>] [--warmup <N>] [--iterations <N>]
                      [--all-modes]
"""

from __future__ import annotations

import argparse
import sys

from thorin.runtime import ThorEngine

BOLD = "\033[1m"
GREEN = "\033[32m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
RED = "\033[31m"
RESET = "\033[0m"
DIM = "\033[2m"
HEADER = "\033[1;37m"

MODES = ["mlp", "snn", "attn", "thor", "transformer"]


def register_subparser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "bench",
        help="Benchmark inference speed",
        description="Measure inference throughput and latency for one or all execution modes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("model", help="Path to model file  (e.g. model.rin)")
    p.add_argument(
        "--mode",
        default="mlp",
        choices=MODES,
        help="Inference mode to benchmark  (default: %(default)s)",
    )
    p.add_argument(
        "--warmup",
        type=int,
        default=10,
        metavar="N",
        help="Number of warm-up iterations  (default: %(default)s)",
    )
    p.add_argument(
        "--iterations",
        type=int,
        default=100,
        metavar="N",
        help="Number of measured iterations  (default: %(default)s)",
    )
    p.add_argument(
        "--all-modes",
        action="store_true",
        help="Benchmark all 5 modes and produce a comparison table",
    )


def _fmt(val: float, unit: str = "") -> str:
    if unit == "ms":
        return f"{val:>10.3f} ms"
    if unit == "tps":
        return f"{val:>10.1f} t/s"
    if unit == "ej":
        return f"{val:>10.6f} J"
    return f"{val:>10.3f}"


def _run_single(
    engine: ThorEngine, mode: str, warmup: int, iterations: int
) -> dict:
    engine.mode = mode
    result = engine.benchmark(mode=mode, warmup=warmup, iterations=iterations)
    energy = engine.energy_joules
    return {
        "mode": mode,
        "ms_per_token": result["ms_per_token"],
        "tokens_per_second": result["tokens_per_second"],
        "energy_joules": energy,
    }


def run(args: argparse.Namespace) -> None:
    print(f"{BOLD}THOR bench{RESET}")
    print(f"{DIM}Model: {args.model}{RESET}")
    print()

    if args.warmup < 0:
        print(f"{RED}Error: --warmup must be >= 0{RESET}", file=sys.stderr)
        sys.exit(1)
    if args.iterations < 1:
        print(f"{RED}Error: --iterations must be >= 1{RESET}", file=sys.stderr)
        sys.exit(1)

    engine = ThorEngine()
    try:
        engine.load_model(args.model)
    except Exception as e:
        print(f"{RED}Failed to load model: {e}{RESET}", file=sys.stderr)
        sys.exit(1)

    info = engine.info
    print(f"{BOLD}Model info:{RESET}")
    print(f"  Parameters : {info.get('num_parameters', '?'):>12,}")
    print(f"  Dimensions : {info.get('model_dim', '?'):>12,}")
    print(f"  Layers     : {info.get('num_layers', '?'):>12,}")
    print(f"  Warmup     : {args.warmup:>12}")
    print(f"  Iterations : {args.iterations:>12}")
    print()

    modes_to_run = MODES if args.all_modes else [args.mode]

    results = []
    for mode in modes_to_run:
        print(f"  Benchmarking mode {CYAN}{mode}{RESET} ... ", end="", flush=True)
        try:
            r = _run_single(engine, mode, args.warmup, args.iterations)
            results.append(r)
            print(f"{GREEN}OK{RESET}")
        except Exception as e:
            print(f"{RED}FAILED ({e}){RESET}")
            results.append({"mode": mode, "error": str(e)})

    print()
    print(f"{HEADER}{'Mode':<14} {'ms/tok':>12} {'tok/s':>10} {'J/tok':>12} {'Status'}{RESET}")
    print(f"{DIM}{'-' * 65}{RESET}")

    for r in results:
        if "error" in r:
            print(
                f"  {r['mode']:<12} {'—':>12} {'—':>10} {'—':>12}  {RED}error{RESET}"
            )
        else:
            ms = r["ms_per_token"]
            tps = r["tokens_per_second"]
            ej = r["energy_joules"] / max(args.iterations, 1)
            print(
                f"  {r['mode']:<12}"
                f" {ms:>10.3f} ms"
                f" {tps:>10.1f}"
                f" {ej:>12.6f}"
                f"  {GREEN}OK{RESET}"
            )

    print()

    if len(results) > 1:
        valid = [r for r in results if "error" not in r]
        if valid:
            best_tps = max(valid, key=lambda x: x["tokens_per_second"])
            best_energy = min(valid, key=lambda x: x["energy_joules"])
            print(f"{BOLD}Best throughput :{RESET} {CYAN}{best_tps['mode']}{RESET}"
                  f"  ({best_tps['tokens_per_second']:.1f} tok/s)")
            print(f"{BOLD}Best efficiency :{RESET} {CYAN}{best_energy['mode']}{RESET}"
                  f"  ({best_energy['energy_joules'] / args.iterations:.6f} J/tok)")

    print(f"{GREEN}Done.{RESET}")
