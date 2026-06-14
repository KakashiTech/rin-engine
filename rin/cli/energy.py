#!/usr/bin/env python3
"""RIN energy — energy consumption profiling.

Usage:
    thor energy <model> [--mode <MODE>] [--runs <N>]

Measures energy usage per inference run using the RIN runtime's built-in
RAPL energy counters (joules), computes average power draw, and estimates
energy per token.
"""

from __future__ import annotations

import argparse
import sys
import time

from rin.runtime import RinEngine
from rin.runtime.energy import EnergyMonitor

BOLD = "\033[1m"
GREEN = "\033[32m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
RED = "\033[31m"
RESET = "\033[0m"
DIM = "\033[2m"

MODES = ["mlp", "snn", "attn", "thor", "transformer"]

RAPL_PATH = "/sys/class/powercap"


def _read_rapl() -> dict[str, float]:
    """Read RAPL powercap energy values (microjoules) from sysfs."""
    readings: dict[str, float] = {}
    try:
        for entry in sorted(os.listdir(RAPL_PATH)):
            path = os.path.join(RAPL_PATH, entry)
            if not os.path.isdir(path):
                continue
            name_file = os.path.join(path, "name")
            energy_file = os.path.join(path, "energy_uj")
            if os.path.isfile(name_file) and os.path.isfile(energy_file):
                with open(name_file) as f:
                    name = f.read().strip()
                with open(energy_file) as f:
                    uj = float(f.read().strip())
                readings[name] = uj / 1e6  # convert to joules
    except (PermissionError, FileNotFoundError, OSError):
        pass
    return readings


import os


def register_subparser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "energy",
        help="Profile energy consumption",
        description=(
            "Profile energy consumption of model inference. "
            "Reads RIN runtime energy counters plus optional RAPL powercap "
            "sensors for system-level power measurement."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("model", help="Path to model file  (e.g. model.rin)")
    p.add_argument(
        "--mode",
        default="mlp",
        choices=MODES,
        help="Inference execution mode  (default: %(default)s)",
    )
    p.add_argument(
        "--runs",
        type=int,
        default=10,
        metavar="N",
        help="Number of inference runs to measure  (default: %(default)s)",
    )


def run(args: argparse.Namespace) -> None:
    print(f"{BOLD}RIN energy{RESET}")
    print(f"{DIM}Model: {args.model}{RESET}")
    print()

    if args.runs < 1:
        print(f"{RED}Error: --runs must be >= 1{RESET}", file=sys.stderr)
        sys.exit(1)

    engine = RinEngine()
    try:
        engine.load_model(args.model)
    except Exception as e:
        print(f"{RED}Failed to load model: {e}{RESET}", file=sys.stderr)
        sys.exit(1)

    engine.mode = args.mode
    info = engine.info

    print(f"{BOLD}Model info:{RESET}")
    print(f"  Parameters: {info.get('num_parameters', '?'):>12,}")
    print(f"  Dimensions: {info.get('model_dim', '?'):>12,}")
    print(f"  Layers    : {info.get('num_layers', '?'):>12,}")
    print(f"  Mode      : {engine.mode:>12}")
    print(f"  Runs      : {args.runs:>12}")
    print()

    rapl_before = _read_rapl()
    monitor = EnergyMonitor(engine)

    # Warm-up: 3 runs (not counted)
    print(f"  Warming up ... ", end="", flush=True)
    for _ in range(3):
        engine.infer([0], max_output=1)
    print(f"{GREEN}done{RESET}")

    monitor.reset()

    print(f"  Running {args.runs} inferences ... ", end="", flush=True)
    t0 = time.perf_counter()
    for _ in range(args.runs):
        engine.infer([0], max_output=1)
    wall = time.perf_counter() - t0
    report = monitor.report()
    rapl_after = _read_rapl()
    print(f"{GREEN}done{RESET}")
    print()

    total_energy = report["consumed_joules"]
    avg_watts = report["average_watts"]
    joules_per_inf = report.get("joules_per_inference", 0.0)
    total_tokens_processed = engine.total_tokens

    print(f"{BOLD}Energy report:{RESET}")
    print(f"  {'Total energy':26s}: {total_energy:<10.6f} J")
    print(f"  {'Wall clock':26s}: {wall:<10.3f} s")
    print(f"  {'Average power':26s}: {avg_watts:<10.3f} W")
    print(f"  {'Energy per inference':26s}: {joules_per_inf:<10.6f} J")
    print(f"  {'Inferences measured':26s}: {args.runs}")
    print(f"  {'Total tokens processed':26s}: {total_tokens_processed}")

    if total_tokens_processed > 0:
        ej_per_tok = total_energy / total_tokens_processed
        print(f"  {'Energy per token':26s}: {ej_per_tok:<10.6f} J")

    print()

    # RAPL system-level readings
    if rapl_before:
        print(f"{BOLD}RAPL powercap readings:{RESET}")
        for domain in sorted(rapl_before):
            dj = rapl_after.get(domain, 0.0) - rapl_before.get(domain, 0.0)
            if dj > 0:
                pw = dj / wall if wall > 0 else 0.0
                print(f"  {domain:26s}: {dj:<10.6f} J  ({pw:<.3f} W)")
        print()
    else:
        print(f"{DIM}RAPL sensors not available (requires root or /sys/class/powercap).{RESET}")
        print()

    print(f"{GREEN}Done.{RESET}")
