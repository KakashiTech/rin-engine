#!/usr/bin/env python3
"""RIN run — load a model and generate text.

Usage:
    thor run <model> [--prompt <text>] [--gen <N>] [--mode <MODE>]
                   [--temp <T>] [--top-k <K>] [--top-p <P>]
"""

from __future__ import annotations

import argparse
import sys
import time

from rin.runtime import RinEngine

BOLD = "\033[1m"
GREEN = "\033[32m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
RED = "\033[31m"
RESET = "\033[0m"
DIM = "\033[2m"

MODES = ["mlp", "snn", "attn", "thor", "transformer"]


def register_subparser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "run",
        help="Run inference and generate text",
        description="Load a model and generate text with optional sampling parameters.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("model", help="Path to model file  (e.g. model.rin)")
    p.add_argument(
        "--prompt",
        default="",
        help="Input prompt text  (default: empty / null token)",
    )
    p.add_argument(
        "--gen",
        type=int,
        default=20,
        metavar="N",
        help="Number of tokens to generate  (default: %(default)s)",
    )
    p.add_argument(
        "--mode",
        default="mlp",
        choices=MODES,
        help="Inference execution mode  (default: %(default)s)",
    )
    p.add_argument(
        "--temp",
        type=float,
        default=None,
        metavar="T",
        help="Sampling temperature  (default: engine default)",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=None,
        metavar="K",
        help="Top-K sampling  (default: engine default, 0 = disabled)",
    )
    p.add_argument(
        "--top-p",
        type=float,
        default=None,
        metavar="P",
        help="Top-P nucleus sampling  (default: engine default, 1.0 = disabled)",
    )


def run(args: argparse.Namespace) -> None:
    print(f"{BOLD}RIN run{RESET}")
    print(f"{DIM}Model: {args.model}{RESET}")
    print()

    if args.gen < 1:
        print(f"{RED}Error: --gen must be >= 1{RESET}", file=sys.stderr)
        sys.exit(1)

    engine = RinEngine()
    try:
        engine.load_model(args.model)
    except Exception as e:
        print(f"{RED}Failed to load model: {e}{RESET}", file=sys.stderr)
        sys.exit(1)

    # Auto-detect mode from model architecture if not explicitly set
    info = engine.info
    arch = info.get("architecture", 0)
    if args.mode == "mlp" and arch == 1:
        engine.mode = "transformer"
    else:
        engine.mode = args.mode

    if args.temp is not None:
        engine.temperature = args.temp
    if args.top_k is not None:
        engine.top_k = args.top_k
    if args.top_p is not None:
        engine.top_p = args.top_p

    info = engine.info

    print(f"{BOLD}Model info:{RESET}")
    print(f"  Parameters : {info.get('num_parameters', '?'):>12,}")
    print(f"  Dimensions : {info.get('model_dim', '?'):>12,}")
    print(f"  Layers     : {info.get('num_layers', '?'):>12,}")
    print(f"  Mode       : {engine.mode:>12}")
    print()

    print(f"{BOLD}Generating up to {args.gen} tokens...{RESET}")
    sys.stdout.flush()

    t0 = time.perf_counter()
    if args.prompt:
        generated = engine.generate(args.prompt, max_tokens=args.gen)
    else:
        generated = engine.generate("", max_tokens=args.gen)
    wall = time.perf_counter() - t0

    tokens_generated = len(engine.decode(engine.encode(generated))) if generated else 0
    try:
        token_count = len(engine.encode(generated))
    except Exception:
        token_count = args.gen

    print()
    if args.prompt:
        print(f"{BOLD}Prompt:{RESET} {args.prompt}")
    print(f"{BOLD}Generated:{RESET} {generated}")
    print()

    energy_j = engine.energy_joules
    tps = token_count / wall if wall > 0 else 0.0

    print(f"{BOLD}Stats:{RESET}")
    print(f"  Wall time       : {wall * 1000:.1f} ms")
    print(f"  Tokens generated: {token_count}")
    print(f"  Speed           : {tps:.1f} tok/s")
    print(f"  Latency         : {wall / max(token_count, 1) * 1000:.1f} ms/tok")
    print(f"  Energy          : {energy_j:.6f} J")
    print(f"  Energy/token    : {energy_j / max(token_count, 1):.6f} J/tok")
    print(f"{GREEN}Done.{RESET}")
