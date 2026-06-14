#!/usr/bin/env python3
"""RIN inspect — display detailed model information.

Usage:
    thor inspect <model>

Shows architecture metadata, layer dimensions, parameter count, memory
footprint, available execution modes, and a sample of the charset.
"""

from __future__ import annotations

import argparse
import sys

from rin.runtime import RinEngine

BOLD = "\033[1m"
GREEN = "\033[32m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
RED = "\033[31m"
RESET = "\033[0m"
DIM = "\033[2m"

ARCH_NAMES = {
    0: "Feed-Forward (MLP)",
    1: "Transformer (GPT-like)",
    2: "Mixture of Experts (MoE)",
    3: "State-Space (Mamba)",
    4: "Hybrid",
}

MODES = ["mlp", "snn", "attn", "thor", "transformer"]


def register_subparser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "inspect",
        help="Display model information",
        description=(
            "Display detailed information about a RIN model file, including "
            "architecture, layer dimensions, parameter count, memory footprint, "
            "available execution modes, and a sample of the token charset."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("model", help="Path to model file  (e.g. model.rin)")


def run(args: argparse.Namespace) -> None:
    print(f"{BOLD}RIN inspect{RESET}")
    print(f"{DIM}Model: {args.model}{RESET}")
    print()

    engine = RinEngine()
    try:
        engine.load_model(args.model)
    except Exception as e:
        print(f"{RED}Failed to load model: {e}{RESET}", file=sys.stderr)
        sys.exit(1)

    info = engine.info

    num_params = info.get("num_parameters", 0)
    arch_code = info.get("architecture", 0)
    arch_name = ARCH_NAMES.get(arch_code, f"Unknown ({arch_code})")

    print(f"{BOLD}Architecture{RESET}")
    print(f"  Type            : {arch_name}")
    print(f"  Architecture ID : {arch_code}")
    print()

    print(f"{BOLD}Dimensions{RESET}")
    print(f"  Number of layers  : {info.get('num_layers', '?'):>12,}")
    print(f"  Model dimension   : {info.get('model_dim', '?'):>12,}")
    print(f"  FFN dimension     : {info.get('ffn_dim', '?'):>12,}")
    print(f"  Number of heads   : {info.get('num_heads', '?'):>12,}")
    print(f"  Vocab size        : {info.get('vocab_size', '?'):>12,}")
    print(f"  Max sequence len  : {info.get('max_seq_len', '?'):>12,}")
    print()

    print(f"{BOLD}Parameters & memory{RESET}")
    print(f"  Parameter count : {num_params:>12,}")
    size_mb = info.get("size_mb", 0.0)
    print(f"  Memory (fp32)   : {size_mb:>10.2f} MB")
    print(f"  Size (int8)     : {size_mb / 4:>10.2f} MB  (estimated)")
    print()

    print(f"{BOLD}Available execution modes{RESET}")
    current = engine.mode
    for m in MODES:
        try:
            engine.mode = m
            marker = f"{GREEN}●{RESET}" if m == current else "○"
            label = m if m != current else f"{CYAN}{m}{RESET}"
            print(f"    {marker}  {label}")
        except Exception:
            print(f"    {RED}✕{RESET}  {m}  (not supported)")
    engine.mode = current
    print()

    print(f"{BOLD}Charset sample{RESET}")
    try:
        charset_str = engine._ctx.get_charset() if hasattr(engine._ctx, 'get_charset') else ""
        vocab_size = len(charset_str)
        sample = charset_str[: min(vocab_size, 80)]
        print(f"  Vocab entries : {vocab_size}")
        print(f"  Sample        : {sample!r}{'...' if vocab_size > 80 else ''}")
    except Exception as e:
        print(f"  {YELLOW}Could not read charset: {e}{RESET}")
    print()

    print(f"{BOLD}Inference counters{RESET}")
    print(f"  Total tokens processed : {engine.total_tokens:>12,}")
    print(f"  Energy consumed        : {engine.energy_joules:<10.6f} J")
    print()

    print(f"{GREEN}Done.{RESET}")
