#!/usr/bin/env python3
"""RIN import — convert external model formats to RIN.

Usage:
    thor import <model> --output <model.rin> [--type auto|pytorch|onnx|gguf]
                      [--quantize int8|int4|fp16]

Detects source format from file extension and dispatches to the appropriate
converter.  Supported source formats:
  - PyTorch  (.pt, .pth)
  - ONNX     (.onnx)
  - GGUF     (.gguf)  (experimental)
"""

from __future__ import annotations

import argparse
import os
import sys
import time

BOLD = "\033[1m"
GREEN = "\033[32m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
RED = "\033[31m"
RESET = "\033[0m"
DIM = "\033[2m"

EXT_MAP: dict[str, str] = {
    ".pt": "pytorch",
    ".pth": "pytorch",
    ".onnx": "onnx",
    ".gguf": "gguf",
}

QUANTIZE_MAP: dict[str, str] = {
    "int8": "int8",
    "int4": "int4",
    "fp16": "fp16",
}


def register_subparser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "import",
        help="Import model from external format",
        description=(
            "Convert a model from an external format (PyTorch, ONNX, GGUF) "
            "to the native RIN format.  Source type is auto-detected "
            "from the file extension when --type is not specified."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "model",
        help="Path to source model file  (e.g. model.pt, model.onnx)",
    )
    p.add_argument(
        "--output",
        "-o",
        required=True,
        metavar="FILE",
        help="Output path for the converted .rin model",
    )
    p.add_argument(
        "--type",
        choices=["auto", "pytorch", "onnx", "gguf"],
        default="auto",
        help="Source model type  (default: auto-detect from extension)",
    )
    p.add_argument(
        "--quantize",
        choices=["int8", "int4", "fp16"],
        default=None,
        help="Target quantization  (default: no quantization)",
    )


def _detect_type(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    fmt = EXT_MAP.get(ext)
    if fmt is None:
        print(
            f"{RED}Error: cannot auto-detect model type from extension "
            f"{ext!r}.  Use --type to specify explicitly.{RESET}",
            file=sys.stderr,
        )
        sys.exit(1)
    return fmt


def _convert_pytorch(src: str, dst: str, quantize: str | None) -> None:
    try:
        from rin.importer import convert_pytorch_model
    except ImportError as e:
        print(
            f"{RED}Error: PyTorch importer not available. "
            f"Install 'torch' and ensure thor.importer is installed.\n"
            f"  Details: {e}{RESET}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"  {DIM}Converting PyTorch model ...{RESET}")
    kwargs = {"quantize": quantize} if quantize else {}
    convert_pytorch_model(src, dst, **kwargs)


def _convert_onnx(src: str, dst: str, quantize: str | None) -> None:
    try:
        from rin.importer import convert_onnx_model
    except ImportError as e:
        print(
            f"{RED}Error: ONNX importer not available. "
            f"Install 'onnx' and ensure thor.importer is installed.\n"
            f"  Details: {e}{RESET}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"  {DIM}Converting ONNX model ...{RESET}")
    kwargs = {"quantize": quantize} if quantize else {}
    convert_onnx_model(src, dst, **kwargs)


def _convert_gguf(src: str, dst: str, quantize: str | None) -> None:
    print(
        f"{YELLOW}Warning: GGUF import is experimental.{RESET}"
    )
    try:
        from rin.importer.gguf import convert_gguf_model as gguf_convert
    except ImportError as e:
        print(
            f"{RED}Error: GGUF importer not available.\n"
            f"  Details: {e}{RESET}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"  {DIM}Converting GGUF model ...{RESET}")
    kwargs = {"quantize": quantize} if quantize else {}
    gguf_convert(src, dst, **kwargs)


_CONVERTERS: dict[str, callable] = {
    "pytorch": _convert_pytorch,
    "onnx": _convert_onnx,
    "gguf": _convert_gguf,
}


def run(args: argparse.Namespace) -> None:
    print(f"{BOLD}RIN import{RESET}")
    print()

    if not os.path.isfile(args.model):
        print(f"{RED}Error: source file not found: {args.model}{RESET}", file=sys.stderr)
        sys.exit(1)

    out_dir = os.path.dirname(args.output)
    if out_dir and not os.path.isdir(out_dir):
        print(f"{RED}Error: output directory does not exist: {out_dir}{RESET}", file=sys.stderr)
        sys.exit(1)

    src_size = os.path.getsize(args.model)
    fmt = args.type if args.type != "auto" else _detect_type(args.model)

    print(f"  Source      : {args.model}")
    print(f"  Source size : {src_size:,} bytes")
    print(f"  Source type : {fmt}")
    print(f"  Output      : {args.output}")
    if args.quantize:
        print(f"  Quantization: {args.quantize}")
    print()

    converter = _CONVERTERS.get(fmt)
    if converter is None:
        print(
            f"{RED}Error: unsupported model type {fmt!r}. "
            f"Supported types: pytorch, onnx, gguf{RESET}",
            file=sys.stderr,
        )
        sys.exit(1)

    t0 = time.perf_counter()
    try:
        converter(args.model, args.output, args.quantize)
    except Exception as e:
        print(f"{RED}Conversion failed: {e}{RESET}", file=sys.stderr)
        sys.exit(1)
    elapsed = time.perf_counter() - t0

    dst_size = os.path.getsize(args.output) if os.path.isfile(args.output) else 0
    ratio = dst_size / src_size if src_size > 0 else 0.0

    print()
    print(f"{BOLD}Conversion summary:{RESET}")
    print(f"  Time      : {elapsed:.2f} s")
    print(f"  Out size  : {dst_size:,} bytes  ({ratio:.2%} of source)")
    print(f"  Output    : {args.output}")
    print(f"{GREEN}Done.{RESET}")
