"""WASM backend detection and SIMD kernel availability.

Provides:
  is_wasm_available()       — True when running under WASM (emscripten/wasi)
  wasm_simd_available()     — True when WASM SIMD kernels are compiled in
"""

from __future__ import annotations

import os
import sys

__all__ = ["is_wasm_available", "wasm_simd_available"]


def is_wasm_available() -> bool:
    if hasattr(sys, "platform"):
        return sys.platform in ("emscripten", "wasi")
    return bool(os.environ.get("RIN_WASM") or os.environ.get("EMSCRIPTEN"))


def wasm_simd_available() -> bool:
    """Return True if the WASM SIMD kernels are compiled into the native
    extension (always False on non-WASM platforms)."""
    try:
        from rin import _cengine as ce
        return bool(ce.wasm_available())
    except ImportError:
        return False
