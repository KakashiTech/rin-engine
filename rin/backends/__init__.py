"""RIN backends — unified SIMD backend detection and auto-routing.

Usage:
    from rin.backends import best_backend, detect, summary

    info = summary()
    # → {'best': 'avx2', 'x86': {...}, 'arm': {...}, 'wasm': {...}}
"""

from __future__ import annotations

from typing import Any, Dict

from rin.backends.x86 import detect_x86_capabilities, x86_kernels_available
from rin.backends.arm import detect_arm_capabilities, neon_kernels_available
from rin.backends.wasm import is_wasm_available, wasm_simd_available

__all__ = ["detect", "best_backend", "summary"]


def detect() -> Dict[str, Any]:
    """Return a dict with all backend capabilities.

    Keys: ``x86``, ``arm``, ``wasm``, ``native_extension``.
    """
    return {
        "x86": detect_x86_capabilities(),
        "arm": detect_arm_capabilities(),
        "wasm": {"available": is_wasm_available(), "simd": wasm_simd_available()},
        "native_extension": x86_kernels_available(),
    }


def best_backend() -> str:
    """Return the name of the best available SIMD backend: ``"avx2"``,
    ``"avx512"``, ``"neon"``, ``"wasm"``, or ``"generic"``."""
    caps = detect()

    # WASM
    if caps["wasm"]["simd"]:
        return "wasm"

    # ARM NEON
    arm = caps["arm"]
    if arm.get("available") and arm.get("kernels_compiled"):
        if arm.get("sve"):
            return "sve"
        return "neon"

    # x86
    x86 = caps["x86"]
    if x86.get("available"):
        if x86.get("avx512_vnni"):
            return "avx512_vnni"
        if x86.get("avx512f"):
            return "avx512"
        if x86.get("avx2") and x86.get("fma"):
            return "avx2"
        if x86.get("sse4_2"):
            return "sse4_2"

    return "generic"


def summary() -> str:
    """Return a formatted one-line summary of the active backend."""
    be = best_backend()
    caps = detect()

    parts = [f"best={be}"]
    if caps["native_extension"]:
        parts.append("native_ext=yes")
    if caps["x86"].get("available"):
        parts.append(f"x86_{caps['x86']['arch']}")
    if caps["arm"].get("available"):
        parts.append(f"arm_{caps['arm']['arch']}")
    return " | ".join(parts)
