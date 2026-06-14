from __future__ import annotations

import os
import platform
from typing import Dict, Set

__all__ = ["detect_x86_capabilities", "get_optimal_mode", "x86_kernels_available"]


def x86_kernels_available() -> bool:
    """Return True if the x86 SIMD kernels are available (native path)."""
    try:
        from thorin import _cengine as ce
        return ce.best_backend() == "x86"
    except ImportError:
        return False


def _read_cpuinfo_flags() -> Set[str]:
    flags: Set[str] = set()
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("flags"):
                    parts = line.split(":", 1)
                    if len(parts) == 2:
                        flags.update(parts[1].strip().split())
    except OSError:
        pass
    return flags


def detect_x86_capabilities() -> Dict[str, bool]:
    arch = platform.machine().lower()
    if arch not in ("x86_64", "amd64", "i686", "i386"):
        return {"available": False, "arch": arch}

    flags = _read_cpuinfo_flags()

    has_avx2 = "avx2" in flags
    has_avx512f = "avx512f" in flags
    has_avx512_vnni = "avx512_vnni" in flags
    has_fma = "fma" in flags
    has_sse4_2 = "sse4_2" in flags
    has_amx = "amx_bf16" in flags or "amx_int8" in flags
    has_avx_vnni = "avx_vnni" in flags

    return {
        "available": True,
        "arch": arch,
        "avx2": has_avx2,
        "avx512f": has_avx512f,
        "avx512_vnni": has_avx512_vnni,
        "avx_vnni": has_avx_vnni,
        "fma": has_fma,
        "sse4_2": has_sse4_2,
        "amx": has_amx,
    }


def get_optimal_mode() -> str:
    caps = detect_x86_capabilities()
    if not caps.get("available"):
        return "mlp"
    if caps.get("avx512_vnni") or caps.get("amx"):
        return "transformer"
    if caps.get("avx2") and caps.get("fma"):
        return "thor"
    if caps.get("avx2"):
        return "attn"
    return "mlp"
