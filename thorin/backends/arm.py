"""ARM backend detection and NEON kernel availability.

Provides:
  detect_arm_capabilities()   — query CPU features from /proc/cpuinfo
  neon_kernels_available()   — True when compiled-in NEON kernels exist
  arm_optimized_gemm()       — placeholder for future NEON-accelerated path
"""

from __future__ import annotations

import platform
from typing import Dict

__all__ = [
    "detect_arm_capabilities",
    "neon_kernels_available",
    "arm_optimized_gemm",
]


def neon_kernels_available() -> bool:
    """Return True if NEON kernels are compiled into the native extension."""
    try:
        from thorin import _cengine as ce
        return bool(ce.neon_available())
    except ImportError:
        return False


def detect_arm_capabilities() -> Dict[str, bool]:
    arch = platform.machine().lower()
    is_arm = arch in ("aarch64", "armv7l", "armv8l", "arm64")
    if not is_arm:
        return {"available": False, "arch": arch}

    has_neon = False
    has_sve = False
    has_dotprod = False

    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("Features"):
                    features = line.split(":", 1)[1].strip().lower()
                    has_neon = "neon" in features
                    has_sve = "sve" in features
                    has_dotprod = "asimddp" in features or "dotprod" in features
    except OSError:
        pass

    return {
        "available": True,
        "arch": arch,
        "neon": has_neon,
        "sve": has_sve,
        "dotprod": has_dotprod,
        "kernels_compiled": neon_kernels_available(),
    }


def arm_optimized_gemm() -> bool:
    """Return True if the ARM NEON GEMM kernel is ready to use."""
    caps = detect_arm_capabilities()
    return caps.get("neon", False) and caps.get("kernels_compiled", False)
