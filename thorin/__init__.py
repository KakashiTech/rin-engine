"""THOR Runtime — Unified backend (native C extension or ctypes fallback).

Usage:
    from thorin import ThorEngine
    engine = ThorEngine("model.rin", mode="transformer")
    text = engine.generate("Hello", max_tokens=50)
"""

from thorin.runtime import ThorEngine, EnergyMonitor, Profiler
from thorin._backend import (
    ThorException,
    MODE_MLP, MODE_SNN, MODE_ATTN, MODE_THOR, MODE_TRANSFORMER,
    HAS_NATIVE,
)

__version__ = "1.0.0"
__all__ = [
    "ThorEngine", "EnergyMonitor", "Profiler",
    "ThorException",
    "MODE_MLP", "MODE_SNN", "MODE_ATTN", "MODE_THOR", "MODE_TRANSFORMER",
    "HAS_NATIVE",
]
