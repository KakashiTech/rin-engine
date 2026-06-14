"""RIN Runtime — Unified backend (native C extension or ctypes fallback).

Usage:
    from rin import RinEngine
    engine = RinEngine("model.rin", mode="transformer")
    text = engine.generate("Hello", max_tokens=50)
"""

from rin.runtime import RinEngine, EnergyMonitor, Profiler
from rin._backend import (
    RinException,
    MODE_MLP, MODE_SNN, MODE_ATTN, MODE_THOR, MODE_TRANSFORMER,
    HAS_NATIVE,
)

__version__ = "1.0.0"
__all__ = [
    "RinEngine", "EnergyMonitor", "Profiler",
    "RinException",
    "MODE_MLP", "MODE_SNN", "MODE_ATTN", "MODE_THOR", "MODE_TRANSFORMER",
    "HAS_NATIVE",
]
