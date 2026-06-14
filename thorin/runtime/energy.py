"""Energy monitoring utilities for THOR inference sessions.

Provides :class:`EnergyMonitor` for measuring energy consumption deltas
during model execution via the THOR C runtime's built-in RAPL / energy
counters.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from thorin.runtime.engine import ThorEngine

# Lazy import of low-level energy functions — they come from whichever
# backend is active (native or ctypes).
_energy_cache: dict = {}

def _get_energy_joules():
    if "joules" not in _energy_cache:
        try:
            from thorin._thor_native import thorin_get_energy_joules as fn
        except Exception:
            # Native path — energy is a property on the context
            fn = None
        _energy_cache["joules"] = fn
    return _energy_cache["joules"]


def _get_inference_count():
    if "count" not in _energy_cache:
        try:
            from thorin._thor_native import thorin_get_inference_count as fn
        except Exception:
            fn = None
        _energy_cache["count"] = fn
    return _energy_cache["count"]

__all__ = ["EnergyMonitor"]


class EnergyMonitor:
    """Track energy consumption deltas for a :class:`ThorEngine`.

    Samples the engine's built-in energy counter at strategic points so the
    caller can measure the energy cost of individual inference calls or
    batches.

    Parameters
    ----------
    engine : ThorEngine, optional
        Engine to monitor.  When ``None`` (default) only the software
        timer is available for wall-clock measurements.
    """

    def __init__(self, engine: Optional[ThorEngine] = None) -> None:
        self._engine = engine
        self._start_joules: float = 0.0
        self._start_timer: float = 0.0
        self._start_count: int = 0
        self.reset()

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(self) -> float:
        """Return the current cumulative energy in joules from the engine.

        Falls back to ``0.0`` when no engine is attached.
        """
        if self._engine is not None:
            return self._engine.energy_joules
        return 0.0

    def reset(self) -> None:
        """Reset the baseline for all tracked metrics."""
        self._start_joules = self.sample()
        self._start_timer = time.perf_counter()
        self._start_count = (
            self._engine.inference_count if self._engine is not None else 0
        )

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def report(self) -> Dict[str, Any]:
        """Return a snapshot of energy and performance deltas since last reset.

        Returns
        -------
        dict
            ``consumed_joules`` — net energy in joules.

            ``wall_clock_seconds`` — elapsed wall time in seconds.

            ``average_watts`` — ``consumed_joules / wall_clock_seconds``.

            ``inference_delta`` — number of inference calls since reset
            (only when an engine is attached).

            ``joules_per_inference`` — energy per inference call (only
            available when *inference_delta* > 0).
        """
        current_joules = self.sample()
        elapsed = time.perf_counter() - self._start_timer
        dj = current_joules - self._start_joules

        result: Dict[str, Any] = {
            "consumed_joules": dj,
            "wall_clock_seconds": elapsed,
            "average_watts": dj / elapsed if elapsed > 0 else 0.0,
        }

        if self._engine is not None:
            di = self._engine.inference_count - self._start_count
            result["inference_delta"] = di
            result["joules_per_inference"] = dj / di if di > 0 else 0.0

        return result

    def __enter__(self) -> EnergyMonitor:
        self.reset()
        return self

    def __exit__(self, *args: Any) -> None:
        pass
