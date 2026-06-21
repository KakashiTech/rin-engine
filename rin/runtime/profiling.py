"""Profiling and debugging utilities for RIN engine inference.

Provides :class:`Profiler` for systematic benchmarking across modes,
including detailed latency, throughput, and energy metrics.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from rin._backend import (
    RinException,
    MODE_MLP, MODE_SNN, MODE_ATTN, MODE_THOR, MODE_TRANSFORMER,
)
from rin.runtime.engine import RinEngine

__all__ = ["Profiler"]


class Profiler:
    """Systematic profiler for RIN engine inference engines.

    Parameters
    ----------
    engine : RinEngine
        An initialised engine with a model already loaded.
    """

    def __init__(self, engine: RinEngine) -> None:
        self._engine = engine

    # ------------------------------------------------------------------
    # Single-mode profiling
    # ------------------------------------------------------------------

    def profile_inference(
        self,
        input_data: Optional[List[int]] = None,
        iterations: int = 100,
        warmup: int = 10,
    ) -> Dict[str, Any]:
        """Profile a single inference mode and return detailed metrics.

        If *input_data* is ``None`` a dummy single-token input (``[0]``) is
        used.  The engine's mode must be set by the caller beforehand.

        Parameters
        ----------
        input_data : list of int, optional
            Input token IDs for inference.
        iterations : int
            Number of measured iterations (default 100).
        warmup : int
            Warm-up iterations before measurement (default 10).

        Returns
        -------
        dict
            ``latency_ms`` — mean latency per call in milliseconds.

            ``latency_std_ms`` — standard deviation of latency.

            ``tokens_per_second`` — mean throughput in tokens/s.

            ``total_tokens_generated`` — total tokens produced.

            ``energy_joules`` — total energy consumed over measured runs.

            ``joules_per_token`` — energy per generated token (when
            tokens > 0).
        """
        if input_data is None:
            input_data = [0]

        # Warm-up
        for _ in range(warmup):
            self._engine.infer(input_data, max_output=1)

        latencies: List[float] = []
        energy_before = self._engine.energy_joules
        count_before = self._engine.inference_count

        for _ in range(iterations):
            t0 = time.perf_counter()
            self._engine.infer(input_data, max_output=1)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)  # ms

        energy_after = self._engine.energy_joules
        count_after = self._engine.inference_count

        mean_lat = sum(latencies) / len(latencies)
        std_lat = (
            (sum((x - mean_lat) ** 2 for x in latencies) / len(latencies)) ** 0.5
            if len(latencies) > 1
            else 0.0
        )
        tokens_per_sec = 1000.0 / mean_lat if mean_lat > 0 else 0.0
        total_energy = energy_after - energy_before
        total_inferences = count_after - count_before

        return {
            "latency_ms": mean_lat,
            "latency_std_ms": std_lat,
            "tokens_per_second": tokens_per_sec,
            "total_tokens_generated": total_inferences,
            "energy_joules": total_energy,
            "joules_per_token": total_energy / total_inferences
            if total_inferences > 0
            else 0.0,
        }

    # ------------------------------------------------------------------
    # Cross-mode comparison
    # ------------------------------------------------------------------

    def compare_modes(
        self,
        modes: Optional[List[str]] = None,
        input_data: Optional[List[int]] = None,
        iterations: int = 100,
        warmup: int = 10,
    ) -> List[Dict[str, Any]]:
        """Benchmark every inference mode and return comparable results.

        Parameters
        ----------
        modes : list of str, optional
            Mode names to compare (default: all five modes).
        input_data : list of int, optional
            Input token IDs for each inference call.
        iterations : int
            Measured iterations per mode (default 100).
        warmup : int
            Warm-up iterations per mode (default 10).

        Returns
        -------
        list of dict
            One entry per mode, each containing the keys from
            :meth:`profile_inference` plus ``mode`` (str).

        Raises
        ------
        RinError
            If a mode string is invalid or any single-mode profile fails.
        """
        if modes is None:
            modes = ["mlp", "snn", "attn", "thor", "transformer"]

        results: List[Dict[str, Any]] = []
        for mode_name in modes:
            try:
                self._engine.mode = mode_name
                profile = self.profile_inference(
                    input_data=input_data,
                    iterations=iterations,
                    warmup=warmup,
                )
                profile["mode"] = mode_name
                results.append(profile)
            except RinException as exc:
                results.append(
                    {
                        "mode": mode_name,
                        "error": str(exc),
                    }
                )

        return results

    # ------------------------------------------------------------------
    # Convenience: full report
    # ------------------------------------------------------------------

    def full_report(
        self,
        modes: Optional[List[str]] = None,
        input_data: Optional[List[int]] = None,
        iterations: int = 100,
        warmup: int = 10,
    ) -> Dict[str, Any]:
        """Run a full multi-mode benchmark and return a structured report.

        Parameters
        ----------
        modes : list of str, optional
            Mode names to compare.
        input_data : list of int, optional
            Input token IDs.
        iterations : int
            Measured iterations per mode (default 100).
        warmup : int
            Warm-up iterations per mode (default 10).

        Returns
        -------
        dict
            ``engine_info`` — model metadata from :attr:`RinEngine.info`.

            ``version`` — C library version string.

            ``results`` — output of :meth:`compare_modes`.
        """
        return {
            "engine_info": self._engine.info,
            "version": self._engine.version(),
            "results": self.compare_modes(
                modes=modes,
                input_data=input_data,
                iterations=iterations,
                warmup=warmup,
            ),
        }
