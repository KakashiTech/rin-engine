from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from rin._backend import RinException
from rin._backend import MODE_MLP, MODE_SNN, MODE_ATTN, MODE_THOR, MODE_TRANSFORMER
from rin.runtime.energy import EnergyMonitor
from rin.runtime.engine import RinEngine

__all__ = ["PowerTuner"]


_MODE_ENERGY_RANKING: List[int] = [
    MODE_MLP,
    MODE_SNN,
    MODE_ATTN,
    MODE_THOR,
    MODE_TRANSFORMER,
]

_MODE_QUALITY_RANKING: List[int] = [
    MODE_TRANSFORMER,
    MODE_THOR,
    MODE_ATTN,
    MODE_SNN,
    MODE_MLP,
]

_MODE_NAMES: Dict[int, str] = {
    MODE_MLP: "mlp",
    MODE_SNN: "snn",
    MODE_ATTN: "attn",
    MODE_THOR: "thor",
    MODE_TRANSFORMER: "transformer",
}


class PowerTuner:
    """
    Automatically tunes inference parameters to meet a power budget.

    Strategy:
    1. Profile each mode (MLP, SNN, ATTN, RIN, Transformer)
    2. Measure energy per token for each mode
    3. Adjust temperature, top-k, top-p to balance quality vs efficiency
    4. Recommend optimal mode + parameters for the power budget

    Usage:
        tuner = PowerTuner(engine)
        config = tuner.optimize(power_budget_watts=5.0)
        engine.set_mode(config['mode'])
        engine.set_temperature(config['temperature'])
    """

    def __init__(self, engine: RinEngine) -> None:
        self._engine = engine
        self._profile_cache: Dict[str, Dict[str, float]] = {}

    def profile_modes(self) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        original_mode = self._engine.mode
        try:
            for mode_val in _MODE_ENERGY_RANKING:
                mode_name = _MODE_NAMES[mode_val]
                try:
                    self._engine.mode = mode_name
                    ept = self.energy_per_token(mode=mode_name, iterations=50)
                    self._profile_cache[mode_name] = {"energy_per_token": ept}
                    results.append({"mode": mode_name, "energy_per_token_joules": ept})
                except RinException as exc:
                    results.append({"mode": mode_name, "error": str(exc)})
        finally:
            try:
                self._engine.mode = original_mode
            except RinException:
                pass
        return results

    def energy_per_token(self, mode: str = "mlp", iterations: int = 50) -> float:
        original_mode = self._engine.mode
        try:
            self._engine.mode = mode
            mon = EnergyMonitor(self._engine)
            for _ in range(iterations):
                self._engine.infer([0], max_output=1)
            report = mon.report()
            total_tokens = report.get("inference_delta", iterations)
            return report["consumed_joules"] / total_tokens if total_tokens > 0 else 0.0
        finally:
            try:
                self._engine.mode = original_mode
            except RinException:
                pass

    def recommend_mode(self, power_budget_watts: float) -> str:
        if not self._profile_cache:
            self.profile_modes()
        for mode_val in _MODE_QUALITY_RANKING:
            mode_name = _MODE_NAMES[mode_val]
            cache_entry = self._profile_cache.get(mode_name)
            if cache_entry is None:
                continue
            ept = cache_entry.get("energy_per_token_joules", float("inf"))
            if ept <= 0.0:
                continue
            max_tokens_per_sec = power_budget_watts / ept
            if max_tokens_per_sec >= 1.0:
                return mode_name
        return "mlp"

    def optimize(
        self, power_budget_watts: float, preference: str = "balanced"
    ) -> Dict[str, Any]:
        if not self._profile_cache:
            self.profile_modes()
        mode = self.recommend_mode(power_budget_watts)
        config: Dict[str, Any] = {"mode": mode, "temperature": 0.8, "top_k": 40, "top_p": 0.9}
        if preference == "efficiency":
            config["temperature"] = 0.6
            config["top_k"] = 20
            config["top_p"] = 0.8
        elif preference == "quality":
            config["temperature"] = 1.0
            config["top_k"] = 60
            config["top_p"] = 0.95
        return config

    @staticmethod
    def estimate_battery_life(
        energy_per_token_joules: float, battery_capacity_wh: float
    ) -> Dict[str, float]:
        battery_joules = battery_capacity_wh * 3600.0
        if energy_per_token_joules <= 0.0:
            return {"tokens": 0, "hours": 0.0, "days": 0.0}
        total_tokens = battery_joules / energy_per_token_joules
        hours = total_tokens / (3600.0 * 10.0)
        return {
            "tokens": int(total_tokens),
            "hours": hours,
            "days": hours / 24.0,
        }
