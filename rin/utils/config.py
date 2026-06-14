from __future__ import annotations

import os
from typing import Any, Dict, Optional

__all__ = [
    "THOR_DEFAULT_CONFIG",
    "get_config",
    "merge_config",
    "config_from_env",
]

THOR_DEFAULT_CONFIG: Dict[str, Any] = {
    "mode": "mlp",
    "temperature": 0.8,
    "top_k": 40,
    "top_p": 0.9,
    "power_budget": 0.0,
}

_ENV_KEYS: Dict[str, str] = {
    "THOR_MODE": "mode",
    "THOR_TEMPERATURE": "temperature",
    "THOR_TOP_K": "top_k",
    "THOR_TOP_P": "top_p",
    "THOR_POWER_BUDGET": "power_budget",
}


def get_config(key: str, default: Any = None) -> Any:
    return THOR_DEFAULT_CONFIG.get(key, default)


def merge_config(overrides: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(THOR_DEFAULT_CONFIG)
    cfg.update(overrides)
    return cfg


def config_from_env() -> Dict[str, Any]:
    cfg = dict(THOR_DEFAULT_CONFIG)
    for env_key, cfg_key in _ENV_KEYS.items():
        val = os.environ.get(env_key)
        if val is None:
            continue
        try:
            if cfg_key in ("top_k",):
                cfg[cfg_key] = int(val)
            elif cfg_key in ("temperature", "top_p", "power_budget"):
                cfg[cfg_key] = float(val)
            else:
                cfg[cfg_key] = val
        except (ValueError, TypeError):
            pass
    return cfg
