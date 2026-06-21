"""Unified backend — tries native CPython extension first, falls back to ctypes.

Module-level constant ``HAS_NATIVE`` indicates whether the C extension was
successfully loaded.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

HAS_NATIVE: bool = False
"""``True`` when the CPython C extension (``_cengine``) is available."""

# We fill these after deciding which backend to use
RinContext: Any = None
RinException: Any = None

# Mode constants — always present regardless of backend
MODE_MLP: int = 0
MODE_SNN: int = 1
MODE_ATTN: int = 2
MODE_THOR: int = 3
MODE_TRANSFORMER: int = 4

STATUS_OK: int = 0
STATUS_ERR_INIT: int = -1
STATUS_ERR_MEMORY: int = -2
STATUS_ERR_WEIGHTS: int = -3
STATUS_ERR_INFERENCE: int = -4
STATUS_ERR_NOT_INITIALIZED: int = -5
STATUS_ERR_INVALID_INPUT: int = -6
STATUS_ERR_UNSUPPORTED: int = -7


def _backend_name() -> str:
    if HAS_NATIVE:
        return "native (CPython C extension)"
    return "ctypes (librin.so)"


# ------------------------------------------------------------------ #
# Native backend                                                     #
# ------------------------------------------------------------------ #

def _try_native() -> bool:
    global HAS_NATIVE, RinContext, RinException
    global MODE_MLP, MODE_SNN, MODE_ATTN, MODE_THOR, MODE_TRANSFORMER
    global STATUS_OK, STATUS_ERR_INIT, STATUS_ERR_MEMORY, STATUS_ERR_WEIGHTS
    global STATUS_ERR_INFERENCE, STATUS_ERR_NOT_INITIALIZED
    global STATUS_ERR_INVALID_INPUT, STATUS_ERR_UNSUPPORTED

    try:
        from . import _cengine as _ce
    except ImportError as e:
        logger.debug("Native C extension not available: %s", e)
        return False

    HAS_NATIVE = True
    RinContext = _ce.RinContext
    RinException = _ce.RinException

    MODE_MLP = _ce.MODE_MLP
    MODE_SNN = _ce.MODE_SNN
    MODE_ATTN = _ce.MODE_ATTN
    MODE_THOR = _ce.MODE_THOR
    MODE_TRANSFORMER = _ce.MODE_TRANSFORMER

    STATUS_OK = _ce.STATUS_OK
    STATUS_ERR_INIT = _ce.STATUS_ERR_INIT
    STATUS_ERR_MEMORY = _ce.STATUS_ERR_MEMORY
    STATUS_ERR_WEIGHTS = _ce.STATUS_ERR_WEIGHTS
    STATUS_ERR_INFERENCE = _ce.STATUS_ERR_INFERENCE
    STATUS_ERR_NOT_INITIALIZED = _ce.STATUS_ERR_NOT_INITED
    STATUS_ERR_INVALID_INPUT = _ce.STATUS_ERR_INVALID
    STATUS_ERR_UNSUPPORTED = _ce.STATUS_ERR_UNSUPPORTED

    return True


# ------------------------------------------------------------------ #
# ctypes fallback backend                                            #
# ------------------------------------------------------------------ #

def _try_ctypes() -> bool:
    global HAS_NATIVE, RinContext, RinException
    global MODE_MLP, MODE_SNN, MODE_ATTN, MODE_THOR, MODE_TRANSFORMER
    global STATUS_OK, STATUS_ERR_INIT, STATUS_ERR_MEMORY, STATUS_ERR_WEIGHTS
    global STATUS_ERR_INFERENCE, STATUS_ERR_NOT_INITIALIZED
    global STATUS_ERR_INVALID_INPUT, STATUS_ERR_UNSUPPORTED

    try:
        from ._rin_native import (
            RinMode as _RM,
            RinStatus as _RS,
            RinError as RinError,
            rin_create,
            rin_destroy,
            rin_load_model,
            rin_get_model_info,
            rin_set_mode,
            rin_get_mode,
            rin_set_temperature,
            rin_set_top_k,
            rin_set_top_p,
            rin_set_power_budget,
            rin_infer,
            rin_free_result,
            rin_encode,
            rin_decode,
            rin_get_charset,
            rin_get_energy_joules,
            rin_get_energy_millijoules,
            rin_get_inference_count,
            rin_get_total_tokens,
            rin_profile,
            rin_version,
            rin_version_numbers,
        )
    except ImportError as e:
        logger.debug("ctypes backend not available: %s", e)
        return False

    RinException = RinError

    # We create a compatibility wrapper so engine.py can use the
    # same interface as the native backend.

    class _CtypesContext:
        """Compatibility wrapper: exposes the same API as native RinContext."""

        def __init__(self) -> None:
            self._ctx: int = rin_create()
            self._closed: bool = False

        def load_model(self, path: str) -> None:
            rin_load_model(self._ctx, path)

        def get_model_info(self) -> Dict[str, Union[int, float]]:
            info = rin_get_model_info(self._ctx)
            return {
                "num_layers": info.num_layers,
                "model_dim": info.model_dim,
                "vocab_size": info.vocab_size,
                "num_heads": info.num_heads,
                "max_seq_len": info.max_seq_len,
                "ffn_dim": info.ffn_dim,
                "num_parameters": info.num_parameters,
                "architecture": info.architecture,
                "size_mb": info.size_mb,
            }

        def infer(
            self, input_ids: List[int], max_output: int = 1
        ) -> Dict[str, Any]:
            result = rin_infer(self._ctx, input_ids, len(input_ids), max_output)
            try:
                tokens = (
                    [result.tokens[i] for i in range(result.num_tokens)]
                    if result.tokens and result.num_tokens
                    else []
                )
                return {
                    "tokens": tokens,
                    "num_tokens": result.num_tokens,
                    "energy_joules": result.energy_joules,
                    "tokens_per_second": result.tokens_per_second,
                    "latency_ns": result.latency_ns,
                }
            finally:
                rin_free_result(result)

        def encode(self, text: str) -> List[int]:
            return rin_encode(self._ctx, text)

        def decode(self, ids: List[int]) -> str:
            return rin_decode(self._ctx, ids)

        def get_charset(self) -> str:
            s, _ = rin_get_charset(self._ctx)
            return s

        def set_temperature(self, v: float) -> None:
            rin_set_temperature(self._ctx, v)

        def set_top_k(self, v: int) -> None:
            rin_set_top_k(self._ctx, v)

        def set_top_p(self, v: float) -> None:
            rin_set_top_p(self._ctx, v)

        def set_power_budget(self, v: float) -> None:
            rin_set_power_budget(self._ctx, v)

        @property
        def mode(self) -> int:
            return rin_get_mode(self._ctx)

        @mode.setter
        def mode(self, value: int) -> None:
            rin_set_mode(self._ctx, value)

        @property
        def energy_joules(self) -> float:
            return rin_get_energy_joules(self._ctx)

        @property
        def energy_millijoules(self) -> float:
            return rin_get_energy_millijoules(self._ctx)

        @property
        def inference_count(self) -> int:
            return rin_get_inference_count(self._ctx)

        @property
        def total_tokens(self) -> int:
            return rin_get_total_tokens(self._ctx)

        def profile(
            self, mode: int, warmup: int = 10, iterations: int = 100
        ) -> Tuple[float, float]:
            return rin_profile(self._ctx, mode, warmup, iterations)

        def close(self) -> None:
            if not self._closed and self._ctx:
                rin_destroy(self._ctx)
            self._closed = True
            self._ctx = 0

        def __del__(self) -> None:
            self.close()

    RinContext = _CtypesContext

    MODE_MLP = _RM.MLP
    MODE_SNN = _RM.SNN
    MODE_ATTN = _RM.ATTN
    MODE_THOR = _RM.THOR
    MODE_TRANSFORMER = _RM.TRANSFORMER

    return True


# ------------------------------------------------------------------ #
# Bootstrap                                                         #
# ------------------------------------------------------------------ #

if not _try_native():
    _try_ctypes()

if RinContext is None:
    raise ImportError(
        "No RIN runtime backend available.\n"
        "Install the package or build librin.so with:  make librin.so"
    )
