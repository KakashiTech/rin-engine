"""High-level :class:`RinEngine` wrapper around the RIN C runtime.

Usage
-----
.. code:: python

    from rin.runtime import RinEngine

    with RinEngine("model.rin", mode="mlp") as engine:
        info = engine.info
        print(f"Model: {info['num_parameters']} parameters")

        output = engine.generate("Hello, world!", max_tokens=50)
        print(output)

        # Raw inference
        ids = engine.encode("Hello")
        result = engine.infer(ids, max_output=5)
        tokens = result["tokens"]
        text = engine.decode(tokens)
        print(text)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

from rin._backend import (
    RinContext as _RinContext,
    RinException,
    MODE_MLP,
    MODE_SNN,
    MODE_ATTN,
    MODE_THOR,
    MODE_TRANSFORMER,
    _backend_name,
)

_MODE_NAMES: Dict[int, str] = {
    MODE_MLP: "mlp",
    MODE_SNN: "snn",
    MODE_ATTN: "attn",
    MODE_THOR: "thor",
    MODE_TRANSFORMER: "transformer",
}
_MODE_VALUES: Dict[str, int] = {v: k for k, v in _MODE_NAMES.items()}

__all__ = ["RinEngine"]


class RinEngine:
    """High-level Python interface to a RIN runtime context.

    Uses either the native CPython extension (``_cengine``) or
    the ctypes fallback (``librin.so``) — selected automatically.

    Parameters
    ----------
    model_path : str, optional
        Path to a model file to load on construction.
    mode : str, optional
        Inference mode string (``"mlp"``, ``"snn"``, ``"attn"``,
        ``"thor"``, ``"transformer"``).  Defaults to ``"mlp"``.

    Raises
    ------
    RinException
        If the context cannot be created or the model cannot be loaded.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        mode: str = "mlp",
    ) -> None:
        self._ctx: _RinContext = _RinContext()
        self._closed: bool = False
        if mode:
            self.mode = mode
        if model_path:
            self.load_model(model_path)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        if not self._closed:
            self._ctx.close()
        self._closed = True

    def __enter__(self) -> RinEngine:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_model(self, path: str) -> None:
        self._ctx.load_model(path)

    # ------------------------------------------------------------------
    # Model info
    # ------------------------------------------------------------------

    @property
    def info(self) -> Dict[str, Union[int, float]]:
        """Return metadata about the currently loaded model as a dictionary.

        Keys: ``num_layers``, ``model_dim``, ``vocab_size``, ``num_heads``,
        ``max_seq_len``, ``ffn_dim``, ``num_parameters``, ``architecture``,
        ``size_mb``.
        """
        return self._ctx.get_model_info()

    # ------------------------------------------------------------------
    # Mode
    # ------------------------------------------------------------------

    @property
    def mode(self) -> str:
        return _MODE_NAMES.get(self._ctx.mode, "unknown")

    @mode.setter
    def mode(self, value: Union[str, int]) -> None:
        if isinstance(value, str):
            try:
                value = _MODE_VALUES[value]
            except KeyError:
                raise ValueError(
                    f"Unknown mode {value!r}.  Valid: {', '.join(_MODE_VALUES)}"
                )
        self._ctx.mode = value

    # ------------------------------------------------------------------
    # Sampling parameters
    # ------------------------------------------------------------------

    @property
    def temperature(self) -> float:
        return getattr(self, "_temperature", 1.0)

    @temperature.setter
    def temperature(self, value: float) -> None:
        self._ctx.set_temperature(value)
        self._temperature = value

    @property
    def top_k(self) -> int:
        return getattr(self, "_top_k", 0)

    @top_k.setter
    def top_k(self, value: int) -> None:
        self._ctx.set_top_k(value)
        self._top_k = value

    @property
    def top_p(self) -> float:
        return getattr(self, "_top_p", 1.0)

    @top_p.setter
    def top_p(self, value: float) -> None:
        self._ctx.set_top_p(value)
        self._top_p = value

    @property
    def power_budget(self) -> float:
        return getattr(self, "_power_budget", 0.0)

    @power_budget.setter
    def power_budget(self, value: float) -> None:
        self._ctx.set_power_budget(value)
        self._power_budget = value

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def infer(
        self,
        input_ids: List[int],
        max_output: int = 1,
    ) -> Dict[str, Any]:
        """Run a single forward pass and return generated token information.

        Parameters
        ----------
        input_ids : list of int
            Input token ID sequence.
        max_output : int
            Maximum number of tokens to generate in this call (default 1).

        Returns
        -------
        dict
            ``tokens`` (list of int), ``num_tokens``, ``energy_joules``,
            ``tokens_per_second``, ``latency_ns``.
        """
        return self._ctx.infer(input_ids, max_output)

    def generate(
        self,
        prompt: str,
        max_tokens: int = 20,
    ) -> str:
        """Generate text from a string prompt.

        Encodes the prompt, auto-regressively samples *max_tokens*
        additional tokens, and decodes the result back to text.

        Parameters
        ----------
        prompt : str
            Input text to condition on.
        max_tokens : int
            Number of new tokens to generate (default 20).

        Returns
        -------
        str
            Generated text (excluding the original prompt).
        """
        ids: List[int] = self.encode(prompt)
        generated: List[int] = []
        context = list(ids)

        for _ in range(max_tokens):
            result = self.infer(context, max_output=1)
            token = result["tokens"][0]
            generated.append(token)
            context.append(token)

            info = self.info
            if info["max_seq_len"] > 0 and len(context) > info["max_seq_len"]:
                context = context[1:]

        return self.decode(generated)

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------

    def encode(self, text: str) -> List[int]:
        return self._ctx.encode(text)

    def decode(self, ids: List[int]) -> str:
        return self._ctx.decode(ids)

    # ------------------------------------------------------------------
    # Energy & counters
    # ------------------------------------------------------------------

    @property
    def energy_joules(self) -> float:
        return self._ctx.energy_joules

    @property
    def energy_millijoules(self) -> float:
        return self._ctx.energy_millijoules

    @property
    def inference_count(self) -> int:
        return self._ctx.inference_count

    @property
    def total_tokens(self) -> int:
        return self._ctx.total_tokens

    # ------------------------------------------------------------------
    # Profiling / benchmarking
    # ------------------------------------------------------------------

    def benchmark(
        self,
        mode: str = "mlp",
        warmup: int = 10,
        iterations: int = 100,
    ) -> Dict[str, float]:
        """Run a micro-benchmark and return throughput metrics.

        Parameters
        ----------
        mode : str
            Inference mode to benchmark (default ``"mlp"``).
        warmup : int
            Warm-up iterations before measurement (default 10).
        iterations : int
            Measured iterations (default 100).

        Returns
        -------
        dict
            ``ms_per_token`` and ``tokens_per_second``.
        """
        mode_val = _MODE_VALUES.get(mode, 0)
        ms, tps = self._ctx.profile(mode_val, warmup, iterations)
        return {"ms_per_token": ms, "tokens_per_second": tps}

    # ------------------------------------------------------------------
    # Version info
    # ------------------------------------------------------------------

    @staticmethod
    def version() -> str:
        """Return the RIN C library version string."""
        try:
            from .. import _cengine as _ce
            return _ce.version()
        except Exception:
            pass
        try:
            from .._rin_native import rin_version
            return rin_version()
        except Exception:
            return "unavailable"

    @staticmethod
    def version_numbers() -> Tuple[int, int, int]:
        """Return the RIN C library version as ``(major, minor, patch)``."""
        try:
            from .. import _cengine as _ce
            return _ce.version_numbers()
        except Exception:
            pass
        try:
            from .._rin_native import rin_version_numbers
            return rin_version_numbers()
        except Exception:
            return (0, 0, 0)

    @staticmethod
    def backend_name() -> str:
        """Return which runtime backend is active (``"native"`` or ``"ctypes"``)."""
        return _backend_name()
