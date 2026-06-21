"""
Low-level ctypes wrapper around librin.so — the RIN C runtime library.

Provides complete ctype definitions for all exported C structs, enums, and
functions, together with automatic library discovery and thread-safe dispatch.

Environment variables
---------------------
RIN_LIB : str
    Explicit path to ``librin.so``. When set, no auto-detection is performed.

Library search order
--------------------
1. ``RIN_LIB`` environment variable.
2. ``<package_root>/../librin.so`` (installed next to the ``rin`` package).
3. Default system linker paths (``ctypes.CDLL("librin.so")``).
"""

from __future__ import annotations

import ctypes
import os
import threading
from pathlib import Path
from typing import ClassVar, List, Optional, Tuple

__all__ = [
    "RinError",
    "RinInitError",
    "RinMemoryError",
    "RinWeightsError",
    "RinInferenceError",
    "RinNotInitializedError",
    "RinInvalidInputError",
    "RinUnsupportedError",
    "RinStatus",
    "RinMode",
    "RinModelInfo",
    "RinResult",
    "RIN_OK",
    "RIN_ERR_INIT",
    "RIN_ERR_MEMORY",
    "RIN_ERR_WEIGHTS",
    "RIN_ERR_INFERENCE",
    "RIN_ERR_NOT_INITIALIZED",
    "RIN_ERR_INVALID_INPUT",
    "RIN_ERR_UNSUPPORTED",
    "RIN_MODE_MLP",
    "RIN_MODE_SNN",
    "RIN_MODE_ATTN",
    "RIN_MODE_THOR",
    "RIN_MODE_TRANSFORMER",
    "load_library",
    "rin_create",
    "rin_destroy",
    "rin_load_model",
    "rin_get_model_info",
    "rin_set_mode",
    "rin_get_mode",
    "rin_set_temperature",
    "rin_set_top_k",
    "rin_set_top_p",
    "rin_set_power_budget",
    "rin_infer",
    "rin_free_result",
    "rin_get_charset",
    "rin_encode",
    "rin_decode",
    "rin_get_energy_joules",
    "rin_get_energy_millijoules",
    "rin_get_inference_count",
    "rin_get_total_tokens",
    "rin_profile",
    "rin_version",
    "rin_version_numbers",
    "native_lock",
]

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class RinError(Exception):
    """Base exception for all RIN runtime errors."""


class RinInitError(RinError):
    """Initialisation failure (``RIN_ERR_INIT``)."""


class RinMemoryError(RinError):
    """Memory allocation failure (``RIN_ERR_MEMORY``)."""


class RinWeightsError(RinError):
    """Model weight loading failure (``RIN_ERR_WEIGHTS``)."""


class RinInferenceError(RinError):
    """Inference execution failure (``RIN_ERR_INFERENCE``)."""


class RinNotInitializedError(RinError):
    """Operation attempted before the context was initialised (``RIN_ERR_NOT_INITIALIZED``)."""


class RinInvalidInputError(RinError):
    """Invalid input argument (``RIN_ERR_INVALID_INPUT``)."""


class RinUnsupportedError(RinError):
    """Unsupported feature or configuration (``RIN_ERR_UNSUPPORTED``)."""


_STATUS_ERROR_MAP: dict[int, type[RinError]] = {
    -1: RinInitError,
    -2: RinMemoryError,
    -3: RinWeightsError,
    -4: RinInferenceError,
    -5: RinNotInitializedError,
    -6: RinInvalidInputError,
    -7: RinUnsupportedError,
}

_STATUS_MESSAGE_MAP: dict[int, str] = {
    0: "success",
    -1: "initialisation failed",
    -2: "memory allocation failed",
    -3: "model weight loading failed",
    -4: "inference execution failed",
    -5: "context not initialised",
    -6: "invalid input",
    -7: "unsupported operation",
}


# ---------------------------------------------------------------------------
# Status / Mode enum-like constants
# ---------------------------------------------------------------------------

RIN_OK: int = 0
RIN_ERR_INIT: int = -1
RIN_ERR_MEMORY: int = -2
RIN_ERR_WEIGHTS: int = -3
RIN_ERR_INFERENCE: int = -4
RIN_ERR_NOT_INITIALIZED: int = -5
RIN_ERR_INVALID_INPUT: int = -6
RIN_ERR_UNSUPPORTED: int = -7

RIN_MODE_MLP: int = 0
RIN_MODE_SNN: int = 1
RIN_MODE_ATTN: int = 2
RIN_MODE_THOR: int = 3
RIN_MODE_TRANSFORMER: int = 4


class RinStatus:
    """Namespace wrapping C status codes with human-readable names."""

    OK: ClassVar[int] = RIN_OK
    ERR_INIT: ClassVar[int] = RIN_ERR_INIT
    ERR_MEMORY: ClassVar[int] = RIN_ERR_MEMORY
    ERR_WEIGHTS: ClassVar[int] = RIN_ERR_WEIGHTS
    ERR_INFERENCE: ClassVar[int] = RIN_ERR_INFERENCE
    ERR_NOT_INITIALIZED: ClassVar[int] = RIN_ERR_NOT_INITIALIZED
    ERR_INVALID_INPUT: ClassVar[int] = RIN_ERR_INVALID_INPUT
    ERR_UNSUPPORTED: ClassVar[int] = RIN_ERR_UNSUPPORTED

    @staticmethod
    def is_error(status: int) -> bool:
        """Return ``True`` when *status* indicates an error."""
        return status < 0

    @staticmethod
    def message(status: int) -> str:
        """Return a short human-readable message for *status*."""
        return _STATUS_MESSAGE_MAP.get(status, f"unknown status code {status}")


class RinMode:
    """Namespace wrapping inference-mode constants."""

    MLP: ClassVar[int] = RIN_MODE_MLP
    SNN: ClassVar[int] = RIN_MODE_SNN
    ATTN: ClassVar[int] = RIN_MODE_ATTN
    THOR: ClassVar[int] = RIN_MODE_THOR
    TRANSFORMER: ClassVar[int] = RIN_MODE_TRANSFORMER

    _NAME_MAP: ClassVar[dict[int, str]] = {
        0: "mlp",
        1: "snn",
        2: "attn",
        3: "thor",
        4: "transformer",
    }

    _VALUE_MAP: ClassVar[dict[str, int]] = {v: k for k, v in _NAME_MAP.items()}

    @staticmethod
    def name(mode: int) -> str:
        """Return the canonical string name for a mode integer."""
        return RinMode._NAME_MAP.get(mode, f"unknown({mode})")

    @staticmethod
    def from_string(name: str) -> int:
        """Parse a mode string (e.g. ``"mlp"``, ``"snn"``) to its integer value.

        Raises ``ValueError`` on unknown names.
        """
        name = name.strip().lower()
        try:
            return RinMode._VALUE_MAP[name]
        except KeyError:
            valid = ", ".join(sorted(RinMode._VALUE_MAP))
            raise ValueError(
                f"Unknown mode {name!r}. Valid modes: {valid}"
            ) from None


# ---------------------------------------------------------------------------
# C-compatible struct definitions
# ---------------------------------------------------------------------------

class RinModelInfo(ctypes.Structure):
    """Mirrors the C ``RinModelInfo`` struct."""

    _fields_ = [
        ("num_layers", ctypes.c_uint32),
        ("model_dim", ctypes.c_uint32),
        ("vocab_size", ctypes.c_uint32),
        ("num_heads", ctypes.c_uint32),
        ("max_seq_len", ctypes.c_uint32),
        ("ffn_dim", ctypes.c_uint32),
        ("num_parameters", ctypes.c_uint32),
        ("architecture", ctypes.c_uint32),
        ("size_mb", ctypes.c_float),
    ]

    def as_dict(self) -> dict[str, int | float]:
        """Export struct fields as a plain dictionary."""
        return {
            "num_layers": self.num_layers,
            "model_dim": self.model_dim,
            "vocab_size": self.vocab_size,
            "num_heads": self.num_heads,
            "max_seq_len": self.max_seq_len,
            "ffn_dim": self.ffn_dim,
            "num_parameters": self.num_parameters,
            "architecture": self.architecture,
            "size_mb": self.size_mb,
        }


class RinResult(ctypes.Structure):
    """Mirrors the C ``RinResult`` struct."""

    _fields_ = [
        ("tokens", ctypes.POINTER(ctypes.c_uint32)),
        ("num_tokens", ctypes.c_uint32),
        ("energy_joules", ctypes.c_double),
        ("tokens_per_second", ctypes.c_float),
        ("latency_ns", ctypes.c_uint64),
    ]

    def as_dict(self) -> dict[str, any]:
        """Copy token data into a plain dictionary and return it.

        The caller *must still* call :func:`rin_free_result` after reading
        the dict.
        """
        token_list: List[int] = (
            [self.tokens[i] for i in range(self.num_tokens)]
            if self.tokens and self.num_tokens
            else []
        )
        return {
            "tokens": token_list,
            "num_tokens": self.num_tokens,
            "energy_joules": self.energy_joules,
            "tokens_per_second": self.tokens_per_second,
            "latency_ns": self.latency_ns,
        }


# ---------------------------------------------------------------------------
# Library loading
# ---------------------------------------------------------------------------

_lib: Optional[ctypes.CDLL] = None
native_lock = threading.RLock()
"""Re-entrant lock protecting every call through the C FFI boundary."""


def _resolve_library_path() -> str:
    """Return the filesystem path of ``librin.so``.

    Resolution order
    -----------------
     1. ``RIN_LIB`` environment variable.
     2. ``<package_root>/../librin.so`` (next to the ``rin`` package).
     3. ``"librin.so"`` (system library path).
    """
    env_path = os.environ.get("RIN_LIB")
    if env_path:
        resolved = os.path.abspath(env_path)
        if os.path.isfile(resolved):
            return resolved
        raise RinError(f"RIN_LIB={env_path} does not point to a file")

    # Relative to this module's location (rin._rin_native.py -> ../librin.so)
    module_dir = Path(__file__).resolve().parent
    sibling = module_dir.parent / "librin.so"
    if sibling.is_file():
        return str(sibling)

    return "librin.so"


def load_library(path: Optional[str] = None) -> ctypes.CDLL:
    """Load the RIN C shared library.

    Parameters
    ----------
    path : str, optional
        Explicit filesystem path.  When ``None`` (default) the library is
        auto-detected via :func:`_resolve_library_path`.

    Returns
    -------
    ctypes.CDLL
        The loaded library handle, cached globally.

    Raises
    ------
    RinError
        If the library cannot be found or loaded.
    """
    global _lib
    if _lib is not None and path is None:
        return _lib

    lib_path = path if path else _resolve_library_path()
    try:
        _lib = ctypes.CDLL(lib_path)
    except OSError as e:
        raise RinError(
            f"Failed to load RIN shared library from {lib_path!r}: {e}"
        ) from e
    return _lib


# ---------------------------------------------------------------------------
# Function signature declarations
# ---------------------------------------------------------------------------

def _init_fn_signatures(lib: ctypes.CDLL) -> None:
    """Set ``argtypes`` / ``restype`` on every exported C function."""
    # The opaque context handle
    lib.rin_create.restype = ctypes.c_void_p
    lib.rin_create.argtypes = []

    lib.rin_destroy.restype = None
    lib.rin_destroy.argtypes = [ctypes.c_void_p]

    # Model loading
    lib.rin_load_model.restype = ctypes.c_int
    lib.rin_load_model.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

    lib.rin_get_model_info.restype = ctypes.c_int
    lib.rin_get_model_info.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(RinModelInfo),
    ]

    # Configuration
    lib.rin_set_mode.restype = None
    lib.rin_set_mode.argtypes = [ctypes.c_void_p, ctypes.c_int]

    lib.rin_get_mode.restype = ctypes.c_int
    lib.rin_get_mode.argtypes = [ctypes.c_void_p]

    lib.rin_set_temperature.restype = None
    lib.rin_set_temperature.argtypes = [ctypes.c_void_p, ctypes.c_float]

    lib.rin_set_top_k.restype = None
    lib.rin_set_top_k.argtypes = [ctypes.c_void_p, ctypes.c_uint32]

    lib.rin_set_top_p.restype = None
    lib.rin_set_top_p.argtypes = [ctypes.c_void_p, ctypes.c_float]

    lib.rin_set_power_budget.restype = None
    lib.rin_set_power_budget.argtypes = [ctypes.c_void_p, ctypes.c_float]

    # Inference
    lib.rin_infer.restype = ctypes.c_int
    lib.rin_infer.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.POINTER(RinResult),
    ]

    lib.rin_free_result.restype = None
    lib.rin_free_result.argtypes = [ctypes.POINTER(RinResult)]

    # Tokenizer
    lib.rin_get_charset.restype = ctypes.c_char_p
    lib.rin_get_charset.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]

    lib.rin_encode.restype = ctypes.c_int
    lib.rin_encode.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_int,
    ]

    lib.rin_decode.restype = None
    lib.rin_decode.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
    ]

    # Energy
    lib.rin_get_energy_joules.restype = ctypes.c_double
    lib.rin_get_energy_joules.argtypes = [ctypes.c_void_p]

    lib.rin_get_energy_millijoules.restype = ctypes.c_double
    lib.rin_get_energy_millijoules.argtypes = [ctypes.c_void_p]

    lib.rin_get_inference_count.restype = ctypes.c_uint64
    lib.rin_get_inference_count.argtypes = [ctypes.c_void_p]

    lib.rin_get_total_tokens.restype = ctypes.c_uint64
    lib.rin_get_total_tokens.argtypes = [ctypes.c_void_p]

    # Profiling
    lib.rin_profile.restype = ctypes.c_int
    lib.rin_profile.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]

    # Version
    lib.rin_version.restype = ctypes.c_char_p
    lib.rin_version.argtypes = []

    lib.rin_version_numbers.restype = None
    lib.rin_version_numbers.argtypes = [
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
    ]


# ---------------------------------------------------------------------------
# Public wrapper functions  (thread-safe)
# ---------------------------------------------------------------------------

def _check(status: int) -> int:
    """Raise the appropriate :class:`RinError` if *status* is an error code."""
    if status == 0:
        return status
    exc_cls = _STATUS_ERROR_MAP.get(status, RinError)
    msg = _STATUS_MESSAGE_MAP.get(status, f"unknown error {status}")
    raise exc_cls(msg)


def rin_create() -> int:
    """Create a new RIN runtime context.

    Returns
    -------
    int
        Opaque pointer (handle) to the ``RinContext``.

    Raises
    ------
    RinError
        On failure.
    """
    with native_lock:
        lib = load_library()
        ptr = lib.rin_create()
        if not ptr:
            raise RinError("rin_create returned NULL")
        return ptr


def rin_destroy(ctx: int) -> None:
    """Destroy a RIN runtime context previously created with :func:`rin_create`.

    Parameters
    ----------
    ctx : int
        Opaque handle returned by :func:`rin_create`.
    """
    with native_lock:
        lib = load_library()
        lib.rin_destroy(ctx)


def rin_load_model(ctx: int, model_path: str) -> None:
    """Load a model file into the runtime context.

    Parameters
    ----------
    ctx : int
        Runtime context handle.
    model_path : str
        Filesystem path to the model file.

    Raises
    ------
    RinWeightsError
        If the model file cannot be loaded.
    """
    with native_lock:
        lib = load_library()
        _check(lib.rin_load_model(ctx, model_path.encode("utf-8")))


def rin_get_model_info(ctx: int) -> RinModelInfo:
    """Retrieve the currently loaded model's metadata.

    Parameters
    ----------
    ctx : int
        Runtime context handle.

    Returns
    -------
    RinModelInfo
        Populated struct mirroring the C struct.

    Raises
    ------
    RinNotInitializedError
        If no model is loaded.
    """
    with native_lock:
        lib = load_library()
        info = RinModelInfo()
        _check(lib.rin_get_model_info(ctx, ctypes.byref(info)))
        return info


def rin_set_mode(ctx: int, mode: int) -> None:
    """Set the inference execution mode.

    Parameters
    ----------
    ctx : int
        Runtime context handle.
    mode : int
        One of ``RIN_MODE_MLP``, ``RIN_MODE_SNN``, etc.
    """
    with native_lock:
        lib = load_library()
        lib.rin_set_mode(ctx, mode)


def rin_get_mode(ctx: int) -> int:
    """Return the currently active inference mode.

    Parameters
    ----------
    ctx : int
        Runtime context handle.

    Returns
    -------
    int
        One of the ``RIN_MODE_*`` constants.
    """
    with native_lock:
        lib = load_library()
        return lib.rin_get_mode(ctx)


def rin_set_temperature(ctx: int, temp: float) -> None:
    """Set the sampling temperature.

    Parameters
    ----------
    ctx : int
        Runtime context handle.
    temp : float
        Temperature value (commonly 0.0 – 2.0).  Lower = more deterministic.
    """
    with native_lock:
        lib = load_library()
        lib.rin_set_temperature(ctx, temp)


def rin_set_top_k(ctx: int, k: int) -> None:
    """Set Top-K sampling parameter.

    Parameters
    ----------
    ctx : int
        Runtime context handle.
    k : int
        Number of highest-probability tokens to consider.
    """
    with native_lock:
        lib = load_library()
        lib.rin_set_top_k(ctx, k)


def rin_set_top_p(ctx: int, p: float) -> None:
    """Set Top-P (nucleus) sampling parameter.

    Parameters
    ----------
    ctx : int
        Runtime context handle.
    p : float
        Cumulative probability threshold (0.0 – 1.0).
    """
    with native_lock:
        lib = load_library()
        lib.rin_set_top_p(ctx, p)


def rin_set_power_budget(ctx: int, watts: float) -> None:
    """Set the power budget for energy-aware scheduling.

    Parameters
    ----------
    ctx : int
        Runtime context handle.
    watts : float
        Power budget in watts.
    """
    with native_lock:
        lib = load_library()
        lib.rin_set_power_budget(ctx, watts)


def rin_infer(
    ctx: int,
    input_ids: List[int],
    num_input: int,
    max_output: int,
) -> RinResult:
    """Run inference on the given input token IDs.

    Parameters
    ----------
    ctx : int
        Runtime context handle.
    input_ids : list of int
        Input token ID sequence.
    num_input : int
        Length of ``input_ids``.
    max_output : int
        Maximum number of tokens to generate.

    Returns
    -------
    RinResult
        Result struct containing generated tokens and performance data.

    Raises
    ------
    RinInferenceError
        If inference fails.
    """
    with native_lock:
        lib = load_library()
        arr = (ctypes.c_uint32 * num_input)(*input_ids)
        result = RinResult()
        _check(lib.rin_infer(ctx, arr, num_input, max_output, ctypes.byref(result)))
        return result


def rin_free_result(result: RinResult) -> None:
    """Free the dynamically allocated token buffer inside a result struct.

    Parameters
    ----------
    result : RinResult
        Struct whose ``tokens`` pointer will be freed.
    """
    with native_lock:
        lib = load_library()
        lib.rin_free_result(ctypes.byref(result))


def rin_get_charset(ctx: int) -> Tuple[str, int]:
    """Return the character set (vocabulary) used by the tokenizer.

    Parameters
    ----------
    ctx : int
        Runtime context handle.

    Returns
    -------
    (charset_string, vocab_size)
        Tuple of charset string and its length.
    """
    with native_lock:
        lib = load_library()
        vocab_size = ctypes.c_int()
        charset = lib.rin_get_charset(ctx, ctypes.byref(vocab_size))
        if not charset:
            raise RinError("rin_get_charset returned NULL")
        return charset.decode("utf-8"), vocab_size.value


def rin_encode(ctx: int, text: str, max_ids: int = 0) -> List[int]:
    """Encode a text string into a sequence of token IDs.

    Parameters
    ----------
    ctx : int
        Runtime context handle.
    text : str
        Input text to tokenise.
    max_ids : int
        Capacity of the output buffer.  If ``<= 0`` a heuristic
        (``len(text) * 4 + 256``) is used.

    Returns
    -------
    list of int
        Encoded token ID sequence.
    """
    with native_lock:
        lib = load_library()
        if max_ids <= 0:
            max_ids = len(text) * 4 + 256
        ids = (ctypes.c_uint32 * max_ids)()
        n = lib.rin_encode(ctx, text.encode("utf-8"), ids, max_ids)
        if n < 0:
            raise RinEncodeError(
                f"rin_encode failed with return code {n}"
            )
        return list(ids[:n])


class RinEncodeError(RinError):
    """Raised when :func:`rin_encode` fails."""


def rin_decode(ctx: int, ids: List[int]) -> str:
    """Decode a sequence of token IDs back into a text string.

    Parameters
    ----------
    ctx : int
        Runtime context handle.
    ids : list of int
        Token ID sequence to decode.

    Returns
    -------
    str
        Decoded text.
    """
    with native_lock:
        lib = load_library()
        arr = (ctypes.c_uint32 * len(ids))(*ids)
        max_text = len(ids) * 16 + 256
        buf = ctypes.create_string_buffer(max_text)
        lib.rin_decode(ctx, arr, len(ids), buf, max_text)
        return buf.value.decode("utf-8")


def rin_get_energy_joules(ctx: int) -> float:
    """Return accumulated energy consumption in joules.

    Parameters
    ----------
    ctx : int
        Runtime context handle.

    Returns
    -------
    float
        Energy in joules.
    """
    with native_lock:
        lib = load_library()
        return lib.rin_get_energy_joules(ctx)


def rin_get_energy_millijoules(ctx: int) -> float:
    """Return accumulated energy consumption in millijoules.

    Parameters
    ----------
    ctx : int
        Runtime context handle.

    Returns
    -------
    float
        Energy in millijoules.
    """
    with native_lock:
        lib = load_library()
        return lib.rin_get_energy_millijoules(ctx)


def rin_get_inference_count(ctx: int) -> int:
    """Return the total number of inference calls made so far.

    Parameters
    ----------
    ctx : int
        Runtime context handle.

    Returns
    -------
    int
        Inference count.
    """
    with native_lock:
        lib = load_library()
        return lib.rin_get_inference_count(ctx)


def rin_get_total_tokens(ctx: int) -> int:
    """Return the total number of tokens processed so far.

    Parameters
    ----------
    ctx : int
        Runtime context handle.

    Returns
    -------
    int
        Total tokens processed.
    """
    with native_lock:
        lib = load_library()
        return lib.rin_get_total_tokens(ctx)


def rin_profile(
    ctx: int,
    mode: int,
    num_warmup: int,
    num_iter: int,
) -> Tuple[float, float]:
    """Run a micro-benchmark and return performance figures.

    Parameters
    ----------
    ctx : int
        Runtime context handle.
    mode : int
        Inference mode to profile (one of ``RIN_MODE_*``).
    num_warmup : int
        Number of warm-up iterations.
    num_iter : int
        Number of measured iterations.

    Returns
    -------
    (ms_per_token, tokens_per_second)
        Throughput metrics.

    Raises
    ------
    RinError
        If profiling fails.
    """
    with native_lock:
        lib = load_library()
        ms_per_token = ctypes.c_double()
        tokens_per_sec = ctypes.c_double()
        _check(
            lib.rin_profile(
                ctx,
                mode,
                num_warmup,
                num_iter,
                ctypes.byref(ms_per_token),
                ctypes.byref(tokens_per_sec),
            )
        )
        return ms_per_token.value, tokens_per_sec.value


def rin_version() -> str:
    """Return the version string of the RIN C library.

    Returns
    -------
    str
        Version string (library-internal format).
    """
    lib = load_library()
    raw = lib.rin_version()
    return raw.decode("utf-8") if raw else ""


def rin_version_numbers() -> Tuple[int, int, int]:
    """Return the version as ``(major, minor, patch)`` integers.

    Returns
    -------
    (major, minor, patch)
        Three-component version tuple.
    """
    with native_lock:
        lib = load_library()
        major = ctypes.c_uint32()
        minor = ctypes.c_uint32()
        patch = ctypes.c_uint32()
        lib.rin_version_numbers(
            ctypes.byref(major), ctypes.byref(minor), ctypes.byref(patch)
        )
        return major.value, minor.value, patch.value


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

_init_fn_signatures(load_library())
