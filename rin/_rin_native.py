"""
Low-level ctypes wrapper around librin.so — the RIN C runtime library.

Provides complete ctype definitions for all exported C structs, enums, and
functions, together with automatic library discovery and thread-safe dispatch.

Environment variables
---------------------
THOR_LIB : str
    Explicit path to ``librin.so``. When set, no auto-detection is performed.

Library search order
--------------------
1. ``THOR_LIB`` environment variable.
2. ``<package_root>/../librin.so`` (installed next to the ``thor`` package).
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
    "ThorMemoryError",
    "ThorWeightsError",
    "ThorInferenceError",
    "ThorNotInitializedError",
    "ThorInvalidInputError",
    "ThorUnsupportedError",
    "ThorStatus",
    "ThorMode",
    "ThorModelInfo",
    "ThorResult",
    "THOR_OK",
    "THOR_ERR_INIT",
    "THOR_ERR_MEMORY",
    "THOR_ERR_WEIGHTS",
    "THOR_ERR_INFERENCE",
    "THOR_ERR_NOT_INITIALIZED",
    "THOR_ERR_INVALID_INPUT",
    "THOR_ERR_UNSUPPORTED",
    "THOR_MODE_MLP",
    "THOR_MODE_SNN",
    "THOR_MODE_ATTN",
    "THOR_MODE_THOR",
    "THOR_MODE_TRANSFORMER",
    "load_library",
    "thor_create",
    "thor_destroy",
    "thor_load_model",
    "thor_get_model_info",
    "thor_set_mode",
    "thor_get_mode",
    "thor_set_temperature",
    "thor_set_top_k",
    "thor_set_top_p",
    "thor_set_power_budget",
    "thor_infer",
    "thor_free_result",
    "thor_get_charset",
    "thor_encode",
    "thor_decode",
    "thor_get_energy_joules",
    "thor_get_energy_millijoules",
    "thor_get_inference_count",
    "thor_get_total_tokens",
    "thor_profile",
    "thor_version",
    "thor_version_numbers",
    "native_lock",
]

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ThorError(Exception):
    """Base exception for all RIN runtime errors."""


class ThorInitError(ThorError):
    """Initialisation failure (``THOR_ERR_INIT``)."""


class ThorMemoryError(ThorError):
    """Memory allocation failure (``THOR_ERR_MEMORY``)."""


class ThorWeightsError(ThorError):
    """Model weight loading failure (``THOR_ERR_WEIGHTS``)."""


class ThorInferenceError(ThorError):
    """Inference execution failure (``THOR_ERR_INFERENCE``)."""


class ThorNotInitializedError(ThorError):
    """Operation attempted before the context was initialised (``THOR_ERR_NOT_INITIALIZED``)."""


class ThorInvalidInputError(ThorError):
    """Invalid input argument (``THOR_ERR_INVALID_INPUT``)."""


class ThorUnsupportedError(ThorError):
    """Unsupported feature or configuration (``THOR_ERR_UNSUPPORTED``)."""


_STATUS_ERROR_MAP: dict[int, type[ThorError]] = {
    -1: ThorInitError,
    -2: ThorMemoryError,
    -3: ThorWeightsError,
    -4: ThorInferenceError,
    -5: ThorNotInitializedError,
    -6: ThorInvalidInputError,
    -7: ThorUnsupportedError,
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

THOR_OK: int = 0
THOR_ERR_INIT: int = -1
THOR_ERR_MEMORY: int = -2
THOR_ERR_WEIGHTS: int = -3
THOR_ERR_INFERENCE: int = -4
THOR_ERR_NOT_INITIALIZED: int = -5
THOR_ERR_INVALID_INPUT: int = -6
THOR_ERR_UNSUPPORTED: int = -7

THOR_MODE_MLP: int = 0
THOR_MODE_SNN: int = 1
THOR_MODE_ATTN: int = 2
THOR_MODE_THOR: int = 3
THOR_MODE_TRANSFORMER: int = 4


class ThorStatus:
    """Namespace wrapping C status codes with human-readable names."""

    OK: ClassVar[int] = THOR_OK
    ERR_INIT: ClassVar[int] = THOR_ERR_INIT
    ERR_MEMORY: ClassVar[int] = THOR_ERR_MEMORY
    ERR_WEIGHTS: ClassVar[int] = THOR_ERR_WEIGHTS
    ERR_INFERENCE: ClassVar[int] = THOR_ERR_INFERENCE
    ERR_NOT_INITIALIZED: ClassVar[int] = THOR_ERR_NOT_INITIALIZED
    ERR_INVALID_INPUT: ClassVar[int] = THOR_ERR_INVALID_INPUT
    ERR_UNSUPPORTED: ClassVar[int] = THOR_ERR_UNSUPPORTED

    @staticmethod
    def is_error(status: int) -> bool:
        """Return ``True`` when *status* indicates an error."""
        return status < 0

    @staticmethod
    def message(status: int) -> str:
        """Return a short human-readable message for *status*."""
        return _STATUS_MESSAGE_MAP.get(status, f"unknown status code {status}")


class ThorMode:
    """Namespace wrapping inference-mode constants."""

    MLP: ClassVar[int] = THOR_MODE_MLP
    SNN: ClassVar[int] = THOR_MODE_SNN
    ATTN: ClassVar[int] = THOR_MODE_ATTN
    THOR: ClassVar[int] = THOR_MODE_THOR
    TRANSFORMER: ClassVar[int] = THOR_MODE_TRANSFORMER

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
        return ThorMode._NAME_MAP.get(mode, f"unknown({mode})")

    @staticmethod
    def from_string(name: str) -> int:
        """Parse a mode string (e.g. ``"mlp"``, ``"snn"``) to its integer value.

        Raises ``ValueError`` on unknown names.
        """
        name = name.strip().lower()
        try:
            return ThorMode._VALUE_MAP[name]
        except KeyError:
            valid = ", ".join(sorted(ThorMode._VALUE_MAP))
            raise ValueError(
                f"Unknown mode {name!r}. Valid modes: {valid}"
            ) from None


# ---------------------------------------------------------------------------
# C-compatible struct definitions
# ---------------------------------------------------------------------------

class ThorModelInfo(ctypes.Structure):
    """Mirrors the C ``ThorModelInfo`` struct."""

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


class ThorResult(ctypes.Structure):
    """Mirrors the C ``ThorResult`` struct."""

    _fields_ = [
        ("tokens", ctypes.POINTER(ctypes.c_uint32)),
        ("num_tokens", ctypes.c_uint32),
        ("energy_joules", ctypes.c_double),
        ("tokens_per_second", ctypes.c_float),
        ("latency_ns", ctypes.c_uint64),
    ]

    def as_dict(self) -> dict[str, any]:
        """Copy token data into a plain dictionary and return it.

        The caller *must still* call :func:`thor_free_result` after reading
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
    1. ``THOR_LIB`` environment variable.
    2. ``<package_root>/../librin.so`` (next to the ``thor`` package).
    3. ``"librin.so"`` (system library path).
    """
    env_path = os.environ.get("THOR_LIB")
    if env_path:
        resolved = os.path.abspath(env_path)
        if os.path.isfile(resolved):
            return resolved
        raise ThorError(f"THOR_LIB={env_path} does not point to a file")

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
    ThorError
        If the library cannot be found or loaded.
    """
    global _lib
    if _lib is not None and path is None:
        return _lib

    lib_path = path if path else _resolve_library_path()
    try:
        _lib = ctypes.CDLL(lib_path)
    except OSError as e:
        raise ThorError(
            f"Failed to load RIN shared library from {lib_path!r}: {e}"
        ) from e
    return _lib


# ---------------------------------------------------------------------------
# Function signature declarations
# ---------------------------------------------------------------------------

def _init_fn_signatures(lib: ctypes.CDLL) -> None:
    """Set ``argtypes`` / ``restype`` on every exported C function."""
    # The opaque context handle
    lib.thor_create.restype = ctypes.c_void_p
    lib.thor_create.argtypes = []

    lib.thor_destroy.restype = None
    lib.thor_destroy.argtypes = [ctypes.c_void_p]

    # Model loading
    lib.thor_load_model.restype = ctypes.c_int
    lib.thor_load_model.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

    lib.thor_get_model_info.restype = ctypes.c_int
    lib.thor_get_model_info.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ThorModelInfo),
    ]

    # Configuration
    lib.thor_set_mode.restype = None
    lib.thor_set_mode.argtypes = [ctypes.c_void_p, ctypes.c_int]

    lib.thor_get_mode.restype = ctypes.c_int
    lib.thor_get_mode.argtypes = [ctypes.c_void_p]

    lib.thor_set_temperature.restype = None
    lib.thor_set_temperature.argtypes = [ctypes.c_void_p, ctypes.c_float]

    lib.thor_set_top_k.restype = None
    lib.thor_set_top_k.argtypes = [ctypes.c_void_p, ctypes.c_uint32]

    lib.thor_set_top_p.restype = None
    lib.thor_set_top_p.argtypes = [ctypes.c_void_p, ctypes.c_float]

    lib.thor_set_power_budget.restype = None
    lib.thor_set_power_budget.argtypes = [ctypes.c_void_p, ctypes.c_float]

    # Inference
    lib.thor_infer.restype = ctypes.c_int
    lib.thor_infer.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.POINTER(ThorResult),
    ]

    lib.thor_free_result.restype = None
    lib.thor_free_result.argtypes = [ctypes.POINTER(ThorResult)]

    # Tokenizer
    lib.thor_get_charset.restype = ctypes.c_char_p
    lib.thor_get_charset.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]

    lib.thor_encode.restype = ctypes.c_int
    lib.thor_encode.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_int,
    ]

    lib.thor_decode.restype = None
    lib.thor_decode.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
    ]

    # Energy
    lib.thor_get_energy_joules.restype = ctypes.c_double
    lib.thor_get_energy_joules.argtypes = [ctypes.c_void_p]

    lib.thor_get_energy_millijoules.restype = ctypes.c_double
    lib.thor_get_energy_millijoules.argtypes = [ctypes.c_void_p]

    lib.thor_get_inference_count.restype = ctypes.c_uint64
    lib.thor_get_inference_count.argtypes = [ctypes.c_void_p]

    lib.thor_get_total_tokens.restype = ctypes.c_uint64
    lib.thor_get_total_tokens.argtypes = [ctypes.c_void_p]

    # Profiling
    lib.thor_profile.restype = ctypes.c_int
    lib.thor_profile.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]

    # Version
    lib.thor_version.restype = ctypes.c_char_p
    lib.thor_version.argtypes = []

    lib.thor_version_numbers.restype = None
    lib.thor_version_numbers.argtypes = [
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
    ]


# ---------------------------------------------------------------------------
# Public wrapper functions  (thread-safe)
# ---------------------------------------------------------------------------

def _check(status: int) -> int:
    """Raise the appropriate :class:`ThorError` if *status* is an error code."""
    if status == 0:
        return status
    exc_cls = _STATUS_ERROR_MAP.get(status, ThorError)
    msg = _STATUS_MESSAGE_MAP.get(status, f"unknown error {status}")
    raise exc_cls(msg)


def thor_create() -> int:
    """Create a new RIN runtime context.

    Returns
    -------
    int
        Opaque pointer (handle) to the ``ThorContext``.

    Raises
    ------
    ThorError
        On failure.
    """
    with native_lock:
        lib = load_library()
        ptr = lib.thor_create()
        if not ptr:
            raise ThorError("thor_create returned NULL")
        return ptr


def thor_destroy(ctx: int) -> None:
    """Destroy a RIN runtime context previously created with :func:`thor_create`.

    Parameters
    ----------
    ctx : int
        Opaque handle returned by :func:`thor_create`.
    """
    with native_lock:
        lib = load_library()
        lib.thor_destroy(ctx)


def thor_load_model(ctx: int, model_path: str) -> None:
    """Load a model file into the runtime context.

    Parameters
    ----------
    ctx : int
        Runtime context handle.
    model_path : str
        Filesystem path to the model file.

    Raises
    ------
    ThorWeightsError
        If the model file cannot be loaded.
    """
    with native_lock:
        lib = load_library()
        _check(lib.thor_load_model(ctx, model_path.encode("utf-8")))


def thor_get_model_info(ctx: int) -> ThorModelInfo:
    """Retrieve the currently loaded model's metadata.

    Parameters
    ----------
    ctx : int
        Runtime context handle.

    Returns
    -------
    ThorModelInfo
        Populated struct mirroring the C struct.

    Raises
    ------
    ThorNotInitializedError
        If no model is loaded.
    """
    with native_lock:
        lib = load_library()
        info = ThorModelInfo()
        _check(lib.thor_get_model_info(ctx, ctypes.byref(info)))
        return info


def thor_set_mode(ctx: int, mode: int) -> None:
    """Set the inference execution mode.

    Parameters
    ----------
    ctx : int
        Runtime context handle.
    mode : int
        One of ``THOR_MODE_MLP``, ``THOR_MODE_SNN``, etc.
    """
    with native_lock:
        lib = load_library()
        lib.thor_set_mode(ctx, mode)


def thor_get_mode(ctx: int) -> int:
    """Return the currently active inference mode.

    Parameters
    ----------
    ctx : int
        Runtime context handle.

    Returns
    -------
    int
        One of the ``THOR_MODE_*`` constants.
    """
    with native_lock:
        lib = load_library()
        return lib.thor_get_mode(ctx)


def thor_set_temperature(ctx: int, temp: float) -> None:
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
        lib.thor_set_temperature(ctx, temp)


def thor_set_top_k(ctx: int, k: int) -> None:
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
        lib.thor_set_top_k(ctx, k)


def thor_set_top_p(ctx: int, p: float) -> None:
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
        lib.thor_set_top_p(ctx, p)


def thor_set_power_budget(ctx: int, watts: float) -> None:
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
        lib.thor_set_power_budget(ctx, watts)


def thor_infer(
    ctx: int,
    input_ids: List[int],
    num_input: int,
    max_output: int,
) -> ThorResult:
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
    ThorResult
        Result struct containing generated tokens and performance data.

    Raises
    ------
    ThorInferenceError
        If inference fails.
    """
    with native_lock:
        lib = load_library()
        arr = (ctypes.c_uint32 * num_input)(*input_ids)
        result = ThorResult()
        _check(lib.thor_infer(ctx, arr, num_input, max_output, ctypes.byref(result)))
        return result


def thor_free_result(result: ThorResult) -> None:
    """Free the dynamically allocated token buffer inside a result struct.

    Parameters
    ----------
    result : ThorResult
        Struct whose ``tokens`` pointer will be freed.
    """
    with native_lock:
        lib = load_library()
        lib.thor_free_result(ctypes.byref(result))


def thor_get_charset(ctx: int) -> Tuple[str, int]:
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
        charset = lib.thor_get_charset(ctx, ctypes.byref(vocab_size))
        if not charset:
            raise ThorError("thor_get_charset returned NULL")
        return charset.decode("utf-8"), vocab_size.value


def thor_encode(ctx: int, text: str, max_ids: int = 0) -> List[int]:
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
        n = lib.thor_encode(ctx, text.encode("utf-8"), ids, max_ids)
        if n < 0:
            raise ThorEncodeError(
                f"thor_encode failed with return code {n}"
            )
        return list(ids[:n])


class ThorEncodeError(ThorError):
    """Raised when :func:`thor_encode` fails."""


def thor_decode(ctx: int, ids: List[int]) -> str:
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
        lib.thor_decode(ctx, arr, len(ids), buf, max_text)
        return buf.value.decode("utf-8")


def thor_get_energy_joules(ctx: int) -> float:
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
        return lib.thor_get_energy_joules(ctx)


def thor_get_energy_millijoules(ctx: int) -> float:
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
        return lib.thor_get_energy_millijoules(ctx)


def thor_get_inference_count(ctx: int) -> int:
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
        return lib.thor_get_inference_count(ctx)


def thor_get_total_tokens(ctx: int) -> int:
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
        return lib.thor_get_total_tokens(ctx)


def thor_profile(
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
        Inference mode to profile (one of ``THOR_MODE_*``).
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
    ThorError
        If profiling fails.
    """
    with native_lock:
        lib = load_library()
        ms_per_token = ctypes.c_double()
        tokens_per_sec = ctypes.c_double()
        _check(
            lib.thor_profile(
                ctx,
                mode,
                num_warmup,
                num_iter,
                ctypes.byref(ms_per_token),
                ctypes.byref(tokens_per_sec),
            )
        )
        return ms_per_token.value, tokens_per_sec.value


def thor_version() -> str:
    """Return the version string of the RIN C library.

    Returns
    -------
    str
        Version string (library-internal format).
    """
    lib = load_library()
    raw = lib.thor_version()
    return raw.decode("utf-8") if raw else ""


def thor_version_numbers() -> Tuple[int, int, int]:
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
        lib.thor_version_numbers(
            ctypes.byref(major), ctypes.byref(minor), ctypes.byref(patch)
        )
        return major.value, minor.value, patch.value


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

_init_fn_signatures(load_library())
