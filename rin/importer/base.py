"""Base classes for the RIN import pipeline.

WeightLayer — a single quantised weight+bias container.
RinGraph  — top-level container that serialises to/from the .rin format.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class WeightLayer:
    """A single quantised weight layer.

    Attributes
    ----------
    name : str
        Human-readable identifier (e.g. "linear_0", "Wq_3").
    shape : (int, int)
        (rows, cols) — the logical dimensions of the original float weight
        matrix **before** flattening.
    scale : float
        Quantisation scale such that
        ``float_weight ≈ (uint8_weight - 128) * scale``.
    weights : np.ndarray (uint8)
        Flattened quantised weights, shape ``(rows * cols,)``.
    bias : np.ndarray (float32)
        Bias vector, shape ``(cols,)``.
    """

    name: str
    shape: Tuple[int, int]
    scale: float
    weights: np.ndarray
    bias: np.ndarray

    def __post_init__(self) -> None:
        rows, cols = self.shape
        expected = rows * cols
        if self.weights.dtype != np.uint8:
            raise TypeError(f"weights must be uint8, got {self.weights.dtype}")
        if self.bias.dtype != np.float32:
            raise TypeError(f"bias must be float32, got {self.bias.dtype}")
        if self.weights.shape != (expected,):
            raise ValueError(
                f"weights shape {self.weights.shape} != ({expected},) "
                f"for shape {self.shape}"
            )
        if self.bias.shape != (cols,):
            raise ValueError(
                f"bias shape {self.bias.shape} != ({cols},) for shape {self.shape}"
            )

    @property
    def rows(self) -> int:
        return self.shape[0]

    @property
    def cols(self) -> int:
        return self.shape[1]

    def dequantize(self) -> np.ndarray:
        """Recover an approximation of the original float32 weight matrix."""
        return (self.weights.astype(np.float32) - 128.0) * self.scale


class RinGraph:
    """Computation graph holding a list of ``WeightLayer`` layers and metadata.

    Serialises to / deserialises from the ``.rin`` binary format.

    Parameters
    ----------
    metadata : dict, optional
        Architecture-level key/value store.  Expected keys include
        ``architecture`` (0=MLP, 1=Transformer), ``num_layers``, ``dim``,
        ``vocab_size``, ``num_heads``, ``max_seq_len``, ``ffn_dim``,
        ``input_dim``, ``output_dim``.
    """

    def __init__(self, metadata: Optional[Dict[str, Any]] = None) -> None:
        self.layers: List[WeightLayer] = []
        self.metadata: Dict[str, Any] = metadata or {}

    # -- helpers ----------------------------------------------------------

    def add_layer(self, layer: WeightLayer) -> None:
        """Append *layer* to the internal layer list."""
        self.layers.append(layer)

    @staticmethod
    def quantize_weights(weights: np.ndarray) -> Tuple[np.ndarray, float]:
        """Uniform uint8 quantisation of a float32 weight array.

        Uses symmetric quantisation:  value ``+max_abs`` maps to ``127``
        and ``-max_abs`` maps to ``-128`` (stored as ``uint8[0..255]``).

        Returns
        -------
        quantized : np.ndarray (uint8)
            Flat array of quantised values.
        scale : float
            Scale factor.
        """
        weights = weights.astype(np.float32, copy=False)
        max_abs = float(np.max(np.abs(weights)))
        scale = max_abs / 127.0 if max_abs > 0.0 else 1.0
        quantized = np.round(weights / scale + 128.0).clip(0, 255).astype(np.uint8)
        return quantized, scale

    # -- serialisation ----------------------------------------------------

    def to_rin(self) -> bytes:
        """Serialise the graph to the ``.rin`` binary format."""
        arch = self.metadata.get("architecture", 0)
        buf = bytearray()
        buf.extend(b"RIN1")
        buf.extend(struct.pack("<I", 1))  # version
        buf.extend(struct.pack("<I", arch))

        if arch == 0:
            self._write_mlp(buf)
        elif arch == 1:
            self._write_transformer(buf)
        else:
            raise ValueError(f"Unknown architecture code: {arch}")
        return bytes(buf)

    def _write_mlp(self, buf: bytearray) -> None:
        buf.extend(struct.pack("<I", self.metadata.get("num_layers", len(self.layers))))
        buf.extend(struct.pack("<I", self.metadata.get("input_dim", 0)))
        buf.extend(struct.pack("<I", self.metadata.get("output_dim", 0)))
        for layer in self.layers:
            _write_weight_layer(buf, layer)

    def _write_transformer(self, buf: bytearray) -> None:
        for key in ("num_layers", "dim", "vocab_size", "num_heads",
                     "max_seq_len", "ffn_dim"):
            buf.extend(struct.pack("<I", self.metadata.get(key, 0)))
        for layer in self.layers:
            _write_weight_layer(buf, layer)

        pe = self.metadata.get("position_embeddings")
        if pe is not None:
            buf.extend(np.asarray(pe, dtype=np.int16).tobytes())

        charset = self.metadata.get("charset", b"")
        pad = ((len(charset) + 3) // 4) * 4
        buf.extend(charset.ljust(pad, b"\x00"))

    def save(self, path: str) -> None:
        """Write the ``.rin`` file to *path*."""
        with open(path, "wb") as f:
            f.write(self.to_rin())

    # -- deserialisation --------------------------------------------------

    @classmethod
    def load(cls, path: str) -> "RinGraph":
        """Load a ``.rin`` file and return a ``RinGraph``."""
        with open(path, "rb") as f:
            data = f.read()
        return cls.from_bytes(data)

    @classmethod
    def from_bytes(cls, data: bytes) -> "RinGraph":
        """Parse a raw byte sequence into a ``RinGraph``."""
        offset = 0
        magic = data[offset:offset + 4]
        offset += 4
        if magic != b"RIN1":
            raise ValueError(f"Bad magic: {magic!r}")
        version = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        arch = struct.unpack_from("<I", data, offset)[0]
        offset += 4

        if arch == 0:
            return cls._read_mlp(data, offset, version)
        elif arch == 1:
            return cls._read_transformer(data, offset, version)
        else:
            raise ValueError(f"Unknown architecture code: {arch}")

    @classmethod
    def _read_mlp(cls, data: bytes, offset: int,
                  version: int) -> "RinGraph":
        num_layers = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        input_dim = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        output_dim = struct.unpack_from("<I", data, offset)[0]
        offset += 4

        meta: Dict[str, Any] = {
            "architecture": 0,
            "num_layers": num_layers,
            "input_dim": input_dim,
            "output_dim": output_dim,
            "version": version,
        }
        graph = cls(meta)
        for i in range(num_layers):
            layer, offset = _read_weight_layer(data, offset)
            layer.name = f"layer_{i}"
            graph.add_layer(layer)
        return graph

    @classmethod
    def _read_transformer(cls, data: bytes, offset: int,
                          version: int) -> "RinGraph":
        num_layers = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        dim = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        vocab_size = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        num_heads = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        max_seq_len = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        ffn_dim = struct.unpack_from("<I", data, offset)[0]
        offset += 4

        total_weights = 2 + 6 * num_layers
        meta: Dict[str, Any] = {
            "architecture": 1,
            "num_layers": num_layers,
            "dim": dim,
            "vocab_size": vocab_size,
            "num_heads": num_heads,
            "max_seq_len": max_seq_len,
            "ffn_dim": ffn_dim,
            "version": version,
        }
        graph = cls(meta)
        for i in range(total_weights):
            layer, offset = _read_weight_layer(data, offset)
            layer.name = f"w_{i}"
            graph.add_layer(layer)

        # position embeddings (int16, shape [max_seq_len, dim])
        pe_bytes = max_seq_len * dim * 2
        if offset + pe_bytes <= len(data):
            pe = (
                np.frombuffer(data[offset:offset + pe_bytes], dtype=np.int16)
                .copy()
                .reshape(max_seq_len, dim)
            )
            offset += pe_bytes
            graph.metadata["position_embeddings"] = pe

        # charset (remaining bytes, null-padded to 4-byte boundary)
        if offset < len(data):
            raw = data[offset:]
            graph.metadata["charset"] = raw.rstrip(b"\x00")

        return graph


# -- internal helpers -----------------------------------------------------

def _write_weight_layer(buf: bytearray, layer: WeightLayer) -> None:
    rows, cols = layer.shape
    buf.extend(struct.pack("<II", rows, cols))
    buf.extend(struct.pack("<f", layer.scale))
    buf.extend(layer.weights.tobytes())
    buf.extend(layer.bias.tobytes())


def _read_weight_layer(data: bytes, offset: int) -> Tuple[WeightLayer, int]:
    rows = struct.unpack_from("<I", data, offset)[0]
    offset += 4
    cols = struct.unpack_from("<I", data, offset)[0]
    offset += 4
    scale = struct.unpack_from("<f", data, offset)[0]
    offset += 4
    n_w = rows * cols
    weights = np.frombuffer(data[offset:offset + n_w], dtype=np.uint8).copy()
    offset += n_w
    bias = np.frombuffer(
        data[offset:offset + 4 * cols], dtype=np.float32
    ).copy()
    offset += 4 * cols
    layer = WeightLayer(
        name="",
        shape=(rows, cols),
        scale=scale,
        weights=weights,
        bias=bias,
    )
    return layer, offset
