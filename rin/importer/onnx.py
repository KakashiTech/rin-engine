"""ONNX model converter for the RIN format.

Extracts weight tensors from ONNX initialisers and builds a ``RinGraph``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


def _import_onnx():
    """Lazy-import the ``onnx`` package; raises ImportError with help text."""
    try:
        import onnx
        from onnx import numpy_helper
        return onnx, numpy_helper
    except ImportError as exc:
        raise ImportError(
            "The ``onnx`` package is not available.  "
            "Install it with:  pip install onnx\n"
            f"Original error: {exc}"
        ) from exc


def convert_onnx_model(
    model_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
) -> "RinGraph":
    """Convert an ONNX model to RIN format.

    Parameters
    ----------
    model_path : str or Path
        Path to an ``.onnx`` file.
    output_path : str or Path, optional
        If given, the resulting ``.rin`` file is written here.

    Returns
    -------
    RinGraph
        Populated graph ready for serialisation.
    """
    onnx, numpy_helper = _import_onnx()

    model = onnx.load(str(model_path))
    graph_proto = model.graph

    # -- collect initialisers (weight tensors) ----------------------------
    initializers: Dict[str, np.ndarray] = {}
    for init in graph_proto.initializer:
        initializers[init.name] = numpy_helper.to_array(init)

    if not initializers:
        raise ValueError(
            "No initializers (weight tensors) found in the ONNX model."
        )

    # -- try to detect architecture from graph structure ------------------
    arch, meta = _detect_onnx_architecture(graph_proto, initializers)

    # -- build RinGraph --------------------------------------------------
    from rin.importer.base import RinGraph, WeightLayer

    graph = RinGraph(meta)

    if arch == "mlp":
        _populate_mlp(graph, graph_proto, initializers)
    elif arch == "transformer":
        _populate_transformer(graph, graph_proto, initializers)
    else:
        _populate_fallback(graph, initializers)

    if output_path is not None:
        graph.save(str(output_path))
        logger.info("Saved ONNX model to %s", output_path)

    return graph


# ---------------------------------------------------------------------------
# Architecture detection for ONNX
# ---------------------------------------------------------------------------

def _detect_onnx_architecture(
    graph_proto: Any,
    initializers: Dict[str, np.ndarray],
) -> tuple[str, Dict[str, Any]]:
    """Return ``('mlp', meta)`` or ``('transformer', meta)``."""
    node_ops = {n.op_type for n in graph_proto.node}

    input_shapes: Dict[str, tuple[int, ...]] = {}
    for vi in graph_proto.input:
        shape = tuple(
            d.dim_value for d in vi.type.tensor_type.shape.dim
        )
        input_shapes[vi.name] = shape

    meta: Dict[str, Any] = {"architecture": 0}

    has_attention = bool(
        node_ops & {"Attention", "MultiHeadAttention"}
    )
    has_gemm = bool(node_ops & {"Gemm", "MatMul"})
    has_layer_norm = bool(
        node_ops & {"LayerNormalization", "SkipLayerNormalization"}
    )

    if has_attention or (has_layer_norm and has_gemm):
        meta["architecture"] = 1
        meta["num_layers"] = _count_transformer_layers(graph_proto)
        meta["num_heads"] = _infer_num_heads(graph_proto)
        meta["dim"] = _infer_dim(graph_proto, initializers)
        meta["vocab_size"] = _infer_vocab_size(initializers)
        meta["ffn_dim"] = _infer_ffn_dim(graph_proto, initializers)
        meta["max_seq_len"] = _infer_max_seq_len(input_shapes)
        return "transformer", meta

    meta["architecture"] = 0
    meta["num_layers"] = 0
    meta["input_dim"] = 0
    meta["output_dim"] = 0
    return "mlp", meta


def _count_transformer_layers(graph_proto: Any) -> int:
    count = 0
    for node in graph_proto.node:
        if node.op_type in ("Attention", "MultiHeadAttention"):
            count += 1
    return max(count, 1)


def _infer_num_heads(graph_proto: Any) -> int:
    for node in graph_proto.node:
        for attr in node.attribute:
            if attr.name in ("num_heads", "num_attention_heads"):
                return attr.i
    return 12


def _infer_dim(
    graph_proto: Any, initializers: Dict[str, np.ndarray]
) -> int:
    for name, arr in initializers.items():
        if arr.ndim == 2 and min(arr.shape) >= 64:
            return min(arr.shape)
    return 768


def _infer_vocab_size(initializers: Dict[str, np.ndarray]) -> int:
    for name, arr in initializers.items():
        if "embed" in name.lower() and arr.ndim == 2:
            return arr.shape[0]
    return 50257


def _infer_ffn_dim(
    graph_proto: Any, initializers: Dict[str, np.ndarray]
) -> int:
    for node in graph_proto.node:
        if node.op_type == "Gemm":
            for inp in node.input:
                if inp in initializers:
                    arr = initializers[inp]
                    if arr.ndim == 2:
                        return max(arr.shape)
    return 3072


def _infer_max_seq_len(
    input_shapes: Dict[str, tuple[int, ...]]
) -> int:
    for name, shape in input_shapes.items():
        if shape and len(shape) >= 2:
            val = shape[1]
            if isinstance(val, int) and val > 1:
                return val
    return 1024


# ---------------------------------------------------------------------------
# Graph population (MLP)
# ---------------------------------------------------------------------------

def _populate_mlp(
    graph: "RinGraph",
    graph_proto: Any,
    initializers: Dict[str, np.ndarray],
) -> None:
    from rin.importer.base import RinGraph as TG, WeightLayer

    weight_nodes = [
        n for n in graph_proto.node if n.op_type in ("Gemm", "MatMul")
    ]

    input_dim = 0
    for idx, node in enumerate(weight_nodes):
        w_name = node.input[1] if len(node.input) > 1 else None
        b_name = node.input[2] if len(node.input) > 2 else None
        if w_name is None or w_name not in initializers:
            logger.warning("Skipping node %s: weight not found", node.name)
            continue

        w = initializers[w_name].astype(np.float32)
        if b_name and b_name in initializers:
            b = initializers[b_name].astype(np.float32)
        else:
            b = np.zeros(
                w.shape[1] if w.ndim == 2 else 1, dtype=np.float32
            )

        if w.ndim == 2:
            rows, cols = w.shape
        else:
            rows, cols = w.shape[0], 1

        quant_w, scale = TG.quantize_weights(w.ravel())

        if idx == 0:
            input_dim = rows

        graph.add_layer(
            WeightLayer(
                name=f"gemm_{idx}",
                shape=(rows, cols),
                scale=scale,
                weights=quant_w,
                bias=b,
            )
        )

    if graph.layers:
        graph.metadata["num_layers"] = len(graph.layers)
        graph.metadata["input_dim"] = input_dim
        graph.metadata["output_dim"] = graph.layers[-1].cols


# ---------------------------------------------------------------------------
# Graph population (Transformer)
# ---------------------------------------------------------------------------

def _populate_transformer(
    graph: "RinGraph",
    graph_proto: Any,
    initializers: Dict[str, np.ndarray],
) -> None:
    from rin.importer.base import RinGraph as TG, WeightLayer

    processed: set = set()

    def _add(
        name: str,
        w_key: str,
        b_key: Optional[str] = None,
    ) -> None:
        if w_key not in initializers:
            return
        w = initializers[w_key].astype(np.float32)
        if b_key and b_key in initializers:
            b = initializers[b_key].astype(np.float32)
        else:
            b = np.zeros(
                w.shape[1] if w.ndim == 2 else 1, dtype=np.float32
            )
        processed.add(w_key)
        if b_key:
            processed.add(b_key)

        if w.ndim == 2 and b.shape[0] == w.shape[0]:
            w = w.T

        rows, cols = w.shape
        quant_w, scale = TG.quantize_weights(w.ravel())
        graph.add_layer(
            WeightLayer(
                name=name, shape=(rows, cols), scale=scale,
                weights=quant_w, bias=b,
            )
        )

    # Embedding
    for name, arr in initializers.items():
        if "embed" in name.lower() and arr.ndim == 2 and name not in processed:
            _add("embedding", name)
            break

    # Attention layers (pattern matching)
    layer_keys: Dict[int, Dict[str, str]] = {}
    for name in initializers:
        base = name.replace(".weight", "").replace("_weight", "")
        parts = base.replace(".", "_").split("_")
        layer_idx = None
        for p in parts:
            if p.isdigit():
                layer_idx = int(p)
                break
        if layer_idx is None:
            continue
        if layer_idx not in layer_keys:
            layer_keys[layer_idx] = {}
        layer_keys[layer_idx][base] = name

    for lidx in sorted(layer_keys):
        keys = layer_keys[lidx]
        for tag, substrings in [
            ("Wq", ("q_proj", "self_attn_q", "q.weight")),
            ("Wk", ("k_proj", "self_attn_k", "k.weight")),
            ("Wv", ("v_proj", "self_attn_v", "v.weight")),
            ("Wo", ("o_proj", "out_proj", "self_attn_o", "o.weight")),
            ("W1", ("fc1", "fc_in", "intermediate", "up_proj")),
            ("W2", ("fc2", "fc_out", "output", "down_proj")),
        ]:
            for base_key, raw_key in keys.items():
                if any(s in base_key.lower() for s in substrings):
                    b_key = raw_key.replace(".weight", ".bias").replace(
                        "_weight", "_bias"
                    )
                    _add(f"{tag}_{lidx}", raw_key, b_key)

    # Output / unembed
    for name in initializers:
        if name not in processed and (
            "lm_head" in name.lower() or "unembed" in name.lower()
        ):
            _add("output", name)

    # Fallback: remaining 2-D initialisers
    for name, arr in initializers.items():
        if name not in processed and arr.ndim == 2:
            _add(f"extra_{name}", name)


# ---------------------------------------------------------------------------
# Fallback
# ---------------------------------------------------------------------------

def _populate_fallback(
    graph: "RinGraph",
    initializers: Dict[str, np.ndarray],
) -> None:
    from rin.importer.base import RinGraph as TG, WeightLayer

    for idx, (name, arr) in enumerate(initializers.items()):
        if arr.ndim != 2:
            continue
        w = arr.astype(np.float32)
        b = np.zeros(w.shape[1], dtype=np.float32)
        rows, cols = w.shape
        quant_w, scale = TG.quantize_weights(w.ravel())
        graph.add_layer(
            WeightLayer(
                name=f"init_{idx}_{name}",
                shape=(rows, cols),
                scale=scale,
                weights=quant_w,
                bias=b,
            )
        )
    graph.metadata["num_layers"] = len(graph.layers)
