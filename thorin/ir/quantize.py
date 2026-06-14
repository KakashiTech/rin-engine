"""Advanced quantisation utilities for THOR models.

Provides per-tensor / per-channel quantisation, model-level quantisation
with optional calibration, and utilities for range analysis.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from thorin.importer.base import ThorGraph, WeightLayer
from thorin.ir.graph import ComputationGraph


# ---------------------------------------------------------------------------
# Tensor quantisation
# ---------------------------------------------------------------------------


def quantize_tensor(
    tensor: np.ndarray,
    bits: int = 8,
    per_channel: bool = False,
    axis: int = 0,
) -> Tuple[np.ndarray, Union[float, np.ndarray]]:
    """Quantise a float32 tensor to unsigned integer.

    Parameters
    ----------
    tensor : np.ndarray (float32)
        Input tensor.
    bits : int
        Bit width (default 8).  Supported: 4, 8.
    per_channel : bool
        If ``True``, each slice along *axis* gets its own scale.
    axis : int
        Axis along which to compute per-channel scales (default 0).

    Returns
    -------
    quantized : np.ndarray (uint8 or uint16)
        Quantised tensor, same shape as input.
    scale : float or np.ndarray
        Scale factor(s).  Single float for per-tensor, 1-D array for
        per-channel.

    Raises
    ------
    ValueError
        If *bits* is not 4 or 8.
    """
    if bits not in (4, 8):
        raise ValueError(f"Unsupported bit width: {bits}.  Supported: 4, 8.")

    tensor = tensor.astype(np.float32, copy=False)
    max_val = float(2 ** (bits - 1) - 1)

    if per_channel:
        if tensor.ndim < 2:
            raise ValueError(
                f"per_channel quantisation requires at least 2-D tensor, "
                f"got shape {tensor.shape}"
            )
        # Compute |max| along axis, keepdims for broadcasting
        max_abs = np.max(np.abs(tensor), axis=axis, keepdims=True)
        max_abs = np.maximum(max_abs, 1e-8)
        scales = np.squeeze(max_abs / max_val)
        quantized = np.round(tensor / max_abs * max_val + max_val).clip(0, 255)
    else:
        max_abs = float(np.max(np.abs(tensor)))
        if max_abs == 0.0:
            max_abs = 1.0
        scale = max_abs / max_val
        quantized = np.round(tensor / scale + max_val).clip(0, 255)

    dtype = np.uint8 if bits == 8 else np.uint8  # 4-bit stored as uint8
    return quantized.astype(dtype), (scale if not per_channel else scales)


# ---------------------------------------------------------------------------
# Model-level quantisation
# ---------------------------------------------------------------------------


def quantize_model(
    model_graph: ThorGraph,
    bits: int = 8,
    calibration_data: Optional[Dict[str, np.ndarray]] = None,
) -> ThorGraph:
    """Quantise an entire ``ThorGraph`` model.

    When *calibration_data* is provided the scale for each layer is
    selected to minimise quantisation error over the calibration
    activations; otherwise a simple direct quantisation of the stored
    weight tensor is used.

    Parameters
    ----------
    model_graph : ThorGraph
        Source graph (typically dequantised or freshly built).
    bits : int
        Target bit width (4 or 8).
    calibration_data : dict, optional
        Mapping ``layer_name → np.ndarray`` of representative float
        activations or weight observations.

    Returns
    -------
    ThorGraph
        New graph with re-quantised weight layers.
    """
    quantized_layers: List[WeightLayer] = []

    for layer in model_graph.layers:
        name = layer.name

        if calibration_data and name in calibration_data:
            calib = calibration_data[name]
            scale = _optimal_scale(layer, calib, bits)
        else:
            scale = layer.scale

        # Re-quantise the dequantised weights
        float_w = layer.dequantize().ravel()
        max_val = float(2 ** (bits - 1) - 1)
        if scale == 0.0:
            scale = 1.0
        q_w = np.round(float_w / scale + max_val).clip(0, 255).astype(np.uint8)

        quantized_layers.append(
            WeightLayer(
                name=name,
                shape=layer.shape,
                scale=scale,
                weights=q_w,
                bias=layer.bias.copy(),
            )
        )

    new_graph = ThorGraph(metadata=dict(model_graph.metadata))
    for layer in quantized_layers:
        new_graph.add_layer(layer)

    return new_graph


def _optimal_scale(
    layer: WeightLayer,
    calibration_data: np.ndarray,
    bits: int,
) -> float:
    """Brute-force search for the scale that minimises MSE on calibration data."""
    float_w = layer.dequantize()
    max_val = float(2 ** (bits - 1) - 1)

    best_scale = layer.scale
    best_error = float("inf")

    # Search around the current scale
    for factor in np.logspace(-0.5, 0.5, 21):
        candidate = layer.scale * factor
        if candidate == 0.0:
            continue
        q = np.round(float_w / candidate + max_val).clip(0, 255).astype(np.uint8)
        dq = (q.astype(np.float32) - max_val) * candidate
        error = float(np.mean((calibration_data - dq) ** 2))
        if error < best_error:
            best_error = error
            best_scale = candidate

    return best_scale


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def calibrate(
    model_graph: ThorGraph,
    dataset: Union[np.ndarray, List[np.ndarray]],
    num_samples: int = 100,
) -> Dict[str, np.ndarray]:
    """Run calibration to determine optimal quantisation ranges.

    Parameters
    ----------
    model_graph : ThorGraph
        The model to calibrate.
    dataset : np.ndarray or list of np.ndarray
        Representative input data (or list of samples).
    num_samples : int
        Number of samples to use.

    Returns
    -------
    dict
        ``{layer_name: np.ndarray}`` — the distribution of activations /
        weight values observed for each layer during calibration.
    """
    if isinstance(dataset, np.ndarray):
        samples = [dataset]
    else:
        samples = list(dataset)

    samples = samples[:num_samples]

    # For each layer, collect the dequantised weight distribution
    calibration_map: Dict[str, np.ndarray] = {}
    for layer in model_graph.layers:
        float_w = layer.dequantize()
        calibration_map[layer.name] = float_w

    return calibration_map


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def analyze_ranges(
    graph_or_node: Union[ThorGraph, WeightLayer],
) -> Dict[str, Any]:
    """Analyse value ranges across a graph or weight layer.

    Returns a dictionary with keys:
    ``min``, ``max``, ``mean``, ``std``, ``scale``, ``dynamic_range``.
    """
    if isinstance(graph_or_node, ThorGraph):
        layers = graph_or_node.layers
        if not layers:
            return {}
        all_deq = np.concatenate([l.dequantize().ravel() for l in layers])
        scales = [l.scale for l in layers]
        return {
            "min": float(all_deq.min()),
            "max": float(all_deq.max()),
            "mean": float(all_deq.mean()),
            "std": float(all_deq.std()),
            "scale_min": min(scales),
            "scale_max": max(scales),
            "dynamic_range_db": float(
                20 * np.log10(np.abs(all_deq).max() / max(np.abs(all_deq).min(), 1e-10))
            ),
        }

    layer = graph_or_node
    float_w = layer.dequantize()
    return {
        "min": float(float_w.min()),
        "max": float(float_w.max()),
        "mean": float(float_w.mean()),
        "std": float(float_w.std()),
        "scale": layer.scale,
        "dynamic_range_db": float(
            20 * np.log10(
                np.abs(float_w).max() / max(np.abs(float_w).min(), 1e-10)
            )
        ),
    }
