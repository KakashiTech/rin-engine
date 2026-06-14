"""Graph optimisation passes for the RIN intermediate representation.

Provides fusion, elimination, and constant-folding passes that operate on
``ComputationGraph``.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set

import numpy as np

from rin.ir.graph import ComputationGraph, RinNode

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fusion passes
# ---------------------------------------------------------------------------


def fuse_bn(g: ComputationGraph) -> ComputationGraph:
    """Fuse batch-normalisation nodes into preceding linear nodes.

    Batch normalisation during inference is a linear transformation:

        y = gamma * (x - mean) / sqrt(var + eps) + beta

    which can be folded into an adjacent ``gemv`` / ``linear`` node by
    adjusting its weights and bias.

    Returns a new ``ComputationGraph`` with BN nodes removed.
    """
    nodes = list(g.nodes)
    removed: Set[int] = set()
    bn_params: Dict[str, Dict[str, np.ndarray]] = {}

    # Collect BN parameter tensors (represented as constant inputs)
    for i, node in enumerate(nodes):
        if node.op_type in ("batch_norm", "batchnorm", "fused_batch_norm"):
            if len(node.inputs) >= 5:
                bn_params[node.outputs[0]] = {
                    "gamma": node.attrs.get("gamma"),
                    "beta": node.attrs.get("beta"),
                    "mean": node.attrs.get("mean"),
                    "var": node.attrs.get("var"),
                    "eps": node.attrs.get("eps", 1e-5),
                }

    # Find linear nodes that feed into BN nodes and fuse
    fused = list(nodes)
    for i, node in enumerate(nodes):
        if node.op_type not in ("gemv", "linear", "dense", "matmul"):
            continue
        if not node.outputs:
            continue
        out_name = node.outputs[0]
        # Check if this output feeds into a BN
        bn_targets = [
            j for j, n in enumerate(nodes)
            if n.op_type in ("batch_norm", "batchnorm", "fused_batch_norm")
            and n.inputs and n.inputs[0] == out_name
        ]
        if not bn_targets:
            continue

        for bn_idx in bn_targets:
            bn_node = nodes[bn_idx]
            params = bn_params.get(bn_node.outputs[0]
                                    if bn_node.outputs else "", {})
            if not params:
                continue

            gamma = params["gamma"]
            beta = params["beta"]
            mean = params["mean"]
            var = params["var"]
            eps = params["eps"]

            if any(p is None for p in (gamma, beta, mean, var)):
                continue

            # Folding:  W' = W * gamma / sqrt(var + eps)
            #            b' = (b - mean) * gamma / sqrt(var + eps) + beta
            scale = gamma / np.sqrt(var + eps)

            if node.weights is not None:
                w_flat = node.weights.dequantize()
                w_mat = w_flat.reshape(node.weights.shape)
                w_fused = w_mat * scale  # broadcast along cols
                new_w, new_scale = ComputationGraph.quantize_weights(w_fused.ravel())
                new_bias = (node.weights.bias - mean) * scale + beta

                fused[i].weights = type(node.weights)(
                    name=node.weights.name,
                    shape=node.weights.shape,
                    scale=new_scale,
                    weights=new_w,
                    bias=new_bias.astype(np.float32),
                )

            removed.add(bn_idx)
            fused[i].outputs = list(bn_node.outputs)
            logger.debug("Fused BN into node %d (%s)", i, node.op_type)

    result = [n for j, n in enumerate(fused) if j not in removed]
    return ComputationGraph(result)


# ---------------------------------------------------------------------------
# No-op removal
# ---------------------------------------------------------------------------


def remove_noop(g: ComputationGraph) -> ComputationGraph:
    """Remove no-operation nodes.

    Eliminates:
      - ``identity`` nodes (pass-through)
      - ``dropout`` in inference mode
      - ``cast`` when source and target types match
      - ``reshape`` when shape does not change
    """
    nodes = list(g.nodes)
    tensors = g.tensors()

    kept: List[RinNode] = []
    skip: Set[int] = set()

    for i, node in enumerate(nodes):
        if i in skip:
            continue

        # identity → wire through (handled by ComputationGraph.optimize)
        if node.op_type == "identity":
            continue

        # dropout in inference → pass-through
        if node.op_type == "dropout":
            continue

        # cast with matching types
        if node.op_type == "cast":
            src = node.attrs.get("from", None)
            dst = node.attrs.get("to", None)
            if src is not None and dst is not None and src == dst:
                continue

        # reshape that preserves total element count with same shape
        if node.op_type == "reshape":
            in_shape = node.attrs.get("input_shape")
            out_shape = node.attrs.get("output_shape")
            if in_shape is not None and out_shape is not None:
                if np.prod(in_shape) == np.prod(out_shape) and in_shape == out_shape:
                    continue

        kept.append(node)

    trimmed = ComputationGraph(kept)
    logger.debug(
        "remove_noop: %d → %d nodes", len(nodes), len(kept)
    )
    return trimmed


# ---------------------------------------------------------------------------
# Constant folding
# ---------------------------------------------------------------------------


def constant_fold(g: ComputationGraph) -> ComputationGraph:
    """Fold constant sub-expressions.

    Identifies nodes whose all inputs are constants (literals embedded
    in ``attrs``) and replaces them with a single constant output.
    """
    nodes = list(g.nodes)
    const_values: Dict[str, np.ndarray] = {}

    # Pre-populate with node-internal constants
    for node in nodes:
        if "const_value" in node.attrs:
            const_values[node.outputs[0]] = node.attrs["const_value"]

    kept: List[RinNode] = []
    for node in nodes:
        # Check if all inputs are known constants
        all_const = all(inp in const_values for inp in node.inputs)

        if all_const and node.op_type in ("add", "mul", "sub", "div", "neg",
                                           "concat", "reshape", "transpose"):
            try:
                inputs = [const_values[inp] for inp in node.inputs]
                result = _evaluate(node, inputs)
                if result is not None:
                    out = node.outputs[0] if node.outputs else "const"
                    const_values[out] = result
                    # Replace with a constant node
                    kept.append(RinNode(
                        op_type="constant",
                        inputs=[],
                        outputs=[out],
                        attrs={"const_value": result},
                    ))
                    continue
            except Exception:
                logger.debug("Constant folding failed for node %s", node.op_type)

        kept.append(node)

    logger.debug("constant_fold: %d → %d nodes", len(nodes), len(kept))
    return ComputationGraph(kept)


def _evaluate(node: RinNode, inputs: List[np.ndarray]) -> Optional[np.ndarray]:
    """Evaluate a simple arithmetic node given constant inputs."""
    if node.op_type == "add":
        return inputs[0] + inputs[1] if len(inputs) >= 2 else None
    elif node.op_type == "mul":
        return inputs[0] * inputs[1] if len(inputs) >= 2 else None
    elif node.op_type == "sub":
        return inputs[0] - inputs[1] if len(inputs) >= 2 else None
    elif node.op_type == "div":
        return inputs[0] / np.maximum(inputs[1], 1e-10) if len(inputs) >= 2 else None
    elif node.op_type == "neg":
        return -inputs[0]
    elif node.op_type == "concat":
        axis = node.attrs.get("axis", 0)
        return np.concatenate(inputs, axis=axis)
    elif node.op_type == "reshape":
        shape = node.attrs.get("shape", None)
        if shape is not None:
            return inputs[0].reshape(shape)
        return inputs[0]
    elif node.op_type == "transpose":
        perm = node.attrs.get("perm", None)
        if perm is not None:
            return inputs[0].transpose(perm)
        return inputs[0].T
    return None


# ---------------------------------------------------------------------------
# Mode-specific optimisation
# ---------------------------------------------------------------------------


def optimize_for_mode(
    g: ComputationGraph, mode: str = "inference"
) -> ComputationGraph:
    """Apply a preset of optimisation passes suitable for *mode*.

    Parameters
    ----------
    g : ComputationGraph
        Input graph.
    mode : str
        One of ``'inference'``, ``'training'``, ``'memory'``.

    Returns
    -------
    ComputationGraph
        Optimised graph.
    """
    if mode == "inference":
        g = remove_noop(g)
        g = fuse_bn(g)
        g = constant_fold(g)
        g = g.optimize()
        logger.info("Optimised graph for inference (%d nodes)", len(g.nodes))
    elif mode == "training":
        # For training, keep dropout etc. but still fuse where safe
        g = constant_fold(g)
        g = g.optimize()
        logger.info("Optimised graph for training (%d nodes)", len(g.nodes))
    elif mode == "memory":
        g = remove_noop(g)
        g = constant_fold(g)
        g = g.optimize()
        logger.info("Optimised graph for memory (%d nodes)", len(g.nodes))
    else:
        logger.warning("Unknown optimise mode '%s'; returning graph unchanged", mode)

    return g
