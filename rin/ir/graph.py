"""Advanced computation graph representation for THOR models.

Provides ``ThorNode`` (a single operation) and ``ComputationGraph`` (a
composable graph of nodes) with ASCII visualisation, operator counting,
and optimisation passes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from rin.importer.base import WeightLayer


# ---------------------------------------------------------------------------
# Node types
# ---------------------------------------------------------------------------

OP_CATEGORIES: Dict[str, str] = {
    # linear / projection
    "gemv": "linear",
    "gemm": "linear",
    "linear": "linear",
    "matmul": "linear",
    "dense": "linear",
    # element-wise
    "relu": "activation",
    "gelu": "activation",
    "silu": "activation",
    "tanh": "activation",
    "sigmoid": "activation",
    "softmax": "activation",
    "layer_norm": "normalisation",
    "rms_norm": "normalisation",
    "batch_norm": "normalisation",
    # attention
    "attention": "attention",
    "multi_head_attention": "attention",
    "self_attention": "attention",
    "qkv_projection": "attention",
    # embedding
    "embedding": "embedding",
    "gather": "embedding",
    # utility
    "reshape": "utility",
    "transpose": "utility",
    "concat": "utility",
    "slice": "utility",
    "add": "utility",
    "mul": "utility",
    "identity": "utility",
    "dropout": "utility",
    "cast": "utility",
}


@dataclass
class ThorNode:
    """A single operation node in the computation graph.

    Attributes
    ----------
    op_type : str
        Operation type (e.g. ``'gemv'``, ``'relu'``, ``'attention'``).
    inputs : list of str
        Names of tensors consumed by this node.
    outputs : list of str
        Names of tensors produced by this node.
    weights : WeightLayer or None
        Associated quantised weight layer, if applicable.
    attrs : dict
        Extra attributes (e.g. ``{"axis": -1}``).
    """

    op_type: str
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    weights: Optional[WeightLayer] = None
    attrs: Dict[str, Any] = field(default_factory=dict)

    @property
    def category(self) -> str:
        """Return the high-level category of this node's operation."""
        return OP_CATEGORIES.get(self.op_type, "other")

    @property
    def flops_estimate(self) -> int:
        """Rough FLOPs estimate for this node.

        For linear layers: 2 * rows * cols  (multiply + add).
        For other ops: 0 (unknown / negligible).
        """
        if self.category == "linear" and self.weights is not None:
            r, c = self.weights.shape
            return 2 * r * c
        return 0


# ---------------------------------------------------------------------------
# Computation graph
# ---------------------------------------------------------------------------

class ComputationGraph:
    """A composable directed graph of ``ThorNode`` nodes.

    Parameters
    ----------
    nodes : iterable of ThorNode, optional
        Initial nodes.
    """

    def __init__(self, nodes: Optional[List[ThorNode]] = None) -> None:
        self.nodes: List[ThorNode] = nodes or []

    def add_node(self, node: ThorNode) -> None:
        """Append *node* to the graph."""
        self.nodes.append(node)

    def tensors(self) -> Set[str]:
        """Return the set of all tensor names referenced in the graph."""
        tens: Set[str] = set()
        for n in self.nodes:
            tens.update(n.inputs)
            tens.update(n.outputs)
        return tens

    def graph_inputs(self) -> List[str]:
        """Return tensor names that are consumed but never produced."""
        produced: Set[str] = set()
        consumed: Set[str] = set()
        for n in self.nodes:
            produced.update(n.outputs)
            consumed.update(n.inputs)
        return sorted(consumed - produced)

    def graph_outputs(self) -> List[str]:
        """Return tensor names that are produced but never consumed."""
        produced: Set[str] = set()
        consumed: Set[str] = set()
        for n in self.nodes:
            produced.update(n.outputs)
            consumed.update(n.inputs)
        return sorted(produced - consumed)

    # -- visualisation ----------------------------------------------------

    def visualize(self) -> str:
        """Render the graph as an ASCII string.

        Each node is shown as::

            [op_type]  inputs → outputs
        """
        lines: List[str] = []
        lines.append(f"ComputationGraph ({len(self.nodes)} nodes)")
        lines.append("=" * 60)

        for i, node in enumerate(self.nodes):
            in_str = ", ".join(node.inputs) if node.inputs else "(none)"
            out_str = ", ".join(node.outputs) if node.outputs else "(none)"
            cat = node.category
            flops = node.flops_estimate

            w_info = ""
            if node.weights is not None:
                r, c = node.weights.shape
                w_info = f"  W[{r}×{c}] scale={node.weights.scale:.4f}"

            flop_str = f"  ~{flops:,} FLOPs" if flops else ""

            lines.append(
                f"  [{i:3d}] {node.op_type:20s}  "
                f"{cat:15s}{w_info}{flop_str}"
            )
            lines.append(f"        inputs :  {in_str}")
            lines.append(f"        outputs:  {out_str}")

        # Aggregate by category
        from collections import Counter
        counts: Counter = Counter(n.category for n in self.nodes)
        lines.append("=" * 60)
        lines.append("Operator summary:")
        for cat, cnt in sorted(counts.items()):
            lines.append(f"  {cat:15s} : {cnt}")
        lines.append("-" * 60)
        lines.append(f"  {'total':15s} : {len(self.nodes)}")

        return "\n".join(lines)

    # -- operator counting ------------------------------------------------

    def count_ops(self) -> Dict[str, int]:
        """Return a dictionary mapping ``category → count``."""
        from collections import Counter
        return dict(Counter(n.category for n in self.nodes))

    def total_flops(self) -> int:
        """Return the sum of estimated FLOPs across all nodes."""
        return sum(n.flops_estimate for n in self.nodes)

    # -- optimisation -----------------------------------------------------

    def optimize(self) -> "ComputationGraph":
        """Run a standard set of fusion and elimination passes.

        Currently applies:
          1. Remove no-op identity nodes.
          2. Fuse consecutive reshapes.
          3. Eliminate duplicate transpose pairs.
        """
        g = self._remove_identity()
        g = g._fuse_reshapes()
        g = g._eliminate_transpose_pairs()
        return g

    def _remove_identity(self) -> "ComputationGraph":
        """Remove ``identity`` nodes, re-wiring inputs to outputs."""
        kept: List[ThorNode] = []
        # Map: identity_output → identity_input
        remap: Dict[str, str] = {}
        for node in self.nodes:
            if node.op_type == "identity" and len(node.inputs) == 1:
                for out in node.outputs:
                    remap[out] = node.inputs[0]
            else:
                kept.append(node)

        # Re-wire remaining nodes
        def _resolve(name: str) -> str:
            while name in remap:
                name = remap[name]
            return name

        for node in kept:
            node.inputs = [_resolve(t) for t in node.inputs]
        return ComputationGraph(kept)

    def _fuse_reshapes(self) -> "ComputationGraph":
        """Merge consecutive reshape nodes operating on the same tensor."""
        kept: List[ThorNode] = []
        reshape_map: Dict[str, str] = {}  # output → input of reshape chain
        i = 0
        while i < len(self.nodes):
            node = self.nodes[i]
            if node.op_type == "reshape" and len(node.inputs) == 1:
                out = node.outputs[0] if node.outputs else ""
                reshape_map[out] = node.inputs[0]
                # Skip subsequent reshapes that consume this output
                while (i + 1 < len(self.nodes)
                       and self.nodes[i + 1].op_type == "reshape"
                       and self.nodes[i + 1].inputs
                       and self.nodes[i + 1].inputs[0] == out):
                    i += 1
                    inner = self.nodes[i]
                    inner_out = inner.outputs[0] if inner.outputs else ""
                    reshape_map[inner_out] = reshape_map.get(out, node.inputs[0])
                    out = inner_out
                # Keep a single fused reshape
                fused = ThorNode(
                    op_type="reshape",
                    inputs=[reshape_map[out]],
                    outputs=[out],
                    attrs=node.attrs,
                )
                kept.append(fused)
            else:
                kept.append(node)
            i += 1

        # Re-wire
        def _resolve(name: str) -> str:
            while name in reshape_map:
                name = reshape_map[name]
            return name

        for node in kept:
            node.inputs = [_resolve(t) for t in node.inputs]
        return ComputationGraph(kept)

    def _eliminate_transpose_pairs(self) -> "ComputationGraph":
        """Remove ``transpose; transpose`` pairs that cancel out."""
        kept: List[ThorNode] = []
        skip: Set[int] = set()
        for i, node in enumerate(self.nodes):
            if i in skip:
                continue
            if (node.op_type == "transpose"
                    and i + 1 < len(self.nodes)
                    and self.nodes[i + 1].op_type == "transpose"):
                # Check if the second transpose uses the first's output
                nxt = self.nodes[i + 1]
                if (node.outputs and nxt.inputs
                        and nxt.inputs[0] == node.outputs[0]):
                    # They cancel: wire the first input to the second output
                    if nxt.outputs and node.inputs:
                        kept.append(ThorNode(
                            op_type="identity",
                            inputs=list(node.inputs),
                            outputs=list(nxt.outputs),
                        ))
                    skip.add(i + 1)
                    continue
            kept.append(node)
        return ComputationGraph(kept)
