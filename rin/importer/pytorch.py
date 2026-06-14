"""PyTorch model converter for the RIN format.

Supports:
  - ``nn.Sequential`` with ``nn.Linear`` layers   -> MLP architecture
  - GPT-2 / GPT-Neo / LLaMA / BERT-style models   -> Transformer architecture
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


def _import_torch():
    """Lazy-import PyTorch; raises ImportError with a helpful message if missing."""
    try:
        import torch
        import torch.nn as nn
        return torch, nn
    except ImportError as exc:
        raise ImportError(
            "PyTorch is not available.  Install it with:  pip install torch\n"
            f"Original error: {exc}"
        ) from exc


def convert_pytorch_model(
    model_or_path: Any,
    output_path: Optional[Union[str, Path]] = None,
    **kwargs: Any,
) -> "RinGraph":
    """Convert a PyTorch model to RIN format.

    Parameters
    ----------
    model_or_path : nn.Module or str or Path
        Either a loaded PyTorch module or a path to a ``.pt`` / ``.pth`` file.
    output_path : str or Path, optional
        If given, the resulting ``.rin`` file is written here.
    **kwargs
        Passed through to architecture-specific converters.

    Returns
    -------
    RinGraph
        Populated graph ready for serialisation.
    """
    torch, nn = _import_torch()

    if isinstance(model_or_path, (str, Path)):
        model = torch.load(str(model_or_path), map_location="cpu",
                           weights_only=False)
    else:
        model = model_or_path

    model.eval()

    arch = _detect_architecture(model, nn)
    if arch == "mlp":
        return _convert_mlp(model, output_path, torch, nn, **kwargs)
    elif arch == "transformer":
        return _convert_transformer(model, output_path, torch, nn, **kwargs)
    else:
        raise ValueError(f"Unsupported architecture type: {arch}")


# ---------------------------------------------------------------------------
# Architecture detection
# ---------------------------------------------------------------------------

def _detect_architecture(model: Any, nn: Any) -> str:
    """Return ``'mlp'`` or ``'transformer'`` based on module structure."""

    if isinstance(model, nn.Sequential):
        return "mlp"

    if hasattr(model, "config"):
        cfg = model.config
        model_type = getattr(cfg, "model_type", "").lower()
        if model_type in ("gpt2", "gpt_neo", "gptj", "llama", "bert",
                          "roberta", "opt", "bloom", "mistral", "falcon",
                          "transformer"):
            return "transformer"
        archs = getattr(cfg, "architectures", None) or []
        if any("ForCausalLM" in a or "Attention" in a for a in archs):
            return "transformer"
        if hasattr(cfg, "hidden_size") and hasattr(cfg, "num_attention_heads"):
            return "transformer"

    for name, _ in model.named_modules():
        lower = name.lower()
        if any(k in lower for k in ("attention", "self_attn", "encoder.block",
                                    "transformer.h")):
            return "transformer"

    for _ in model.modules():
        if isinstance(_, nn.Linear):
            return "mlp"

    raise ValueError(
        "Could not detect architecture.  The model must contain at least "
        "one nn.Linear layer."
    )


# ---------------------------------------------------------------------------
# MLP converter
# ---------------------------------------------------------------------------

def _convert_mlp(
    model: Any,
    output_path: Optional[str],
    torch: Any,
    nn: Any,
    **kwargs: Any,
) -> "RinGraph":
    from rin.importer.base import RinGraph, WeightLayer

    layers: List[WeightLayer] = []
    input_dim: Optional[int] = None
    linear_idx = 0

    for module in model.modules():
        if not isinstance(module, nn.Linear):
            continue

        w = module.weight.detach().cpu().numpy().astype(np.float32)
        b = (
            module.bias.detach().cpu().numpy().astype(np.float32)
            if module.bias is not None
            else np.zeros(w.shape[0], dtype=np.float32)
        )

        w = w.T
        rows, cols = w.shape
        quant_w, scale = RinGraph.quantize_weights(w.ravel())

        if input_dim is None:
            input_dim = rows

        layers.append(
            WeightLayer(
                name=f"linear_{linear_idx}",
                shape=(rows, cols),
                scale=scale,
                weights=quant_w,
                bias=b,
            )
        )
        linear_idx += 1

    if not layers:
        raise ValueError("No nn.Linear layers found in the model.")

    output_dim = layers[-1].cols

    meta: Dict[str, Any] = {
        "architecture": 0,
        "num_layers": len(layers),
        "input_dim": input_dim or 0,
        "output_dim": output_dim,
    }
    graph = RinGraph(meta)
    for layer in layers:
        graph.add_layer(layer)

    if output_path is not None:
        graph.save(str(output_path))
        logger.info("Saved MLP model to %s", output_path)

    return graph


# ---------------------------------------------------------------------------
# Transformer converter
# ---------------------------------------------------------------------------

def _convert_transformer(
    model: Any,
    output_path: Optional[str],
    torch: Any,
    nn: Any,
    **kwargs: Any,
) -> "RinGraph":
    from rin.importer.base import RinGraph, WeightLayer

    cfg = model.config if hasattr(model, "config") else None
    sd = model.state_dict()

    dim = _cfg(cfg, "hidden_size", "n_embd", "d_model", default=768)
    num_layers_meta = _cfg(cfg, "num_hidden_layers", "n_layer",
                           "num_layers", default=12)
    num_heads = _cfg(cfg, "num_attention_heads", "n_head", default=12)
    vocab_size = _cfg(cfg, "vocab_size", default=50257)
    max_seq_len = _cfg(cfg, "max_position_embeddings", "n_positions",
                       "max_seq_len", default=1024)
    ffn_dim = _cfg(cfg, "intermediate_size", "n_inner",
                   "ffn_dim", default=dim * 4)

    meta: Dict[str, Any] = {
        "architecture": 1,
        "num_layers": num_layers_meta,
        "dim": dim,
        "vocab_size": vocab_size,
        "num_heads": num_heads,
        "max_seq_len": max_seq_len,
        "ffn_dim": ffn_dim,
    }
    graph = RinGraph(meta)

    def _add_layer(
        name: str, w: np.ndarray, b: Optional[np.ndarray] = None
    ) -> None:
        w = np.ascontiguousarray(w, dtype=np.float32)
        if b is not None:
            b = np.ascontiguousarray(b, dtype=np.float32)
        else:
            b = np.zeros(
                w.shape[1] if w.ndim == 2 else 1, dtype=np.float32
            )

        if w.ndim == 2:
            if b.shape[0] == w.shape[0]:
                w = w.T
            rows, cols = w.shape
        else:
            rows, cols = w.shape[0], 1

        quant_w, scale = RinGraph.quantize_weights(w.ravel())
        graph.add_layer(
            WeightLayer(
                name=name, shape=(rows, cols), scale=scale,
                weights=quant_w, bias=b,
            )
        )

    # 1) token embedding
    emb_key = _find_key(sd, "embed", "wte", "tok_embeddings")
    if emb_key:
        _add_layer("embedding", sd[emb_key].cpu().numpy())

    # 2) per-layer attention + FFN
    for layer_idx in range(num_layers_meta):
        prefix = _find_layer_prefix(sd, layer_idx)
        if prefix is None:
            logger.warning(
                "Could not find prefix for layer %d, skipping", layer_idx
            )
            continue

        fused_key = f"{prefix}attn.c_attn.weight"
        if fused_key in sd:
            fused_w = sd[fused_key].cpu().numpy()
            fused_b = sd.get(fused_key.replace(".weight", ".bias"))
            fused_b = (
                fused_b.cpu().numpy()
                if fused_b is not None
                else np.zeros(dim, dtype=np.float32)
            )
            q_w, k_w, v_w = np.split(fused_w, 3, axis=0)
            q_b, k_b, v_b = np.split(fused_b, 3, axis=0)
            _add_layer(f"Wq_{layer_idx}", q_w, q_b)
            _add_layer(f"Wk_{layer_idx}", k_w, k_b)
            _add_layer(f"Wv_{layer_idx}", v_w, v_b)
        else:
            for tag, proj in [
                ("q", "q_proj"),
                ("k", "k_proj"),
                ("v", "v_proj"),
            ]:
                w_key = _find_key(
                    sd,
                    f"{prefix}attn.{proj}.weight",
                    f"{prefix}attention.{proj}.weight",
                    f"{prefix}self_attn.{proj}.weight",
                )
                if w_key:
                    w = sd[w_key].cpu().numpy()
                    b_key = w_key.replace(".weight", ".bias")
                    b = sd[b_key].cpu().numpy() if b_key in sd else None
                    _add_layer(f"W{tag}_{layer_idx}".upper(), w, b)

        o_key = _find_key(
            sd,
            f"{prefix}attn.out_proj.weight",
            f"{prefix}attn.c_proj.weight",
            f"{prefix}attention.out_proj.weight",
            f"{prefix}self_attn.out_proj.weight",
        )
        if o_key:
            w = sd[o_key].cpu().numpy()
            b_key = o_key.replace(".weight", ".bias")
            b = sd[b_key].cpu().numpy() if b_key in sd else None
            _add_layer(f"Wo_{layer_idx}", w, b)

        w1_key = _find_key(
            sd,
            f"{prefix}mlp.c_fc.weight",
            f"{prefix}mlp.fc1.weight",
            f"{prefix}mlp.up_proj.weight",
            f"{prefix}ffn.linear1.weight",
            f"{prefix}mlp.dense_h_to_4h.weight",
            f"{prefix}mlp.gate_proj.weight",
        )
        if w1_key:
            w = sd[w1_key].cpu().numpy()
            b_key = w1_key.replace(".weight", ".bias")
            b = sd[b_key].cpu().numpy() if b_key in sd else None
            _add_layer(f"W1_{layer_idx}", w, b)

        w2_key = _find_key(
            sd,
            f"{prefix}mlp.c_proj.weight",
            f"{prefix}mlp.fc2.weight",
            f"{prefix}mlp.down_proj.weight",
            f"{prefix}ffn.linear2.weight",
            f"{prefix}mlp.dense_4h_to_h.weight",
        )
        if w2_key:
            w = sd[w2_key].cpu().numpy()
            b_key = w2_key.replace(".weight", ".bias")
            b = sd[b_key].cpu().numpy() if b_key in sd else None
            _add_layer(f"W2_{layer_idx}", w, b)

    # 3) output / unembed
    out_key = _find_key(sd, "lm_head.weight", "embed_out.weight")
    if out_key is None and emb_key is not None:
        out_key = emb_key
    if out_key:
        w = sd[out_key].cpu().numpy()
        b_key = out_key.replace(".weight", ".bias")
        b = sd[b_key].cpu().numpy() if b_key in sd else None
        _add_layer("output", w, b)

    # 4) position embeddings
    pe_key = _find_key(sd, "position_embeddings.weight", "wpe.weight")
    if pe_key is None:
        pe_key = _find_key(sd, "embed_positions.weight")
    if pe_key:
        graph.metadata["position_embeddings"] = (
            sd[pe_key].cpu().numpy().astype(np.float32)
        )

    if output_path is not None:
        graph.save(str(output_path))
        logger.info("Saved Transformer model to %s", output_path)

    return graph


# ---------------------------------------------------------------------------
# internal helpers
# ---------------------------------------------------------------------------

def _cfg(config: Any, *names: str, default: Any = None) -> Any:
    if config is None:
        return default
    for name in names:
        val = getattr(config, name, None)
        if val is not None:
            return val
    return default


def _find_key(sd: Dict[str, Any], *candidates: str) -> Optional[str]:
    for key in candidates:
        if key in sd:
            return key
    return None


def _find_layer_prefix(
    sd: Dict[str, Any], layer_idx: int
) -> Optional[str]:
    templates = [
        f"transformer.h.{layer_idx}.",
        f"h.{layer_idx}.",
        f"model.layers.{layer_idx}.",
        f"layers.{layer_idx}.",
        f"encoder.layer.{layer_idx}.",
        f"transformer.layer.{layer_idx}.",
        f"decoder.layer.{layer_idx}.",
        f"decoder.layers.{layer_idx}.",
        f"bert.encoder.layer.{layer_idx}.",
    ]
    for prefix in templates:
        if any(k.startswith(prefix) for k in sd):
            return prefix
    return None
