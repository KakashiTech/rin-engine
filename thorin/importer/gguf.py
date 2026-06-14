"""GGUF model converter for the THOR format (experimental)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, Union

logger = logging.getLogger(__name__)


def convert_gguf_model(
    model_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    **kwargs: Any,
) -> "ThorGraph":
    raise ImportError(
        "GGUF support requires the ``gguf`` package.\n"
        "Install it with:  pip install gguf\n"
        "Note: GGUF import is experimental and may not work with all models."
    )
