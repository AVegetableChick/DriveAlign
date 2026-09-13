"""Load the base Qwen2.5-VL model and its matching processor.

Purpose:
    Provide the single model-loading entry point used by DriveAlign inference
    stages, keeping the model and processor on the same local checkpoint.

Startup command:
    This module is imported by the Stage 01 smoke CLI and is not started
    directly. Run it with:
    ``conda activate autovla_codeclean && PYTHONPATH=DriveAlign/src python -m \
    drivealign.cli.smoke_model --config DriveAlign/configs/model/qwen25_vl_3b.yaml``
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch


@dataclass
class LoadedModel:
    """The model and processor loaded from one local model revision."""

    model: Any
    processor: Any
    model_path: Path
    dtype: torch.dtype


def _torch_dtype(value: str | torch.dtype) -> torch.dtype:
    if isinstance(value, torch.dtype):
        return value
    try:
        return getattr(torch, value.lower())
    except AttributeError as exc:
        raise ValueError(f"Unsupported torch dtype: {value}") from exc


def load_model_and_processor(config: Mapping[str, Any]) -> LoadedModel:
    """Load Qwen2.5-VL and AutoProcessor using one shared local path.

    The imports are intentionally inside the function so configuration and
    validation tools can still import DriveAlign without Transformers.
    """
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    model_config = config["model"]
    model_path = Path(model_config["path"]).expanduser()
    if not model_path.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {model_path}")

    dtype = _torch_dtype(model_config.get("dtype", "bfloat16"))
    processor_kwargs = dict(model_config.get("processor", {}))
    processor = AutoProcessor.from_pretrained(
        model_path,
        revision=model_config.get("revision"),
        local_files_only=True,
        **processor_kwargs,
    )

    load_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "local_files_only": True,
    }
    device_map = model_config.get("device_map")
    if device_map is not None:
        load_kwargs["device_map"] = device_map
    attn_implementation = model_config.get("attn_implementation")
    if attn_implementation is not None:
        load_kwargs["attn_implementation"] = attn_implementation
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        revision=model_config.get("revision"),
        **load_kwargs,
    )
    model.eval()

    return LoadedModel(
        model=model,
        processor=processor,
        model_path=model_path,
        dtype=dtype,
    )