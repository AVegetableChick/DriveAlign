"""Generic single-image Qwen2.5-VL inference runner.

Purpose:
    Convert one image and prompt into Qwen2.5-VL inputs, perform one greedy
    generation, and return raw text with inference telemetry. This module does
    not provide a standalone command or task-specific output parsing.

Startup command:
    Import it through the Stage 02 CLI with:
    ``conda activate autovla_codeclean && PYTHONPATH=DriveAlign/src python -m \
    drivealign.cli.infer_image --prompt "请描述图像中的道路场景。"``
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

import torch
from PIL import Image

from drivealign.inference.model_loader import LoadedModel


@dataclass
class GenerationResult:
    """Raw generated text and measurements for one inference call."""

    text: str
    input_tokens: int
    output_tokens: int
    preprocess_seconds: float
    generation_seconds: float
    peak_cuda_memory_bytes: int | None


def _validate_image(image: str | Path) -> Path:
    image_path = Path(image).expanduser()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file does not exist: {image_path}")
    try:
        with Image.open(image_path) as opened_image:
            opened_image.verify()
    except (OSError, ValueError) as exc:
        raise ValueError(f"Image cannot be read: {image_path}") from exc
    return image_path


def _model_device(model: Any) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration as exc:
        raise RuntimeError("Model has no parameters to determine its device") from exc


def generate_one(
    loaded: LoadedModel,
    image: str | Path,
    prompt: str,
    generation_config: Mapping[str, Any] | None = None,
) -> GenerationResult:
    """Generate one response for an image and prompt.

    This runner returns raw model text and telemetry only. Task-specific parsing
    and structured output validation belong to later stages.
    """
    image_path = _validate_image(image)
    if not prompt or not prompt.strip():
        raise ValueError("Prompt must not be empty")

    from qwen_vl_utils import process_vision_info

    preprocess_started = perf_counter()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = loaded.processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = loaded.processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    preprocess_seconds = perf_counter() - preprocess_started

    device = _model_device(loaded.model)
    inputs = {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }
    input_tokens = int(inputs["input_ids"].shape[-1])

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    generation_started = perf_counter()
    generate_kwargs = dict(generation_config or {})
    generate_kwargs["do_sample"] = False
    with torch.inference_mode():
        generated_ids = loaded.model.generate(
            **inputs,
            **generate_kwargs,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    generation_seconds = perf_counter() - generation_started

    generated_tokens = generated_ids[:, input_tokens:]
    output_tokens = int(generated_tokens.shape[-1])
    generated_text = loaded.processor.batch_decode(
        generated_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    peak_memory = (
        torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
    )
    return GenerationResult(
        text=generated_text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        preprocess_seconds=preprocess_seconds,
        generation_seconds=generation_seconds,
        peak_cuda_memory_bytes=peak_memory,
    )