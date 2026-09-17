"""Structured inference runner for the Stage 03 output contract.

Purpose:
    Combine the frozen Stage 03 prompt, the Stage 02 generic runner, and the
    strict v1 output contract into per-sample records. Every sample returns a
    record: generation problems become generation_failure records instead of
    raised exceptions, so one bad sample never aborts a batch. Parsing is the
    strict contract parser; no field is ever repaired silently.

Startup command:
    This module is a library imported by the Stage 03 smoke CLI and tests.
    A real inference run needs the local model; parse behavior can be checked
    from the workspace root with a stubbed generation:
    ``conda activate autovla_codeclean && PYTHONPATH=DriveAlign/src python -c \
    "from drivealign.inference.structured_runner import SampleRequest; print(SampleRequest('demo', 'x.jpg'))"``
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from drivealign.contracts.output import (
    DEFAULT_SCHEMA_VERSION,
    OutputContractError,
    OutputErrorCategory,
    ParseResult,
    StructuredDrivingOutput,
    parse_structured_output,
)
from drivealign.contracts.prompt import PROMPT_VERSION, build_structured_prompt
from drivealign.inference.base_runner import GenerationResult, generate_one
from drivealign.inference.model_loader import LoadedModel


OK_STATUS = "ok"


@dataclass(frozen=True)
class SampleRequest:
    """One batch item: a traced sample id, its image, and causal inputs.

    ``prompt_version``/``schema_version`` default to the frozen pairing
    (prompt v3 + schema v2). Explicit values exist for reproducibility of
    earlier pairings, e.g. prompt v2 + schema v1 for the recorded S03 run.
    """

    sample_id: str
    image: str | Path
    available_speed: str | None = None
    image_size: tuple[float, float] | None = None
    prompt_version: str | None = None
    schema_version: str | None = None


@dataclass(frozen=True)
class SampleRecord:
    """Per-sample outcome with raw text, parse result, and categorized errors.

    ``status`` is ``ok`` or one ``OutputErrorCategory`` value. ``errors`` keeps
    every categorized failure so one sample id traces raw text, parse result,
    and validation errors together. ``normalized_terms``/``unmapped_terms``
    expose the v2 synonym-table rewrites; ``derived_positions`` holds the v2
    deterministic coarse positions per object index.
    """

    sample_id: str
    image: str
    prompt: str
    status: str
    parse_ok: bool
    raw_text: str | None
    output: StructuredDrivingOutput | None
    errors: tuple[OutputContractError, ...]
    generation_error: str | None
    telemetry: dict[str, Any] | None
    available_speed: str | None
    extracted_from_fences: bool = False
    prompt_version: str = PROMPT_VERSION
    schema_version: str = DEFAULT_SCHEMA_VERSION
    normalized_terms: tuple[tuple[str, str], ...] = ()
    unmapped_terms: tuple[str, ...] = ()
    derived_positions: tuple[str | None, ...] = ()


def _telemetry_to_dict(generation: GenerationResult) -> dict[str, Any]:
    return {
        "input_tokens": generation.input_tokens,
        "output_tokens": generation.output_tokens,
        "preprocess_seconds": generation.preprocess_seconds,
        "generation_seconds": generation.generation_seconds,
        "peak_cuda_memory_bytes": generation.peak_cuda_memory_bytes,
    }


def _make_record(
    request: SampleRequest,
    prompt: str,
    status: str,
    *,
    parse_ok: bool = False,
    raw_text: str | None = None,
    output: StructuredDrivingOutput | None = None,
    errors: tuple[OutputContractError, ...] = (),
    generation_error: str | None = None,
    telemetry: dict[str, Any] | None = None,
    extracted_from_fences: bool = False,
    schema_version: str | None = None,
    normalized_terms: tuple[tuple[str, str], ...] = (),
    unmapped_terms: tuple[str, ...] = (),
    derived_positions: tuple[str | None, ...] = (),
) -> SampleRecord:
    return SampleRecord(
        sample_id=request.sample_id,
        image=str(Path(request.image).expanduser()),
        prompt=prompt,
        status=status,
        parse_ok=parse_ok,
        raw_text=raw_text,
        output=output,
        errors=errors,
        generation_error=generation_error,
        telemetry=telemetry,
        available_speed=request.available_speed,
        extracted_from_fences=extracted_from_fences,
        prompt_version=request.prompt_version or PROMPT_VERSION,
        schema_version=schema_version or request.schema_version or DEFAULT_SCHEMA_VERSION,
        normalized_terms=normalized_terms,
        unmapped_terms=unmapped_terms,
        derived_positions=derived_positions,
    )


def run_structured_inference(
    loaded: LoadedModel,
    request: SampleRequest,
    *,
    generation_config: Mapping[str, Any] | None = None,
) -> SampleRecord:
    """Run one structured sample: prompt, generation, strict parse.

    Generation exceptions are downgraded to a generation_failure record with
    the exception summary; they are never raised to the caller.
    """
    prompt = build_structured_prompt(
        request.available_speed, version=request.prompt_version
    )
    try:
        generation = generate_one(
            loaded,
            request.image,
            prompt,
            generation_config=generation_config,
        )
    except Exception as exc:  # recorded per the frozen S03 taxonomy
        message = f"{type(exc).__name__}: {exc}"
        return _make_record(
            request,
            prompt,
            OutputErrorCategory.GENERATION_FAILURE.value,
            errors=(
                OutputContractError(
                    category=OutputErrorCategory.GENERATION_FAILURE,
                    message=message,
                ),
            ),
            generation_error=message,
        )

    result: ParseResult = parse_structured_output(
        generation.text,
        schema_version=request.schema_version,
        image_size=request.image_size,
    )
    status = OK_STATUS if result.ok else result.errors[0].category.value
    return _make_record(
        request,
        prompt,
        status,
        parse_ok=result.ok,
        raw_text=result.raw_text,
        output=result.output,
        errors=result.errors,
        telemetry=_telemetry_to_dict(generation),
        extracted_from_fences=result.extracted_from_fences,
        schema_version=result.schema_version,
        normalized_terms=result.normalized_terms,
        unmapped_terms=result.unmapped_terms,
        derived_positions=result.derived_positions,
    )


def run_structured_batch(
    loaded: LoadedModel,
    requests: Sequence[SampleRequest],
    *,
    generation_config: Mapping[str, Any] | None = None,
) -> list[SampleRecord]:
    """Run every sample and return one record per sample, never aborting.

    Any unexpected exception while handling one sample is recorded as a
    generation_failure for that sample; remaining samples still run.
    """
    records: list[SampleRecord] = []
    for request in requests:
        try:
            records.append(
                run_structured_inference(
                    loaded,
                    request,
                    generation_config=generation_config,
                )
            )
        except Exception as exc:  # defensive: batch must continue
            message = f"{type(exc).__name__}: {exc}"
            records.append(
                _make_record(
                    request,
                    build_structured_prompt(
                        request.available_speed, version=request.prompt_version
                    ),
                    OutputErrorCategory.GENERATION_FAILURE.value,
                    errors=(
                        OutputContractError(
                            category=OutputErrorCategory.GENERATION_FAILURE,
                            message=message,
                        ),
                    ),
                    generation_error=message,
                )
            )
    return records


def record_to_dict(record: SampleRecord) -> dict[str, Any]:
    """Serialize one record for JSON/JSONL persistence and error reports."""
    return {
        "sample_id": record.sample_id,
        "image": record.image,
        "prompt": record.prompt,
        "prompt_version": record.prompt_version,
        "schema_version": record.schema_version,
        "available_speed": record.available_speed,
        "status": record.status,
        "parse_ok": record.parse_ok,
        "extracted_from_fences": record.extracted_from_fences,
        "normalized_terms": [list(pair) for pair in record.normalized_terms],
        "unmapped_terms": list(record.unmapped_terms),
        "derived_coarse_positions": list(record.derived_positions),
        "raw_text": record.raw_text,
        "generation_error": record.generation_error,
        "output": record.output.to_dict() if record.output is not None else None,
        "errors": [
            {
                "category": error.category.value,
                "field": error.field,
                "message": error.message,
            }
            for error in record.errors
        ],
        "telemetry": record.telemetry,
    }


def make_blank_image(image: str | Path, out_path: str | Path) -> Path:
    """Create a same-size mid-gray image that destroys all scene content."""
    from PIL import Image

    source = Path(image).expanduser()
    out = Path(out_path).expanduser()
    with Image.open(source) as opened:
        blank = Image.new("RGB", opened.size, (128, 128, 128))
    out.parent.mkdir(parents=True, exist_ok=True)
    blank.save(out)
    return out


def make_shuffled_image(
    image: str | Path,
    out_path: str | Path,
    *,
    seed: int = 0,
    grid: tuple[int, int] = (4, 4),
) -> Path:
    """Create a deterministic patch-shuffled image that breaks scene layout.

    The source is cut into ``grid`` patches and their positions are shuffled
    with a fixed seed, keeping colors and textures but destroying spatial
    semantics for the Stage 03 sanity check.
    """
    import random

    from PIL import Image

    source = Path(image).expanduser()
    out = Path(out_path).expanduser()
    cols, rows = grid
    if cols < 2 and rows < 2:
        raise ValueError("Shuffle grid must have at least two patches")
    with Image.open(source) as opened:
        rgb = opened.convert("RGB")
        width, height = rgb.size
        patch_w, patch_h = width // cols, height // rows
        patches = [
            (x * patch_w, y * patch_h, (x + 1) * patch_w, (y + 1) * patch_h)
            for y in range(rows)
            for x in range(cols)
        ]
        order = list(range(len(patches)))
        random.Random(seed).shuffle(order)
        canvas = Image.new("RGB", rgb.size)
        for destination, source_index in zip(patches, order):
            canvas.paste(rgb.crop(patches[source_index]), destination[:2])
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    return out
