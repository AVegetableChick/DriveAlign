"""Stage 06 serializer: model_inputs -> 1F/4F model requests (causal guard).

Purpose:
    The only bridge between a DriveAlignRecord and what the model ever sees.
    :func:`serialize` accepts ONLY the ``model_inputs`` namespace (enforced by
    a runtime type check), so future evidence stored in ``oracle_only``
    physically cannot reach the prompt, the processor inputs or the batch.
    Both input policies share the frozen Stage 03 v3 prompt (never rewritten)
    plus one causally available ego-speed line; 1F takes only the newest
    frame image, 4F takes all four (oldest -> newest). Machine-readable frame
    metadata (frame tokens, time offsets) travels outside the prompt text so
    the prompt stays identical across policies.

    Leak scanning: :func:`future_fingerprints` extracts the real future
    evidence strings from ``oracle_only`` (future sample tokens and their
    timestamps; no synthetic canary fields) and :func:`scan_future_fingerprints`
    reports which text payloads contain any of them. A clean pipeline must
    yield zero hits at the prompt, processor and batch levels.

Example launch command:
    ``conda activate autovla_codeclean && cd /root/autodl-tmp/drivealign_workspace && \
    PYTHONPATH=DriveAlign/src python -c \
    "from drivealign.records.record import FrameInput, ModelInputs; \
    from drivealign.records.serializer import InputPolicy, serialize, \
    to_processor_inputs, collate, future_fingerprints, scan_future_fingerprints; \
    frames = [FrameInput('samples/CAM_FRONT/n%d.jpg' % i, 'tok%d' % i, \
    1510000000000000 + i * 500000, (i - 3) * 0.5) for i in range(4)]; \
    mi = ModelInputs(frames=frames, ego_speed_mps=5.2); \
    reqs = [serialize(mi, p) for p in InputPolicy]; \
    batch = collate([to_processor_inputs(r) for r in reqs]); \
    assert reqs[0].prompt == reqs[1].prompt; \
    print(batch.canonical_hash())"``
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum

from drivealign.contracts.prompt import build_structured_prompt
from drivealign.contracts.versions import DEFAULT_CONTRACT_VERSION
from drivealign.records.record import DriveAlignRecord, ModelInputs

#: Decimals used for the ego-speed line of the prompt (matches the record).
SPEED_LINE_PRECISION = 3


class InputPolicy(str, Enum):
    """How many history frames the serializer exposes to the model."""

    ONE_FRAME = "1F"
    FOUR_FRAME = "4F"


@dataclass(frozen=True)
class ModelRequest:
    """Everything the model may see for one sample under one input policy.

    ``prompt`` is the frozen contract prompt (plus the causal ego-speed
    line); frame tokens and time offsets are machine-readable metadata that
    must never be rewritten into the prompt text.
    """

    policy: InputPolicy
    contract_version: str
    prompt: str
    image_relpaths: tuple[str, ...]  # oldest -> newest; len 1 for 1F
    frame_tokens: tuple[str, ...]  # aligned with image_relpaths
    time_offsets_s: tuple[float, ...]  # aligned with image_relpaths
    ego_speed_mps: float

    def to_dict(self) -> dict:
        return {
            "policy": self.policy.value,
            "contract_version": self.contract_version,
            "prompt": self.prompt,
            "image_relpaths": list(self.image_relpaths),
            "frame_tokens": list(self.frame_tokens),
            "time_offsets_s": list(self.time_offsets_s),
            "ego_speed_mps": self.ego_speed_mps,
        }

    def canonical_hash(self) -> str:
        """sha256 of the sorted-keys compact JSON; same input, same hash."""
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True)
class ProcessorInputs:
    """Minimal processor-level contract for one request.

    Carries the prompt plus image paths only; tensor-level vision processing
    (pixel blocks) lands with the Stage 07 training integration. Kept
    hashable so the smoke can scan it for future fingerprints.
    """

    prompt: str
    image_relpaths: tuple[str, ...]
    policy: str
    contract_version: str

    def to_dict(self) -> dict:
        return {
            "prompt": self.prompt,
            "image_relpaths": list(self.image_relpaths),
            "policy": self.policy,
            "contract_version": self.contract_version,
        }

    def canonical_hash(self) -> str:
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True)
class Batch:
    """Minimal collated batch: per-sample fields stacked in sample order."""

    prompts: tuple[str, ...]
    image_relpaths: tuple[tuple[str, ...], ...]
    policies: tuple[str, ...]
    contract_versions: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "prompts": list(self.prompts),
            "image_relpaths": [list(p) for p in self.image_relpaths],
            "policies": list(self.policies),
            "contract_versions": list(self.contract_versions),
        }

    def canonical_hash(self) -> str:
        return _canonical_hash(self.to_dict())


def serialize(model_inputs: ModelInputs, policy: InputPolicy) -> ModelRequest:
    """Derive the model request from ``model_inputs`` under ``policy``.

    The ``model_inputs``-only signature is the causal guard: the serializer
    has no way to reach ``training_targets`` or ``oracle_only``. The prompt
    is the frozen contract text plus the ego-speed line derived from the
    backward-difference speed; identical ``model_inputs`` therefore yield
    identical 1F and 4F prompts.
    """
    if not isinstance(model_inputs, ModelInputs):
        raise TypeError(
            "serialize accepts only drivealign.records.record.ModelInputs, "
            f"got {type(model_inputs).__name__}"
        )
    if policy is InputPolicy.ONE_FRAME:
        anchor = model_inputs.frames[-1]
        relpaths = (anchor.image_relpath,)
        tokens = (anchor.frame_token,)
        offsets = (anchor.time_offset_s,)
    else:
        relpaths = tuple(f.image_relpath for f in model_inputs.frames)
        tokens = tuple(f.frame_token for f in model_inputs.frames)
        offsets = tuple(f.time_offset_s for f in model_inputs.frames)
    prompt = build_structured_prompt(
        available_speed=(
            f"{model_inputs.ego_speed_mps:.{SPEED_LINE_PRECISION}f} m/s "
            "(backward difference from the previous keyframe)"
        )
    )
    return ModelRequest(
        policy=policy,
        contract_version=DEFAULT_CONTRACT_VERSION,
        prompt=prompt,
        image_relpaths=relpaths,
        frame_tokens=tokens,
        time_offsets_s=offsets,
        ego_speed_mps=model_inputs.ego_speed_mps,
    )


def to_processor_inputs(request: ModelRequest) -> ProcessorInputs:
    """Project a request onto the processor-level contract (lossless pass)."""
    return ProcessorInputs(
        prompt=request.prompt,
        image_relpaths=request.image_relpaths,
        policy=request.policy.value,
        contract_version=request.contract_version,
    )


def collate(inputs: Sequence[ProcessorInputs]) -> Batch:
    """Stack processor inputs into a minimal batch, preserving sample order."""
    return Batch(
        prompts=tuple(i.prompt for i in inputs),
        image_relpaths=tuple(i.image_relpaths for i in inputs),
        policies=tuple(i.policy for i in inputs),
        contract_versions=tuple(i.contract_version for i in inputs),
    )


def future_fingerprints(record: DriveAlignRecord) -> tuple[str, ...]:
    """Extract real future-evidence strings from ``oracle_only``.

    One fingerprint per future sample token and one per future timestamp;
    these are the only probe strings (no synthetic canary fields, so record
    bytes stay canonical).
    """
    fingerprints: list[str] = []
    for pose in record.oracle_only.future_ego_poses:
        fingerprints.append(pose.sample_token)
        fingerprints.append(str(pose.timestamp_us))
    return tuple(fingerprints)


def scan_future_fingerprints(
    fingerprints: Sequence[str],
    payloads: Mapping[str, str],
) -> list[str]:
    """Return the sorted names of payloads containing any fingerprint.

    An empty result means the payload layer is leak-free. Timestamps are
    scanned as their decimal string form, matching how they would appear if
    serialized into prompt or metadata text.
    """
    return sorted(
        name
        for name, text in payloads.items()
        if any(fingerprint in text for fingerprint in fingerprints)
    )


def _canonical_hash(payload: dict) -> str:
    """sha256 over sorted-keys compact JSON of ``payload``."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
