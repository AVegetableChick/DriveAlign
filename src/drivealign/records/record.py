"""DriveAlignRecord v1: the standard temporal sample unit (Stage 06).

Purpose:
    Define the frozen v1 record contract shared by training and evaluation.
    A record is the complete per-sample archive (all namespaces), while the
    model only ever sees what the Stage 06 serializer derives from the
    ``model_inputs`` namespace, so the namespace split is the causal guard:

    - ``model_inputs``: everything the model may see (four CAM_FRONT
      keyframes oldest -> newest plus the backward-difference ego speed).
    - ``training_targets``: final supervision labels only (filled by S07/S08).
    - ``oracle_only``: future evidence used for evaluation GT and leak
      scanning (future ego poses along ``sample.next``); must never reach
      prompt, processor or batch.

    The canonical hash is sha256 over the sorted-keys compact JSON of the
    full record (no excluded fields). Floats are rounded at construction
    time (time offsets 6 decimals, speed/pose values 3 decimals) so the same
    inputs and configuration always yield identical bytes and hashes.

Example launch command:
    ``conda activate autovla_codeclean && cd /root/autodl-tmp/drivealign_workspace && \
    PYTHONPATH=DriveAlign/src python -c \
    "from drivealign.records.record import DriveAlignRecord, validate_record; \
    f = [{'image_relpath': f'samples/CAM_FRONT/n{i}.jpg', 'frame_token': f't{i}', \
    'timestamp_us': 1510000000000000 + i * 500000, 'time_offset_s': (i - 3) * 0.5} \
    for i in range(4)]; \
    d = {'record_version': 'v1', 'sample_token': 't3', 'scene_token': 's0', \
    'model_inputs': {'frames': f, 'ego_speed_mps': 5.2}, \
    'training_targets': {'expected_output': None, 'language_reference': None}, \
    'oracle_only': {'future_ego_poses': []}, \
    'provenance': {'contract_version': 'v3', 'temporal_policy_version': 'v1', \
    'builder_commit': 'deadbeef', 'nuscenes_version': 'v1.0-mini'}}; \
    r = DriveAlignRecord.from_dict(d); \
    print(validate_record(r), r.canonical_hash())"``
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from drivealign.contracts.versions import AVAILABLE_CONTRACT_VERSIONS

RECORD_VERSION = "v1"
TEMPORAL_POLICY_VERSION = "v1"
NUM_HISTORY_FRAMES = 4

# Fixed float precision so canonical bytes stay stable across runs.
OFFSET_PRECISION = 6  # decimals for time_offset_s (matches nuscenes_io)
SPEED_PRECISION = 3  # decimals for ego_speed_mps
POSE_PRECISION = 3  # decimals for future x, y, yaw

_TOP_LEVEL_KEYS = frozenset(
    {
        "record_version",
        "sample_token",
        "scene_token",
        "model_inputs",
        "training_targets",
        "oracle_only",
        "provenance",
    }
)
_MODEL_INPUTS_KEYS = frozenset({"frames", "ego_speed_mps"})
_FRAME_KEYS = frozenset(
    {"image_relpath", "frame_token", "timestamp_us", "time_offset_s"}
)
_TARGET_KEYS = frozenset({"expected_output", "language_reference"})
_ORACLE_KEYS = frozenset({"future_ego_poses"})
_POSE_KEYS = frozenset({"sample_token", "timestamp_us", "x", "y", "yaw"})
_PROVENANCE_KEYS = frozenset(
    {
        "contract_version",
        "temporal_policy_version",
        "builder_commit",
        "nuscenes_version",
    }
)


@dataclass(frozen=True)
class FrameInput:
    """One CAM_FRONT keyframe of model_inputs.frames (oldest -> newest)."""

    image_relpath: str  # relative to the nuScenes dataroot
    frame_token: str  # keyframe sample token
    timestamp_us: int
    time_offset_s: float  # relative to the anchor (newest) frame; anchor 0.0

    def __post_init__(self):
        object.__setattr__(
            self, "time_offset_s", round(float(self.time_offset_s), OFFSET_PRECISION)
        )


@dataclass(frozen=True)
class ModelInputs:
    """Everything the model may see; the serializer's only input."""

    frames: tuple[FrameInput, ...]
    ego_speed_mps: float  # backward difference over the past 0.5 s

    def __post_init__(self):
        object.__setattr__(self, "frames", tuple(self.frames))
        object.__setattr__(
            self, "ego_speed_mps", round(float(self.ego_speed_mps), SPEED_PRECISION)
        )


@dataclass(frozen=True)
class TrainingTargets:
    """Final supervision labels; filled by Stage 07/08, None until then.

    Both values must be treated as immutable once set.
    """

    expected_output: dict | None = None
    language_reference: dict | None = None


@dataclass(frozen=True)
class FutureEgoPose:
    """One future ego pose along sample.next (evaluation GT evidence)."""

    sample_token: str
    timestamp_us: int
    x: float  # global frame, meters
    y: float  # global frame, meters
    yaw: float  # global frame, radians

    def __post_init__(self):
        for name in ("x", "y", "yaw"):
            object.__setattr__(
                self, name, round(float(getattr(self, name)), POSE_PRECISION)
            )


@dataclass(frozen=True)
class OracleOnly:
    """Future evidence for evaluation GT and leak scanning; never model input."""

    future_ego_poses: tuple[FutureEgoPose, ...]


@dataclass(frozen=True)
class Provenance:
    """Version traceability; the record's only versioning surface."""

    contract_version: str  # binds prompt + schema, see contracts/versions.py
    temporal_policy_version: str  # 1F/4F policy taxonomy version
    builder_commit: str  # git commit of the repo that built this record
    nuscenes_version: str  # e.g. "v1.0-mini"


@dataclass(frozen=True)
class DriveAlignRecord:
    """Stage 06 sample unit: exactly seven top-level namespaces."""

    record_version: str
    sample_token: str
    scene_token: str
    model_inputs: ModelInputs
    training_targets: TrainingTargets
    oracle_only: OracleOnly
    provenance: Provenance

    def to_dict(self) -> dict:
        """Plain-dict form with exactly the seven top-level keys."""
        return {
            "record_version": self.record_version,
            "sample_token": self.sample_token,
            "scene_token": self.scene_token,
            "model_inputs": {
                "frames": [
                    {
                        "image_relpath": f.image_relpath,
                        "frame_token": f.frame_token,
                        "timestamp_us": f.timestamp_us,
                        "time_offset_s": f.time_offset_s,
                    }
                    for f in self.model_inputs.frames
                ],
                "ego_speed_mps": self.model_inputs.ego_speed_mps,
            },
            "training_targets": {
                "expected_output": self.training_targets.expected_output,
                "language_reference": self.training_targets.language_reference,
            },
            "oracle_only": {
                "future_ego_poses": [
                    {
                        "sample_token": p.sample_token,
                        "timestamp_us": p.timestamp_us,
                        "x": p.x,
                        "y": p.y,
                        "yaw": p.yaw,
                    }
                    for p in self.oracle_only.future_ego_poses
                ]
            },
            "provenance": {
                "contract_version": self.provenance.contract_version,
                "temporal_policy_version": self.provenance.temporal_policy_version,
                "builder_commit": self.provenance.builder_commit,
                "nuscenes_version": self.provenance.nuscenes_version,
            },
        }

    def canonical_hash(self) -> str:
        """sha256 of the sorted-keys compact JSON of the full record."""
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DriveAlignRecord:
        """Rebuild a record from its plain-dict form (strict structural check).

        Enforces the exact key whitelist of every namespace, the v1 record
        version, and exactly four history frames. Semantic consistency (time
        ordering, offsets, leak-free namespaces) is checked by
        :func:`validate_record`.
        """
        _check_keys(data, _TOP_LEVEL_KEYS, "record")
        if data["record_version"] != RECORD_VERSION:
            raise ValueError(
                f"record_version must be {RECORD_VERSION!r}, "
                f"got {data['record_version']!r}"
            )
        sample_token = _require_str(data["sample_token"], "sample_token")
        scene_token = _require_str(data["scene_token"], "scene_token")

        model_inputs_raw = data["model_inputs"]
        _check_keys(model_inputs_raw, _MODEL_INPUTS_KEYS, "model_inputs")
        raw_frames = model_inputs_raw["frames"]
        if not isinstance(raw_frames, (list, tuple)):
            raise ValueError(
                f"model_inputs.frames must be a list, got {type(raw_frames).__name__}"
            )
        if len(raw_frames) != NUM_HISTORY_FRAMES:
            raise ValueError(
                f"model_inputs.frames must contain exactly {NUM_HISTORY_FRAMES} "
                f"frames, got {len(raw_frames)}"
            )
        frames = []
        for i, raw in enumerate(raw_frames):
            ctx = f"model_inputs.frames[{i}]"
            _check_keys(raw, _FRAME_KEYS, ctx)
            frames.append(
                FrameInput(
                    image_relpath=_require_str(
                        raw["image_relpath"], f"{ctx}.image_relpath"
                    ),
                    frame_token=_require_str(raw["frame_token"], f"{ctx}.frame_token"),
                    timestamp_us=_require_int(raw["timestamp_us"], f"{ctx}.timestamp_us"),
                    time_offset_s=_require_float(
                        raw["time_offset_s"], f"{ctx}.time_offset_s"
                    ),
                )
            )
        ego_speed_mps = _require_float(
            model_inputs_raw["ego_speed_mps"], "model_inputs.ego_speed_mps"
        )
        model_inputs = ModelInputs(frames=tuple(frames), ego_speed_mps=ego_speed_mps)

        targets_raw = data["training_targets"]
        _check_keys(targets_raw, _TARGET_KEYS, "training_targets")
        for name in ("expected_output", "language_reference"):
            value = targets_raw[name]
            if value is not None and not isinstance(value, dict):
                raise ValueError(
                    f"training_targets.{name} must be a dict or None, "
                    f"got {type(value).__name__}"
                )
        training_targets = TrainingTargets(
            expected_output=targets_raw["expected_output"],
            language_reference=targets_raw["language_reference"],
        )

        oracle_raw = data["oracle_only"]
        _check_keys(oracle_raw, _ORACLE_KEYS, "oracle_only")
        raw_poses = oracle_raw["future_ego_poses"]
        if not isinstance(raw_poses, (list, tuple)):
            raise ValueError(
                f"oracle_only.future_ego_poses must be a list, "
                f"got {type(raw_poses).__name__}"
            )
        poses = []
        for i, raw in enumerate(raw_poses):
            ctx = f"oracle_only.future_ego_poses[{i}]"
            _check_keys(raw, _POSE_KEYS, ctx)
            poses.append(
                FutureEgoPose(
                    sample_token=_require_str(raw["sample_token"], f"{ctx}.sample_token"),
                    timestamp_us=_require_int(raw["timestamp_us"], f"{ctx}.timestamp_us"),
                    x=_require_float(raw["x"], f"{ctx}.x"),
                    y=_require_float(raw["y"], f"{ctx}.y"),
                    yaw=_require_float(raw["yaw"], f"{ctx}.yaw"),
                )
            )
        oracle_only = OracleOnly(future_ego_poses=tuple(poses))

        provenance_raw = data["provenance"]
        _check_keys(provenance_raw, _PROVENANCE_KEYS, "provenance")
        provenance = Provenance(
            contract_version=_require_str(
                provenance_raw["contract_version"], "provenance.contract_version"
            ),
            temporal_policy_version=_require_str(
                provenance_raw["temporal_policy_version"],
                "provenance.temporal_policy_version",
            ),
            builder_commit=_require_str(
                provenance_raw["builder_commit"], "provenance.builder_commit"
            ),
            nuscenes_version=_require_str(
                provenance_raw["nuscenes_version"], "provenance.nuscenes_version"
            ),
        )

        return cls(
            record_version=data["record_version"],
            sample_token=sample_token,
            scene_token=scene_token,
            model_inputs=model_inputs,
            training_targets=training_targets,
            oracle_only=oracle_only,
            provenance=provenance,
        )


def _check_keys(mapping: Any, expected: frozenset, ctx: str) -> None:
    """Raise ValueError unless ``mapping`` has exactly ``expected`` keys."""
    if not isinstance(mapping, Mapping):
        raise ValueError(f"{ctx} must be a mapping, got {type(mapping).__name__}")
    keys = set(mapping)
    missing = sorted(expected - keys)
    extra = sorted(keys - expected)
    if missing or extra:
        raise ValueError(f"{ctx} key mismatch; missing={missing}, unexpected={extra}")


def _require_str(value: Any, ctx: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{ctx} must be a non-empty string, got {value!r}")
    return value


def _require_int(value: Any, ctx: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{ctx} must be an int, got {type(value).__name__}")
    return value


def _require_float(value: Any, ctx: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{ctx} must be a number, got {type(value).__name__}")
    return float(value)


def validate_record(record: DriveAlignRecord) -> list[str]:
    """Return every contract violation found; an empty list means valid."""
    problems: list[str] = []

    if record.record_version != RECORD_VERSION:
        problems.append(
            f"record_version must be {RECORD_VERSION!r}, got {record.record_version!r}"
        )
    if not (isinstance(record.sample_token, str) and record.sample_token):
        problems.append("sample_token must be a non-empty string")
    if not (isinstance(record.scene_token, str) and record.scene_token):
        problems.append("scene_token must be a non-empty string")

    frames = record.model_inputs.frames
    ts_last = None
    if len(frames) != NUM_HISTORY_FRAMES:
        problems.append(
            f"model_inputs.frames must contain exactly {NUM_HISTORY_FRAMES} "
            f"frames, got {len(frames)}"
        )
    else:
        ts_last = frames[-1].timestamp_us
        for i, frame in enumerate(frames):
            if not (isinstance(frame.image_relpath, str) and frame.image_relpath):
                problems.append(
                    f"frames[{i}].image_relpath must be a non-empty string"
                )
            if not (isinstance(frame.frame_token, str) and frame.frame_token):
                problems.append(f"frames[{i}].frame_token must be a non-empty string")
            if not math.isfinite(frame.time_offset_s):
                problems.append(f"frames[{i}].time_offset_s must be finite")
        for i, (older, newer) in enumerate(zip(frames, frames[1:])):
            if newer.timestamp_us <= older.timestamp_us:
                problems.append(
                    f"frames[{i + 1}].timestamp_us must be strictly greater "
                    f"than frames[{i}]"
                )
        frame_tokens = [f.frame_token for f in frames]
        if len(set(frame_tokens)) != len(frame_tokens):
            problems.append("model_inputs.frames frame_token values must be unique")
        if frames[-1].frame_token != record.sample_token:
            problems.append(
                "frames[-1].frame_token must equal the anchor sample_token "
                f"({record.sample_token!r})"
            )
        for i, frame in enumerate(frames):
            expected_offset = round(
                (frame.timestamp_us - ts_last) / 1e6, OFFSET_PRECISION
            )
            if frame.time_offset_s != expected_offset:
                problems.append(
                    f"frames[{i}].time_offset_s must equal "
                    f"(timestamp_us - anchor timestamp)/1e6 rounded to "
                    f"{OFFSET_PRECISION} decimals"
                )

    speed = record.model_inputs.ego_speed_mps
    if not math.isfinite(speed):
        problems.append("model_inputs.ego_speed_mps must be finite")
    elif speed < 0:
        problems.append("model_inputs.ego_speed_mps must be non-negative")

    for name in ("expected_output", "language_reference"):
        value = getattr(record.training_targets, name)
        if value is not None and not isinstance(value, dict):
            problems.append(f"training_targets.{name} must be a dict or None")

    history_tokens = {f.frame_token for f in frames}
    poses = record.oracle_only.future_ego_poses
    previous_ts = ts_last
    for i, pose in enumerate(poses):
        if not (isinstance(pose.sample_token, str) and pose.sample_token):
            problems.append(
                f"future_ego_poses[{i}].sample_token must be a non-empty string"
            )
        if not (
            math.isfinite(pose.x) and math.isfinite(pose.y) and math.isfinite(pose.yaw)
        ):
            problems.append(f"future_ego_poses[{i}] x/y/yaw must be finite")
        if previous_ts is not None:
            if pose.timestamp_us <= previous_ts:
                problems.append(
                    "future_ego_poses timestamps must be strictly increasing "
                    "and later than the anchor frame"
                )
            previous_ts = pose.timestamp_us
    pose_tokens = [p.sample_token for p in poses]
    if len(set(pose_tokens)) != len(pose_tokens):
        problems.append(
            "oracle_only.future_ego_poses sample_token values must be unique"
        )
    leaked = sorted(history_tokens & set(pose_tokens))
    if leaked:
        problems.append(
            "oracle_only.future_ego_poses sample_token must not overlap the "
            "history frames: " + ", ".join(leaked)
        )

    provenance = record.provenance
    if provenance.contract_version not in AVAILABLE_CONTRACT_VERSIONS:
        problems.append(
            "provenance.contract_version must be one of: "
            + ", ".join(AVAILABLE_CONTRACT_VERSIONS)
        )
    if provenance.temporal_policy_version != TEMPORAL_POLICY_VERSION:
        problems.append(
            f"provenance.temporal_policy_version must be "
            f"{TEMPORAL_POLICY_VERSION!r}, got "
            f"{provenance.temporal_policy_version!r}"
        )
    if not (isinstance(provenance.builder_commit, str) and provenance.builder_commit):
        problems.append("provenance.builder_commit must be a non-empty string")
    if not (
        isinstance(provenance.nuscenes_version, str) and provenance.nuscenes_version
    ):
        problems.append("provenance.nuscenes_version must be a non-empty string")

    return problems
