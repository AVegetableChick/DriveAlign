"""Stage 06 adapter: build a DriveAlignRecord v1 from a nuScenes anchor.

Purpose:
    Turn one anchor sample into the frozen v1 record contract by reusing the
    Stage 04 keyframe-chain reader (``read_keyframe_chain``). The adapter is
    the only producer of :class:`DriveAlignRecord` instances and never raises
    for data-level problems: every failure returns a stable quarantine reason
    code so smoke runs can categorize outcomes deterministically.

    Derivations (past data only, so ``model_inputs`` stays causal):
    - ``model_inputs.ego_speed_mps``: planar (x, y) backward difference of the
      anchor and previous keyframe global ego poses divided by their actual
      time delta (~0.5 s at the 2 Hz keyframe rate).
    - ``oracle_only.future_ego_poses``: global ego poses along ``sample.next``
      for ``NUM_FUTURE_POSES`` steps (3.0 s), truncated naturally at the scene
      end (scene end is NOT a quarantine condition).

    Quarantine taxonomy: the six ``nuscenes_io`` chain codes
    (INSUFFICIENT_HISTORY / CROSS_SCENE / NON_INCREASING_TIME / MISSING_IMAGE
    / BAD_GAP / NO_CAM_FRONT) plus two adapter codes:
    - ``MISSING_CALIBRATION``: ego_pose / calibrated_sensor of a touched
      frame cannot be resolved (dangling token or degenerate quaternion).
    - ``NUMERIC_ANOMALY``: backward-difference speed or a future pose
      coordinate is non-finite. A non-increasing future timestamp chain
      reuses ``NON_INCREASING_TIME``.

    The built record is re-checked with ``validate_record`` before being
    returned; a failure there means the derivation itself is buggy, so the
    adapter raises ValueError instead of quarantining.

Example launch command:
    ``conda activate autovla_codeclean && cd /root/autodl-tmp/drivealign_workspace && \
    PYTHONPATH=DriveAlign/src python -c \
    "from pathlib import Path; \
    from drivealign.data.nuscenes_io import load_nuscenes, resolve_anchor; \
    from drivealign.records.adapter import build_record; \
    nusc = load_nuscenes(Path('data/nuscenes/mini')); \
    token = resolve_anchor(nusc); \
    result = build_record(nusc, token, builder_commit='deadbeef'); \
    print(result.reason_code if result.record is None \
    else result.record.canonical_hash())"``
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion

from drivealign.contracts.versions import DEFAULT_CONTRACT_VERSION
from drivealign.data.nuscenes_io import (
    NO_CAM_FRONT,
    NON_INCREASING_TIME,
    ChainResult,
    read_keyframe_chain,
)
from drivealign.records.record import (
    RECORD_VERSION,
    TEMPORAL_POLICY_VERSION,
    DriveAlignRecord,
    FrameInput,
    FutureEgoPose,
    ModelInputs,
    OracleOnly,
    Provenance,
    TrainingTargets,
    validate_record,
)

# Quarantine codes added on top of the nuscenes_io chain codes.
MISSING_CALIBRATION = "MISSING_CALIBRATION"
NUMERIC_ANOMALY = "NUMERIC_ANOMALY"

#: Future ego poses per record: 6 keyframes = 3.0 s at the 2 Hz keyframe rate.
NUM_FUTURE_POSES = 6


@dataclass(frozen=True)
class BuildResult:
    """Outcome of :func:`build_record`: a record or a quarantine code.

    ``record`` is None exactly when ``reason_code`` is not None; ``detail``
    carries scene / anomaly context for the experiment record.
    """

    record: Optional[DriveAlignRecord]
    reason_code: Optional[str]
    detail: Optional[str] = None


@dataclass(frozen=True)
class _EgoPose:
    """Global-frame ego pose resolved at one keyframe sample."""

    timestamp_us: int
    x: float
    y: float
    yaw: float


def build_record(
    nusc: NuScenes,
    anchor_token: str,
    builder_commit: str,
) -> BuildResult:
    """Build the v1 record for ``anchor_token`` or classify a quarantine.

    Reads the 4-frame CAM_FRONT history chain (anchor + 3 prevs), derives the
    backward-difference ego speed from the two most recent global ego poses,
    and collects up to ``NUM_FUTURE_POSES`` future ego poses along
    ``sample.next``. Returns BuildResult with ``record=None`` and a stable
    reason code on any data-level failure; raises ValueError only if the
    assembled record fails ``validate_record`` (derivation bug).
    """
    chain = read_keyframe_chain(nusc, anchor_token)
    if chain.reason_code is not None:
        return BuildResult(
            record=None, reason_code=chain.reason_code, detail=_chain_detail(chain)
        )
    assert chain.frames is not None and chain.scene_token is not None
    detail = _chain_detail(chain)

    frames = [
        FrameInput(
            image_relpath=frame.image_relpath,
            frame_token=frame.sample_token,
            timestamp_us=frame.timestamp_us,
            time_offset_s=frame.offset_s,
        )
        for frame in chain.frames
    ]

    # Global ego pose of every history frame (used for the backward speed).
    history_poses = []
    for frame in chain.frames:
        sample = nusc.get("sample", frame.sample_token)
        pose, reason = _resolve_ego_pose(nusc, sample)
        if pose is None:
            return BuildResult(record=None, reason_code=reason, detail=detail)
        history_poses.append(pose)

    anchor_pose = history_poses[-1]
    prev_pose = history_poses[-2]
    dt_s = (anchor_pose.timestamp_us - prev_pose.timestamp_us) / 1e6
    speed = math.hypot(anchor_pose.x - prev_pose.x, anchor_pose.y - prev_pose.y) / dt_s
    if not math.isfinite(speed) or speed < 0:
        return BuildResult(
            record=None,
            reason_code=NUMERIC_ANOMALY,
            detail=f"{detail} backward-difference speed={speed!r}",
        )

    # Future evidence: walk sample.next, truncated at the scene end.
    future: list[FutureEgoPose] = []
    next_token = nusc.get("sample", anchor_token)["next"]
    while next_token and len(future) < NUM_FUTURE_POSES:
        sample = nusc.get("sample", next_token)
        pose, reason = _resolve_ego_pose(nusc, sample)
        if pose is None:
            return BuildResult(record=None, reason_code=reason, detail=detail)
        if future and pose.timestamp_us <= future[-1].timestamp_us:
            return BuildResult(
                record=None,
                reason_code=NON_INCREASING_TIME,
                detail=f"{detail} future_ego_poses not strictly increasing",
            )
        if not all(map(math.isfinite, (pose.x, pose.y, pose.yaw))):
            return BuildResult(
                record=None,
                reason_code=NUMERIC_ANOMALY,
                detail=f"{detail} non-finite future pose sample={sample['token']}",
            )
        future.append(
            FutureEgoPose(
                sample_token=sample["token"],
                timestamp_us=pose.timestamp_us,
                x=pose.x,
                y=pose.y,
                yaw=pose.yaw,
            )
        )
        next_token = sample["next"]

    record = DriveAlignRecord(
        record_version=RECORD_VERSION,
        sample_token=anchor_token,
        scene_token=chain.scene_token,
        model_inputs=ModelInputs(frames=frames, ego_speed_mps=speed),
        training_targets=TrainingTargets(),
        oracle_only=OracleOnly(future_ego_poses=tuple(future)),
        provenance=Provenance(
            contract_version=DEFAULT_CONTRACT_VERSION,
            temporal_policy_version=TEMPORAL_POLICY_VERSION,
            builder_commit=builder_commit,
            nuscenes_version=nusc.version,
        ),
    )
    problems = validate_record(record)
    if problems:
        raise ValueError(
            "adapter produced an invalid record for "
            f"{anchor_token}; derivation bug: {'; '.join(problems)}"
        )
    return BuildResult(
        record=record, reason_code=None, detail=f"{detail} future={len(future)}"
    )


def _resolve_ego_pose(
    nusc: NuScenes, sample: dict
) -> Tuple[Optional[_EgoPose], Optional[str]]:
    """Resolve the global ego pose at one sample's CAM_FRONT keyframe.

    Returns (pose, None) on success or (None, reason_code) when the sample
    has no CAM_FRONT data (NO_CAM_FRONT) or its ego_pose / calibrated_sensor
    links cannot be resolved (MISSING_CALIBRATION).
    """
    sd_token = sample["data"].get("CAM_FRONT")
    if sd_token is None:
        return None, NO_CAM_FRONT
    try:
        sd_record = nusc.get("sample_data", sd_token)
        ego_pose = nusc.get("ego_pose", sd_record["ego_pose_token"])
        nusc.get("calibrated_sensor", sd_record["calibrated_sensor_token"])
        x, y, _ = ego_pose["translation"]
        yaw = Quaternion(ego_pose["rotation"]).yaw_pitch_roll[0]
    except (KeyError, IndexError, ValueError):
        return None, MISSING_CALIBRATION
    return (
        _EgoPose(
            timestamp_us=sample["timestamp"], x=float(x), y=float(y), yaw=float(yaw)
        ),
        None,
    )


def _chain_detail(chain: ChainResult) -> str:
    """Short scene/history context string for BuildResult.detail."""
    parts = []
    if chain.scene_name is not None:
        parts.append(f"scene={chain.scene_name}")
    if chain.frames is not None:
        parts.append(f"history={len(chain.frames)}")
    return " ".join(parts) if parts else "chain detail unavailable"
