"""Read-only temporal access to nuScenes CAM_FRONT keyframe chains.

Purpose:
    Stage 04 helper that loads the official NuScenes index, resolves a
    deterministic anchor sample, and reads a 2 Hz keyframe history chain
    strictly along ``sample.prev`` (never along the high-frequency camera
    sweep chain ``sample_data.prev``). Every failure mode returns a stable
    reason code instead of raising, so smoke runs can categorize errors.

Example launch command:
    ``conda activate autovla_codeclean && cd /root/autodl-tmp/drivealign_workspace && \
    PYTHONPATH=DriveAlign/src python -c \
    "from drivealign.data.nuscenes_io import *; \
    nusc = load_nuscenes(Path('data/nuscenes/mini')); \
    token = resolve_anchor(nusc); print(read_keyframe_chain(nusc, token))"``
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from nuscenes.nuscenes import NuScenes

# Reason codes returned by read_keyframe_chain on failure.
INSUFFICIENT_HISTORY = "INSUFFICIENT_HISTORY"
CROSS_SCENE = "CROSS_SCENE"
NON_INCREASING_TIME = "NON_INCREASING_TIME"
MISSING_IMAGE = "MISSING_IMAGE"
BAD_GAP = "BAD_GAP"
NO_CAM_FRONT = "NO_CAM_FRONT"

# Target CAM_FRONT keyframe period is 0.5 s; allow some slack.
DEFAULT_MAX_GAP_S = 0.6


@dataclass(frozen=True)
class FrameRecord:
    """One keyframe of the CAM_FRONT history chain (oldest -> newest)."""

    sample_token: str
    sample_data_token: str
    timestamp_us: int
    offset_s: float  # relative to the anchor (newest) frame; anchor is 0.0
    image_relpath: str  # relative to the nuScenes dataroot
    image_size: tuple[int, int]  # (width, height) from the sample_data record
    is_anchor: bool


@dataclass(frozen=True)
class ChainResult:
    """Result of reading a keyframe history chain.

    frames is None whenever reason_code is not None.
    """

    frames: Optional[tuple[FrameRecord, ...]]
    scene_token: Optional[str]
    scene_name: Optional[str]
    reason_code: Optional[str]


def load_nuscenes(dataroot: Path, version: str = "v1.0-mini") -> NuScenes:
    """Load the official NuScenes index (read-only)."""
    return NuScenes(version=version, dataroot=str(dataroot), verbose=False)


def resolve_anchor(
    nusc: NuScenes,
    scene_name: Optional[str] = None,
    sample_token: Optional[str] = None,
    num_prev: int = 3,
) -> str:
    """Deterministically pick an anchor sample token.

    Precedence: explicit --sample-token, then --scene-name, then the first
    scene in table order. Within a scene, walk forward from the first sample
    and return the first sample that has ``num_prev`` keyframe predecessors.
    Raises ValueError if no such sample exists.
    """
    if sample_token is not None:
        nusc.get("sample", sample_token)  # raises if the token is invalid
        return sample_token

    if scene_name is not None:
        scene_tokens = nusc.field2token("scene", "name", scene_name)
        if not scene_tokens:
            raise ValueError(f"Unknown scene name: {scene_name}")
        scene = nusc.get("scene", scene_tokens[0])
    else:
        scene = nusc.scene[0]

    sample_token_iter = scene["first_sample_token"]
    while sample_token_iter:
        sample = nusc.get("sample", sample_token_iter)
        if _count_prevs(nusc, sample["token"]) >= num_prev:
            return sample["token"]
        sample_token_iter = sample["next"]
    raise ValueError(
        f"Scene {scene['name']} has no sample with {num_prev} predecessors"
    )


def _count_prevs(nusc: NuScenes, sample_token: str) -> int:
    """Count keyframe predecessors along sample.prev."""
    count = 0
    current = nusc.get("sample", sample_token).get("prev")
    while current:
        count += 1
        current = nusc.get("sample", current).get("prev")
    return count


def read_keyframe_chain(
    nusc: NuScenes,
    sample_token: str,
    num_prev: int = 3,
    max_gap_s: float = DEFAULT_MAX_GAP_S,
) -> ChainResult:
    """Read anchor + ``num_prev`` keyframe history of CAM_FRONT.

    Walks ``sample.prev`` only. Returns ChainResult with reason_code set and
    frames=None on any validation failure (insufficient history, cross-scene,
    non-increasing timestamps, abnormal gap, missing CAM_FRONT or image).
    On success frames is ordered oldest -> newest and offsets are relative to
    the anchor (about [-1.5, -1.0, -0.5, 0.0] for num_prev=3).
    """
    # 1) Walk back along sample.prev to collect keyframe samples.
    samples = [nusc.get("sample", sample_token)]
    for _ in range(num_prev):
        prev = samples[-1].get("prev")
        if not prev:
            return ChainResult(None, None, None, INSUFFICIENT_HISTORY)
        samples.append(nusc.get("sample", prev))
    samples.reverse()  # oldest -> newest

    # 2) Same-scene check.
    scene_tokens = {s["scene_token"] for s in samples}
    if len(scene_tokens) != 1:
        return ChainResult(None, None, None, CROSS_SCENE)
    scene_token = scene_tokens.pop()
    scene = nusc.get("scene", scene_token)

    # 3) Attach each sample's CAM_FRONT keyframe sample_data.
    records: list[FrameRecord] = []
    anchor_ts = samples[-1]["timestamp"]
    for sample in samples:
        sd_token = sample["data"].get("CAM_FRONT")
        if sd_token is None:
            return ChainResult(None, scene_token, scene["name"], NO_CAM_FRONT)
        sd_record = nusc.get("sample_data", sd_token)
        if not sd_record["is_key_frame"]:
            raise ValueError(
                "sample.data must reference a keyframe; got a sweep for "
                f"sample {sample['token']}"
            )
        records.append(
            FrameRecord(
                sample_token=sample["token"],
                sample_data_token=sd_token,
                timestamp_us=sample["timestamp"],
                offset_s=round((sample["timestamp"] - anchor_ts) / 1e6, 6),
                image_relpath=sd_record["filename"],
                image_size=(sd_record["width"], sd_record["height"]),
                is_anchor=sample["token"] == sample_token,
            )
        )

    # 4) Strictly increasing timestamps (oldest -> newest).
    for older, newer in zip(records, records[1:]):
        if newer.timestamp_us <= older.timestamp_us:
            return ChainResult(None, scene_token, scene["name"], NON_INCREASING_TIME)

    # 5) Consecutive gap sanity (target ~0.5 s for CAM_FRONT keyframes).
    for older, newer in zip(records, records[1:]):
        gap_s = (newer.timestamp_us - older.timestamp_us) / 1e6
        if gap_s > max_gap_s:
            return ChainResult(None, scene_token, scene["name"], BAD_GAP)

    # 6) Image existence on disk.
    dataroot = Path(nusc.dataroot)
    for record in records:
        if not (dataroot / record.image_relpath).is_file():
            return ChainResult(None, scene_token, scene["name"], MISSING_IMAGE)

    return ChainResult(
        frames=tuple(records),
        scene_token=scene_token,
        scene_name=scene["name"],
        reason_code=None,
    )
