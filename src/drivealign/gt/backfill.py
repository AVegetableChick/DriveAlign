"""expected_output backfill: evidence collection + record assembly (S08).

Purpose:
    Fill ``training_targets.expected_output`` for one valid record. Two
    layers keep the rule code unit-testable without nuScenes:

    1. :func:`collect_anchor_evidence` — the ONLY nuScenes-touching part.
       Reads the anchor CAM_FRONT frame (sensor-frame GT boxes, intrinsics,
       image size), the anchor global ego pose, the global position /
       velocity / category of every annotation referenced by the anchor and
       its future frames (walks ``sample.next`` up to ``num_future_frames``
       keyframes = 3.0 s at 2 Hz, aligned with oracle_only), and indexes
       future annotations by instance token. Velocities come from the
       devkit ``box_velocity`` finite difference; NaN (isolated instances)
       degrades to zero velocity = stationary.

    2. :func:`build_expected_output` — the pure rule pipeline:
       observability gating -> pool top-8 -> motion_state -> risk_factors
       (counterfactual corridor family + actual-trajectory evidence) ->
       speed_action (actual future ego trajectory) -> yield_required
       (counterfactual corridor family only) -> deterministic reasoning
       render -> jsonschema (v4 output_schema) validation -> P4.5
       zero-contradiction gate. Any gate failure raises ValueError
       (fail-fast, per the S08 gate plan); the caller records the anchor.

Example launch command:
    ``conda activate autovla_codeclean && cd /root/autodl-tmp/drivealign_workspace && \
    PYTHONPATH=DriveAlign/src python -c \
    "from pathlib import Path; \
    from drivealign.data.nuscenes_io import load_nuscenes, resolve_anchor; \
    from drivealign.gt.backfill import backfill_anchor; \
    nusc = load_nuscenes(Path('data/nuscenes/mini')); \
    out = backfill_anchor(nusc, resolve_anchor(nusc), ego_speed_mps=5.0, \
    future_ego_poses=[], sample_token=resolve_anchor(nusc)); \
    print(out['speed_action'], out['risk_factors'])"``
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box

from drivealign.evaluation.projection import camera_boxes_for_frame, project_box
from drivealign.gt.action_yield import (
    derive_speed_action,
    derive_yield_required,
    future_ego_speeds,
)
from drivealign.gt.config import (
    RuleConfig,
    canonical_category,
    load_phrase_map,
    load_rule_config,
    load_templates,
)
from drivealign.gt.motion_state import classify_motion_state, relative_angle_deg
from drivealign.gt.observability import INFERABLE, OBSERVABLE, classify_observability
from drivealign.gt.pool import PoolCandidate, azimuth_deg_from_center, select_objects
from drivealign.gt.render import check_zero_contradiction, render_reasoning
from drivealign.gt.risk import (
    EgoState,
    ObjectState,
    ObjectTrack,
    derive_risk_factors,
)

#: Future keyframes per record: 3.0 s at the 2 Hz keyframe rate.
NUM_FUTURE_FRAMES = 6


@dataclass(frozen=True)
class AnnRecord:
    """Global-frame facts of one sample_annotation (t0 or future frame)."""

    instance_token: str
    category_full: str
    x: float
    y: float
    velocity_x_mps: float  # NaN folded to 0.0
    velocity_y_mps: float


@dataclass(frozen=True)
class FutureFrame:
    """One future keyframe: annotations indexed by instance token."""

    sample_token: str
    timestamp_us: int
    ann_by_instance: Dict[str, str]


@dataclass(frozen=True)
class AnchorEvidence:
    """Everything the pure pipeline needs for one anchor."""

    sample_token: str
    image_size: Tuple[int, int]
    intrinsics: np.ndarray  # 3x3 CAM_FRONT camera intrinsics
    camera_boxes: Tuple[Box, ...]  # sensor frame (devkit get_sample_data)
    ann_records: Dict[str, AnnRecord]  # ann_token -> record (t0 + future)
    anchor_ann_tokens: Tuple[str, ...]
    future_frames: Tuple[FutureFrame, ...]  # chronological
    ego_x: float
    ego_y: float
    ego_yaw: float
    ego_timestamp_us: int


def collect_anchor_evidence(
    nusc: NuScenes,
    anchor_token: str,
    num_future_frames: int = NUM_FUTURE_FRAMES,
) -> AnchorEvidence:
    """Read all nuScenes evidence for one anchor (the only devkit-facing layer)."""
    sample = nusc.get("sample", anchor_token)
    sd_token = sample["data"].get("CAM_FRONT")
    if sd_token is None:
        raise ValueError(f"anchor {anchor_token} has no CAM_FRONT keyframe")
    sd_record = nusc.get("sample_data", sd_token)
    camera = camera_boxes_for_frame(nusc, sd_token)
    image_size = (int(sd_record["width"]), int(sd_record["height"]))

    ego_pose = nusc.get("ego_pose", sd_record["ego_pose_token"])
    from pyquaternion import Quaternion

    ego_yaw = float(Quaternion(ego_pose["rotation"]).yaw_pitch_roll[0])

    ann_records: Dict[str, AnnRecord] = {}

    def _record(ann_token: str) -> str:
        if ann_token not in ann_records:
            ann = nusc.get("sample_annotation", ann_token)
            vel = nusc.box_velocity(ann_token)
            vx = float(vel[0]) if math.isfinite(float(vel[0])) else 0.0
            vy = float(vel[1]) if math.isfinite(float(vel[1])) else 0.0
            ann_records[ann_token] = AnnRecord(
                instance_token=ann["instance_token"],
                category_full=ann["category_name"],
                x=float(ann["translation"][0]),
                y=float(ann["translation"][1]),
                velocity_x_mps=vx,
                velocity_y_mps=vy,
            )
        return ann_token

    anchor_anns = tuple(_record(t) for t in sample["anns"])

    future_frames: List[FutureFrame] = []
    next_token = sample["next"]
    while next_token and len(future_frames) < num_future_frames:
        fut = nusc.get("sample", next_token)
        ann_by_instance = {}
        for ann_token in fut["anns"]:
            _record(ann_token)
            ann_by_instance[ann_records[ann_token].instance_token] = ann_token
        future_frames.append(
            FutureFrame(
                sample_token=fut["token"],
                timestamp_us=int(fut["timestamp"]),
                ann_by_instance=ann_by_instance,
            )
        )
        next_token = fut["next"]

    return AnchorEvidence(
        sample_token=anchor_token,
        image_size=image_size,
        intrinsics=camera.intrinsics,
        camera_boxes=camera.boxes,
        ann_records=ann_records,
        anchor_ann_tokens=anchor_anns,
        future_frames=tuple(future_frames),
        ego_x=float(ego_pose["translation"][0]),
        ego_y=float(ego_pose["translation"][1]),
        ego_yaw=ego_yaw,
        ego_timestamp_us=int(sample["timestamp"]),
    )


def build_expected_output(
    evidence: AnchorEvidence,
    ego_speed_mps: float,
    future_ego_poses: Sequence[Tuple[float, float, int]],
    config: RuleConfig,
    templates: Dict,
    phrase_map: Dict,
) -> Dict:
    """Pure S08 pipeline: evidence -> expected_output dict (gate-enforced).

    ``ego_speed_mps`` is the record's backward-difference t0 speed;
    ``future_ego_poses`` are (x, y, timestamp_us) from oracle_only.
    """
    # 1) Observability gating over the anchor's camera boxes.
    candidates: List[PoolCandidate] = []
    for box in evidence.camera_boxes:
        category = canonical_category(box.name)
        if category is None:
            continue  # outside the closed 10-class vocabulary
        ann = evidence.ann_records.get(box.token)
        if ann is None:
            continue
        projection = project_box(box, evidence.intrinsics)
        gated = classify_observability(
            projection.corners_2d,
            projection.depth,
            evidence.image_size,
            config.observability.min_depth_m,
            config.observability.min_visible_area_frac,
        )
        if gated.state not in (OBSERVABLE, INFERABLE):
            continue
        candidates.append(
            PoolCandidate(
                ann_token=box.token,
                instance_token=ann.instance_token,
                category=category,
                distance_m=projection.distance_m,
                azimuth_deg=azimuth_deg_from_center(float(box.center[0]), float(box.center[2])),
                bbox_2d=gated.bbox_2d,
                visibility_state=gated.state,
                global_x=ann.x,
                global_y=ann.y,
                velocity_x_mps=ann.velocity_x_mps,
                velocity_y_mps=ann.velocity_y_mps,
            )
        )

    # 2) Pool: geometric filter + distance-ascending top-8.
    pool = select_objects(candidates, config.pool)

    # 3) motion_state from t0 kinematics (ego heading vs box velocity).
    motion_by_token = {
        c.ann_token: classify_motion_state(
            c.speed_mps,
            relative_angle_deg(c.velocity_x_mps, c.velocity_y_mps, evidence.ego_yaw),
            config.motion.stationary_speed_mps,
            config.motion.band_half_width_deg,
        )
        for c in pool
    }

    # 4) Future evidence tracks per pool instance (<= 8 x 6 lookups).
    tracks: Dict[str, ObjectTrack] = {}
    for cand in pool:
        times, speeds, positions = [], [], []
        for frame in evidence.future_frames:
            ann_token = frame.ann_by_instance.get(cand.instance_token)
            if ann_token is None:
                continue
            ann = evidence.ann_records[ann_token]
            times.append((frame.timestamp_us - evidence.ego_timestamp_us) / 1e6)
            speeds.append(math.hypot(ann.velocity_x_mps, ann.velocity_y_mps))
            positions.append((ann.x, ann.y))
        if times:
            tracks[cand.ann_token] = ObjectTrack(
                rel_times_s=tuple(times),
                speeds_mps=tuple(round(s, 3) for s in speeds),
                positions_m=tuple(positions),
            )

    # 5) risk_factors (counterfactual corridor family + evidence windows).
    ego_state = EgoState(
        x=evidence.ego_x,
        y=evidence.ego_y,
        yaw=evidence.ego_yaw,
        speed_mps=ego_speed_mps,
        length_m=config.corridor.ego_length_m,
        width_m=config.corridor.ego_width_m,
    )
    object_states = [
        ObjectState(
            ann_token=c.ann_token,
            category=c.category,
            x=c.global_x,
            y=c.global_y,
            velocity_x_mps=c.velocity_x_mps,
            velocity_y_mps=c.velocity_y_mps,
            distance_m=c.distance_m,
            azimuth_deg=c.azimuth_deg,
        )
        for c in pool
    ]
    risk_factors = list(
        derive_risk_factors(
            object_states,
            tracks,
            motion_by_token,
            ego_state,
            config.pool.group_of,
            config,
        )
    )

    # 6) speed_action (actual future ego trajectory) + yield (counterfactual).
    speeds = future_ego_speeds(
        evidence.ego_x,
        evidence.ego_y,
        evidence.ego_timestamp_us,
        future_ego_poses,
        config.action.window_s,
    )
    speed_action = derive_speed_action(
        ego_speed_mps, speeds, config.action.delta, config.action.stop_speed_mps
    )
    yield_required = derive_yield_required(risk_factors)

    # 7) critical_objects in pool order (distance ascending).
    critical_objects = [
        {
            "category": c.category,
            "bbox_2d": list(c.bbox_2d),
            "motion_state": motion_by_token[c.ann_token],
        }
        for c in pool
    ]

    # 8) Deterministic reasoning render.
    reasoning = render_reasoning(
        evidence.sample_token,
        critical_objects,
        risk_factors,
        speed_action,
        yield_required,
        templates,
        phrase_map,
    )

    expected_output = {
        "critical_objects": critical_objects,
        "risk_factors": risk_factors,
        "reasoning": reasoning,
        "yield_required": yield_required,
        "speed_action": speed_action,
    }

    # 9) Gates: schema + zero-contradiction (fail-fast).
    _validate_schema(expected_output)
    problems = check_zero_contradiction(
        reasoning, critical_objects, risk_factors, phrase_map
    )
    if problems:
        raise ValueError(
            f"zero-contradiction gate failed for {evidence.sample_token}: "
            + "; ".join(problems)
        )
    return expected_output


def backfill_anchor(
    nusc: NuScenes,
    anchor_token: str,
    ego_speed_mps: float,
    future_ego_poses: Sequence[Tuple[float, float, int]],
    config: Optional[RuleConfig] = None,
) -> Dict:
    """One-shot helper: collect evidence and build the expected_output dict.

    ``ego_speed_mps`` / ``future_ego_poses`` must come from the SAME
    record's model_inputs / oracle_only so the GT matches the frozen
    derivation; the render salt is the anchor token.
    """
    cfg = config or load_rule_config()
    evidence = collect_anchor_evidence(nusc, anchor_token)
    return build_expected_output(
        evidence,
        ego_speed_mps,
        future_ego_poses,
        cfg,
        load_templates(),
        load_phrase_map(),
    )


def _validate_schema(expected_output: Dict) -> None:
    """jsonschema validation against the v4 output_schema (draft 2020-12)."""
    import json

    from jsonschema import Draft202012Validator
    from drivealign.contracts.versions import contract_file

    schema = json.loads(
        contract_file("v4", "output_schema.json").read_text(encoding="utf-8")
    )
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(expected_output), key=lambda e: list(e.path))
    if errors:
        details = "; ".join(f"{list(e.path)}: {e.message}" for e in errors)
        raise ValueError(f"expected_output violates the v4 output schema: {details}")
