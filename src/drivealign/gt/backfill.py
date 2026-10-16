"""expected_output backfill: evidence collection + record assembly (S13 v7).

Purpose:
    Fill ``training_targets.expected_output`` for one valid record. Two
    layers keep the rule code unit-testable without nuScenes:

    1. :func:`collect_anchor_evidence` — the ONLY nuScenes-touching part.
       Reads the anchor CAM_FRONT frame (sensor-frame GT boxes, intrinsics,
       image size), the anchor global ego pose, the global position /
       velocity / box geometry (size + yaw) / category of every annotation
       referenced by the anchor and its future frames (walks ``sample.next``
       up to ``num_future_frames`` keyframes = 3.0 s at 2 Hz, aligned with
       oracle_only), and indexes future annotations by instance token.
       Velocities come from the devkit ``box_velocity`` finite difference;
       NaN (isolated instances) degrades to zero velocity = stationary.
       v7 additionally reads each annotation's size (width, length) and
       global yaw — deterministic inputs of the firing support function and
       of the BEV nearest-point GT.

    2. :func:`build_expected_output` — the pure rule pipeline:
       observability gating -> pool top-8 -> motion_state -> risk_factors +
       attribution + any_firing (v7 firing engine, rules §0.4-0.7; §12
       attribution, ruling V7-A1) -> speed_action (actual future ego
       trajectory) -> yield_required (= firing union, §7) -> critical_objects
       with the id anchor (0-7 by pool order, D-A1-2) and the BEV
       nearest-point GT (§3.3 of the BEV rules doc) -> jsonschema (v7
       output_schema) validation. The v7 contract has
       FOUR fields — the reasoning render chain (templates / phrase map /
       zero-contradiction gate) is retired with the T0 atomic batch. Any
       gate failure raises ValueError (fail-fast); the caller records the
       anchor.

Example launch command:
    ``conda activate autovla_codeclean && cd /root/autodl-tmp/drivealign_workspace && \
    PYTHONPATH=DriveAlign/src python -c \
    "from drivealign.gt.backfill import bev_nearest_point; \
    print(bev_nearest_point(20.0, 0.0, 4.5, 2.0, 0.0))"``
    # -> (17.8, 0.0)  BEV 验算例：20 m 正前 L=4.5 → 最近点 17.75 → 0.1 m 取整
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion

from drivealign.evaluation.projection import camera_boxes_for_frame, project_box
from drivealign.gt.action_yield import (
    derive_speed_action,
    derive_yield_required,
    future_ego_speeds,
)
from drivealign.gt.config import RuleConfig, canonical_category, load_rule_config
from drivealign.gt.motion_state import classify_motion_state, relative_angle_deg
from drivealign.gt.observability import INFERABLE, OBSERVABLE, classify_observability
from drivealign.gt.pool import PoolCandidate, azimuth_deg_from_center, select_objects
from drivealign.gt.risk import (
    EgoState,
    ObjectState,
    ObjectTrack,
    derive_risk_factors,
    ego_frame_xy,
)

#: Future keyframes per record: 3.0 s at the 2 Hz keyframe rate.
NUM_FUTURE_FRAMES = 6


@dataclass(frozen=True)
class AnnRecord:
    """Global-frame facts of one sample_annotation (t0 or future frame).

    v7 新增标注盒几何三字段：``size_w_m`` / ``size_l_m`` 为 nuScenes size
    前两维（width, length；height 不消费），``yaw_global_rad`` 为全局系
    朝向（pyquaternion yaw 分量）。firing 支撑函数与 BEV GT 的确定性输入。
    """

    instance_token: str
    category_full: str
    x: float
    y: float
    velocity_x_mps: float  # NaN folded to 0.0
    velocity_y_mps: float
    size_w_m: float
    size_l_m: float
    yaw_global_rad: float


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


def bev_nearest_point(
    x_c: float,
    y_c: float,
    box_length_m: float,
    box_width_m: float,
    box_yaw_ego_deg: float,
) -> Tuple[float, float]:
    """BEV GT: nearest point on the annotation box to the ego origin (BEV rules §3.3).

    盒局部系（x = 盒纵向，原点 = 盒中心，θ = 盒朝向，均在 ego 系下）：

    1. ego 中心（= ego 系原点）旋到盒局部系：``p_local = R(−θ)·(−x_c, −y_c)``，
       其中 (−x_c, −y_c) = 从盒中心指向 ego 中心的向量；
    2. clamp 到盒矩形：``q_local = (clamp(p_x, ±L/2), clamp(p_y, ±W/2))``；
    3. 旋回 ego 系并平移：``p_near = (x_c, y_c) + R(θ)·q_local``；
    4. 0.1 m 取整：``round(v, 1)`` —— 与 output_schema parser
       (contracts/output.py) 严格同款取整（不是 round(v*10)/10），保证
       GT 与模型输出落在同一离散空间（含 banker's rounding 边界行为：
       round(127.5) = 128 → 12.8）。
    """
    theta = math.radians(box_yaw_ego_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    # 第 1 步：R(−θ)·(−x_c, −y_c)
    px = cos_t * (-x_c) + sin_t * (-y_c)
    py = -sin_t * (-x_c) + cos_t * (-y_c)
    # 第 2 步：clamp 到盒矩形
    qx = max(-box_length_m / 2.0, min(box_length_m / 2.0, px))
    qy = max(-box_width_m / 2.0, min(box_width_m / 2.0, py))
    # 第 3 步：R(θ)·q 回 ego 系 + 盒中心平移
    nx = x_c + cos_t * qx - sin_t * qy
    ny = y_c + sin_t * qx + cos_t * qy
    # 第 4 步：0.1 m 取整（与 parser 同款 round(v, 1)）；``+ 0.0`` 把
    # round 产生的 -0.0 归一为 0.0（数值相等，避免 GT 面出现 -0.0 字面量）。
    return round(nx, 1) + 0.0, round(ny, 1) + 0.0


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
    ego_yaw = float(Quaternion(ego_pose["rotation"]).yaw_pitch_roll[0])

    ann_records: Dict[str, AnnRecord] = {}

    def _record(ann_token: str) -> str:
        if ann_token not in ann_records:
            ann = nusc.get("sample_annotation", ann_token)
            vel = nusc.box_velocity(ann_token)
            vx = float(vel[0]) if math.isfinite(float(vel[0])) else 0.0
            vy = float(vel[1]) if math.isfinite(float(vel[1])) else 0.0
            # nuScenes size 惯例 = [width, length, height]；朝向取全局 yaw 分量。
            width_m, length_m = float(ann["size"][0]), float(ann["size"][1])
            yaw_rad = float(Quaternion(ann["rotation"]).yaw_pitch_roll[0])
            ann_records[ann_token] = AnnRecord(
                instance_token=ann["instance_token"],
                category_full=ann["category_name"],
                x=float(ann["translation"][0]),
                y=float(ann["translation"][1]),
                velocity_x_mps=vx,
                velocity_y_mps=vy,
                size_w_m=width_m,
                size_l_m=length_m,
                yaw_global_rad=yaw_rad,
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
) -> Dict:
    """Pure v7 pipeline: evidence -> expected_output dict (schema-gated).

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
                # v7 标注盒几何：尺寸来自标注 size，朝向 = 全局 yaw − ego yaw。
                box_length_m=ann.size_l_m,
                box_width_m=ann.size_w_m,
                box_yaw_ego_deg=math.degrees(ann.yaw_global_rad - evidence.ego_yaw),
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

    # 5) risk_factors + attribution + any_firing（v7 firing 引擎，三元组返回；
    #    归因 = 全部合格对象的 ann_token 元组，rules §12 / V7-A1）。
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
            box_length_m=c.box_length_m,
            box_width_m=c.box_width_m,
            box_yaw_ego_deg=c.box_yaw_ego_deg,
        )
        for c in pool
    ]
    _factors, attribution, any_firing = derive_risk_factors(
        object_states,
        tracks,
        motion_by_token,
        ego_state,
        config,
    )

    # 6) speed_action（真实未来行为证据）+ yield_required（firing 并集，§7）。
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
    yield_required = derive_yield_required(any_firing)

    # 7) critical_objects in pool order + id anchor + BEV nearest-point GT
    #    （BEV §3.3；归因规则稿 §3.3）：id = 池序下标 0..7（D-A1-2 的 GT 分
    #    配，risk_factors 条目按该 id 引用对象），ego 系中心经共享原语
    #    ego_frame_xy（与因子判据同一变换），最近点经 bev_nearest_point 的
    #    0.1 m 取整（与 parser 同离散空间）。
    id_by_token = {c.ann_token: idx for idx, c in enumerate(pool)}
    critical_objects = []
    for idx, c in enumerate(pool):
        x_c, y_c = ego_frame_xy(ego_state, c.global_x, c.global_y)
        bev_x, bev_y = bev_nearest_point(
            x_c, y_c, c.box_length_m, c.box_width_m, c.box_yaw_ego_deg
        )
        critical_objects.append(
            {
                "id": idx,
                "category": c.category,
                "bbox_2d": list(c.bbox_2d),
                "motion_state": motion_by_token[c.ann_token],
                "bev_x": bev_x,
                "bev_y": bev_y,
            }
        )

    # 归因落字段（D-A1-1）：因子 ann_token 集合 → id 列表（升序）。
    # attribution 已按 RISK_TAXONOMY_ORDER 排列，因子存在 ⟺ objects 非空。
    risk_attribution = [
        {
            "factor": factor,
            "objects": sorted(id_by_token[token] for token in tokens),
        }
        for factor, tokens in attribution.items()
    ]

    expected_output = {
        "critical_objects": critical_objects,
        "risk_factors": risk_attribution,
        "yield_required": yield_required,
        "speed_action": speed_action,
    }

    # 8) Gate: v7 output_schema validation (fail-fast)。
    _validate_schema(expected_output)
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
    derivation.
    """
    cfg = config or load_rule_config()
    evidence = collect_anchor_evidence(nusc, anchor_token)
    return build_expected_output(evidence, ego_speed_mps, future_ego_poses, cfg)


def _validate_schema(expected_output: Dict) -> None:
    """jsonschema validation against the DEFAULT contract's output_schema (2020-12)."""
    import json

    from jsonschema import Draft202012Validator
    from drivealign.contracts.versions import DEFAULT_CONTRACT_VERSION, contract_file

    schema = json.loads(
        contract_file(DEFAULT_CONTRACT_VERSION, "output_schema.json").read_text(encoding="utf-8")
    )
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(expected_output), key=lambda e: list(e.path))
    if errors:
        details = "; ".join(f"{list(e.path)}: {e.message}" for e in errors)
        raise ValueError(
            f"expected_output violates the {DEFAULT_CONTRACT_VERSION} output schema: {details}"
        )
