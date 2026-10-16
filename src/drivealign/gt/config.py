"""Loader for the v7 GT-side rule-config contract asset.

Purpose:
    Single access point for the S13 rule-engine inputs, frozen under
    ``configs/contracts/v7/``:

    - ``gt_rule_config.json``: every threshold used by the v7 GT derivation.
      Parsed into a frozen :class:`RuleConfig` tree so no module can invent
      its own numbers. The five calibration slots (``firing.base_m``,
      ``crossing_gate.lateral_thr_mps``, ``object_near.radius_m``,
      ``braking.y_brake_m``, ``stationary_obstacle.y_st_m``) are FROZEN
      since 2026-10-12 (Stage C of the S13 train-only calibration sweep;
      status flipped provisional -> frozen, ``crossing_gate.t_arrive_max_s``
      added by the rules v2.4 crossing revision).

    v7 structural changes vs. the v4 layout (frozen in
    ``drivealign.gt.legacy_v4`` for S08 audit reproducibility):
    - pool regrouped into vru/vehicle/static with a per-class cone-range
      override (``cone_range_by_class_m``, bicycle/motorcycle keep 45 m);
    - corridor ``d_risk_*`` removed — replaced by the firing engine
      (``firing`` + ``crossing_gate`` + per-factor in-path half-widths);
    - congestion removed (starved in the P1.2 calibration);
    - ``object_near`` added (instantaneous spatial evidence).

    Also hosts the nuScenes full-category -> canonical 10-class mapping
    (mirrors the devkit detection-name folding; anything outside the closed
    v3 category vocabulary is dropped from GT derivation).

Example launch command:
    ``conda activate autovla_codeclean && cd /root/autodl-tmp/drivealign_workspace && \
    PYTHONPATH=DriveAlign/src python -c \
    "from drivealign.gt.config import load_rule_config, canonical_category; \
    cfg = load_rule_config(); \
    print(cfg.rule_version, cfg.firing.base_m, canonical_category('human.pedestrian.adult'))"``
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, Optional, Tuple

from drivealign.contracts.versions import DEFAULT_CONTRACT_VERSION, contract_file

RULE_CONFIG_FILE = "gt_rule_config.json"

# nuScenes full category -> canonical term of the v3 category vocabulary.
# Mirrors nuscenes_eval category_to_detection_name; entries folding to none
# (animal, debris, stroller, ...) are excluded from GT derivation entirely.
NUSCENES_CATEGORY_MAP: Dict[str, Optional[str]] = {
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "human.pedestrian.police_officer": "pedestrian",
    "human.pedestrian.wheelchair": None,
    "human.pedestrian.stroller": None,
    "human.pedestrian.personal_mobility": None,
    "movable_object.barrier": "barrier",
    "movable_object.trafficcone": "traffic_cone",
    "movable_object.pushable_pullable": None,
    "movable_object.debris": None,
    "static_object.bicycle_rack": None,
    "vehicle.bicycle": "bicycle",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.car": "car",
    "vehicle.construction": "construction_vehicle",
    "vehicle.emergency.ambulance": None,
    "vehicle.emergency.police": None,
    "vehicle.motorcycle": "motorcycle",
    "vehicle.trailer": "trailer",
    "vehicle.truck": "truck",
}


def canonical_category(full_name: str) -> Optional[str]:
    """Map a nuScenes full category name to the closed v3 vocabulary."""
    return NUSCENES_CATEGORY_MAP.get(full_name)


@dataclass(frozen=True)
class PoolConfig:
    """对象池几何：视角锥 + 近距圆盘 + 三组锥距 + per-class 覆盖。"""

    cone_half_angle_deg: float
    proximity_disc_radius_m: float
    cone_range_by_group_m: Dict[str, float]
    class_groups: Dict[str, Tuple[str, ...]]
    # per-class 锥距覆盖（v7 新增）：键必须落在 class_groups 的成员并集内，
    # 未覆盖的类别回落到其所属组的锥距。bicycle/motorcycle 借此保持 45 m。
    cone_range_by_class_m: Dict[str, float] = field(default_factory=dict)

    def group_of(self, category: str) -> str:
        """返回类别所属的三组之一（vru/vehicle/static）。"""
        for group, members in self.class_groups.items():
            if category in members:
                return group
        raise ValueError(f"category {category!r} not in any class group")

    def cone_range_for(self, category: str) -> float:
        """该类别的有效锥距：per-class 覆盖优先，否则取组值。"""
        override = self.cone_range_by_class_m.get(category)
        if override is not None:
            return override
        return self.cone_range_by_group_m[self.group_of(category)]


@dataclass(frozen=True)
class ObservabilityConfig:
    min_depth_m: float
    min_visible_area_frac: float


@dataclass(frozen=True)
class MotionConfig:
    stationary_speed_mps: float
    band_half_width_deg: float


@dataclass(frozen=True)
class CorridorConfig:
    """CV 采样与 ego 几何（v7 删除 d_risk_*，走廊语义由 firing 引擎接管）。"""

    t_risk_s: float
    time_step_s: float
    ego_length_m: float
    ego_width_m: float
    width_margin_m: float


@dataclass(frozen=True)
class FiringConfig:
    """firing 引擎：clearance* < base 即触发；assumption 控制分族 base。"""

    base_m: float
    assumption: str  # "single"（全类别统一 base）或 "split"（vru 单独 base）
    base_vru_m: float  # 仅 assumption == "split" 时消费


@dataclass(frozen=True)
class GateConfig:
    """crossing 到达时间判据（crossing_gate，rules doc v2.4 §0.7）。

    v_toward 定义不变：带内取 ``|v_lat|``、带外取 ``-sign(y_c) · v_lat``；
    判据 ⟺ ``v_toward > lateral_thr_mps`` 且 ``t_arrive <= t_arrive_max_s``，
    其中 ``t_arrive = (|y_c| - band_half_width_m) / v_toward``、带内记 0。
    v2.4 起 crossing 与 firing 解耦：thr 角色变为 crossing 速度门槛（不再
    与 firing 联动构成"统一横向 gate"）。
    """

    lateral_thr_mps: float
    band_half_width_m: float
    t_arrive_max_s: float


@dataclass(frozen=True)
class BrakingConfig:
    evidence_window_s: float
    speed_drop_mps: float
    y_brake_m: float  # 刹车因子在路径条件的横向半宽（v7 新增标定槽位）


@dataclass(frozen=True)
class StationaryObstacleConfig:
    evidence_window_s: float
    max_speed_mps: float
    y_st_m: float  # 静止障碍因子在路径条件的横向半宽（v7 新增标定槽位）


@dataclass(frozen=True)
class ObjectNearConfig:
    """object_near：瞬时空间邻近证据（全类别，distance_m <= radius）。"""

    radius_m: float


@dataclass(frozen=True)
class OncomingConfig:
    range_m: float


@dataclass(frozen=True)
class ActionConfig:
    window_s: float
    delta: float
    stop_speed_mps: float


@dataclass(frozen=True)
class RuleConfig:
    """v7 规则引擎配置树（frozen；见 rules doc §0.4-0.6 与 §7-§9）。"""

    rule_version: str
    status: str
    contract_version: str
    pool: PoolConfig
    observability: ObservabilityConfig
    motion: MotionConfig
    corridor: CorridorConfig
    firing: FiringConfig
    crossing_gate: GateConfig
    braking: BrakingConfig
    stationary_obstacle: StationaryObstacleConfig
    object_near: ObjectNearConfig
    oncoming: OncomingConfig
    action: ActionConfig
    calibration_targets: Tuple[str, ...] = field(default_factory=tuple)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@lru_cache(maxsize=None)
def _load_json(version: str, filename: str) -> Tuple[dict, ...]:
    payload = json.loads(contract_file(version, filename).read_text(encoding="utf-8"))
    return payload,  # single-element tuple so the cache stays hashable-safe


def load_rule_config(version: str | None = None) -> RuleConfig:
    """Parse and validate the v7 ``gt_rule_config.json`` into a frozen tree.

    ``version`` 默认取 ``DEFAULT_CONTRACT_VERSION``（当前为 v7），与
    contracts/versions.py 的默认 lineage 保持单一事实源。
    """
    version = version or DEFAULT_CONTRACT_VERSION
    payload = _load_json(version, RULE_CONFIG_FILE)[0]
    _require(payload["status"] in ("provisional", "frozen"), "status invalid")

    pool = payload["pool"]
    groups = {g: tuple(members) for g, members in pool["class_groups"].items()}
    range_by_class = {
        c: float(v) for c, v in pool.get("cone_range_by_class_m", {}).items()
    }
    pool_cfg = PoolConfig(
        cone_half_angle_deg=float(pool["cone_half_angle_deg"]),
        proximity_disc_radius_m=float(pool["proximity_disc_radius_m"]),
        cone_range_by_group_m={
            g: float(v) for g, v in pool["cone_range_by_group_m"].items()
        },
        class_groups=groups,
        cone_range_by_class_m=range_by_class,
    )
    # 三组锥距键必须与组键一致；per-class 覆盖键必须落在已知类别并集内。
    _require(
        set(pool_cfg.cone_range_by_group_m) == set(groups),
        "cone_range_by_group_m keys must match class_groups keys",
    )
    known_categories = {c for members in groups.values() for c in members}
    _require(
        set(range_by_class) <= known_categories,
        "cone_range_by_class_m keys must be known categories",
    )

    corridor = payload["corridor"]
    firing = payload["firing"]
    gate = payload["crossing_gate"]
    braking = payload["braking"]
    stationary = payload["stationary_obstacle"]
    action = payload["action"]

    # firing.assumption 只允许两种分族策略（P-4/P-5：分族按类别集合实现）。
    _require(
        firing["assumption"] in ("single", "split"),
        "firing.assumption must be 'single' or 'split'",
    )
    # Y_st >= Y_brake：静止障碍的横向包容必须不窄于刹车（规划稿 §2.3 红线）。
    _require(
        float(stationary["y_st_m"]) >= float(braking["y_brake_m"]),
        "stationary_obstacle.y_st_m must be >= braking.y_brake_m",
    )
    # 统一 gate 带半宽必须与 ego 半宽 + margin 严格一致（P-3 锚定推导）。
    band_expected = (float(corridor["ego_width_m"]) + float(corridor["width_margin_m"])) / 2.0
    _require(
        float(gate["band_half_width_m"]) == band_expected,
        "crossing_gate.band_half_width_m must equal (ego_width_m + width_margin_m) / 2",
    )

    return RuleConfig(
        rule_version=str(payload["version"]),
        status=str(payload["status"]),
        contract_version=str(payload["contract_version"]),
        pool=pool_cfg,
        observability=ObservabilityConfig(
            min_depth_m=float(payload["observability"]["min_depth_m"]),
            min_visible_area_frac=float(payload["observability"]["min_visible_area_frac"]),
        ),
        motion=MotionConfig(
            stationary_speed_mps=float(payload["motion"]["stationary_speed_mps"]),
            band_half_width_deg=float(payload["motion"]["band_half_width_deg"]),
        ),
        corridor=CorridorConfig(
            t_risk_s=float(corridor["t_risk_s"]),
            time_step_s=float(corridor["time_step_s"]),
            ego_length_m=float(corridor["ego_length_m"]),
            ego_width_m=float(corridor["ego_width_m"]),
            width_margin_m=float(corridor["width_margin_m"]),
        ),
        firing=FiringConfig(
            base_m=float(firing["base_m"]),
            assumption=str(firing["assumption"]),
            base_vru_m=float(firing["base_vru_m"]),
        ),
        crossing_gate=GateConfig(
            lateral_thr_mps=float(gate["lateral_thr_mps"]),
            band_half_width_m=float(gate["band_half_width_m"]),
            t_arrive_max_s=float(gate["t_arrive_max_s"]),
        ),
        braking=BrakingConfig(
            evidence_window_s=float(braking["evidence_window_s"]),
            speed_drop_mps=float(braking["speed_drop_mps"]),
            y_brake_m=float(braking["y_brake_m"]),
        ),
        stationary_obstacle=StationaryObstacleConfig(
            evidence_window_s=float(stationary["evidence_window_s"]),
            max_speed_mps=float(stationary["max_speed_mps"]),
            y_st_m=float(stationary["y_st_m"]),
        ),
        object_near=ObjectNearConfig(radius_m=float(payload["object_near"]["radius_m"])),
        oncoming=OncomingConfig(range_m=float(payload["oncoming"]["range_m"])),
        action=ActionConfig(
            window_s=float(action["window_s"]),
            delta=float(action["delta"]),
            stop_speed_mps=float(action["stop_speed_mps"]),
        ),
        calibration_targets=tuple(payload.get("calibration_targets", ())),
    )
