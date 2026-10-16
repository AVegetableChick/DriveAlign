"""Frozen S08 v4 reproduction layer (S13 v7 migration, read-only).

Purpose:
    v7 把 GT 引擎切换到 firing 语义（见 :mod:`drivealign.gt.risk`）并删除了
    v4/v6 的走廊机制与 d_risk_* 阈值。但两个 S08 审计 CLI——
    ``cli/gt_calibration.py``（P1.2 标定）与 ``cli/gt_truncation_check.py``
    （截断风险核验）——的历史结论都建立在 v4 规则之上；删掉 v4 机制后
    这两份报告将无法复现。本模块因此把 v4 规则引擎的**数据类树 + 加载器 +
    走廊 CV 机制 + 池选择**原样冻结在一个自包含文件里。

    消费边界（勿扩展）：
        - 唯一消费者 = 上述两个 S08 审计 CLI；
        - v4 语义在此冻结，不随 v7 演进；任何新功能一律进 v7 模块；
        - 共享的三个符号（``EgoState`` / ``ObjectState`` / ``PoolCandidate``）
          从 v7 模块导入——它们只是证据容器，v4 函数只读旧字段，容器新增
          字段不影响本层行为。

    v4 走廊语义（t0 双侧恒速反事实，2026-10-02 ledger）：
        ego 走廊 = ego 若保持 t0 状态前进 T_risk 秒扫过的矩形带
        (长 x (宽 + margin))；物体侧按 t0 全局速度 CV 外推。conflict =
        采样时刻上物体 CV 点轨迹到 ego CV 矩形带的最小距离 < 该类组的
        d_risk。语义是"双方都不反应时是否冲突"——绝不用真实未来轨迹，
        否则好司机会擦除冲突证据（self-defeating trap）。

Example launch command:
    ``conda activate autovla_codeclean && cd /root/autodl-tmp/drivealign_workspace && \
    PYTHONPATH=DriveAlign/src python -c \
    "from drivealign.gt.legacy_v4 import load_v4_rule_config; \
    cfg = load_v4_rule_config(); \
    print(cfg.rule_version, cfg.corridor.d_risk_vehicle_m)"``
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Sequence, Tuple

from drivealign.contracts.versions import contract_file
# canonical_category（nuScenes 全类 -> 10 类映射）v4/v7 语义一致，re-export
# 让两个 S08 CLI 只依赖本模块一处。
from drivealign.gt.config import canonical_category
# 共享证据容器：v4 规则只消费下列旧字段，容器后续加字段与本层无关。
from drivealign.gt.pool import MAX_POOL_SIZE, PoolCandidate
from drivealign.gt.risk import EgoState, ObjectState
from drivealign.gt.observability import INFERABLE, OBSERVABLE

RULE_CONFIG_FILE = "gt_rule_config.json"
V4_CONTRACT_VERSION = "v4"


# --- v4 配置数据类树（原样冻结，勿增删字段） ---------------------------------


@dataclass(frozen=True)
class PoolConfig:
    """v4 对象池几何：锥 + 近距圆盘 + 按组锥距（无 per-class 覆盖）。"""

    cone_half_angle_deg: float
    proximity_disc_radius_m: float
    cone_range_by_group_m: Dict[str, float]
    class_groups: Dict[str, Tuple[str, ...]]

    def group_of(self, category: str) -> str:
        for group, members in self.class_groups.items():
            if category in members:
                return group
        raise ValueError(f"category {category!r} not in any class group")


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
    """v4 走廊配置：d_risk_* 是 conflict 判定的 per-group 缓冲半径。"""

    t_risk_s: float
    time_step_s: float
    ego_length_m: float
    ego_width_m: float
    width_margin_m: float
    d_risk_vehicle_m: float
    d_risk_pedestrian_m: float


@dataclass(frozen=True)
class BrakingConfig:
    evidence_window_s: float
    speed_drop_mps: float


@dataclass(frozen=True)
class StationaryObstacleConfig:
    evidence_window_s: float
    max_speed_mps: float


@dataclass(frozen=True)
class CongestionConfig:
    lead_speed_mps: float
    min_queue_vehicles: int


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
    """v4 规则配置树（frozen；仅由 :func:`load_v4_rule_config` 构造）。"""

    rule_version: str
    status: str
    contract_version: str
    pool: PoolConfig
    observability: ObservabilityConfig
    motion: MotionConfig
    corridor: CorridorConfig
    braking: BrakingConfig
    stationary_obstacle: StationaryObstacleConfig
    congestion: CongestionConfig
    oncoming: OncomingConfig
    action: ActionConfig
    calibration_targets: Tuple[str, ...] = ()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@lru_cache(maxsize=None)
def _load_json(version: str, filename: str) -> Tuple[dict, ...]:
    payload = json.loads(contract_file(version, filename).read_text(encoding="utf-8"))
    return payload,  # single-element tuple so the cache stays hashable-safe


def load_v4_rule_config() -> RuleConfig:
    """Parse and validate the frozen v4 ``gt_rule_config.json`` (fixed version)."""
    payload = _load_json(V4_CONTRACT_VERSION, RULE_CONFIG_FILE)[0]
    _require(payload["status"] in ("provisional", "frozen"), "status invalid")

    pool = payload["pool"]
    groups = {g: tuple(members) for g, members in pool["class_groups"].items()}
    pool_cfg = PoolConfig(
        cone_half_angle_deg=float(pool["cone_half_angle_deg"]),
        proximity_disc_radius_m=float(pool["proximity_disc_radius_m"]),
        cone_range_by_group_m={
            g: float(v) for g, v in pool["cone_range_by_group_m"].items()
        },
        class_groups=groups,
    )
    _require(
        set(pool_cfg.cone_range_by_group_m) == set(groups),
        "cone_range_by_group_m keys must match class_groups keys",
    )
    obs = payload["observability"]
    motion = payload["motion"]
    corridor = payload["corridor"]
    braking = payload["braking"]
    stationary = payload["stationary_obstacle"]
    congestion = payload["congestion"]
    oncoming = payload["oncoming"]
    action = payload["action"]
    return RuleConfig(
        rule_version=str(payload["version"]),
        status=str(payload["status"]),
        contract_version=str(payload["contract_version"]),
        pool=pool_cfg,
        observability=ObservabilityConfig(
            min_depth_m=float(obs["min_depth_m"]),
            min_visible_area_frac=float(obs["min_visible_area_frac"]),
        ),
        motion=MotionConfig(
            stationary_speed_mps=float(motion["stationary_speed_mps"]),
            band_half_width_deg=float(motion["band_half_width_deg"]),
        ),
        corridor=CorridorConfig(
            t_risk_s=float(corridor["t_risk_s"]),
            time_step_s=float(corridor["time_step_s"]),
            ego_length_m=float(corridor["ego_length_m"]),
            ego_width_m=float(corridor["ego_width_m"]),
            width_margin_m=float(corridor["width_margin_m"]),
            d_risk_vehicle_m=float(corridor["d_risk_vehicle_m"]),
            d_risk_pedestrian_m=float(corridor["d_risk_pedestrian_m"]),
        ),
        braking=BrakingConfig(
            evidence_window_s=float(braking["evidence_window_s"]),
            speed_drop_mps=float(braking["speed_drop_mps"]),
        ),
        stationary_obstacle=StationaryObstacleConfig(
            evidence_window_s=float(stationary["evidence_window_s"]),
            max_speed_mps=float(stationary["max_speed_mps"]),
        ),
        congestion=CongestionConfig(
            lead_speed_mps=float(congestion["lead_speed_mps"]),
            min_queue_vehicles=int(congestion["min_queue_vehicles"]),
        ),
        oncoming=OncomingConfig(range_m=float(oncoming["range_m"])),
        action=ActionConfig(
            window_s=float(action["window_s"]),
            delta=float(action["delta"]),
            stop_speed_mps=float(action["stop_speed_mps"]),
        ),
        calibration_targets=tuple(payload.get("calibration_targets", ())),
    )


# --- v4 走廊机制（t0 双侧 CV 反事实） ----------------------------------------


def _rect_distance(
    local_x: float, local_y: float, half_len: float, half_wid: float
) -> float:
    """Distance from a point (ego frame) to the axis-aligned ego rectangle."""
    dx = max(abs(local_x) - half_len, 0.0)
    dy = max(abs(local_y) - half_wid, 0.0)
    return math.hypot(dx, dy)


def corridor_distance_curve(
    ego: EgoState,
    obj: ObjectState,
    t_risk_s: float,
    time_step_s: float,
    width_margin_m: float,
) -> Tuple[Tuple[float, float], ...]:
    """(t, distance) between the object's CV point trajectory and the ego CV band.

    Both sides extrapolate from t0 states with constant velocity; the ego
    band is a rectangle of (length x (width + margin)) sliding along the
    heading line. t=0 is included, so an object already overlapping the band
    scores distance 0.
    """
    half_len = ego.length_m / 2.0
    half_wid = (ego.width_m + width_margin_m) / 2.0
    cos_y = math.cos(ego.yaw)
    sin_y = math.sin(ego.yaw)
    # ego 系下的相对 CV 运动：d(obj - ego)/dt。
    rel_vx = obj.velocity_x_mps - ego.speed_mps * cos_y
    rel_vy = obj.velocity_y_mps - ego.speed_mps * sin_y
    dx0 = obj.x - ego.x
    dy0 = obj.y - ego.y

    points = []
    n_steps = int(round(t_risk_s / time_step_s))
    for i in range(n_steps + 1):
        t = i * time_step_s
        gx = dx0 + rel_vx * t
        gy = dy0 + rel_vy * t
        local_x = cos_y * gx + sin_y * gy
        local_y = -sin_y * gx + cos_y * gy
        points.append((round(t, 6), _rect_distance(local_x, local_y, half_len, half_wid)))
    return tuple(points)


def corridor_conflict(
    curve: Sequence[Tuple[float, float]], d_risk_m: float
) -> bool:
    """True iff the CV curve dips under d_risk at any sampled time."""
    return any(d < d_risk_m for _, d in curve)


# --- v4 池选择（按组锥距，无 per-class 覆盖） --------------------------------


def select_objects(
    candidates: Sequence[PoolCandidate], pool_config: PoolConfig
) -> Tuple[PoolCandidate, ...]:
    """Gate survivors -> geometric filter -> distance-ascending top-8."""
    cone_half = pool_config.cone_half_angle_deg
    disc = pool_config.proximity_disc_radius_m
    kept = []
    for cand in candidates:
        if cand.visibility_state not in (OBSERVABLE, INFERABLE):
            continue
        group = pool_config.group_of(cand.category)
        in_cone = (
            abs(cand.azimuth_deg) <= cone_half
            and cand.distance_m < pool_config.cone_range_by_group_m[group]
        )
        in_disc = cand.distance_m < disc
        if in_cone or in_disc:
            kept.append(cand)
    # 距离升序；精确并列时按 annotation token 字典序打破（确定性要求）。
    kept.sort(key=lambda c: (round(c.distance_m, 3), c.ann_token))
    return tuple(kept[:MAX_POOL_SIZE])


def truncated_count(candidates: Sequence[PoolCandidate], pool_config: PoolConfig) -> int:
    """Number of geometric survivors BEFORE the top-8 cut (truncation audit)."""
    cone_half = pool_config.cone_half_angle_deg
    disc = pool_config.proximity_disc_radius_m
    n = 0
    for cand in candidates:
        if cand.visibility_state not in (OBSERVABLE, INFERABLE):
            continue
        group = pool_config.group_of(cand.category)
        in_cone = (
            abs(cand.azimuth_deg) <= cone_half
            and cand.distance_m < pool_config.cone_range_by_group_m[group]
        )
        if in_cone or cand.distance_m < disc:
            n += 1
    return n


__all__ = [
    "MAX_POOL_SIZE",
    "PoolCandidate",
    "EgoState",
    "ObjectState",
    "PoolConfig",
    "ObservabilityConfig",
    "MotionConfig",
    "CorridorConfig",
    "BrakingConfig",
    "StationaryObstacleConfig",
    "CongestionConfig",
    "OncomingConfig",
    "ActionConfig",
    "RuleConfig",
    "canonical_category",
    "load_v4_rule_config",
    "corridor_distance_curve",
    "corridor_conflict",
    "select_objects",
    "truncated_count",
]
