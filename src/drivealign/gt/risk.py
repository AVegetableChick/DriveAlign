"""risk_factors GT: the v7 six-item taxonomy + the firing engine (S13).

Purpose:
    Implement the v7 risk derivations (rules doc ``S13_v7_risk_factor_rules.md``
    v2.4; decision ledger V7-R1 + the 2026-10-12 crossing revision). The
    v4/v6 corridor machinery (bilateral-CV point-to-band distance vs. d_risk)
    is replaced by the firing engine; the v4 semantics stay frozen in
    :mod:`drivealign.gt.legacy_v4` for the two S08 audit CLIs.

    FIRING ENGINE (rules §0.4-0.5, P-2):
        Bilateral constant-velocity counterfactual, K = 13 samples over
        ``t_risk_s`` at ``time_step_s``. Per sample k the object center moves
        on its CV line in the ego frame and we measure the distance ``d_k``
        to the sweep band rectangle (ego length x (width + margin), i.e.
        2.05 x 1.185 half-dimensions). The annotation box's support half
        extent ``h_k`` along the gap direction (from the band nearest point
        toward the center) converts the center distance into a physical
        surface gap: ``clearance_k = d_k - h_k``. A sample with ``d_k = 0``
        (center inside the band) fires directly. ``firing(o)`` iff
        ``min_k(d_k - h_k) < base(o)`` (strict). ``h`` is deterministic
        (annotation L/W + t0 yaw) — zero calibration. v7 firing has NO
        taxonomy label of its own: since rules v2.4 it feeds the yield union
        ONLY (rules §0.6 / §7) — the crossing factors no longer consume it.

    SIX-ITEM TAXONOMY (canonical output order = v7 output_schema enum order):
        - vru_crossing (§1): a vru-group object (pedestrian/bicycle/motorcycle
          as a CATEGORY SET, not a pool group literal) passing the crossing
          arrival test — NOT gated by firing since rules v2.4;
        - vehicle_crossing (§2): same with the five vehicle categories
          (renamed from v6 vehicle_merging);
        - lead_vehicle_braking (§3): unchanged logic; in-path widened to
          ``x_c > 0 AND |y_c| <= Y_brake`` (P-6); family = vehicle five ∪
          {bicycle, motorcycle} (P-5, category set);
        - stationary_obstacle (§4): in-path widened to ``Y_st >= Y_brake``
          (P-6); ALL categories one criterion; no track / empty window
          degrades to the t0-only test (asymmetric with braking on purpose);
        - object_near (§5, new): instantaneous spatial evidence,
          ``distance_m <= r``, all categories;
        - oncoming_traffic (§6): heading rule, family per P-4 = the same
          category set as braking, strict ``distance_m < range_m``.

    CROSSING ARRIVAL TEST (§0.7, rules v2.4 — replaces the v2.2 unified
    lateral gate): ``v_lat`` = lateral component of the relative velocity in
    the ego frame. OUTSIDE the band (|y_c| > band_half_width_m = 1.185) use
    ``v_toward = -sign(y_c) * v_lat`` and ``t_arrive = (|y_c| - 1.185) /
    v_toward``; INSIDE the band use ``v_toward = |v_lat|`` and
    ``t_arrive = 0``. Crossing iff ``v_toward > thr`` AND
    ``t_arrive <= t_arrive_max_s`` — DECOUPLED from firing: firing answers
    "already too close under the 3 s CV counterfactual", the arrival time
    answers "entering the corridor within 5 s"; a far crossing object that
    never fires still labels, a slow drifter with t_arrive > T does not.

    ATTRIBUTION (§12, V7-A1 / D-A1-3): the return value is
    ``(factors, attribution, any_firing)`` — ``attribution`` maps each fired
    factor to the tuple of ann_tokens of ALL its qualifying objects. The
    crossing factors collect every category-family object passing the
    arrival test (pool order); the three first-hit loops (braking /
    stationary / oncoming) dropped their ``break`` and collect every
    qualifying object instead, so a factor exists iff its attribution set
    is non-empty. Ordering is deterministic: braking = distance ascending
    (the loop's sort key), the rest = pool order (distance-ascending top-8).

    YIELD COUPLING (§7): ``yield_required = any_firing`` is the union of
    firing over the whole pool, INCLUDING static objects and label-less
    firings (decoupled from the taxonomy labels, P1.6 evidence split).

Example launch command:
    ``conda activate autovla_codeclean && cd /root/autodl-tmp/drivealign_workspace && \
    PYTHONPATH=DriveAlign/src python -c \
    "from drivealign.gt.config import load_rule_config; \
    from drivealign.gt.risk import EgoState, ObjectState, clearance_curve, object_firing; \
    cfg = load_rule_config(); \
    ego = EgoState(x=0., y=0., yaw=0., speed_mps=0., length_m=4.1, width_m=1.87); \
    obj = ObjectState(ann_token='a', category='car', x=6.3, y=0., \
    velocity_x_mps=0., velocity_y_mps=0., distance_m=6.3, azimuth_deg=0., \
    box_length_m=4.5, box_width_m=2.0, box_yaw_ego_deg=0.); \
    print(object_firing(clearance_curve(ego, obj, cfg.corridor), cfg.firing.base_m))"``
    # -> False（clearance = 6.3 - 2.05 - 2.25 = 2.0，恰等于 base，严格 < 不触发）
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from drivealign.gt.config import CorridorConfig, GateConfig, RuleConfig
from drivealign.gt.motion_state import ONCOMING

# Closed taxonomy, canonical output order (v7 output_schema enum order).
RISK_TAXONOMY_ORDER = (
    "vru_crossing",
    "vehicle_crossing",
    "lead_vehicle_braking",
    "stationary_obstacle",
    "object_near",
    "oncoming_traffic",
)

# P-4/P-5 族口径按"类别集合"实现，不写"vehicle 组"字面——两轮车在 v7 池的
# 三组拆分里属 vru 组，但 braking / oncoming 的族仍包含它们（动力学证据
# 不随保护类别改变）。
BRAKING_AND_ONCOMING_CATEGORIES = frozenset(
    {"car", "truck", "bus", "trailer", "construction_vehicle", "bicycle", "motorcycle"}
)
# crossing 两判定的对象族（§1 / §2）：由类别集合天然互斥，static 类别
# （barrier / traffic_cone）两者皆不属——其 firing 只喂 yield（§0.6）。
VRU_CATEGORIES = frozenset({"pedestrian", "bicycle", "motorcycle"})
VEHICLE_CROSSING_CATEGORIES = frozenset(
    {"car", "truck", "bus", "trailer", "construction_vehicle"}
)


@dataclass(frozen=True)
class EgoState:
    """t0 ego state in the global frame (CV extrapolation seeds)."""

    x: float
    y: float
    yaw: float  # global heading, radians
    speed_mps: float  # backward-difference speed at t0
    length_m: float
    width_m: float


@dataclass(frozen=True)
class ObjectState:
    """t0 state of one pool object (global frame + sensor-frame geometry).

    v7 新增三个标注盒几何必填字段（firing 支撑函数的确定性输入，零标定）：
    ``box_length_m`` / ``box_width_m`` 为标注盒长宽（米），``box_yaw_ego_deg``
    为盒朝向在 ego 系下的平面角（度；全局 yaw 减 ego yaw）。
    """

    ann_token: str
    category: str  # canonical
    x: float
    y: float
    velocity_x_mps: float
    velocity_y_mps: float
    distance_m: float  # 3D norm of the sensor-frame center
    azimuth_deg: float
    box_length_m: float
    box_width_m: float
    box_yaw_ego_deg: float

    @property
    def speed_mps(self) -> float:
        return math.hypot(self.velocity_x_mps, self.velocity_y_mps)


@dataclass(frozen=True)
class ObjectTrack:
    """Actual future evidence window of one object (global frame)."""

    rel_times_s: Tuple[float, ...]  # seconds after t0, ascending
    speeds_mps: Tuple[float, ...]
    positions_m: Tuple[Tuple[float, float], ...]

    def window(self, max_s: float) -> "ObjectTrack":
        idx = [i for i, t in enumerate(self.rel_times_s) if t <= max_s]
        return ObjectTrack(
            rel_times_s=tuple(self.rel_times_s[i] for i in idx),
            speeds_mps=tuple(self.speeds_mps[i] for i in idx),
            positions_m=tuple(self.positions_m[i] for i in idx),
        )


# --- shared frame / geometry primitives --------------------------------------


def ego_frame_xy(ego: EgoState, x_global: float, y_global: float) -> Tuple[float, float]:
    """Global point -> ego frame (origin = ego t0, x forward, y left).

    公开原语：六因子的 ``x_c`` / ``y_c`` 判据与 BEV GT（backfill）共用同一
    坐标变换，保证两侧几何完全一致。
    """
    cos_y = math.cos(ego.yaw)
    sin_y = math.sin(ego.yaw)
    gx, gy = x_global - ego.x, y_global - ego.y
    return cos_y * gx + sin_y * gy, -sin_y * gx + cos_y * gy


def _rect_distance(local_x: float, local_y: float, half_len: float, half_wid: float) -> float:
    """Distance from a point (ego frame) to the axis-aligned band rectangle."""
    dx = max(abs(local_x) - half_len, 0.0)
    dy = max(abs(local_y) - half_wid, 0.0)
    return math.hypot(dx, dy)


def _band_half_dimensions(corridor: CorridorConfig) -> Tuple[float, float]:
    """Sweep band half dimensions: ego half length x (width + margin) / 2."""
    return corridor.ego_length_m / 2.0, (corridor.ego_width_m + corridor.width_margin_m) / 2.0


def support_half_extent(
    box_length_m: float, box_width_m: float, box_yaw_ego_deg: float, gap_dir_ego_rad: float
) -> float:
    """Support half extent of the annotation box along the gap direction (rules §0.5).

    ``h = 0.5·L·|cos(θ_obj − θ_gap)| + 0.5·W·|sin(θ_obj − θ_gap)|``——车身沿
    间隙方向伸出的半长：垂直横穿 → 半长主导；平行并入 → 半宽主导；斜向 →
    前外角主导（连续插值）。``box_yaw_ego_deg``（度）与 ``gap_dir_ego_rad``
    （弧度）都在 ego 系下，公式只消费差值（内部统一成弧度）。
    Step 3 的 extract.py 特征侧复用同一原语，保证两侧口径一致。
    """
    theta_obj = math.radians(box_yaw_ego_deg)
    delta = theta_obj - gap_dir_ego_rad
    cos_a = abs(math.cos(delta))
    sin_a = abs(math.sin(delta))
    return 0.5 * box_length_m * cos_a + 0.5 * box_width_m * sin_a


def clearance_curve(
    ego: EgoState, obj: ObjectState, corridor: CorridorConfig
) -> Tuple[Tuple[float, float, float], ...]:
    """Firing-engine samples ``((t_k, d_k, h_k), ...)`` per rules §0.4-0.5.

    K = 13 个双侧 CV 采样：对象中心在 ego 系沿相对速度匀速外推，``d_k`` 为
    中心点到扫掠带矩形（2.05 x 1.185 半维）的距离（带内取 0）。``d_k > 0``
    时计算间隙方向的支撑半延展 ``h_k``（间隙方向 = 从带最近点指向中心的
    单位向量；h 只用差角绝对值，方向符号无关）；``d_k = 0`` 的样本直接计为
    firing，h 不参与（记 0）。
    """
    half_len, half_wid = _band_half_dimensions(corridor)
    cos_y = math.cos(ego.yaw)
    sin_y = math.sin(ego.yaw)
    # ego 系下的相对 CV 运动：d(obj - ego)/dt（双侧恒速，"双方均不反应"）。
    rel_vx = obj.velocity_x_mps - ego.speed_mps * cos_y
    rel_vy = obj.velocity_y_mps - ego.speed_mps * sin_y
    dx0 = obj.x - ego.x
    dy0 = obj.y - ego.y

    samples = []
    n_steps = int(round(corridor.t_risk_s / corridor.time_step_s))
    for i in range(n_steps + 1):
        t = round(i * corridor.time_step_s, 6)
        gx = dx0 + rel_vx * t
        gy = dy0 + rel_vy * t
        local_x = cos_y * gx + sin_y * gy
        local_y = -sin_y * gx + cos_y * gy
        d = _rect_distance(local_x, local_y, half_len, half_wid)
        if d == 0.0:
            # 中心已在带内：无需间隙方向与 h，直接 firing（间隙必然 <= 0 < base）。
            samples.append((t, 0.0, 0.0))
            continue
        # 带矩形上最近点（ego 系，轴对齐 -> 逐轴 clamp）；间隙方向角取
        # "最近点 -> 中心"方向（ego 系）。
        qx = max(-half_len, min(half_len, local_x))
        qy = max(-half_wid, min(half_wid, local_y))
        gap_dir = math.atan2(local_y - qy, local_x - qx)
        h = support_half_extent(
            obj.box_length_m, obj.box_width_m, obj.box_yaw_ego_deg, gap_dir
        )
        samples.append((t, d, h))
    return tuple(samples)


def object_firing(samples: Sequence[Tuple[float, float, float]], base_m: float) -> bool:
    """P-2 firing test: exists d_k = 0, or min(d_k - h_k) < base (strict)."""
    min_clearance = math.inf
    for _, d, h in samples:
        if d == 0.0:
            return True  # 中心在带内的样本直接 firing
        clearance = d - h
        if clearance < min_clearance:
            min_clearance = clearance
    return min_clearance < base_m


def firing_base_for(category: str, config: RuleConfig) -> float:
    """Per-object base (rules §0.5): split 假设下 vru 族用 base_vru_m。

    split 时 static 归 base_vehicle 口径（= ``base_m``，v6 ``_d_risk_for``
    先例：非 pedestrian 一律同值）；single 假设下全类别统一 ``base_m``。
    """
    firing = config.firing
    if firing.assumption == "split" and category in VRU_CATEGORIES:
        return firing.base_vru_m
    return firing.base_m


def crossing_kinematics(
    ego: EgoState, obj: ObjectState, gate: GateConfig
) -> Tuple[float, float]:
    """Crossing arrival kinematics ``(v_toward, t_arrive)`` (rules §0.7, v2.4).

    ``v_lat`` = 相对速度在 ego 系的横向分量；带外（|y_c| > 半宽）取
    ``v_toward = -sign(y_c) · v_lat``、``t_arrive = (|y_c| − 半宽) / v_toward``
    （v_toward ≤ 0 即远离走廊，记 inf）；带内（|y_c| <= 半宽，中心已在带内）
    取 ``|v_lat|``、``t_arrive = 0``。判据本体见 :func:`crossing_hit`。
    """
    cos_y = math.cos(ego.yaw)
    sin_y = math.sin(ego.yaw)
    dvx = obj.velocity_x_mps - ego.speed_mps * cos_y
    dvy = obj.velocity_y_mps - ego.speed_mps * sin_y
    v_lat = -sin_y * dvx + cos_y * dvy
    _, y_c = ego_frame_xy(ego, obj.x, obj.y)
    if abs(y_c) <= gate.band_half_width_m:
        return abs(v_lat), 0.0
    v_toward = -math.copysign(1.0, y_c) * v_lat
    if v_toward <= 0.0:
        return v_toward, math.inf
    return v_toward, (abs(y_c) - gate.band_half_width_m) / v_toward


def crossing_hit(ego: EgoState, obj: ObjectState, gate: GateConfig) -> bool:
    """Crossing test (rules §0.7, v2.4): ``v_toward > thr`` AND ``t_arrive <= T``.

    两族共用同一判据；与 firing 解耦（v2.4 拍板）：firing 是"3s 双侧 CV
    反事实下物理间隙已不足"的证据，本判据是"沿当前速度多久进入走廊"的
    预测——远处横穿对象（3s 窗内不 firing）同样获得标签，缓慢漂移
    （t_arrive > T）者不获得。static 类别由调用方的族集合排除。
    """
    v_toward, t_arrive = crossing_kinematics(ego, obj, gate)
    return v_toward > gate.lateral_thr_mps and t_arrive <= gate.t_arrive_max_s


def in_path(ego: EgoState, obj: ObjectState, y_half_width_m: float) -> bool:
    """P-6 in-path condition: ``x_c > 0 AND |y_c| <= Y``（t0 中心点判据）。

    "前方" = x_c > 0（相对 ego 中心，非车头）；横向半宽由调用方传
    ``braking.y_brake_m`` 或 ``stationary_obstacle.y_st_m``。
    """
    x_c, y_c = ego_frame_xy(ego, obj.x, obj.y)
    return x_c > 0.0 and abs(y_c) <= y_half_width_m


# --- risk assembly -----------------------------------------------------------


def derive_risk_factors(
    pool: Sequence[ObjectState],
    tracks: Dict[str, ObjectTrack],
    motion_by_token: Dict[str, str],
    ego: EgoState,
    config: RuleConfig,
) -> Tuple[Tuple[str, ...], Dict[str, Tuple[str, ...]], bool]:
    """Compute the v7 taxonomy-ordered risk_factors + attribution for one anchor.

    ``pool`` objects must already be the selected top-8; ``tracks`` maps
    ann_token -> ObjectTrack (may be empty for objects without future
    annotations); ``motion_by_token`` maps ann_token -> motion_state.

    返回 ``(factors, attribution, any_firing)`` 三元组：``attribution``
    把每个命中因子映射到其全部合格对象的 ann_token 元组（rules §12 归因，
    D-A1-3 全量合格口径——归因机制为全收集，首中即止三因子去 break，因子
    存在 ⟺ 归因集合非空；v2.4 起 crossing 判据改为到达时间口径 §0.7，
    归因机制本身不变）；``any_firing`` 是池上 firing 的并集（含 static 与
    无标签 firing），直接喂 ``yield_required``（§7，与因子标签解耦）。
    """
    corridor = config.corridor
    hits = set()
    attribution: Dict[str, List[str]] = {}

    # Firing engine over the whole pool（v7 firing 无自有标签；v2.4 起
    # crossing 不再消费 firing，firing 只喂 yield 并集）。
    curves = {o.ann_token: clearance_curve(ego, o, corridor) for o in pool}
    firing_by_token = {
        o.ann_token: object_firing(curves[o.ann_token], firing_base_for(o.category, config))
        for o in pool
    }
    any_firing = any(firing_by_token.values())

    # §1/§2 vru_crossing / vehicle_crossing：到达时间判据（rules §0.7 v2.4：
    # v_toward > thr ∧ t_arrive ≤ T），与 firing 解耦——未 firing 的远处
    # 接近对象同样产生标签；类别集合分组天然互斥，static 两族皆不属。
    for obj in pool:
        if not crossing_hit(ego, obj, config.crossing_gate):
            continue
        if obj.category in VRU_CATEGORIES:
            hits.add("vru_crossing")
            attribution.setdefault("vru_crossing", []).append(obj.ann_token)
        elif obj.category in VEHICLE_CROSSING_CATEGORIES:
            hits.add("vehicle_crossing")
            attribution.setdefault("vehicle_crossing", []).append(obj.ann_token)

    # §3 lead_vehicle_braking：族 = vehicle 五类 ∪ 两轮车（P-5 集合），
    # 在路径放宽到 Y_brake（P-6），距离升序逐一检查、全部合格者归因（§12，
    # D-A1-3：原首中即止去 break，判定判据不动）。
    brake_cfg = config.braking
    leads = sorted(
        (
            o
            for o in pool
            if o.category in BRAKING_AND_ONCOMING_CATEGORIES
            and in_path(ego, o, brake_cfg.y_brake_m)
        ),
        key=lambda o: (round(o.distance_m, 3), o.ann_token),
    )
    for lead in leads:
        track = tracks.get(lead.ann_token)
        if track is None:
            continue  # 无未来轨迹：减速需正向证据，跳过
        if lead.speed_mps <= config.motion.stationary_speed_mps:
            continue  # 静止"前车"不是制动
        window = track.window(brake_cfg.evidence_window_s)
        if not window.speeds_mps:
            continue  # 证据窗为空：跳过
        if min(window.speeds_mps) <= lead.speed_mps - brake_cfg.speed_drop_mps:
            hits.add("lead_vehicle_braking")
            attribution.setdefault("lead_vehicle_braking", []).append(lead.ann_token)

    # §4 stationary_obstacle：全类别同一判据（P-6），"停着"两段式
    # （t0 停 ∧ 未来窗全停；无 track / 空窗回退仅 t0 判），在路径 Y_st；
    # 全部合格者归因（§12：原首中即止去 break，判定判据不动）。
    so_cfg = config.stationary_obstacle
    for obj in pool:
        if obj.speed_mps > so_cfg.max_speed_mps:
            continue  # t0 已在动：不满足"现在停"
        track = tracks.get(obj.ann_token)
        window_speeds = (
            track.window(so_cfg.evidence_window_s).speeds_mps if track is not None else ()
        )
        # "将继续停着"：未来窗最大速度 <= 0.5；窗缺失（无 track 或空窗）
        # 回退仅用 t0 速度——"保持停止"是缺省状态（与 braking 的不对称
        # 为有意设计）。
        max_speed = max(window_speeds) if window_speeds else obj.speed_mps
        if max_speed > so_cfg.max_speed_mps:
            continue
        if in_path(ego, obj, so_cfg.y_st_m):
            hits.add("stationary_obstacle")
            attribution.setdefault("stationary_obstacle", []).append(obj.ann_token)

    # §5 object_near：即时空间压力，全类别、无运动条件、含等号；
    # 归因 = 全部 ≤ r 的池内对象（§12）。
    near_tokens = [
        o.ann_token for o in pool if o.distance_m <= config.object_near.radius_m
    ]
    if near_tokens:
        hits.add("object_near")
        attribution["object_near"] = near_tokens

    # §6 oncoming_traffic：族按 P-4 集合，heading 规则 + 严格 < range；
    # 全部合格者归因（§12：原首中即止去 break，判定判据不动）。
    for obj in pool:
        if (
            obj.category in BRAKING_AND_ONCOMING_CATEGORIES
            and motion_by_token.get(obj.ann_token) == ONCOMING
            and obj.distance_m < config.oncoming.range_m
        ):
            hits.add("oncoming_traffic")
            attribution.setdefault("oncoming_traffic", []).append(obj.ann_token)

    factors = tuple(r for r in RISK_TAXONOMY_ORDER if r in hits)
    attribution_out = {
        r: tuple(attribution[r]) for r in RISK_TAXONOMY_ORDER if r in attribution
    }
    return factors, attribution_out, any_firing
