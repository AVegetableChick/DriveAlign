"""Unit tests for the v7 risk_factors GT derivation + firing engine (S13).

Purpose:
    Freeze the v7 rules (``S13_v7_risk_factor_rules.md`` v2.4, including the
    §12 attribution clause and the 2026-10-12 crossing revision), with the
    SELF-DEFEATING-TRAP REGRESSION as the core case: the firing engine must
    use the t0 bilateral constant-velocity counterfactual, so an ego that
    reacts (decelerates for a crossing pedestrian) still carries the conflict
    evidence. Also freezes: the clearance boundary (strict ``< base``), the
    center-inside-band direct firing, the support-function half extent, the
    crossing arrival test (in-band ``t_arrive = 0`` vs out-of-band
    ``(|y_c| - 1.185) / v_toward``, strict ``> thr`` and ``<= T``, DECOUPLED
    from firing), the six-factor taxonomy order, the
    ``(factors, attribution, any_firing)`` return contract (label-less +
    static firings feed yield; attribution = ALL qualifying objects per
    factor, V7-A1 / D-A1-3), the widened in-path bands (``Y_brake`` /
    ``Y_st``, P-6), the P-4/P-5 category-set families, and the split-base
    assumption.

Startup command:
    ``conda activate autovla_codeclean && PYTHONPATH=DriveAlign/src python -m \
    pytest DriveAlign/tests/unit/test_gt_risk.py -q``
"""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from drivealign.gt.config import load_rule_config
from drivealign.gt.motion_state import ONCOMING
from drivealign.gt.risk import (
    BRAKING_AND_ONCOMING_CATEGORIES,
    RISK_TAXONOMY_ORDER,
    VEHICLE_CROSSING_CATEGORIES,
    VRU_CATEGORIES,
    EgoState,
    ObjectState,
    ObjectTrack,
    clearance_curve,
    crossing_hit,
    crossing_kinematics,
    derive_risk_factors,
    firing_base_for,
    in_path,
    object_firing,
    support_half_extent,
)


@pytest.fixture(scope="module")
def config():
    return load_rule_config()


def _ego(x=0.0, y=0.0, yaw=0.0, speed=10.0, config=None):
    return EgoState(
        x=x,
        y=y,
        yaw=yaw,
        speed_mps=speed,
        length_m=config.corridor.ego_length_m,
        width_m=config.corridor.ego_width_m,
    )


def _obj(
    ann_token="a",
    category="pedestrian",
    x=15.0,
    y=-2.0,
    vx=0.0,
    vy=2.0,
    distance=15.0,
    azimuth=0.0,
    length=0.7,
    width=0.5,
    yaw_ego=0.0,
):
    return ObjectState(
        ann_token=ann_token,
        category=category,
        x=x,
        y=y,
        velocity_x_mps=vx,
        velocity_y_mps=vy,
        distance_m=distance,
        azimuth_deg=azimuth,
        box_length_m=length,
        box_width_m=width,
        box_yaw_ego_deg=yaw_ego,
    )


def _run(config, pool, tracks, motions, ego):
    """二元组视图 ``(factors, any_firing)``——既有断言面沿用；归因断言
    （§12 / V7-A1）直接调 ``derive_risk_factors`` 取三元组。"""
    factors, _attribution, any_firing = derive_risk_factors(
        pool, tracks, motions, ego, config
    )
    return factors, any_firing


def _car():
    """默认 car 盒几何（v6 测试同款）：L=4.5, W=2.0，朝向与 ego 平行。"""
    return dict(category="car", length=4.5, width=2.0, yaw_ego=0.0)


# --- core: self-defeating trap regression ------------------------------------


def test_firing_cv_survives_ego_reaction(config):
    """The anchor case: ego decelerates for a crossing pedestrian, the GT must
    still fire (t0 bilateral CV), while an actual-trajectory corridor would
    erase it (implemented inline as the contrast)."""
    ego = _ego(speed=10.0, config=config)
    ped = _obj()  # 15 m ahead, crossing +y at 2 m/s
    assert object_firing(clearance_curve(ego, ped, config.corridor), config.firing.base_m)

    # Contrast: the ACTUAL braking trajectory (10 -> 0 m/s over 1 s, then 0)
    # keeps the ego band far from the crossing point; if the rule had used
    # actual futures, the conflict would vanish. The CV curve only reads t0
    # state, so it cannot be defeated by ego behavior.
    half_wid = (config.corridor.ego_width_m + config.corridor.width_margin_m) / 2.0
    actual_min = math.inf
    for i in range(int(config.corridor.t_risk_s / config.corridor.time_step_s) + 1):
        t = i * config.corridor.time_step_s
        ego_x = 10.0 * t - 5.0 * t**2 if t <= 1.0 else 5.0  # braking profile
        ped_y = -2.0 + 2.0 * t
        dx = max(abs(ped.x - ego_x) - config.corridor.ego_length_m / 2.0, 0.0)
        dy = max(abs(ped_y) - half_wid, 0.0)
        actual_min = min(actual_min, math.hypot(dx, dy))
    assert actual_min >= config.firing.base_m


def test_object_side_symmetric_cv(config):
    """A cut-in vehicle that (in reality) aborts keeps its conflict evidence:
    the object side is also CV extrapolated from t0, not actual tracks."""
    ego = _ego(speed=10.0, config=config)
    cutter = _obj(ann_token="v", **_car(), x=20.0, y=4.0, vx=-1.0, vy=-3.0, distance=20.6)
    assert object_firing(clearance_curve(ego, cutter, config.corridor), config.firing.base_m)


# --- firing engine: boundary + direct firing + support half extent ------------


def test_clearance_boundary_is_strict(config):
    """firing ⟺ min(d_k − h_k) < base（严格）。同车道静止车正对 ego：
    clearance = x_c − 2.05（带半长）− 2.25（车半长），x=6.3 恰等于 base
    2.0 不触发，x=6.29 触发。"""
    ego = _ego(speed=0.0, config=config)
    at_base = _obj(ann_token="at", **_car(), x=6.3, y=0.0, vy=0.0, distance=6.3)
    below_base = _obj(ann_token="below", **_car(), x=6.29, y=0.0, vy=0.0, distance=6.29)
    assert not object_firing(
        clearance_curve(ego, at_base, config.corridor), config.firing.base_m
    )
    assert object_firing(
        clearance_curve(ego, below_base, config.corridor), config.firing.base_m
    )


def test_center_inside_band_fires_directly(config):
    """d_k = 0（中心已在扫掠带内）的样本直接计为 firing，无需 h。"""
    ego = _ego(speed=0.0, config=config)
    inside = _obj(ann_token="in", **_car(), x=1.0, y=0.0, vy=0.0, distance=1.0)
    samples = clearance_curve(ego, inside, config.corridor)
    assert any(d == 0.0 for _, d, _ in samples)
    assert object_firing(samples, config.firing.base_m)


def test_support_half_extent_orientation():
    """支撑函数：垂直横穿半长主导、平行并入半宽主导、斜向连续插值。"""
    # gap 方向 = 盒纵向（θ_obj = θ_gap）：h = 0.5·L（平行，半长主导）
    assert support_half_extent(4.5, 2.0, 0.0, 0.0) == pytest.approx(2.25)
    # gap 方向 ⊥ 盒纵向（θ_obj = θ_gap + 90°）：h = 0.5·W（垂直横穿，半宽主导）
    assert support_half_extent(4.5, 2.0, 90.0, 0.0) == pytest.approx(1.0)
    # 斜 45°：两轴等权插值
    expected = 0.5 * 4.5 * math.sqrt(2) / 2 + 0.5 * 2.0 * math.sqrt(2) / 2
    assert support_half_extent(4.5, 2.0, 45.0, 0.0) == pytest.approx(expected)
    # 绝对值口径：θ_obj − θ_gap 取 ±、超 180° 均不影响（225° ≡ 45°）
    assert support_half_extent(4.5, 2.0, -45.0, 0.0) == pytest.approx(expected)
    assert support_half_extent(4.5, 2.0, 225.0, 0.0) == pytest.approx(expected)


def test_clearance_curve_sample_grid(config):
    """K = 13 个采样（t_risk 3.0 s / step 0.25 s），t=0 含在内。"""
    ego = _ego(speed=10.0, config=config)
    obj = _obj(**_car(), x=15.0, y=0.0, vy=0.0, distance=15.0)
    samples = clearance_curve(ego, obj, config.corridor)
    assert len(samples) == 13
    assert samples[0][0] == 0.0 and samples[-1][0] == pytest.approx(3.0)


# --- crossing arrival test (§0.7, rules v2.4) ------------------------------------


def test_crossing_kinematics_directionality(config):
    """v_toward 方向性不变：带外朝向走廊为正 / 远离为负（t_arrive = inf）；
    带内取 |v_lat| 且 t_arrive = 0（中心已在带内，即时到达）。"""
    ego = _ego(speed=10.0, config=config)
    approaching = _obj(ann_token="v", **_car(), x=15.0, y=3.0, vy=-1.0, distance=15.3)
    v_toward, t_arrive = crossing_kinematics(ego, approaching, config.crossing_gate)
    assert v_toward == pytest.approx(1.0)
    assert t_arrive == pytest.approx((3.0 - 1.185) / 1.0)
    departing = _obj(ann_token="w", **_car(), x=15.0, y=3.0, vy=1.0, distance=15.3)
    v_toward, t_arrive = crossing_kinematics(ego, departing, config.crossing_gate)
    assert v_toward == pytest.approx(-1.0)
    assert t_arrive == math.inf

    inside = _obj(ann_token="i", **_car(), x=15.0, y=1.0, vy=-2.0, distance=15.0)
    v_toward, t_arrive = crossing_kinematics(ego, inside, config.crossing_gate)
    assert v_toward == pytest.approx(2.0)
    assert t_arrive == 0.0


def test_crossing_hit_in_band_only_needs_thr(config):
    """带内：t_arrive = 0，判据只剩 |v_lat| > thr（严格）。v_toward = 0.8
    恰等于阈值不命中，0.9 命中——原 P-3 带内规则原样保留。"""
    ego = _ego(speed=10.0, config=config)
    inside_at_thr = _obj(ann_token="w", **_car(), x=15.0, y=1.0, vy=0.8, distance=15.0)
    assert not crossing_hit(ego, inside_at_thr, config.crossing_gate)
    inside_slow = _obj(ann_token="v", **_car(), x=15.0, y=1.0, vy=-0.9, distance=15.0)
    assert crossing_hit(ego, inside_slow, config.crossing_gate)


def test_crossing_hit_thr_and_arrival_boundaries(config):
    """带外两道门槛：v_toward > thr（严格）；t_arrive <= T（用可分离数值
    避免浮点贴边，等号语义由 ≤ 给出）。"""
    ego = _ego(speed=10.0, config=config)
    # thr 边界：v_toward = 0.8 恰等阈值 → 不命中；0.81 → t = 1.815/0.81 ≈ 2.2 ≤ 5 命中。
    at_thr = _obj(ann_token="t", **_car(), x=15.0, y=3.0, vy=-0.8, distance=15.3)
    assert not crossing_hit(ego, at_thr, config.crossing_gate)
    above_thr = _obj(ann_token="a", **_car(), x=15.0, y=3.0, vy=-0.81, distance=15.3)
    assert crossing_hit(ego, above_thr, config.crossing_gate)
    # T 边界：v_toward = 1.0，|y_c| = 1.185 + 4.9 → t = 4.9 ≤ 5 命中；
    # |y_c| = 1.185 + 5.1 → t = 5.1 > 5 不命中（缓慢漂移被反向保护剔除）。
    within_t = _obj(ann_token="wt", **_car(), x=15.0, y=1.185 + 4.9, vy=-1.0, distance=16.1)
    assert crossing_hit(ego, within_t, config.crossing_gate)
    beyond_t = _obj(ann_token="bt", **_car(), x=15.0, y=1.185 + 5.1, vy=-1.0, distance=16.3)
    assert not crossing_hit(ego, beyond_t, config.crossing_gate)


def test_crossing_decoupled_from_firing(config):
    """v2.4 解耦回归：远处横穿对象（3s CV 反事实窗外，永不 firing）仍获得
    vru_crossing 标签；yield 并集只看 firing，故 any_firing = False。"""
    ego = _ego(speed=10.0, config=config)
    ped = _obj(ann_token="p", x=40.0, y=3.0, vy=-2.0, distance=40.1)
    factors, attribution, any_firing = derive_risk_factors(
        [ped], {}, {"p": "crossing"}, ego, config
    )
    assert factors == ("vru_crossing",)
    assert attribution == {"vru_crossing": ("p",)}
    assert any_firing is False


def test_crossing_family_split_static_never_labeled(config):
    """族划分：同一判据下行人 → vru_crossing、car → vehicle_crossing；
    static（barrier）即使通过到达时间判据也不进任何因子（其 firing 仍喂
    yield——本场景 barrier CV 外推 1.5s 后中心进带，firing 为 True）。"""
    ego = _ego(speed=10.0, config=config)
    ped = _obj(ann_token="p", x=15.0, y=3.0, vy=-2.0, distance=15.3)
    car = _obj(ann_token="c", **_car(), x=15.0, y=3.0, vy=-2.0, distance=15.3)
    barrier = _obj(
        ann_token="b", category="barrier", x=15.0, y=3.0, vy=-2.0, distance=15.3,
        length=2.0, width=0.5, yaw_ego=0.0,
    )
    factors, attribution, any_firing = derive_risk_factors(
        [ped, car, barrier], {}, {t: "crossing" for t in ("p", "c", "b")}, ego, config
    )
    assert factors == ("vru_crossing", "vehicle_crossing")
    assert attribution == {"vru_crossing": ("p",), "vehicle_crossing": ("c",)}
    assert any_firing is True


# --- in-path conditions (P-6) ----------------------------------------------------


def test_in_path_conditions(config):
    """x_c > 0 ∧ |y_c| ≤ Y：前方判定相对 ego 中心；Y 由调用方传入。"""
    ego = _ego(speed=10.0, config=config)
    assert in_path(ego, _obj(**_car(), x=8.0, y=0.5), config.braking.y_brake_m)
    assert not in_path(ego, _obj(**_car(), x=8.0, y=2.0), config.braking.y_brake_m)
    assert in_path(ego, _obj(**_car(), x=8.0, y=2.0), config.stationary_obstacle.y_st_m)
    assert not in_path(ego, _obj(**_car(), x=-5.0, y=0.0), config.braking.y_brake_m)


# --- evidence-window rules --------------------------------------------------------


def test_lead_vehicle_braking_fires_on_actual_deceleration(config):
    ego = _ego(speed=10.0, config=config)
    lead = _obj(ann_token="lead", **_car(), x=10.0, y=0.0, vx=8.0, vy=0.0, distance=10.0)
    track = ObjectTrack(
        rel_times_s=(0.5, 1.0, 1.5),
        speeds_mps=(8.0, 7.4, 7.0),  # drops 1.0 m/s below t0 speed
        positions_m=((12.0, 0.0), (15.7, 0.0), (19.2, 0.0)),
    )
    risks, _ = _run(config, [lead], {"lead": track}, {"lead": "same_direction"}, ego)
    assert risks == ("lead_vehicle_braking",)


def test_no_braking_without_speed_drop(config):
    ego = _ego(speed=10.0, config=config)
    lead = _obj(ann_token="lead", **_car(), x=10.0, y=0.0, vx=8.0, vy=0.0, distance=10.0)
    track = ObjectTrack(
        rel_times_s=(0.5, 1.0), speeds_mps=(8.0, 7.9), positions_m=((12.0, 0.0), (15.9, 0.0))
    )
    risks, _ = _run(config, [lead], {"lead": track}, {"lead": "same_direction"}, ego)
    assert "lead_vehicle_braking" not in risks


def test_braking_skips_trackless_stationary_and_off_path(config):
    """braking 跳过：无 track（需正向证据）、t0 静止、在路径外（Y_brake）。"""
    ego = _ego(speed=10.0, config=config)
    trackless = _obj(ann_token="t0", **_car(), x=10.0, y=0.0, vx=8.0, vy=0.0, distance=10.0)
    risks, _ = _run(config, [trackless], {}, {"t0": "same_direction"}, ego)
    assert "lead_vehicle_braking" not in risks

    stationary_lead = _obj(ann_token="s0", **_car(), x=10.0, y=0.0, vy=0.0, distance=10.0)
    risks, _ = _run(config, [stationary_lead], {}, {"s0": "stationary"}, ego)
    assert "lead_vehicle_braking" not in risks

    off_path = _obj(ann_token="o0", **_car(), x=10.0, y=2.0, vx=8.0, vy=0.0, distance=10.2)
    track = ObjectTrack(
        rel_times_s=(0.5, 1.0), speeds_mps=(8.0, 7.0), positions_m=((12.0, 2.0), (15.9, 2.0))
    )
    risks, _ = _run(config, [off_path], {"o0": track}, {"o0": "same_direction"}, ego)
    assert "lead_vehicle_braking" not in risks


def test_braking_family_includes_two_wheelers(config):
    """P-5 族口径按类别集合：bicycle 减速同样是 braking 证据（v7 池里
    两轮车属 vru 组，但 braking 族仍包含它们）。"""
    assert "bicycle" in BRAKING_AND_ONCOMING_CATEGORIES
    assert "motorcycle" in BRAKING_AND_ONCOMING_CATEGORIES
    ego = _ego(speed=10.0, config=config)
    bike = _obj(
        ann_token="bike", category="bicycle", x=10.0, y=0.0, vx=8.0, vy=0.0,
        distance=10.0, length=1.6, width=0.6,
    )
    track = ObjectTrack(
        rel_times_s=(0.5, 1.0), speeds_mps=(8.0, 6.9), positions_m=((12.0, 0.0), (15.7, 0.0))
    )
    risks, _ = _run(config, [bike], {"bike": track}, {"bike": "same_direction"}, ego)
    assert risks == ("lead_vehicle_braking",)


def test_stationary_obstacle_needs_all_window_stopped(config):
    """'停着'两段式：t0 停 ∧ 未来窗全停；窗内有动样本 → 不命中。"""
    ego = _ego(speed=10.0, config=config)
    parked = _obj(ann_token="park", **_car(), x=10.0, y=0.0, vy=0.0, distance=10.0)
    stopped_track = ObjectTrack(
        rel_times_s=(0.5, 1.0), speeds_mps=(0.1, 0.0), positions_m=((10.0, 0.0), (10.0, 0.0))
    )
    risks, _ = _run(config, [parked], {"park": stopped_track}, {"park": "stationary"}, ego)
    assert "stationary_obstacle" in risks

    moving_track = ObjectTrack(
        rel_times_s=(0.5,), speeds_mps=(3.0,), positions_m=((11.5, 0.0),)
    )
    risks, _ = _run(config, [parked], {"park": moving_track}, {"park": "same_direction"}, ego)
    assert "stationary_obstacle" not in risks


def test_stationary_obstacle_falls_back_to_t0_without_track(config):
    """无 track / 空窗回退仅 t0 判（与 braking 的跳过不对称，有意设计）。"""
    ego = _ego(speed=10.0, config=config)
    parked = _obj(ann_token="park", **_car(), x=10.0, y=0.0, vy=0.0, distance=10.0)
    risks, _ = _run(config, [parked], {}, {"park": "stationary"}, ego)
    assert "stationary_obstacle" in risks


def test_stationary_obstacle_widened_band_all_categories(config):
    """P-6：在路径放宽到 Y_st（> Y_brake），全类别同一判据（barrier 亦然）。
    out 档取 Y_st + 0.1（从 config 读取），不依赖具体占位值。"""
    assert config.stationary_obstacle.y_st_m > config.braking.y_brake_m
    y_out = config.stationary_obstacle.y_st_m + 0.1
    ego = _ego(speed=10.0, config=config)
    boxes = (
        _car(),  # car：vehicle 五类之一
        dict(category="barrier", length=2.0, width=0.5, yaw_ego=0.0),  # static
    )
    for box in boxes:
        in_widened = _obj(ann_token="w0", **box, x=12.0, y=2.0, vy=0.0, distance=12.2)
        risks, _ = _run(config, [in_widened], {}, {"w0": "stationary"}, ego)
        assert "stationary_obstacle" in risks
        out_widened = _obj(
            ann_token="w1", **box, x=12.0, y=y_out, vy=0.0, distance=12.0 + y_out
        )
        risks, _ = _run(config, [out_widened], {}, {"w1": "stationary"}, ego)
        assert "stationary_obstacle" not in risks


def test_object_near_boundary_inclusive(config):
    """object_near：distance_m ≤ r（含等号），全类别无运动条件。边界取值
    从 config 读取（r 已随 Stage C 冻结为 5.5），不硬编码。"""
    ego = _ego(speed=0.0, config=config)
    r = config.object_near.radius_m
    # 横向移动的行人（speed 0.7 > 0.5 避开 stationary；带内 |v_lat| = 0.7
    # < thr 0.8 不触发 crossing，保持本用例聚焦 object_near 边界）。
    at_r = _obj(ann_token="n0", x=r, y=0.0, vy=0.7, distance=r)
    risks, firing = _run(config, [at_r], {}, {"n0": "crossing"}, ego)
    assert risks == ("object_near",)
    assert firing is False  # t0 中心距 r - 2.05，clearance > base：不 firing
    beyond = _obj(ann_token="n1", x=r + 0.01, y=0.0, vy=0.7, distance=r + 0.01)
    risks, _ = _run(config, [beyond], {}, {"n1": "crossing"}, ego)
    assert risks == ()


def test_oncoming_heading_rule_and_range(config):
    ego = _ego(speed=10.0, config=config)
    oncomer = _obj(ann_token="on", **_car(), x=18.0, y=2.0, vx=-10.0, vy=0.0, distance=18.1)
    risks, _ = _run(config, [oncomer], {}, {"on": ONCOMING}, ego)
    assert risks == ("oncoming_traffic",)
    risks_same, _ = _run(config, [oncomer], {}, {"on": "same_direction"}, ego)
    assert "oncoming_traffic" not in risks_same
    # 严格 < range（20 m）。
    at_range = _obj(ann_token="edge", **_car(), x=20.0, y=0.0, vx=-10.0, vy=0.0, distance=20.0)
    risks, _ = _run(config, [at_range], {}, {"edge": ONCOMING}, ego)
    assert "oncoming_traffic" not in risks


def test_oncoming_family_includes_two_wheelers_excludes_vru_and_static(config):
    """P-4 族口径：两轮车在族；行人 / static 不在（行人 oncoming 只可能
    经 firing 喂 yield，不产生标签）。"""
    assert "bicycle" in BRAKING_AND_ONCOMING_CATEGORIES
    assert "pedestrian" not in BRAKING_AND_ONCOMING_CATEGORIES
    assert "barrier" not in BRAKING_AND_ONCOMING_CATEGORIES
    # braking / oncoming 族 = vehicle 五类 ∪ 两轮车（P-4/P-5 同一口径）。
    assert BRAKING_AND_ONCOMING_CATEGORIES == (
        VEHICLE_CROSSING_CATEGORIES | {"bicycle", "motorcycle"}
    )
    assert VRU_CATEGORIES == {"pedestrian", "bicycle", "motorcycle"}
    ego = _ego(speed=10.0, config=config)
    bike = _obj(
        ann_token="bike", category="bicycle", x=18.0, y=0.0, vx=-10.0, vy=0.0,
        distance=18.0, length=1.6, width=0.6, yaw_ego=180.0,
    )
    risks, _ = _run(config, [bike], {}, {"bike": ONCOMING}, ego)
    assert risks == ("oncoming_traffic",)

    ped = _obj(ann_token="ped", x=18.0, y=0.0, vx=-10.0, vy=0.0, distance=18.0, yaw_ego=180.0)
    risks, firing = _run(config, [ped], {}, {"ped": ONCOMING}, ego)
    assert risks == ()  # 族外：无标签
    assert firing is True  # 但 firing 仍喂 yield


# --- assembly: taxonomy order + (factors, any_firing) ---------------------------


def test_taxonomy_output_order_and_tuple_return(config):
    ego = _ego(speed=10.0, config=config)
    ped = _obj(ann_token="p", x=15.0, y=-2.0, vy=2.0, distance=15.1)
    lead = _obj(ann_token="lead", **_car(), x=10.0, y=0.0, vx=8.0, vy=0.0, distance=10.0)
    lead_track = ObjectTrack(
        rel_times_s=(0.5, 1.0, 1.5),
        speeds_mps=(8.0, 7.4, 7.0),  # min = v0 − drop：命中证据窗判据
        positions_m=((12.0, 0.0), (15.7, 0.0), (19.2, 0.0)),
    )
    factors, any_firing = _run(
        config, [ped, lead], {"lead": lead_track}, {"p": "crossing", "lead": "same_direction"}, ego
    )
    # taxonomy 顺序：vru_crossing(0) 在 lead_vehicle_braking(2) 之前。
    assert factors == ("vru_crossing", "lead_vehicle_braking")
    assert any_firing is True


def test_label_less_firing_feeds_yield_only(config):
    """R-OPT-1 修复回归：同道接近（v_lat ≈ 0）不过 gate —— 无 crossing
    标签，但 firing 并集（any_firing）仍为 True 喂 yield。"""
    ego = _ego(speed=10.0, config=config)
    same_lane = _obj(ann_token="v", **_car(), x=20.0, y=0.0, vx=-4.0, vy=0.0, distance=20.0)
    factors, any_firing = _run(config, [same_lane], {}, {"v": "same_direction"}, ego)
    assert factors == ()
    assert any_firing is True


def test_static_firing_feeds_yield_without_label(config):
    """static 对象 firing 同样只喂 yield（P-6：不参与任何因子标签）。
    barrier 放在路径外（|y_c| = Y_st + 0.1，从 config 读取）避免
    stationary_obstacle 干扰。"""
    ego = _ego(speed=10.0, config=config)
    y_out = config.stationary_obstacle.y_st_m + 0.1
    barrier = _obj(
        ann_token="b", category="barrier", x=8.0, y=y_out, vy=0.0, distance=8.0 + y_out,
        length=2.0, width=0.5, yaw_ego=0.0,
    )
    factors, any_firing = _run(config, [barrier], {}, {"b": "stationary"}, ego)
    assert factors == ()
    assert any_firing is True


def test_firing_base_for_split_assumption(config):
    """P-4 分裂假设：vru 族用 base_vru_m；vehicle / static 用 base_m。"""
    split_cfg = replace(
        config,
        firing=replace(config.firing, assumption="split", base_m=2.0, base_vru_m=1.0),
    )
    assert firing_base_for("pedestrian", split_cfg) == 1.0
    assert firing_base_for("bicycle", split_cfg) == 1.0
    assert firing_base_for("car", split_cfg) == 2.0
    assert firing_base_for("barrier", split_cfg) == 2.0  # static 归 vehicle 口径
    assert firing_base_for("car", config) == 2.0  # single 假设统一 base_m

    # 边界行为：区分假设必须用 vru 对象（vehicle 的 split base 仍是
    # base_m，无区分度）。行人 clearance ≈ 1.50，落在 (base_vru=1.0,
    # base=2.0) 区间内——single 假设下 firing → yield=true，split 假设下
    # 不 firing；v2.4 起 crossing 与 firing 解耦（带内 |v_lat| = 1.0 >
    # 0.8，t_arrive = 0），故两假设下 vru_crossing 标签完全一致。
    ego = _ego(speed=0.0, config=config)
    ped = _obj(ann_token="p", x=3.9, y=0.0, vy=-1.0, distance=3.9)
    factors_single, firing_single = _run(config, [ped], {}, {"p": "crossing"}, ego)
    assert factors_single == ("vru_crossing", "object_near")
    assert firing_single is True
    factors_split, firing_split = _run(split_cfg, [ped], {}, {"p": "crossing"}, ego)
    assert factors_split == factors_single  # crossing 判据与 firing 分族假设无关
    assert firing_split is False


# --- attribution (rules §12, V7-A1 / D-A1-3) -------------------------------------


def test_attribution_collects_all_qualifying_objects(config):
    """归因口径（D-A1-3）：因子 → 全部合格对象的 ann_token 元组。
    vru_crossing / object_near = 池序；braking = 距离升序（原循环排序键）。
    池故意非距离序，验证两条确定性顺序各自成立。"""
    ego = _ego(speed=10.0, config=config)
    ped_far = _obj(ann_token="pf", x=15.0, y=-2.0, vy=2.0, distance=15.1)
    ped_near = _obj(ann_token="pn", x=4.0, y=-2.0, vy=2.0, distance=4.5)
    lead_far = _obj(ann_token="lf", **_car(), x=12.0, y=0.0, vx=8.0, vy=0.0, distance=12.0)
    lead_near = _obj(ann_token="ln", **_car(), x=10.0, y=0.0, vx=8.0, vy=0.0, distance=10.0)

    def _brake_track():
        return ObjectTrack(
            rel_times_s=(0.5, 1.0),
            speeds_mps=(8.0, 7.0),  # min = v0 − drop：两车都合格
            positions_m=((12.0, 0.0), (15.7, 0.0)),
        )

    tracks = {"lf": _brake_track(), "ln": _brake_track()}
    motions = {
        "pf": "crossing",
        "pn": "crossing",
        "lf": "same_direction",
        "ln": "same_direction",
    }
    pool = [ped_far, lead_far, ped_near, lead_near]  # 故意非距离序
    factors, attribution, any_firing = derive_risk_factors(
        pool, tracks, motions, ego, config
    )
    assert factors == ("vru_crossing", "lead_vehicle_braking", "object_near")
    assert attribution["vru_crossing"] == ("pf", "pn")  # 池序
    assert attribution["lead_vehicle_braking"] == ("ln", "lf")  # 距离升序
    assert attribution["object_near"] == ("pn",)  # 仅 4.5 ≤ r=5.0
    assert any_firing is True


def test_stationary_attribution_collects_all_in_path(config):
    """D-A1-3：两辆在径静止车都归入 stationary_obstacle（原首中即止去
    break 改全收集，判定判据不动——第二辆在 Y_st 带内、t0 停）。"""
    ego = _ego(speed=0.0, config=config)
    parked_a = _obj(ann_token="pa", **_car(), x=10.0, y=0.0, vy=0.0, distance=10.0)
    parked_b = _obj(ann_token="pb", **_car(), x=14.0, y=1.0, vy=0.0, distance=14.0)
    factors, attribution, _ = derive_risk_factors(
        [parked_a, parked_b],
        {},
        {"pa": "stationary", "pb": "stationary"},
        ego,
        config,
    )
    assert factors == ("stationary_obstacle",)
    assert attribution["stationary_obstacle"] == ("pa", "pb")


def test_attribution_empty_when_no_factors(config):
    """因子存在 ⟺ 归因集合非空：无因子帧的归因映射为空 dict。"""
    ego = _ego(speed=10.0, config=config)
    far = _obj(ann_token="f", **_car(), x=40.0, y=6.0, vy=0.0, distance=40.4)
    factors, attribution, any_firing = derive_risk_factors(
        [far], {}, {"f": "same_direction"}, ego, config
    )
    assert factors == ()
    assert attribution == {}
    assert any_firing is False


# --- frozen config invariants -----------------------------------------------------


def test_frozen_config_invariants(config):
    """加载断言的当前值校验：band = (ego_width + margin)/2、t_arrive 上限
    v2.4 冻结值 5.0、Y_st ≥ Y_brake、assumption 枚举、per-class 覆盖键
    ⊆ 已知类别。"""
    corridor = config.corridor
    gate = config.crossing_gate
    assert gate.band_half_width_m == pytest.approx(
        (corridor.ego_width_m + corridor.width_margin_m) / 2.0
    )
    assert gate.t_arrive_max_s == 5.0  # v2.4 冻结：crossing 到达时间上限
    assert config.stationary_obstacle.y_st_m >= config.braking.y_brake_m
    assert config.firing.assumption in ("single", "split")
    known = {c for members in config.pool.class_groups.values() for c in members}
    assert set(config.pool.cone_range_by_class_m) <= known
    assert set(RISK_TAXONOMY_ORDER) == {
        "vru_crossing",
        "vehicle_crossing",
        "lead_vehicle_braking",
        "stationary_obstacle",
        "object_near",
        "oncoming_traffic",
    }
