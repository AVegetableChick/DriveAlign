"""Unit tests for the S08 risk_factors GT derivation.

Purpose:
    Freeze the P1.5 rules, with the SELF-DEFEATING-TRAP REGRESSION as the
    core case: the corridor family must use the t0 bilateral constant-
    velocity counterfactual, so an ego that reacts (decelerates for a
    crossing pedestrian) still carries the conflict evidence — an actual-
    trajectory corridor would erase it. Also freezes: the ego-stationary
    degenerate band, per-class d_risk separation, the lead membership
    column test, braking / stationary / congestion evidence windows, the
    oncoming heading rule, and the taxonomy output order.

Startup command:
    ``conda activate autovla_codeclean && PYTHONPATH=DriveAlign/src python -m \
    pytest DriveAlign/tests/unit/test_gt_risk.py -q``
"""

from __future__ import annotations

import math

import pytest

from drivealign.gt.config import load_rule_config
from drivealign.gt.motion_state import ONCOMING
from drivealign.gt.risk import (
    EgoState,
    ObjectState,
    ObjectTrack,
    bilateral_cv_center_distance,
    corridor_conflict,
    corridor_distance_curve,
    derive_risk_factors,
    in_ego_corridor_column,
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


def _obj(ann_token="a", category="pedestrian", x=15.0, y=-2.0, vx=0.0, vy=2.0, distance=15.0, azimuth=0.0):
    return ObjectState(
        ann_token=ann_token,
        category=category,
        x=x,
        y=y,
        velocity_x_mps=vx,
        velocity_y_mps=vy,
        distance_m=distance,
        azimuth_deg=azimuth,
    )


def _group_of(category, config):
    return config.pool.group_of(category)


# --- core: self-defeating trap regression ------------------------------------


def test_corridor_cv_survives_ego_reaction(config):
    """The anchor case: ego decelerates for a crossing pedestrian, the GT must
    still record the conflict (t0 CV), while an actual-trajectory corridor
    would erase it (implemented inline as the contrast)."""
    ego = _ego(speed=10.0, config=config)
    ped = _obj()  # 15 m ahead, crossing +y at 2 m/s
    curve = corridor_distance_curve(
        ego, ped, config.corridor.t_risk_s, config.corridor.time_step_s,
        config.corridor.width_margin_m,
    )
    assert corridor_conflict(curve, config.corridor.d_risk_pedestrian_m)

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
    assert actual_min >= config.corridor.d_risk_pedestrian_m


def test_object_side_symmetric_cv(config):
    """A cut-in vehicle that (in reality) aborts keeps its conflict evidence:
    the object side is also CV extrapolated from t0, not actual tracks."""
    ego = _ego(speed=10.0, config=config)
    cutter = _obj(ann_token="v", category="car", x=20.0, y=4.0, vx=-1.0, vy=-3.0, distance=20.6)
    curve = corridor_distance_curve(
        ego, cutter, config.corridor.t_risk_s, config.corridor.time_step_s,
        config.corridor.width_margin_m,
    )
    assert corridor_conflict(curve, config.corridor.d_risk_vehicle_m)


def test_ego_stationary_degenerates_to_ego_box_plus_buffer(config):
    ego = _ego(speed=0.0, config=config)
    near = _obj(x=3.0, y=0.0, vy=2.0, distance=3.0)  # just in front of the bumper
    far = _obj(ann_token="far", x=30.0, y=0.0, vy=2.0, distance=30.0)
    near_curve = corridor_distance_curve(
        ego, near, config.corridor.t_risk_s, config.corridor.time_step_s,
        config.corridor.width_margin_m,
    )
    far_curve = corridor_distance_curve(
        ego, far, config.corridor.t_risk_s, config.corridor.time_step_s,
        config.corridor.width_margin_m,
    )
    assert corridor_conflict(near_curve, config.corridor.d_risk_pedestrian_m)
    assert not corridor_conflict(far_curve, config.corridor.d_risk_pedestrian_m)


# --- per-class thresholds ------------------------------------------------------


def test_d_risk_separation_between_pedestrian_and_vehicle():
    curve = [(0.0, 3.0), (0.5, 1.8), (1.0, 2.4)]  # window minimum 1.8 m
    assert corridor_conflict(curve, 1.5) is False  # pedestrian buffer
    assert corridor_conflict(curve, 2.0) is True  # vehicle buffer


# --- lead membership column ----------------------------------------------------


def test_lead_column_requires_ahead_and_lateral_fit(config):
    ego = _ego(speed=10.0, config=config)
    assert in_ego_corridor_column(ego, _obj(category="car", x=8.0, y=0.5, vx=8.0, vy=0.0), config.corridor.width_margin_m)
    assert not in_ego_corridor_column(ego, _obj(category="car", x=8.0, y=3.0, vx=8.0, vy=0.0), config.corridor.width_margin_m)
    behind = _obj(category="car", x=-5.0, y=0.0, vx=8.0, vy=0.0)
    assert not in_ego_corridor_column(ego, behind, config.corridor.width_margin_m)


def test_lead_eight_meters_ahead_is_in_column_not_in_rect(config):
    ego = _ego(speed=10.0, config=config)
    lead = _obj(category="car", x=8.0, y=0.0, vx=8.0, vy=0.0)
    assert in_ego_corridor_column(ego, lead, config.corridor.width_margin_m)


# --- evidence-window rules ------------------------------------------------------


def _run(config, pool, tracks, motions, ego):
    return derive_risk_factors(
        pool, tracks, motions, ego, lambda c: _group_of(c, config), config
    )


def test_lead_vehicle_braking_fires_on_actual_deceleration(config):
    ego = _ego(speed=10.0, config=config)
    lead = _obj(ann_token="lead", category="car", x=10.0, y=0.0, vx=8.0, vy=0.0, distance=10.0)
    track = ObjectTrack(
        rel_times_s=(0.5, 1.0, 1.5),
        speeds_mps=(8.0, 7.4, 7.0),  # drops 1.0 m/s below t0 speed
        positions_m=((12.0, 0.0), (15.7, 0.0), (19.2, 0.0)),
    )
    risks = _run(config, [lead], {"lead": track}, {"lead": "same_direction"}, ego)
    assert "lead_vehicle_braking" in risks


def test_no_braking_without_speed_drop(config):
    ego = _ego(speed=10.0, config=config)
    lead = _obj(ann_token="lead", category="car", x=10.0, y=0.0, vx=8.0, vy=0.0, distance=10.0)
    track = ObjectTrack(rel_times_s=(0.5, 1.0), speeds_mps=(8.0, 7.9), positions_m=((12.0, 0.0), (15.9, 0.0)))
    risks = _run(config, [lead], {"lead": track}, {"lead": "same_direction"}, ego)
    assert "lead_vehicle_braking" not in risks


def test_stationary_obstacle_needs_zero_speed_window(config):
    ego = _ego(speed=10.0, config=config)
    parked = _obj(ann_token="park", category="car", x=10.0, y=0.0, vx=0.0, vy=0.0, distance=10.0)
    track = ObjectTrack(rel_times_s=(0.5, 1.0), speeds_mps=(0.1, 0.0), positions_m=((10.0, 0.0), (10.0, 0.0)))
    risks = _run(config, [parked], {"park": track}, {"park": "stationary"}, ego)
    assert "stationary_obstacle" in risks
    moving = _run(
        config,
        [parked],
        {"park": ObjectTrack(rel_times_s=(0.5,), speeds_mps=(3.0,), positions_m=((11.5, 0.0),))},
        {"park": "same_direction"},
        ego,
    )
    assert "stationary_obstacle" not in moving


def test_congestion_needs_queue_and_slow_lead(config):
    ego = _ego(speed=10.0, config=config)
    queue = [
        _obj(ann_token=f"q{i}", category="car", x=6.0 + 5.0 * i, y=0.0, vx=1.0, vy=0.0, distance=6.0 + 5.0 * i)
        for i in range(3)
    ]
    tracks = {
        f"q{i}": ObjectTrack(rel_times_s=(0.5, 1.0), speeds_mps=(1.0, 1.0), positions_m=((0.0, 0.0), (0.0, 0.0)))
        for i in range(3)
    }
    motions = {f"q{i}": "stationary" for i in range(3)}
    risks = _run(config, queue, tracks, motions, ego)
    assert "congestion" in risks
    risks_two = _run(config, queue[:2], {k: tracks[k] for k in ("q0", "q1")}, {k: motions[k] for k in ("q0", "q1")}, ego)
    assert "congestion" not in risks_two


def test_oncoming_heading_rule(config):
    ego = _ego(speed=10.0, config=config)
    oncomer = _obj(ann_token="on", category="car", x=18.0, y=2.0, vx=-10.0, vy=0.0, distance=18.1)
    risks = _run(config, [oncomer], {}, {"on": ONCOMING}, ego)
    assert "oncoming_traffic" in risks
    risks_moving_same = _run(config, [oncomer], {}, {"on": "same_direction"}, ego)
    assert "oncoming_traffic" not in risks_moving_same


# --- assembly -------------------------------------------------------------------


def test_taxonomy_output_order_and_corridor_family(config):
    ego = _ego(speed=10.0, config=config)
    ped = _obj(ann_token="p", category="pedestrian", x=15.0, y=-2.0, vx=0.0, vy=2.0, distance=15.1)
    cutter = _obj(ann_token="v", category="car", x=20.0, y=4.0, vx=-1.0, vy=-3.0, distance=20.6)
    risks = _run(config, [ped, cutter], {}, {"p": "crossing", "v": "crossing"}, ego)
    # A crossing pedestrian plus a cut-in vehicle fire the whole corridor
    # family, output in the frozen taxonomy order.
    assert risks == ("pedestrian_crossing", "vehicle_merging", "corridor_conflict")
    # Pool with ONLY the pedestrian: no vehicle-group object -> no merging.
    risks_ped_only = _run(config, [ped], {}, {"p": "crossing"}, ego)
    assert risks_ped_only == ("pedestrian_crossing", "corridor_conflict")


def test_bilateral_cv_center_distance_helper(config):
    ego = _ego(speed=10.0, config=config)
    obj = _obj(x=20.0, y=0.0, vx=-2.0, vy=0.0)  # ahead, head-on closing
    assert bilateral_cv_center_distance(ego, obj, 1.0) == pytest.approx(8.0)
