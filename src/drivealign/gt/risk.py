"""risk_factors GT: the closed 8-item taxonomy (S08 P1.5).

Purpose:
    Implement the frozen risk derivations. Two evidence regimes, per the
    2026-10-02 ledger:

    CORRIDOR FAMILY — t0 bilateral constant-velocity (CV) counterfactual
    (corridor_conflict / pedestrian_crossing / vehicle_merging):
        The ego corridor is the band the ego vehicle would sweep if it kept
        its t0 state (position, heading, backward-difference speed) for
        T_risk seconds: a rectangle (ego length x (ego width + margin))
        sliding along the heading line. The object side is also CV
        extrapolated from its t0 global velocity. Conflict = the minimum
        over sampled times of the distance between the object's CV point
        trajectory and the ego CV rectangle < the per-class-group d_risk.
        Semantics: "would they conflict within T_risk if NEITHER reacted" —
        the standard safety-analysis conflict definition. Using actual
        future trajectories would self-defeat: an ego that brakes for a
        crossing pedestrian would shorten its corridor and erase the very
        evidence that justified braking (regression-tested). Ego stationary
        degenerates naturally to the ego box + buffer (v=0 sweeps nothing).
        Runs ONLY on pool objects (visibility-first: no pool object, no
        risk factor).

    ACTUAL-TRAJECTORY EVIDENCE WINDOWS (lead_vehicle_braking /
    stationary_obstacle / congestion): direct observables that do not
    vanish when ego reacts, so they keep the actual future tracks.
        - lead_vehicle_braking: a vehicle-group lead (inside the ego
          rectangle at t0, ahead of ego) whose window speeds drop by >=
          speed_drop_mps below its t0 speed.
        - stationary_obstacle: an object inside the ego rectangle at t0
          (ahead) with t0 speed and ALL window speeds <= max_speed_mps.
        - congestion: >= min_queue_vehicles vehicle-group objects inside
          the ego rectangle at t0 (ahead) and the nearest one's window
          speeds all <= lead_speed_mps.
    oncoming_traffic stays a pure heading rule (motion_state == oncoming
    within range_m). small_following_gap was REMOVED from the taxonomy by
    the P1.2 calibration (<= 0.64% detection, starved label).

    Output is the taxonomy-ordered unique list (deterministic order).

Example launch command:
    ``conda activate autovla_codeclean && cd /root/autodl-tmp/drivealign_workspace && \
    PYTHONPATH=DriveAlign/src python -c \
    "from drivealign.gt.risk import EgoState, ObjectState, corridor_distance_curve; \
    ego = EgoState(x=0., y=0., yaw=0., speed_mps=10., length_m=4.1, width_m=1.87); \
    obj = ObjectState(ann_token='a', category='pedestrian', x=15., y=0., \
    velocity_x_mps=0., velocity_y_mps=2., distance_m=15., azimuth_deg=0.); \
    print(min(d for _, d in corridor_distance_curve(ego, obj, 3.0, 0.25, 0.5)) < 1.5)"``
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from drivealign.gt.config import CorridorConfig, RuleConfig
from drivealign.gt.motion_state import ONCOMING

# Closed taxonomy, canonical output order (v4 output_schema enum order).
RISK_TAXONOMY_ORDER = (
    "pedestrian_crossing",
    "vehicle_merging",
    "lead_vehicle_braking",
    "corridor_conflict",
    "stationary_obstacle",
    "congestion",
    "oncoming_traffic",
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
    """t0 state of one pool object (global frame + sensor-frame geometry)."""

    ann_token: str
    category: str  # canonical
    x: float
    y: float
    velocity_x_mps: float
    velocity_y_mps: float
    distance_m: float  # 3D norm of the sensor-frame center
    azimuth_deg: float

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


# --- corridor machinery (t0 bilateral CV) ------------------------------------


def _rect_distance(
    local_x: float, local_y: float, half_len: float, half_wid: float
) -> float:
    """Distance from a point (ego frame) to the axis-aligned ego rectangle."""
    dx = max(abs(local_x) - half_len, 0.0)
    dy = max(abs(local_y) - half_wid, 0.0)
    return math.hypot(dx, dy)


def ego_half_dimensions(corridor: CorridorConfig) -> Tuple[float, float]:
    """Half length and half width of the ego CV band (width + margin)."""
    return corridor.ego_length_m / 2.0, (corridor.ego_width_m + corridor.width_margin_m) / 2.0


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
    # Relative CV motion in the ego frame: d(obj - ego)/dt.
    rel_vx = obj.velocity_x_mps - ego.speed_mps * cos_y
    rel_vy = obj.velocity_y_mps - ego.speed_mps * sin_y
    dx0 = obj.x - ego.x
    dy0 = obj.y - ego.y

    points: List[Tuple[float, float]] = []
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


def point_in_ego_band(
    ego: EgoState,
    obj: ObjectState,
    width_margin_m: float,
    require_ahead: bool = True,
) -> bool:
    """Whether the object center lies inside the ego rectangle at t0."""
    cos_y = math.cos(ego.yaw)
    sin_y = math.sin(ego.yaw)
    gx, gy = obj.x - ego.x, obj.y - ego.y
    local_x = cos_y * gx + sin_y * gy
    local_y = -sin_y * gx + cos_y * gy
    half_len = ego.length_m / 2.0
    half_wid = (ego.width_m + width_margin_m) / 2.0
    if abs(local_x) > half_len or abs(local_y) > half_wid:
        return False
    return local_x > 0.0 if require_ahead else True


def in_ego_corridor_column(
    ego: EgoState, obj: ObjectState, width_margin_m: float
) -> bool:
    """Whether the object is AHEAD of ego within the driving band's lateral half-width.

    The "lead" membership test for braking / congestion / stationary
    evidence: the object sits in the same corridor the ego drives in (lateral
    offset within (ego width + margin) / 2) and in front of the ego center.
    Unlike :func:`point_in_ego_band` there is no longitudinal containment —
    a lead vehicle 8 m ahead is in the corridor even though its center is
    outside the ego rectangle.
    """
    cos_y = math.cos(ego.yaw)
    sin_y = math.sin(ego.yaw)
    gx, gy = obj.x - ego.x, obj.y - ego.y
    local_x = cos_y * gx + sin_y * gy
    local_y = -sin_y * gx + cos_y * gy
    half_wid = (ego.width_m + width_margin_m) / 2.0
    return local_x > 0.0 and abs(local_y) <= half_wid


def bilateral_cv_center_distance(
    ego: EgoState, obj: ObjectState, rel_time_s: float
) -> float:
    """Planar center-to-center distance under t0 bilateral CV at ``rel_time_s``."""
    cos_y = math.cos(ego.yaw)
    sin_y = math.sin(ego.yaw)
    ex = ego.x + ego.speed_mps * rel_time_s * cos_y
    ey = ego.y + ego.speed_mps * rel_time_s * sin_y
    ox = obj.x + obj.velocity_x_mps * rel_time_s
    oy = obj.y + obj.velocity_y_mps * rel_time_s
    return math.hypot(ox - ex, oy - ey)


# --- risk assembly -----------------------------------------------------------


def _d_risk_for(category: str, corridor: CorridorConfig) -> float:
    return (
        corridor.d_risk_pedestrian_m
        if category == "pedestrian"
        else corridor.d_risk_vehicle_m
    )


def derive_risk_factors(
    pool: Sequence[ObjectState],
    tracks: Dict[str, ObjectTrack],
    motion_by_token: Dict[str, str],
    ego: EgoState,
    group_of,
    config: RuleConfig,
) -> Tuple[str, ...]:
    """Compute the taxonomy-ordered risk_factors for one anchor.

    ``pool`` objects must already be the selected top-8; ``tracks`` maps
    ann_token -> ObjectTrack (may be empty for objects without future
    annotations); ``motion_by_token`` maps ann_token -> motion_state;
    ``group_of`` maps a canonical category to its class group (from the
    pool config).
    """
    corridor = config.corridor
    curves = {
        obj.ann_token: corridor_distance_curve(
            ego, obj, corridor.t_risk_s, corridor.time_step_s, corridor.width_margin_m
        )
        for obj in pool
    }
    firing = {
        obj.ann_token: corridor_conflict(curves[obj.ann_token], _d_risk_for(obj.category, corridor))
        for obj in pool
    }

    hits = set()

    # Corridor family (counterfactual; only pool objects by construction).
    if any(firing.values()):
        hits.add("corridor_conflict")
    if any(firing[o.ann_token] for o in pool if o.category == "pedestrian"):
        hits.add("pedestrian_crossing")
    if any(
        firing[o.ann_token]
        for o in pool
        if group_of(o.category) == "vehicle"
    ):
        hits.add("vehicle_merging")

    # Leads: vehicle-group objects inside the ego band at t0, ahead of ego,
    # nearest first (deterministic order by (distance, ann_token)).
    leads = sorted(
        (
            o
            for o in pool
            if group_of(o.category) == "vehicle"
            and in_ego_corridor_column(ego, o, corridor.width_margin_m)
        ),
        key=lambda o: (round(o.distance_m, 3), o.ann_token),
    )

    # lead_vehicle_braking: actual-trajectory deceleration evidence.
    brake_cfg = config.braking
    for lead in leads:
        track = tracks.get(lead.ann_token)
        if track is None:
            continue
        window = track.window(brake_cfg.evidence_window_s)
        if not window.speeds_mps:
            continue
        if lead.speed_mps <= config.motion.stationary_speed_mps:
            continue  # a stationary "lead" is not braking
        if min(window.speeds_mps) <= lead.speed_mps - brake_cfg.speed_drop_mps:
            hits.add("lead_vehicle_braking")
            break

    # stationary_obstacle: zero-speed evidence window inside the band.
    so_cfg = config.stationary_obstacle
    for obj in pool:
        if not in_ego_corridor_column(ego, obj, corridor.width_margin_m):
            continue
        if obj.speed_mps > so_cfg.max_speed_mps:
            continue
        track = tracks.get(obj.ann_token)
        max_speed = (
            max(track.window(so_cfg.evidence_window_s).speeds_mps)
            if track is not None and track.window(so_cfg.evidence_window_s).speeds_mps
            else obj.speed_mps
        )
        if max_speed <= so_cfg.max_speed_mps:
            hits.add("stationary_obstacle")
            break

    # congestion: low-speed lead with a queue of vehicles inside the band.
    cong_cfg = config.congestion
    if len(leads) >= cong_cfg.min_queue_vehicles:
        nearest = leads[0]
        track = tracks.get(nearest.ann_token)
        window = (
            track.window(brake_cfg.evidence_window_s)
            if track is not None
            else None
        )
        lead_speeds = window.speeds_mps if window is not None and window.speeds_mps else (nearest.speed_mps,)
        if max(lead_speeds) <= cong_cfg.lead_speed_mps:
            hits.add("congestion")

    # oncoming_traffic: heading rule within range (unchanged from v3).
    for obj in pool:
        if (
            group_of(obj.category) == "vehicle"
            and motion_by_token.get(obj.ann_token) == ONCOMING
            and obj.distance_m < config.oncoming.range_m
        ):
            hits.add("oncoming_traffic")
            break

    return tuple(r for r in RISK_TAXONOMY_ORDER if r in hits)
