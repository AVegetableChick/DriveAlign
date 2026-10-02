"""Candidate pool construction: geometric relevance + top-8 (S08 P1.1).

Purpose:
    Reduce the observability gate survivors to the per-frame critical object
    set (aligned with the model contract ``maxItems: 8`` — truncation and
    scoring use the same set). Pure geometry, ordered AFTER the gating:

    1. forward cone: |azimuth| <= ``cone_half_angle_deg`` (azimuth around the
       CAM_FRONT optical axis, the practical proxy for the ego heading) AND
       distance < the class-group cone range;
    2. OR omnidirectional proximity disc: distance < ``proximity_disc_radius_m``
       regardless of azimuth. The disc never pulls in gate survivors that the
       cone rejects at long range — it only waives the ANGLE restriction for
       near objects (side-front large targets partially in frame);
    3. distance-ascending top-8; exact distance ties break by the annotation
       token lexicographically (determinism requirement).

    ``select_objects`` is parameterized by the pool config so the P1.2
    calibration can sweep the grid without touching the rule code.

Example launch command:
    ``conda activate autovla_codeclean && cd /root/autodl-tmp/drivealign_workspace && \
    PYTHONPATH=DriveAlign/src python -c \
    "from drivealign.gt.config import load_rule_config; \
    from drivealign.gt.pool import select_objects; \
    cfg = load_rule_config().pool; \
    print(select_objects([], cfg))"``
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Tuple

from drivealign.gt.config import PoolConfig
from drivealign.gt.observability import INFERABLE, OBSERVABLE

MAX_POOL_SIZE = 8  # aligns with output_schema critical_objects.maxItems


@dataclass(frozen=True)
class PoolCandidate:
    """One gate survivor with the facts the pool geometry needs.

    ``azimuth_deg`` is the planar angle around the CAM_FRONT optical axis
    (0 = straight ahead, positive to the right) computed from the sensor-
    frame box center; ``distance_m`` is the 3D norm of that center.
    """

    ann_token: str
    instance_token: str
    category: str  # canonical (closed 10-class vocabulary)
    distance_m: float
    azimuth_deg: float
    bbox_2d: Tuple[float, float, float, float]
    visibility_state: str  # observable | inferable
    # Global-frame facts for downstream rules (filled by the collector).
    global_x: float = 0.0
    global_y: float = 0.0
    velocity_x_mps: float = 0.0
    velocity_y_mps: float = 0.0

    @property
    def speed_mps(self) -> float:
        return math.hypot(self.velocity_x_mps, self.velocity_y_mps)


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
    # Distance ascending; exact ties break by annotation token (lexicographic,
    # so the order is machine-independent).
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


def azimuth_deg_from_center(center_x: float, center_z: float) -> float:
    """Planar azimuth around the optical axis from a sensor-frame center."""
    return math.degrees(math.atan2(center_x, center_z))


__all__ = [
    "MAX_POOL_SIZE",
    "PoolCandidate",
    "azimuth_deg_from_center",
    "select_objects",
    "truncated_count",
]
