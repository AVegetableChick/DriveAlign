"""speed_action and yield_required GT (S08 P1.6).

Purpose:
    The two decision-level labels with the ledger's evidence split:

    - ``speed_action`` — BEHAVIOR evidence: derived from the ACTUAL future
      ego trajectory over ``window_s`` (3.0 s = 6 keyframes at 2 Hz, aligned
      with oracle_only.future_ego_poses). Priority STOP > DECELERATE >
      ACCELERATE > KEEP_SPEED:
        STOP       : min future speed < stop_speed_mps
        DECELERATE : min future speed < v0 * (1 - delta), STOP not triggered
        ACCELERATE : max future speed > v0 * (1 + delta), neither above
        KEEP_SPEED : otherwise
      With an empty future window (scene-end truncation is NOT a quarantine)
      the label degrades to KEEP_SPEED and the caller can count the case.
    - ``yield_required`` — CONFLICT evidence: pure counterfactual, true iff
      the corridor family fired (corridor_conflict / pedestrian_crossing /
      vehicle_merging). The actual "did ego yield" behavior is deliberately
      NOT mixed in — that would re-create the self-defeating trap where a
      good driver erases the conflict evidence (P1.6 evidence split).

Example launch command:
    ``conda activate autovla_codeclean && cd /root/autodl-tmp/drivealign_workspace && \
    PYTHONPATH=DriveAlign/src python -c \
    "from drivealign.gt.action_yield import derive_speed_action, derive_yield_required; \
    print(derive_speed_action(8.0, [7.9, 6.0, 4.0], delta=0.15, stop_speed_mps=0.5)); \
    print(derive_yield_required(['corridor_conflict']))"``
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple

ACCELERATE = "ACCELERATE"
KEEP_SPEED = "KEEP_SPEED"
DECELERATE = "DECELERATE"
STOP = "STOP"

# Corridor family (the counterfactual conflict evidence for yield_required).
CORRIDOR_FAMILY = (
    "corridor_conflict",
    "pedestrian_crossing",
    "vehicle_merging",
)


def future_ego_speeds(
    anchor_x: float,
    anchor_y: float,
    anchor_timestamp_us: int,
    future_poses: Sequence[Tuple[float, float, int]],
    window_s: float,
) -> Tuple[float, ...]:
    """Planar speeds between consecutive ego poses inside ``window_s``.

    ``future_poses`` are (x, y, timestamp_us) in chronological order; the
    anchor pose closes the first finite difference. Speeds are rounded to
    3 decimals (same precision as the record's ego speed).
    """
    speeds = []
    prev_x, prev_y, prev_ts = anchor_x, anchor_y, anchor_timestamp_us
    for x, y, ts in future_poses:
        dt_s = (ts - prev_ts) / 1e6
        if dt_s <= 0.0:
            continue
        rel_s = (ts - anchor_timestamp_us) / 1e6
        if rel_s > window_s:
            break
        speed = round(math.hypot(x - prev_x, y - prev_y) / dt_s, 3)
        speeds.append(speed)
        prev_x, prev_y, prev_ts = x, y, ts
    return tuple(speeds)


def derive_speed_action(
    v0_mps: float,
    future_speeds_mps: Sequence[float],
    delta: float,
    stop_speed_mps: float,
) -> str:
    """Actual-future-trajectory behavior label with the frozen priority."""
    if not future_speeds_mps:
        return KEEP_SPEED  # scene-end truncation: no future behavior visible
    v_min = min(future_speeds_mps)
    v_max = max(future_speeds_mps)
    if v_min < stop_speed_mps:
        return STOP
    if v_min < v0_mps * (1.0 - delta):
        return DECELERATE
    if v_max > v0_mps * (1.0 + delta):
        return ACCELERATE
    return KEEP_SPEED


def derive_yield_required(risk_factors: Sequence[str]) -> bool:
    """True iff any corridor-family risk fired (pure counterfactual)."""
    return any(r in CORRIDOR_FAMILY for r in risk_factors)


__all__ = [
    "ACCELERATE",
    "KEEP_SPEED",
    "DECELERATE",
    "STOP",
    "CORRIDOR_FAMILY",
    "future_ego_speeds",
    "derive_speed_action",
    "derive_yield_required",
]
