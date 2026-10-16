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
    - ``yield_required`` — CONFLICT evidence: true iff ANY pool object fired
      under the v7 firing engine (``risk.derive_risk_factors`` returns
      ``(factors, any_firing)``). The union INCLUDES static objects and
      label-less firings — the v7 firing mechanism has no taxonomy label of
      its own, it feeds only the two crossing factors and this union (rules
      doc §0.6 / §7). The actual "did ego yield" behavior is deliberately
      NOT mixed in — that would re-create the self-defeating trap where a
      good driver erases the conflict evidence (P1.6 evidence split).

Example launch command:
    ``conda activate autovla_codeclean && cd /root/autodl-tmp/drivealign_workspace && \
    PYTHONPATH=DriveAlign/src python -c \
    "from drivealign.gt.action_yield import derive_speed_action, derive_yield_required; \
    print(derive_speed_action(8.0, [7.9, 6.0, 4.0], delta=0.15, stop_speed_mps=0.5)); \
    print(derive_yield_required(True))"``
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple

ACCELERATE = "ACCELERATE"
KEEP_SPEED = "KEEP_SPEED"
DECELERATE = "DECELERATE"
STOP = "STOP"


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


def derive_yield_required(any_firing: bool) -> bool:
    """v7 yield = firing 并集（rules §7）：risk 引擎直接给出布尔入参。

    与因子标签完全解耦：static 对象与无标签 firing 同样贡献 true。
    """ 
    return bool(any_firing)


__all__ = [
    "ACCELERATE",
    "KEEP_SPEED",
    "DECELERATE",
    "STOP",
    "future_ego_speeds",
    "derive_speed_action",
    "derive_yield_required",
]
