"""motion_state GT: pure t0 kinematics, three symmetric bands (S08 P1.4).

Purpose:
    Classify one pool object's motion state from its t0 box velocity and the
    ego heading. The four v3 enum values are unchanged; the derivation is
    the 2026-10-02 revision:

    - judgment order (special before general): stationary -> crossing ->
      oncoming -> same_direction;
    - ``stationary``: planar box speed < ``stationary_speed_mps``;
    - three bands fill [0, 180] symmetrically so every moving object falls
      into exactly one band (no dead zones for diagonal cutters):
      same_direction < theta <= crossing <= 180 - theta < oncoming, with
      theta = ``band_half_width_deg``;
    - ``crossing`` is PURE kinematics now (relative angle in
      [theta, 180 - theta]); the corridor-intersection condition moved to
      the risk layer (pedestrian_crossing / corridor_conflict own it).

    The relative angle is the absolute planar angle between the object's
    global velocity and the ego heading; unavailable velocities (NaN from
    the devkit finite difference) are treated as zero speed -> stationary.

Example launch command:
    ``conda activate autovla_codeclean && cd /root/autodl-tmp/drivealign_workspace && \
    PYTHONPATH=DriveAlign/src python -c \
    "from drivealign.gt.motion_state import classify_motion_state; \
    print(classify_motion_state(2.0, 90.0, 0.5, 45.0))"``
"""

from __future__ import annotations

import math

STATIONARY = "stationary"
CROSSING = "crossing"
ONCOMING = "oncoming"
SAME_DIRECTION = "same_direction"

# Judgment order (first match wins; with the symmetric bands the bands are
# disjoint, the order is kept for ledger fidelity and future band changes).
JUDGMENT_ORDER = (STATIONARY, CROSSING, ONCOMING, SAME_DIRECTION)


def relative_angle_deg(vx: float, vy: float, ego_yaw: float) -> float:
    """Absolute planar angle [0, 180] between global velocity and ego heading.

    A zero (or non-finite) velocity vector yields 0.0 — callers must check
    speed first for the stationary decision.
    """
    if not (math.isfinite(vx) and math.isfinite(vy)) or (vx == 0.0 and vy == 0.0):
        return 0.0
    heading = math.degrees(math.atan2(vy, vx)) - math.degrees(ego_yaw)
    # Wrap to (-180, 180], then fold to [0, 180].
    heading = (heading + 180.0) % 360.0 - 180.0
    return abs(heading)


def classify_motion_state(
    object_speed_mps: float,
    rel_angle_deg: float,
    stationary_speed_mps: float,
    band_half_width_deg: float,
) -> str:
    """One motion_state enum value per the P1.4 revised derivation."""
    if object_speed_mps < stationary_speed_mps:
        return STATIONARY
    theta = band_half_width_deg
    if theta <= rel_angle_deg <= 180.0 - theta:
        return CROSSING
    if rel_angle_deg > 180.0 - theta:
        return ONCOMING
    return SAME_DIRECTION
