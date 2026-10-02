"""Unit tests for the S08 motion_state GT derivation.

Purpose:
    Freeze the P1.4 revised rules: stationary threshold first, then the
    three symmetric bands filling [0, 180] with NO dead zones (band edges
    theta / 180 - theta belong to crossing), and the relative-angle helper.

Startup command:
    ``conda activate autovla_codeclean && PYTHONPATH=DriveAlign/src python -m \
    pytest DriveAlign/tests/unit/test_gt_motion.py -q``
"""

from __future__ import annotations

import math

from drivealign.gt.motion_state import (
    CROSSING,
    JUDGMENT_ORDER,
    ONCOMING,
    SAME_DIRECTION,
    STATIONARY,
    classify_motion_state,
    relative_angle_deg,
)


def test_stationary_threshold():
    assert classify_motion_state(0.4, 90.0, 0.5, 45.0) == STATIONARY
    assert classify_motion_state(0.5, 90.0, 0.5, 45.0) == CROSSING  # not stationary
    assert classify_motion_state(0.0, 0.0, 0.5, 45.0) == STATIONARY


def test_band_edges_45_135_are_crossing():
    # theta = 45: [45, 135] is crossing; edges included.
    for rel in (44.9, 0.0):
        assert classify_motion_state(2.0, rel, 0.5, 45.0) == SAME_DIRECTION
    for rel in (45.0, 90.0, 135.0):
        assert classify_motion_state(2.0, rel, 0.5, 45.0) == CROSSING
    for rel in (135.1, 180.0):
        assert classify_motion_state(2.0, rel, 0.5, 45.0) == ONCOMING


def test_bands_fill_the_full_range_no_dead_zones():
    theta = 60.0
    for deg in range(0, 181):
        state = classify_motion_state(2.0, float(deg), 0.5, theta)
        assert state in (SAME_DIRECTION, CROSSING, ONCOMING)
        if deg < theta:
            assert state == SAME_DIRECTION
        elif deg <= 180 - theta:
            assert state == CROSSING
        else:
            assert state == ONCOMING


def test_theta_30_and_60_boundaries():
    assert classify_motion_state(2.0, 30.0, 0.5, 30.0) == CROSSING
    assert classify_motion_state(2.0, 150.0, 0.5, 30.0) == CROSSING
    assert classify_motion_state(2.0, 150.1, 0.5, 30.0) == ONCOMING
    assert classify_motion_state(2.0, 60.0, 0.5, 60.0) == CROSSING
    assert classify_motion_state(2.0, 59.9, 0.5, 60.0) == SAME_DIRECTION


def test_judgment_order_special_before_general():
    assert JUDGMENT_ORDER == (STATIONARY, CROSSING, ONCOMING, SAME_DIRECTION)


def test_relative_angle_helper():
    assert relative_angle_deg(0.0, 5.0, 0.0) == 90.0  # pure lateral
    assert relative_angle_deg(5.0, 0.0, 0.0) == 0.0  # along heading
    assert relative_angle_deg(-5.0, 0.0, 0.0) == 180.0  # head-on
    assert relative_angle_deg(5.0, 0.0, math.pi / 2) == 90.0  # rotated heading
    assert relative_angle_deg(3.0, 4.0, 0.0) == pytest_approx(53.13)
    assert relative_angle_deg(0.0, 0.0, 0.0) == 0.0  # degenerate -> caller gates on speed
    assert relative_angle_deg(float("nan"), 1.0, 0.0) == 0.0  # NaN velocity


def pytest_approx(value):
    import pytest

    return pytest.approx(value, abs=1e-2)
