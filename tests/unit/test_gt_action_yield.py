"""Unit tests for speed_action / yield_required GT (S08 P1.6).

Purpose:
    Freeze the action priority (STOP > DECELERATE > ACCELERATE > KEEP_SPEED)
    with the STOP and +/- delta boundary cases, the window truncation of the
    future ego speed helper, and the v7 yield contract: ``yield_required``
    is the firing union passed in as a plain bool (static / label-less
    firings already folded in by the risk engine) — the actual "did ego
    yield" behavior must never enter the label.

Startup command:
    ``conda activate autovla_codeclean && PYTHONPATH=DriveAlign/src python -m \
    pytest DriveAlign/tests/unit/test_gt_action_yield.py -q``
"""

from __future__ import annotations

from drivealign.gt.action_yield import (
    ACCELERATE,
    DECELERATE,
    KEEP_SPEED,
    STOP,
    derive_speed_action,
    derive_yield_required,
    future_ego_speeds,
)

DELTA = 0.15
STOP_SPEED = 0.5


def test_stop_boundary():
    # v0 = 10 -> decelerate threshold 8.5, accelerate 11.5.
    assert derive_speed_action(10.0, [0.49], DELTA, STOP_SPEED) == STOP
    assert derive_speed_action(10.0, [0.5], DELTA, STOP_SPEED) == DECELERATE


def test_decelerate_boundary():
    assert derive_speed_action(10.0, [8.4], DELTA, STOP_SPEED) == DECELERATE
    assert derive_speed_action(10.0, [8.5], DELTA, STOP_SPEED) == KEEP_SPEED


def test_accelerate_boundary():
    assert derive_speed_action(10.0, [11.6], DELTA, STOP_SPEED) == ACCELERATE
    assert derive_speed_action(10.0, [11.5], DELTA, STOP_SPEED) == KEEP_SPEED


def test_stop_has_priority_over_accelerate():
    assert derive_speed_action(10.0, [0.2, 11.6], DELTA, STOP_SPEED) == STOP


def test_keep_speed_window():
    assert derive_speed_action(10.0, [9.0, 10.5, 11.0], DELTA, STOP_SPEED) == KEEP_SPEED


def test_empty_future_degrades_to_keep_speed():
    assert derive_speed_action(10.0, [], DELTA, STOP_SPEED) == KEEP_SPEED


def test_from_standstill_any_motion_is_accelerate():
    # v0 = 0: the plan formula makes any forward motion ACCELERATE.
    assert derive_speed_action(0.0, [0.6, 1.2], DELTA, STOP_SPEED) == ACCELERATE


def test_future_ego_speeds_and_window_truncation():
    t0 = 1_510_000_000_000_000
    poses = [(5.0, 0.0, t0 + 1_000_000), (5.0, 1.0, t0 + 2_000_000), (8.0, 1.0, t0 + 3_000_000)]
    speeds = future_ego_speeds(0.0, 0.0, t0, poses, window_s=3.0)
    assert speeds == (5.0, 1.0, 3.0)  # 5 m over 1 s; 1 m; 3 m
    short = future_ego_speeds(0.0, 0.0, t0, poses, window_s=1.5)
    assert short == (5.0,)  # poses beyond the window are truncated


def test_yield_is_firing_union_bool():
    """v7 §7：yield = firing 并集（risk 引擎已归约为布尔入参），与因子
    标签解耦——static / 无标签 firing 同样贡献 True（在 risk 引擎测试里
    冻结场景级回归，此处冻结纯函数契约）。"""
    assert derive_yield_required(True) is True
    assert derive_yield_required(False) is False
