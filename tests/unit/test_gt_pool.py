"""Unit tests for the S08 candidate pool construction.

Purpose:
    Freeze the P1.1 geometry: cone filtering with per-class-group ranges,
    the omnidirectional proximity disc as an ANGLE-only exemption for near
    gate survivors, top-8 truncation, and the sample-token lexicographic
    tie-break for exact distance ties (determinism).

Startup command:
    ``conda activate autovla_codeclean && PYTHONPATH=DriveAlign/src python -m \
    pytest DriveAlign/tests/unit/test_gt_pool.py -q``
"""

from __future__ import annotations

import pytest

from drivealign.gt.config import load_rule_config
from drivealign.gt.pool import MAX_POOL_SIZE, PoolCandidate, select_objects, truncated_count


def _cand(ann_token, category="car", distance=30.0, azimuth=0.0, state="observable"):
    return PoolCandidate(
        ann_token=ann_token,
        instance_token=f"inst-{ann_token}",
        category=category,
        distance_m=distance,
        azimuth_deg=azimuth,
        bbox_2d=(0.0, 0.0, 10.0, 10.0),
        visibility_state=state,
    )


@pytest.fixture(scope="module")
def pool_cfg():
    return load_rule_config().pool


def test_cone_rejects_large_azimuth(pool_cfg):
    kept = select_objects(
        [_cand("a", azimuth=40.0, distance=30.0)], pool_cfg
    )
    assert kept == ()  # 40 deg > 30 deg cone and outside the 10 m disc


def test_disc_waives_the_angle_limit_for_near_objects(pool_cfg):
    kept = select_objects(
        [_cand("a", azimuth=80.0, distance=9.0)], pool_cfg
    )
    assert len(kept) == 1  # inside the 10 m disc despite the 80 deg azimuth


def test_cone_range_is_per_class_group(pool_cfg):
    pedestrian_far = _cand("p", category="pedestrian", distance=40.0)
    vehicle_far = _cand("v", category="car", distance=40.0)
    kept = select_objects([pedestrian_far, vehicle_far], pool_cfg)
    assert [c.ann_token for c in kept] == ["v"]  # ped range 30 < 40, vehicle 45 > 40


def test_unscorable_states_never_enter_the_pool(pool_cfg):
    kept = select_objects([_cand("a", state="unscorable")], pool_cfg)
    assert kept == ()


def test_distance_ascending_top8_truncation(pool_cfg):
    candidates = [_cand(f"tok{i:02d}", distance=10.0 + i) for i in range(12)]
    kept = select_objects(candidates, pool_cfg)
    assert len(kept) == MAX_POOL_SIZE == 8
    assert [c.distance_m for c in kept] == sorted(c.distance_m for c in kept)
    assert [c.ann_token for c in kept] == [f"tok{i:02d}" for i in range(8)]
    assert truncated_count(candidates, pool_cfg) == 12


def test_distance_tie_breaks_by_ann_token_lexicographic(pool_cfg):
    tied = [
        _cand("token-b", distance=20.0),
        _cand("token-a", distance=20.0),
        _cand("token-c", distance=20.0),
    ]
    kept = select_objects(tied, pool_cfg)
    assert [c.ann_token for c in kept] == ["token-a", "token-b", "token-c"]


def test_select_is_parameterized_by_config(pool_cfg):
    from dataclasses import replace

    wide = replace(pool_cfg, cone_half_angle_deg=90.0)
    cand = _cand("a", azimuth=80.0, distance=30.0)
    assert select_objects([cand], pool_cfg) == ()
    assert len(select_objects([cand], wide)) == 1
