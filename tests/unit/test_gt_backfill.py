"""Unit tests for the S08 backfill pipeline over SYNTHETIC evidence.

Purpose:
    Exercise :func:`build_expected_output` end to end without nuScenes:
    a stationary-car scene (stationary_obstacle evidence window, STOP,
    yield=False), a crossing-pedestrian scene (counterfactual corridor
    family fires -> pedestrian_crossing + corridor_conflict, yield=True,
    DECELERATE), the empty scene (empty-state reasoning passes the gate),
    byte-identical determinism of a rebuild, and the closed-vocabulary
    drop of unmapped categories. A final integration test runs
    :func:`backfill_anchor` on the real v1.0-mini dataroot (skipped when
    mini is absent).

Startup command:
    ``conda activate autovla_codeclean && PYTHONPATH=DriveAlign/src python -m \
    pytest DriveAlign/tests/unit/test_gt_backfill.py -q``
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion

from drivealign.gt.backfill import (
    AnchorEvidence,
    AnnRecord,
    FutureFrame,
    backfill_anchor,
    build_expected_output,
)
from drivealign.gt.config import load_phrase_map, load_rule_config, load_templates
from drivealign.gt.render import check_zero_contradiction

MINI_ROOT = Path(__file__).resolve().parents[3] / "data" / "nuscenes" / "mini"

INTRINSICS = np.array(
    [[500.0, 0.0, 400.0], [0.0, 500.0, 300.0], [0.0, 0.0, 1.0]]
)
IMAGE_SIZE = (800, 600)
CFG = load_rule_config()
TEMPLATES = load_templates()
PHRASES = load_phrase_map()


def _box(name, token, x, y, z, w=2.0, l=4.5, h=1.5):
    return Box(
        center=(float(x), float(y), float(z)),
        size=(w, l, h),
        orientation=Quaternion([1.0, 0.0, 0.0, 0.0]),
        name=name,
        token=token,
    )


def _evidence(camera_boxes, ann_records, anchor_anns, future_frames):
    return AnchorEvidence(
        sample_token="tok-test",
        image_size=IMAGE_SIZE,
        intrinsics=INTRINSICS,
        camera_boxes=tuple(camera_boxes),
        ann_records=dict(ann_records),
        anchor_ann_tokens=tuple(anchor_anns),
        future_frames=tuple(future_frames),
        ego_x=0.0,
        ego_y=0.0,
        ego_yaw=0.0,
        ego_timestamp_us=1_000_000,
    )


def _valid_bbox(bbox):
    assert len(bbox) == 4
    assert all(v >= 0 for v in bbox)
    assert bbox[0] < bbox[2] and bbox[1] < bbox[3]
    assert bbox[2] <= IMAGE_SIZE[0] and bbox[3] <= IMAGE_SIZE[1]


def _gate(out):
    return check_zero_contradiction(
        out["reasoning"], out["critical_objects"], out["risk_factors"], PHRASES
    )


def _stationary_car_evidence():
    """Car parked 15 m dead ahead; ego stationary; one future keyframe."""
    anns = {
        "ann-car": AnnRecord("inst-car", "vehicle.car", 15.0, 0.0, 0.0, 0.0),
        "ann-car-f1": AnnRecord("inst-car", "vehicle.car", 15.0, 0.0, 0.0, 0.0),
    }
    frames = [FutureFrame("tok-f1", 1_500_000, {"inst-car": "ann-car-f1"})]
    boxes = [_box("vehicle.car", "ann-car", 0.0, 0.0, 15.0)]
    return _evidence(boxes, anns, ["ann-car"], frames)


def _crossing_ped_evidence():
    """Pedestrian 8 m ahead crossing right-to-left; ego closes at 5 m/s."""
    anns = {
        "ann-ped": AnnRecord("inst-ped", "human.pedestrian.adult", 8.0, 5.0, 0.0, -1.5),
        "ann-ped-f1": AnnRecord("inst-ped", "human.pedestrian.adult", 8.0, 4.25, 0.0, -1.5),
        "ann-ped-f2": AnnRecord("inst-ped", "human.pedestrian.adult", 8.0, 3.5, 0.0, -1.5),
    }
    frames = [
        FutureFrame("tok-f1", 1_500_000, {"inst-ped": "ann-ped-f1"}),
        FutureFrame("tok-f2", 2_000_000, {"inst-ped": "ann-ped-f2"}),
    ]
    # Sensor frame: z forward -> global x; x right -> -global y (ped at 5 m left).
    boxes = [_box("human.pedestrian.adult", "ann-ped", -5.0, -0.9, 8.0, w=0.6, l=0.5, h=1.8)]
    return _evidence(boxes, anns, ["ann-ped"], frames)


# Ego poses every 0.5 s braking 5.0 -> 2.5 m/s (finite differences).
BRAKING_POSES = [
    (2.25, 0.0, 1_500_000),
    (4.0, 0.0, 2_000_000),
    (5.25, 0.0, 2_500_000),
]


def test_stationary_car_scene_full_pipeline():
    out = build_expected_output(
        _stationary_car_evidence(), 0.0, [(0.0, 0.0, 1_500_000)], CFG, TEMPLATES, PHRASES
    )
    # Pipeline passed the v4 schema and the P4.5 gate (fail-fast would raise).
    assert set(out) == {
        "critical_objects", "risk_factors", "reasoning", "yield_required", "speed_action"
    }
    assert len(out["critical_objects"]) == 1
    obj = out["critical_objects"][0]
    assert obj["category"] == "car" and obj["motion_state"] == "stationary"
    _valid_bbox(obj["bbox_2d"])
    assert out["risk_factors"] == ["stationary_obstacle"]
    assert out["speed_action"] == "STOP"  # future window is all-zero speed
    assert out["yield_required"] is False
    assert "a stationary car" in out["reasoning"]
    assert _gate(out) == []


def test_crossing_pedestrian_fires_counterfactual_corridor_family():
    out = build_expected_output(
        _crossing_ped_evidence(), 5.0, BRAKING_POSES, CFG, TEMPLATES, PHRASES
    )
    # Taxonomy order; the counterfactual CV corridor dips under d_risk_ped.
    assert out["risk_factors"] == ["pedestrian_crossing", "corridor_conflict"]
    assert out["yield_required"] is True
    assert out["speed_action"] == "DECELERATE"  # 2.5 < 5.0 * (1 - 0.15), >= stop
    obj = out["critical_objects"][0]
    assert obj["category"] == "pedestrian" and obj["motion_state"] == "crossing"
    _valid_bbox(obj["bbox_2d"])
    assert "a crossing pedestrian" in out["reasoning"]
    assert _gate(out) == []


def test_empty_scene_renders_empty_state():
    out = build_expected_output(_evidence([], {}, [], []), 5.0, [], CFG, TEMPLATES, PHRASES)
    assert out["critical_objects"] == []
    assert out["risk_factors"] == []
    assert out["speed_action"] == "KEEP_SPEED"  # empty future window degrades
    assert out["yield_required"] is False
    assert len(out["reasoning"]) >= 1
    assert _gate(out) == []


def test_rebuild_is_byte_identical():
    kwargs = (5.0, BRAKING_POSES, CFG, TEMPLATES, PHRASES)
    first = build_expected_output(_crossing_ped_evidence(), *kwargs)
    second = build_expected_output(_crossing_ped_evidence(), *kwargs)
    assert first == second
    assert first["reasoning"].encode("utf-8") == second["reasoning"].encode("utf-8")


def test_unmapped_category_dropped_from_pool():
    anns = {
        "ann-barrier": AnnRecord("inst-bar", "movable_object.barrier", 12.0, 3.0, 0.0, 0.0),
    }
    boxes = [
        # bicycle_rack folds to None in the category map -> dropped entirely.
        _box("static_object.bicycle_rack", "ann-rack", -1.0, -0.5, 10.0),
        _box("movable_object.barrier", "ann-barrier", -3.0, -1.0, 12.0),
    ]
    out = build_expected_output(
        _evidence(boxes, anns, ["ann-barrier"], []), 0.0, [], CFG, TEMPLATES, PHRASES
    )
    assert [o["category"] for o in out["critical_objects"]] == ["barrier"]
    # Barrier is 3 m off the corridor column -> no risk fires.
    assert out["risk_factors"] == []
    assert _gate(out) == []


# --- real-data integration (v1.0-mini, skipped when absent) ------------------


def _ego_motion(nusc, anchor_token, num_future=6):
    """Backward-difference speed + future poses from the mini dataroot."""

    def pose(sample):
        sd = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
        ep = nusc.get("ego_pose", sd["ego_pose_token"])
        return float(ep["translation"][0]), float(ep["translation"][1]), int(sample["timestamp"])

    sample = nusc.get("sample", anchor_token)
    prev = nusc.get("sample", sample["prev"])
    ax, ay, at = pose(sample)
    px, py, pt = pose(prev)
    speed = round(math.hypot(ax - px, ay - py) / ((at - pt) / 1e6), 3)
    poses = []
    cur = sample
    while len(poses) < num_future and cur["next"]:
        cur = nusc.get("sample", cur["next"])
        poses.append(pose(cur))
    return speed, poses


def test_backfill_anchor_on_real_mini():
    if not MINI_ROOT.is_dir():
        pytest.skip("mini dataroot not present")
    from drivealign.data.nuscenes_io import load_nuscenes, resolve_anchor

    nusc = load_nuscenes(MINI_ROOT)
    anchor = resolve_anchor(nusc)
    speed, poses = _ego_motion(nusc, anchor)
    out = backfill_anchor(nusc, anchor, speed, poses)
    assert set(out) == {
        "critical_objects", "risk_factors", "reasoning", "yield_required", "speed_action"
    }
    assert len(out["critical_objects"]) <= 8
    assert len(out["reasoning"]) >= 1
    assert check_zero_contradiction(
        out["reasoning"], out["critical_objects"], out["risk_factors"], PHRASES
    ) == []
