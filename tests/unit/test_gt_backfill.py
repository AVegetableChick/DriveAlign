"""Unit tests for the S13 v7 backfill pipeline over SYNTHETIC evidence.

Purpose:
    Exercise :func:`build_expected_output` end to end without nuScenes:
    a stationary-car scene (stationary_obstacle, STOP, yield=False), a
    crossing-pedestrian scene (v7 firing engine -> vru_crossing,
    yield=True, DECELERATE), the empty scene, the closed-vocabulary drop
    of unmapped categories, byte-identical determinism of a rebuild, and
    the BEV nearest-point GT (BEV rules §3.3) both as a unit (the frozen
    verification example ``20 m ahead, L=4.5 -> (17.8, 0.0)``, the rotated
    box, the -0.0 normalization) and through the assembled
    ``critical_objects`` (each carrying an integer ``id`` 0..N-1, V7-A1).
    ``risk_factors`` is the attribution form ``[{factor, objects}]`` (V7-A1):
    ``objects`` holds the critical-object ids the factor is attributed to,
    sorted ascending. The v7 contract has FOUR fields — the reasoning
    render chain is retired (V7-C1) and the v7 output_schema validation
    runs inline as the fail-fast gate. A final integration test runs
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
    bev_nearest_point,
    build_expected_output,
)
from drivealign.gt.config import load_rule_config

MINI_ROOT = Path(__file__).resolve().parents[3] / "data" / "nuscenes" / "mini"

INTRINSICS = np.array(
    [[500.0, 0.0, 400.0], [0.0, 500.0, 300.0], [0.0, 0.0, 1.0]]
)
IMAGE_SIZE = (800, 600)
CFG = load_rule_config()


def _box(name, token, x, y, z, w=2.0, l=4.5, h=1.5):
    return Box(
        center=(float(x), float(y), float(z)),
        size=(w, l, h),
        orientation=Quaternion([1.0, 0.0, 0.0, 0.0]),
        name=name,
        token=token,
    )


def _ann(inst, cat, x, y, vx=0.0, vy=0.0, w=2.0, l=4.5, yaw=0.0):
    """Global-frame annotation record（v7：size + yaw 为 firing/BEV 必填几何）。"""
    return AnnRecord(inst, cat, x, y, vx, vy, w, l, yaw)


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


def _assert_v7_shape(out):
    """V7-C1：expected_output 收敛为四个契约字段（reasoning 已退役）；
    V7-A1：critical_object 带 ``id`` 整数、risk_factors 为归因对象列表。"""
    assert set(out) == {
        "critical_objects", "risk_factors", "yield_required", "speed_action"
    }
    for idx, obj in enumerate(out["critical_objects"]):
        assert set(obj) == {
            "id", "category", "bbox_2d", "motion_state", "bev_x", "bev_y"
        }
        # GT 侧 id = 池序下标（0..N-1），确定性分配（V7-A1 / D-A1-2）。
        assert obj["id"] == idx
    for entry in out["risk_factors"]:
        # 归因条目 {factor, objects}；objects 为升序 id 列表且都指向池内对象。
        assert set(entry) == {"factor", "objects"}
        assert 0 < len(entry["objects"]) <= len(out["critical_objects"])
        assert entry["objects"] == sorted(entry["objects"])
        assert all(0 <= i < len(out["critical_objects"]) for i in entry["objects"])


def _stationary_car_evidence():
    """Car parked 15 m dead ahead; ego stationary; one future keyframe."""
    anns = {
        "ann-car": _ann("inst-car", "vehicle.car", 15.0, 0.0),
        "ann-car-f1": _ann("inst-car", "vehicle.car", 15.0, 0.0),
    }
    frames = [FutureFrame("tok-f1", 1_500_000, {"inst-car": "ann-car-f1"})]
    boxes = [_box("vehicle.car", "ann-car", 0.0, 0.0, 15.0)]
    return _evidence(boxes, anns, ["ann-car"], frames)


def _crossing_ped_evidence():
    """Pedestrian 8 m ahead crossing right-to-left; ego closes at 5 m/s."""
    anns = {
        "ann-ped": _ann("inst-ped", "human.pedestrian.adult", 8.0, 5.0, 0.0, -1.5, w=0.6, l=0.5),
        "ann-ped-f1": _ann("inst-ped", "human.pedestrian.adult", 8.0, 4.25, 0.0, -1.5, w=0.6, l=0.5),
        "ann-ped-f2": _ann("inst-ped", "human.pedestrian.adult", 8.0, 3.5, 0.0, -1.5, w=0.6, l=0.5),
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
    out = build_expected_output(_stationary_car_evidence(), 0.0, [(0.0, 0.0, 1_500_000)], CFG)
    _assert_v7_shape(out)  # schema gate ran inline (fail-fast would raise)
    assert len(out["critical_objects"]) == 1
    obj = out["critical_objects"][0]
    assert obj["category"] == "car" and obj["motion_state"] == "stationary"
    _valid_bbox(obj["bbox_2d"])
    assert out["risk_factors"] == [{"factor": "stationary_obstacle", "objects": [0]}]
    assert out["speed_action"] == "STOP"  # future window is all-zero speed
    assert out["yield_required"] is False
    # BEV GT：15 m 正前 L=4.5 → 最近点 x = 15 − 2.25 = 12.75 → 0.1 取整 12.8。
    assert obj["bev_x"] == pytest.approx(12.8)
    assert obj["bev_y"] == pytest.approx(0.0)


def test_crossing_pedestrian_fires_vru_crossing():
    out = build_expected_output(_crossing_ped_evidence(), 5.0, BRAKING_POSES, CFG)
    _assert_v7_shape(out)
    # 到达时间判据（rules v2.4）：v_toward = 1.5 > 0.8，t_arrive =
    # (5 − 1.185)/1.5 ≈ 2.5 s ≤ 5 → vru_crossing（与 firing 解耦，
    # firing 仍为 True 一并喂 yield）。
    assert out["risk_factors"] == [{"factor": "vru_crossing", "objects": [0]}]
    assert out["yield_required"] is True
    assert out["speed_action"] == "DECELERATE"  # 2.5 < 5.0 * (1 - 0.15), >= stop
    obj = out["critical_objects"][0]
    assert obj["category"] == "pedestrian" and obj["motion_state"] == "crossing"
    _valid_bbox(obj["bbox_2d"])
    # BEV GT：盒中心 (8, 5)，L=0.5/W=0.6 → 最近点 (7.75, 4.7) → (7.8, 4.7)。
    assert obj["bev_x"] == pytest.approx(7.8)
    assert obj["bev_y"] == pytest.approx(4.7)


def test_empty_scene_yields_empty_state():
    out = build_expected_output(_evidence([], {}, [], []), 5.0, [], CFG)
    _assert_v7_shape(out)
    assert out["critical_objects"] == []
    assert out["risk_factors"] == []
    assert out["speed_action"] == "KEEP_SPEED"  # empty future window degrades
    assert out["yield_required"] is False


def test_rebuild_is_byte_identical():
    kwargs = (5.0, BRAKING_POSES, CFG)
    first = build_expected_output(_crossing_ped_evidence(), *kwargs)
    second = build_expected_output(_crossing_ped_evidence(), *kwargs)
    assert first == second


def test_unmapped_category_dropped_from_pool():
    anns = {
        "ann-barrier": _ann("inst-bar", "movable_object.barrier", 12.0, 3.0, w=0.5, l=2.0),
    }
    boxes = [
        # bicycle_rack folds to None in the category map -> dropped entirely.
        _box("static_object.bicycle_rack", "ann-rack", -1.0, -0.5, 10.0),
        _box("movable_object.barrier", "ann-barrier", -3.0, -1.0, 12.0),
    ]
    out = build_expected_output(
        _evidence(boxes, anns, ["ann-barrier"], []), 0.0, [], CFG
    )
    assert [o["category"] for o in out["critical_objects"]] == ["barrier"]
    # Barrier |y_c| = 3.0 恰落入 v7 放宽后的在路径带（Y_st = 3.0，Stage C
    # 冻结值；占位值 2.5 时代此处无因子）→ stationary_obstacle 命中且
    # 归因到该 barrier；distance ≈ 12.4 > r，不 firing，yield 仍为 False。
    assert out["risk_factors"] == [{"factor": "stationary_obstacle", "objects": [0]}]
    assert out["yield_required"] is False
    # BEV GT：盒中心 (12, 3)，L=2.0/W=0.5 → 最近点 (11.0, 2.75) → (11.0, 2.8)。
    assert out["critical_objects"][0]["bev_x"] == pytest.approx(11.0)
    assert out["critical_objects"][0]["bev_y"] == pytest.approx(2.8)


# --- BEV nearest point unit contract (BEV rules §3.3) -------------------------


def test_bev_verification_example_frozen():
    """冻结验算例：20 m 正前 L=4.5 的车 → 最近点 17.75 → 0.1 取整 17.8
    （banker's rounding，与 parser output.py 严格同款 round(v, 1)）。"""
    assert bev_nearest_point(20.0, 0.0, 4.5, 2.0, 0.0) == (17.8, 0.0)


def test_bev_rotated_box_uses_width_edge():
    """盒纵向垂直于视线（yaw 90°）：ego 侧露出的是宽度半边 W/2=1.0 →
    最近点 (9.0, 0.0)，与支撑函数的垂直/平行互换一致。"""
    assert bev_nearest_point(10.0, 0.0, 4.5, 2.0, 90.0) == (9.0, 0.0)


def test_bev_center_inside_box_is_the_center():
    """ego 中心已在盒内：clamp 后 q=0 → 最近点 = 盒中心本身。"""
    assert bev_nearest_point(0.0, 0.0, 4.5, 2.0, 0.0) == (0.0, 0.0)


def test_bev_normalizes_negative_zero():
    """正侧方情形 round 出 -0.0：``+ 0.0`` 归一后 GT 面不出现 -0.0 字面量
    （数值相等，用 copysign 检查符号位）。"""
    bev_x, bev_y = bev_nearest_point(0.0, 10.0, 4.5, 2.0, 0.0)
    assert (bev_x, bev_y) == (0.0, 9.0)
    assert math.copysign(1.0, bev_x) == 1.0
    assert math.copysign(1.0, bev_y) == 1.0


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
    _assert_v7_shape(out)
    assert len(out["critical_objects"]) <= 8
