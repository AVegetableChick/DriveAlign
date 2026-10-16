"""Unit tests for Eval v3 BEV metrics (S13 eval v3 additions §2).

覆盖 G7 验收口径的 BEV 成员：**(17.75, 0) 验算例**（BEV rules §4.2
既有算例：前车 20 m 正前最近点 (17.75, 0)，构造预测点验证
``bev_dxy = hypot`` 与分桶归属）、三族全非负公式、d_gt 四桶边界
（[0,10)/[10,25)/[25,40)/40+ 开放桶，2026-10-14 四桶拍板）、
p50/p90 线性插值口径（与 matched_iou_p90 同一 _percentile）、
signed 方向分解（正/负子集分别求均值——严禁代数均值的拍板由
+2/−2 对抗例钉死）、空集全 0.0 口径、匹配对收集与 bev 缺失跳过。

Run:
    ``conda activate autovla_codeclean && cd /root/autodl-tmp/drivealign_workspace && \
    PYTHONPATH=DriveAlign/src python -m pytest DriveAlign/tests/unit/test_eval_metrics_bev.py -q``
"""

from __future__ import annotations

import math

import pytest

from drivealign.evaluation.matching import match_objects
from drivealign.evaluation.metrics_bev import (
    _BEV_BUCKETS,
    bev_statistics,
    collect_bev_pairs,
)

# 冻结键集（草案 §2.3/§5 + §6-3）：顶层 9 + 分桶 36 + signed 4 = 49。
FAMILIES = ("dx", "dy", "dxy")
TOP_KEYS = {f"bev_{f}_{s}" for f in FAMILIES for s in ("mean", "p50", "p90")}
BUCKET_KEYS = {
    f"bev_{f}_{s}_bucket_{tag}"
    for f in FAMILIES
    for s in ("mean", "p50", "p90")
    for tag in ("0_10", "10_25", "25_40", "40p")
}
SIGNED_KEYS = {
    f"bev_signed_{axis}_{direction}_mean"
    for axis in ("dx", "dy")
    for direction in ("pos", "neg")
}
EXPECTED_KEYS = TOP_KEYS | BUCKET_KEYS | SIGNED_KEYS


class TestKeySetShape:
    def test_exactly_49_frozen_keys(self):
        """键集形状 pin：49 键（顶层 9 + 分桶 36 + signed 4），无多余。"""
        stats = bev_statistics([(18.25, 0.0, 17.75, 0.0)])
        assert set(stats) == EXPECTED_KEYS
        assert len(stats) == 49

    def test_bucket_tags_are_the_frozen_four(self):
        """分桶 tag 与草案 §2.3 四桶表逐字对应（bucket_0_10/10_25/25_40/40p）。"""
        assert [tag for _, _, tag in _BEV_BUCKETS] == ["0_10", "10_25", "25_40", "40p"]
        assert [(lo, hi) for lo, hi, _ in _BEV_BUCKETS] == [
            (0.0, 10.0), (10.0, 25.0), (25.0, 40.0), (40.0, math.inf),
        ]


class TestVerificationExample:
    def test_g7_17_75_example_hypot_and_bucket(self):
        """G7-1 验算例：GT 前车 20 m 正前最近点 (17.75, 0)（BEV rules
        §4.2），预测纵向偏 0.5 m：dx=0.5 / dy=0 / dxy=hypot=0.5，
        d_gt=17.75 落 [10,25) 桶。"""
        stats = bev_statistics([(18.25, 0.0, 17.75, 0.0)])
        assert stats["bev_dx_mean"] == pytest.approx(0.5)
        assert stats["bev_dy_mean"] == pytest.approx(0.0)
        assert stats["bev_dxy_mean"] == pytest.approx(0.5)  # hypot(0.5, 0)
        assert stats["bev_dxy_p50_bucket_10_25"] == pytest.approx(0.5)
        assert stats["bev_dxy_mean_bucket_0_10"] == 0.0

    def test_hypot_combines_both_axes(self):
        """横向偏移例：预测 (17.75, 1.5) → dy=1.5 / dxy=hypot(0, 1.5)。"""
        stats = bev_statistics([(17.75, 1.5, 17.75, 0.0)])
        assert stats["bev_dx_mean"] == pytest.approx(0.0)
        assert stats["bev_dy_mean"] == pytest.approx(1.5)
        assert stats["bev_dxy_mean"] == pytest.approx(1.5)
        assert stats["bev_dxy_mean_bucket_10_25"] == pytest.approx(1.5)

    def test_all_non_negative_error_semantics(self):
        """全非负语义（2026-10-13 拍板）：预测在 GT 另一侧，三族仍非负。"""
        stats = bev_statistics([(16.75, -1.0, 17.75, 0.0)])
        assert stats["bev_dx_mean"] == pytest.approx(1.0)   # abs(-1.0)
        assert stats["bev_dy_mean"] == pytest.approx(1.0)   # abs(-1.0)
        assert stats["bev_dxy_mean"] == pytest.approx(math.hypot(1.0, 1.0))


class TestBucketBoundaries:
    def test_d_gt_boundary_values_left_closed(self):
        """四桶左闭右开：d_gt=10 → 10_25、25 → 25_40、40 → 40p、
        9.999 → 0_10；桶归属只看 GT 距离（预测取同点使误差为 0 以隔离
        归属信号）。"""
        boundary_pairs = [
            (10.0, 0.0, 10.0, 0.0),     # d_gt = 10 → [10,25)，误差 0
            (25.0, 0.0, 25.0, 0.0),     # d_gt = 25 → [25,40)，误差 0
            (40.0, 0.0, 40.0, 0.0),     # d_gt = 40 → [40, ∞)，误差 0
            (10.0, 0.0, 9.999, 0.0),    # d_gt = 9.999 → [0,10)，误差 0.001
        ]
        stats = bev_statistics(boundary_pairs)
        # [0,10) 桶唯一样本误差 = hypot(10-9.999, 0) = 0.001；其余桶零误差。
        assert stats["bev_dxy_p90_bucket_0_10"] == pytest.approx(0.001)
        assert stats["bev_dxy_p90_bucket_10_25"] == pytest.approx(0.0)
        assert stats["bev_dxy_p90_bucket_25_40"] == pytest.approx(0.0)
        assert stats["bev_dxy_p90_bucket_40p"] == pytest.approx(0.0)

    def test_open_bucket_absorbs_beyond_45(self):
        """40+ 开放桶吸收量程外尾巴（2026-10-14 拍板）：d_gt=45.5 有归桶。"""
        stats = bev_statistics([(45.5, 0.0, 45.5, 0.0), (47.0, 1.0, 46.0, 1.0)])
        # 两样本 dx 误差 {0, 1} → p50 = 0.5（dxy 同）。
        assert stats["bev_dxy_p50_bucket_40p"] == pytest.approx(0.5)
        assert stats["bev_dx_p50_bucket_40p"] == pytest.approx(0.5)

    def test_bucket_condition_uses_gt_distance_not_pred(self):
        """分桶条件量 = d_gt（GT 距离），与预测误差无关：预测严重偏位
        但 GT 近距 → 仍归近距桶（模型间可比的保证）。"""
        stats = bev_statistics([(40.0, 0.0, 5.0, 0.0)])
        assert stats["bev_dxy_mean_bucket_0_10"] == pytest.approx(35.0)
        assert stats["bev_dxy_mean_bucket_40p"] == 0.0


class TestPercentileConvention:
    def test_p50_p90_linear_interpolation(self):
        """p50/p90 与 matched_iou_p90 同一 _percentile 线性插值口径。"""
        # dx 误差序列 [1, 2, 3, 4]：p50 rank=(4-1)*0.5=1.5 → 2.5；p90 rank=2.7 → 3.7。
        pairs = []
        for i, gt_x in enumerate((10.0, 10.0, 10.0, 10.0)):
            pred_x = gt_x + (1.0, 2.0, 3.0, 4.0)[i]
            pairs.append((pred_x, 0.0, gt_x, 0.0))
        stats = bev_statistics(pairs)
        assert stats["bev_dx_p50"] == pytest.approx(2.5)
        assert stats["bev_dx_p90"] == pytest.approx(3.7)
        assert stats["bev_dx_mean"] == pytest.approx(2.5)


class TestSignedBiasKeys:
    def test_positive_and_negative_subsets_averaged_separately(self):
        """方向分解拍板（2026-10-14）：+2/−2 对抗例——代数均值会得 0
        丢光偏置；pos 子集均值恒 >0、neg 子集均值恒 <0。"""
        # dx 有符号误差：+2、-2；dy 有符号误差：+1、-3。
        pairs = [
            (12.0, 1.0, 10.0, 0.0),   # sdx=+2, sdy=+1
            (8.0, -3.0, 10.0, 0.0),   # sdx=-2, sdy=-3
        ]
        stats = bev_statistics(pairs)
        assert stats["bev_signed_dx_pos_mean"] == pytest.approx(2.0)
        assert stats["bev_signed_dx_neg_mean"] == pytest.approx(-2.0)
        assert stats["bev_signed_dy_pos_mean"] == pytest.approx(1.0)
        assert stats["bev_signed_dy_neg_mean"] == pytest.approx(-3.0)
        # 全非负三族不受 signed 键影响：dx 误差 2/2 → mean 2。
        assert stats["bev_dx_mean"] == pytest.approx(2.0)

    def test_zero_errors_enter_neither_subset(self):
        """零误差不进任何子集：全零样本 → signed 四键全 0.0。"""
        stats = bev_statistics([(10.0, 0.0, 10.0, 0.0)])
        assert stats["bev_signed_dx_pos_mean"] == 0.0
        assert stats["bev_signed_dx_neg_mean"] == 0.0
        assert stats["bev_signed_dy_pos_mean"] == 0.0
        assert stats["bev_signed_dy_neg_mean"] == 0.0

    def test_empty_subset_is_zero_not_null(self):
        """空子集置 0.0（不置 null）：只有正向误差时 neg 键 = 0.0。"""
        stats = bev_statistics([(12.0, 1.0, 10.0, 0.0)])
        assert stats["bev_signed_dx_pos_mean"] == pytest.approx(2.0)
        assert stats["bev_signed_dx_neg_mean"] == 0.0


class TestEmptySet:
    def test_no_pairs_all_zeros(self):
        """空集口径（G7-3）：无匹配对切片 49 键全 0.0，不置 null。"""
        stats = bev_statistics([])
        assert len(stats) == 49
        assert all(value == 0.0 for value in stats.values())


class TestCollectBevPairs:
    def test_matched_pairs_resolved_from_both_sides(self):
        """匹配对几何解析：pred 与 GT 的 bev 按匹配对取（非按下标对齐）。"""
        gt_objects = [
            {"category": "car", "bbox_2d": [0, 0, 10, 10], "bev_x": 10.0, "bev_y": 0.0},
            {"category": "car", "bbox_2d": [50, 50, 60, 60], "bev_x": 30.0, "bev_y": 2.0},
        ]
        pred_objects = [
            {"category": "car", "bbox_2d": [50.5, 50, 60, 60], "bev_x": 31.0, "bev_y": 1.0},
            {"category": "car", "bbox_2d": [0.5, 0, 10, 10], "bev_x": 10.5, "bev_y": -0.5},
        ]
        outcome = match_objects(gt_objects, pred_objects, iou_threshold=0.5)
        pairs = collect_bev_pairs(gt_objects, pred_objects, outcome)
        assert sorted(pairs) == [(10.5, -0.5, 10.0, 0.0), (31.0, 1.0, 30.0, 2.0)]

    def test_missing_bev_on_either_side_skips_pair(self):
        """任一侧 bev 缺失（v1–v6 面混入）→ 该对不产生 BEV 样本
        （缺样本 ≠ 零误差，语义必须分离）。"""
        gt_objects = [{"category": "car", "bbox_2d": [0, 0, 10, 10], "bev_x": 10.0, "bev_y": 0.0}]
        pred_objects = [{"category": "car", "bbox_2d": [0.5, 0, 10, 10]}]  # 无 bev
        outcome = match_objects(gt_objects, pred_objects, iou_threshold=0.5)
        assert collect_bev_pairs(gt_objects, pred_objects, outcome) == []

    def test_unmatched_objects_produce_no_samples(self):
        """未匹配对象不产生 BEV 样本（漏检/幻觉由 object_f1 族量，
        不向 BEV 族泄漏）。"""
        gt_objects = [{"category": "car", "bbox_2d": [0, 0, 10, 10], "bev_x": 10.0, "bev_y": 0.0}]
        pred_objects = [{"category": "car", "bbox_2d": [100, 100, 110, 110], "bev_x": 99.0, "bev_y": 1.0}]
        outcome = match_objects(gt_objects, pred_objects, iou_threshold=0.5)
        assert outcome.matched == ()
        assert collect_bev_pairs(gt_objects, pred_objects, outcome) == []
