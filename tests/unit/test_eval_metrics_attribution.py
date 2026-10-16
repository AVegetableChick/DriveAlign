"""Unit tests for Eval v3 risk-factor attribution metrics (S13 eval v3 §3).

覆盖 G7 验收口径的归因成员：conditional 双向闸门（预测多报因子 /
GT 有而预测未报均不计入，2026-10-14 拍板）、id 几何桥接（两侧独立
编号空间，严禁编号直比——经 primary IoU 匹配桥接，编号错位例钉死）、
边界规则表全部七行（草案 §3.3）、micro 池化 / macro 无原子因子跳过 /
per_factor 六字段嵌套块、合计与聚合共用同一份定义。

Run:
    ``conda activate autovla_codeclean && cd /root/autodl-tmp/drivealign_workspace && \
    PYTHONPATH=DriveAlign/src python -m pytest DriveAlign/tests/unit/test_eval_metrics_attribution.py -q``
"""

from __future__ import annotations

import pytest

from drivealign.evaluation.matching import match_objects
from drivealign.evaluation.metrics_attribution import (
    _PER_FACTOR_FIELDS,
    aggregate_attribution,
    anchor_attribution_counts,
    attribution_totals,
)


def car(number, *, object_id=None):
    """bbox 互不重叠的 car 对象（number 控制横向位置，保证匹配可控）。"""
    entry = {"category": "car", "bbox_2d": [number * 100, 0, number * 100 + 10, 10]}
    if object_id is not None:
        entry["id"] = object_id
    return entry


def run_attribution(gt_objects, gt_risks, pred_objects, pred_risks, *, threshold=0.5):
    """构造 primary 匹配并计数（test helper，非生产路径）。"""
    outcome = match_objects(gt_objects, pred_objects, iou_threshold=threshold)
    return anchor_attribution_counts(
        {"critical_objects": gt_objects, "risk_factors": gt_risks},
        {"critical_objects": pred_objects, "risk_factors": pred_risks},
        outcome,
    )


class TestConditionalBidirectionalGate:
    def test_over_reported_factor_not_counted(self):
        """预测多报因子（∉ GT 因子集）：不计入（因子误报由
        risk_factors_f1_macro 计 FP，归因侧不重复惩罚）。"""
        counts = run_attribution(
            [car(0, object_id=0)],
            [{"factor": "object_near", "objects": [0]}],
            [car(0, object_id=5)],
            [
                {"factor": "object_near", "objects": [5]},
                {"factor": "vru_crossing", "objects": [5]},  # 多报
            ],
        )
        assert counts == {"object_near": (1, 0, 0)}

    def test_gt_factor_not_reported_not_counted(self):
        """GT 有而预测未报：不计入（因子漏报同理由因子指标计 FN）。"""
        counts = run_attribution(
            [car(0, object_id=0)],
            [
                {"factor": "object_near", "objects": [0]},
                {"factor": "oncoming_traffic", "objects": [0]},  # 预测未报
            ],
            [car(0, object_id=5)],
            [{"factor": "object_near", "objects": [5]}],
        )
        assert counts == {"object_near": (1, 0, 0)}

    def test_intersection_only_factors_produce_atoms(self):
        """闸门 = 两侧因子名集合之交：两侧各有独占因子时只交集成原子。"""
        counts = run_attribution(
            [car(0, object_id=0)],
            [
                {"factor": "object_near", "objects": [0]},
                {"factor": "stationary_obstacle", "objects": [0]},  # 预测未报
            ],
            [car(0, object_id=5)],
            [
                {"factor": "object_near", "objects": [5]},
                {"factor": "vru_crossing", "objects": [5]},  # GT 没有
            ],
        )
        assert counts == {"object_near": (1, 0, 0)}


class TestIdBridging:
    def test_disjoint_id_spaces_bridge_through_iou_match(self):
        """id 几何桥接（关键设计）：GT id 与 pred id 是两个独立编号空间，
        引用 id 直接比较必然全错——经 IoU 匹配桥接后正确落点。"""
        counts = run_attribution(
            [car(0, object_id=0)],
            [{"factor": "object_near", "objects": [0]}],
            [car(0, object_id=7)],  # pred id 空间完全不同
            [{"factor": "object_near", "objects": [7]}],
        )
        assert counts == {"object_near": (1, 0, 0)}

    def test_reference_without_landing_is_fp(self):
        """引用幻觉对象 / IoU<0.5 对象：无 GT 落点 → FP。"""
        counts = run_attribution(
            [car(0, object_id=0)],
            [{"factor": "object_near", "objects": [0]}],
            [car(0, object_id=5), car(9, object_id=6)],  # car(9) 与 GT 无交集
            [{"factor": "object_near", "objects": [5, 6]}],
        )
        assert counts == {"object_near": (1, 1, 0)}

    def test_same_pred_object_referenced_by_two_factors(self):
        """同一 pred 对象被多因子引用：每 (factor, object) 独立判定，
        各自按落点是否 ∈ 该因子 GT 引用集计 TP。"""
        counts = run_attribution(
            [car(0, object_id=0)],
            [
                {"factor": "object_near", "objects": [0]},
                {"factor": "stationary_obstacle", "objects": [0]},
            ],
            [car(0, object_id=5)],
            [
                {"factor": "object_near", "objects": [5]},
                {"factor": "stationary_obstacle", "objects": [5]},
            ],
        )
        assert counts == {"object_near": (1, 0, 0), "stationary_obstacle": (1, 0, 0)}

    def test_landing_outside_factor_reference_set_is_fp(self):
        """落点 GT 对象存在但 ∉ 该因子 GT 引用集 → FP（对象指错了因子）。"""
        # GT 两对象：object_near 只引用 gt0；预测把 gt1 也指给 object_near。
        counts = run_attribution(
            [car(0, object_id=0), car(1, object_id=1)],
            [{"factor": "object_near", "objects": [0]}],
            [car(0, object_id=5), car(1, object_id=6)],
            [{"factor": "object_near", "objects": [5, 6]}],
        )
        assert counts == {"object_near": (1, 1, 0)}


class TestBoundaryRules:
    def test_reported_factor_with_empty_objects_all_fn(self):
        """报了因子没指对象：O_gt(f) 全体计 FN（全漏归因口径）。"""
        counts = run_attribution(
            [car(0, object_id=0)],
            [{"factor": "object_near", "objects": [0]}],
            [car(0, object_id=5)],
            [{"factor": "object_near", "objects": []}],
        )
        assert counts == {"object_near": (0, 0, 1)}

    def test_missed_gt_object_is_fn(self):
        """GT 对象漏检（无预测落点）：该对象计 FN；其他对象匹配不受
        串位影响（贪心 bbox 对 bbox，一对一）。"""
        counts = run_attribution(
            [car(0, object_id=0), car(1, object_id=1)],
            [{"factor": "object_near", "objects": [0, 1]}],
            [car(0, object_id=5)],  # gt1 漏检
            [{"factor": "object_near", "objects": [5]}],
        )
        assert counts == {"object_near": (1, 0, 1)}

    def test_empty_gt_objects_all_references_fp(self):
        """GT objects 空（理论不发生，防御口径）：预测引用全 FP。"""
        counts = run_attribution(
            [car(0, object_id=0)],
            [{"factor": "object_near", "objects": []}],
            [car(0, object_id=5)],
            [{"factor": "object_near", "objects": [5]}],
        )
        assert counts == {"object_near": (0, 1, 0)}

    def test_both_sides_empty_objects_factor_skipped(self):
        """两侧 objects 均空：跳过该因子（不产生原子、不出现在计数中）。"""
        counts = run_attribution(
            [car(0, object_id=0)],
            [{"factor": "object_near", "objects": []}],
            [car(0, object_id=5)],
            [{"factor": "object_near", "objects": []}],
        )
        assert counts == {}

    def test_bare_string_factor_treated_as_empty_objects(self):
        """v6- 裸名字符串条目防御：按 objects 空处理（报了因子没指对象
        = 全漏归因口径），与边界表第 3 行一致。"""
        counts = run_attribution(
            [car(0, object_id=0)],
            ["object_near"],  # v6 形状 GT
            [car(0, object_id=5)],
            [{"factor": "object_near", "objects": [5]}],
        )
        assert counts == {"object_near": (0, 1, 0)}


class TestAggregation:
    def test_micro_pools_all_atoms(self):
        """micro = 全原子池直接计 F1：跨因子池化 tp/fp/fn。"""
        totals = {"a": (2, 0, 0), "b": (0, 2, 0), "c": (0, 0, 2)}
        result = aggregate_attribution(totals)
        # 池化 tp=2, fp=2, fn=2 → P=0.5, R=0.5, F1=0.5。
        assert result["risk_attribution_f1_micro"] == pytest.approx(0.5)

    def test_macro_skips_atomless_factors(self):
        """macro = 有原子因子 F1 的均值，无原子因子跳过（不置 0 摊薄）。"""
        totals = {"a": (1, 0, 0), "b": (1, 1, 1), "dead": (0, 0, 0)}
        result = aggregate_attribution(totals)
        f1_a = 1.0
        f1_b = 0.5  # P=0.5, R=0.5
        assert result["risk_attribution_f1_macro"] == pytest.approx((f1_a + f1_b) / 2)
        assert "dead" not in result["risk_attribution_per_factor"]

    def test_empty_totals_all_zero_and_empty_block(self):
        """空集口径：无原子 → 两标量 0.0，per_factor 空块（不置 null）。"""
        result = aggregate_attribution({})
        assert result["risk_attribution_f1_micro"] == 0.0
        assert result["risk_attribution_f1_macro"] == 0.0
        assert result["risk_attribution_per_factor"] == {}

    def test_per_factor_block_six_frozen_fields(self):
        """per_factor 嵌套块：每因子恰有 tp/fp/fn/precision/recall/f1
        六字段（键集 pin）。"""
        result = aggregate_attribution({"vru_crossing": (2, 1, 1)})
        block = result["risk_attribution_per_factor"]["vru_crossing"]
        assert tuple(block) == _PER_FACTOR_FIELDS
        assert block["precision"] == pytest.approx(2 / 3)
        assert block["recall"] == pytest.approx(2 / 3)
        assert block["f1"] == pytest.approx(2 / 3)

    def test_totals_pool_across_anchors(self):
        """attribution_totals 跨 anchor 池化（与 aggregate/bootstrap 共用
        同一份定义）：parse 失败不进格子。"""
        scores = [
            {"parse_ok": True, "risk_attribution_counts": {"a": (1, 0, 1)}},
            {"parse_ok": True, "risk_attribution_counts": {"a": (1, 1, 0), "b": (0, 0, 0)}},
            {"parse_ok": False, "risk_attribution_counts": {"a": (9, 9, 9)}},
        ]
        assert attribution_totals(scores) == {"a": (2, 1, 1), "b": (0, 0, 0)}
