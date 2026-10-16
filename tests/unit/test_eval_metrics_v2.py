"""Unit tests for Eval v2 metrics (S13 口径修订台账).

覆盖四类新增 counting helper（T1 motion 逐类 F1、T3 stop 方向计数、
空集方向四格、基数直方图）、T2/T3 方向率的三种塌缩情形，以及
``aggregate_v2`` 的报告键集（T6 漏斗 / 改名键退役与新增 / 直方图与
空集块 / denominators 复算）。

Run:
    ``conda activate autovla_codeclean && cd /root/autodl-tmp/drivealign_workspace && \
    PYTHONPATH=DriveAlign/src python -m pytest DriveAlign/tests/unit/test_eval_metrics_v2.py -q``
"""

from __future__ import annotations

import pytest

from drivealign.evaluation.metrics import score_anchor
from drivealign.evaluation.metrics_v2 import (
    aggregate_v2,
    cardinality_histogram,
    empty_direction_block,
    motion_state_per_class_f1,
    stop_counts,
    stop_direction_rates,
    yield_direction_rates,
)

STATES = ["stationary", "same_direction", "oncoming", "crossing"]
ACTIONS = ["ACCELERATE", "KEEP_SPEED", "DECELERATE", "STOP"]
RISKS = [
    "pedestrian_crossing",
    "vehicle_merging",
    "lead_vehicle_braking",
    "corridor_conflict",
    "stationary_obstacle",
    "congestion",
    "oncoming_traffic",
]
TAX = {"motion_states": STATES, "speed_actions": ACTIONS, "risk_factors": RISKS}
IoU = [0.3, 0.5, 0.7]
PRIMARY = 0.5
MAX_ITEMS = 8


def scene_object(category, bbox, motion_state):
    return {"category": category, "bbox_2d": list(bbox), "motion_state": motion_state}


def anchor_output(objects, risks, *, yield_required, speed_action):
    return {
        "critical_objects": list(objects),
        "risk_factors": list(risks),
        "reasoning": "r",
        "yield_required": yield_required,
        "speed_action": speed_action,
    }


def pred_row(output, *, parse_ok=True):
    return {
        "sample_token": "t",
        "parse_ok": parse_ok,
        "status": "ok" if parse_ok else "schema_failure",
        "errors": [],
        "output": output,
    }


def anchor(gt, pred_output, *, parse_ok=True):
    return score_anchor(
        gt,
        pred_row(pred_output, parse_ok=parse_ok),
        iou_thresholds=IoU,
        primary_iou_threshold=PRIMARY,
        max_items=MAX_ITEMS,
    )


class TestMotionStatePerClassF1:
    def test_hand_computed_macro_one_sixth(self):
        """T1 手算钉死案例：仅 stationary 有 tp=2/fp=1/fn=1（F1=2/3），
        其余三类 (0,0,0) → F1=0；macro = (2/3)/4 = 1/6。"""
        confusion = {
            "stationary": {"stationary": 2, "same_direction": 1},
            "same_direction": {"stationary": 1},
        }
        per_class = motion_state_per_class_f1(confusion, STATES)
        assert per_class["stationary"] == pytest.approx(2.0 / 3.0)
        assert per_class["same_direction"] == 0.0
        assert per_class["oncoming"] == 0.0
        assert per_class["crossing"] == 0.0
        assert sum(per_class.values()) / len(STATES) == pytest.approx(1.0 / 6.0)

    def test_absent_gt_state_counts_as_zero_row(self):
        """混淆矩阵缺某 GT 态 → 按全零行处理；返回仍为固定 4 态形状。"""
        per_class = motion_state_per_class_f1({"stationary": {"stationary": 1}}, STATES)
        assert sorted(per_class) == sorted(STATES)
        assert per_class["stationary"] == pytest.approx(1.0)
        assert per_class["crossing"] == 0.0


class TestStopCounts:
    def test_four_cell_counts_and_parse_fail_excluded(self):
        """表 5 四格：a GT KEEP/ pred STOP → fp；b GT DECEL/ pred KEEP →
        fn；c GT ACCEL / pred ACCEL → 只进 keep 分母；d GT STOP /
        pred STOP → 只进 stop 分母；parse 失败不进任何格子。"""
        cases = [
            ("KEEP_SPEED", "STOP"),
            ("DECELERATE", "KEEP_SPEED"),
            ("ACCELERATE", "ACCELERATE"),
            ("STOP", "STOP"),
        ]
        scores = []
        for index, (gt_action, pred_action) in enumerate(cases):
            objects = [scene_object("car", (0, 0, 10, 10), "stationary")]
            gt = anchor_output(
                objects, [], yield_required=False, speed_action=gt_action
            )
            pred = anchor_output(
                objects, [], yield_required=False, speed_action=pred_action
            )
            score = anchor(gt, pred)
            score["sample_token"] = f"tok_{index}"
            scores.append(score)
        failed_gt = anchor_output([], [], yield_required=False, speed_action="STOP")
        failed_pred = anchor_output([], [], yield_required=False, speed_action="STOP")
        scores.append(anchor(failed_gt, failed_pred, parse_ok=False))
        assert stop_counts(scores) == {
            "stop_fp": 1,
            "stop_gt_keep": 2,
            "stop_fn": 1,
            "stop_gt_stop": 2,
        }


class TestEmptyDirectionBlock:
    def test_four_cells_and_rates(self):
        counts, hall, miss = empty_direction_block([(0, 0), (0, 2), (3, 0), (1, 1)])
        assert counts == {
            "gt_empty_pred_empty": 1,
            "gt_empty_pred_nonempty": 1,
            "gt_nonempty_pred_empty": 1,
            "gt_nonempty_pred_nonempty": 1,
        }
        assert hall == pytest.approx(0.5)
        assert miss == pytest.approx(0.5)

    def test_zero_denominator_guards(self):
        """无 GT 空 anchor → hall 分母 0 取 0.0；无 GT 非空 → miss 同理。"""
        _, hall, miss = empty_direction_block([(1, 1), (2, 0)])
        assert hall == 0.0
        assert miss == pytest.approx(0.5)
        _, hall, miss = empty_direction_block([(0, 1)])
        assert hall == pytest.approx(1.0)
        assert miss == 0.0


class TestCardinalityHistogram:
    def test_full_bucket_coverage(self):
        assert cardinality_histogram([0, 1, 1, 3], 4) == {
            "0": 1, "1": 2, "2": 0, "3": 1, "4": 0,
        }

    def test_cap_truncates_defensively(self):
        """契约 maxItems 保证基数不超封顶；min 截断仅为防御性兜底。"""
        assert cardinality_histogram([9], 8) == {
            **{str(k): 0 for k in range(8)}, "8": 1,
        }


class TestYieldDirectionRates:
    def test_collapse_cases(self):
        """T2 塌缩三情形（0.0 守卫下的定义条款）。"""
        # 模型恒输出 false（GT 混合）→ miss = 1.0 / overcons = 0.0
        assert yield_direction_rates({"tp": 0, "fp": 0, "fn": 3, "tn": 2}) == (1.0, 0.0)
        # 模型恒输出 true（GT 混合）→ miss = 0.0 / overcons = 1.0
        assert yield_direction_rates({"tp": 3, "fp": 2, "fn": 0, "tn": 0}) == (0.0, 1.0)
        # 世界塌缩且模型全对（GT 恒 true）：两率同 0.0
        assert yield_direction_rates({"tp": 3, "fp": 0, "fn": 0, "tn": 0}) == (0.0, 0.0)
        # 世界塌缩且模型全对（GT 恒 false）：两率同 0.0（分母 0 守卫）
        assert yield_direction_rates({"tp": 0, "fp": 0, "fn": 0, "tn": 3}) == (0.0, 0.0)


class TestStopDirectionRates:
    def test_collapse_cases(self):
        """T3 塌缩三情形，与 T2 同构。"""
        # 模型恒输出 STOP（GT 混合）→ overcons = 1.0 / miss = 0.0
        rates = stop_direction_rates(
            {"stop_fp": 2, "stop_gt_keep": 2, "stop_fn": 0, "stop_gt_stop": 3}
        )
        assert rates == (1.0, 0.0)
        # 模型恒不输出 STOP（GT 混合）→ overcons = 0.0 / miss = 1.0
        rates = stop_direction_rates(
            {"stop_fp": 0, "stop_gt_keep": 2, "stop_fn": 3, "stop_gt_stop": 3}
        )
        assert rates == (0.0, 1.0)
        # 世界塌缩且模型全对（GT 恒 stop）：两率同 0.0（分母 0 守卫）
        rates = stop_direction_rates(
            {"stop_fp": 0, "stop_gt_keep": 0, "stop_fn": 0, "stop_gt_stop": 3}
        )
        assert rates == (0.0, 0.0)


class TestAggregateV2:
    """4-anchor 合成世界：a/c 完美帧，b 过度保守（yield miss + stop fp +
    risk fn），d parse 失败。全部率值手算钉死。"""

    @staticmethod
    def _scores():
        car = scene_object("car", (0, 0, 10, 10), "stationary")
        gt_b = anchor_output(
            [car], ["corridor_conflict"], yield_required=True, speed_action="KEEP_SPEED"
        )
        pred_b = anchor_output(
            [car], [], yield_required=False, speed_action="STOP"
        )
        gt_perfect = anchor_output(
            [car], [], yield_required=False, speed_action="KEEP_SPEED"
        )
        return [
            anchor(gt_perfect, gt_perfect),
            anchor(gt_b, pred_b),
            anchor(gt_perfect, gt_perfect),
            anchor(gt_perfect, gt_perfect, parse_ok=False),
        ]

    def test_report_keyset_and_values(self):
        report = aggregate_v2(
            self._scores(),
            n_anchors_expected=4,
            taxonomies=TAX,
            primary_iou_threshold=PRIMARY,
            iou_thresholds=IoU,
            max_items=MAX_ITEMS,
            risk_max_items=8,
        )
        # T6 三级漏斗：anchors_scored 退役，parse_rate 分母 = anchors_expected
        assert report["coverage"] == {
            "anchors_expected": 4, "parse_ok": 3, "parse_failed": 1,
        }
        assert "anchors_scored" not in report["coverage"]
        assert report["parse_rate"] == pytest.approx(3 / 4)
        # T1/T2/T3 键集变化：accuracy 与旧 overconservative_* 退役
        assert "motion_state_accuracy" not in report
        assert "overconservative_yield_rate" not in report
        assert "overconservative_stop_rate" not in report
        # T1：accuracy 视角是 1.0，macro-F1 视角 = (1 + 0 + 0 + 0)/4 = 0.25
        assert report["motion_state_f1_per_class"]["stationary"] == pytest.approx(1.0)
        assert report["motion_state_f1_macro"] == pytest.approx(0.25)
        # T2：GT yield true 仅 b 且预测 false → miss = 1/1；GT false 全预测 false
        assert report["yield_required_f1"] == 0.0
        assert report["yield_miss_rate"] == pytest.approx(1.0)
        assert report["yield_overconservative_rate"] == 0.0
        # T3：b 是唯一 GT keep/accel 预测 STOP → 1/3；无 GT stop → miss 守卫 0.0
        assert report["stop_overconservative_rate"] == pytest.approx(1.0 / 3.0)
        assert report["stop_miss_rate"] == 0.0
        # speed_action 逐类：KEEP tp=2/fp=0/fn=1 → 0.8，其余 0 → macro 0.2
        assert report["speed_action_per_class_f1"]["KEEP_SPEED"] == pytest.approx(0.8)
        assert report["speed_action_f1_macro"] == pytest.approx(0.2)
        # risk：唯一 GT risk（corridor_conflict, b）预测缺席 → 全类 F1 = 0
        assert report["risk_factors_f1_macro"] == 0.0

    def test_micro_grid_and_per_category(self):
        report = aggregate_v2(
            self._scores(),
            n_anchors_expected=4,
            taxonomies=TAX,
            primary_iou_threshold=PRIMARY,
            iou_thresholds=IoU,
            max_items=MAX_ITEMS,
            risk_max_items=8,
        )
        for tag in ("0.3", "0.5", "0.7"):
            assert report[f"object_precision_micro@{tag}"] == pytest.approx(1.0)
            assert report[f"object_recall_micro@{tag}"] == pytest.approx(1.0)
        assert report["object_f1_micro@0.5"] == pytest.approx(1.0)
        assert report["object_f1_macro@0.5"] == pytest.approx(1.0)
        assert sorted(report["object_f1_per_category@0.5"]) == ["car"]

    def test_histograms_full_buckets_and_empty_direction(self):
        report = aggregate_v2(
            self._scores(),
            n_anchors_expected=4,
            taxonomies=TAX,
            primary_iou_threshold=PRIMARY,
            iou_thresholds=IoU,
            max_items=MAX_ITEMS,
            risk_max_items=8,
        )
        for key in ("n_pred_objects", "n_gt_objects", "n_gt_risk_factors",
                    "n_pred_risk_factors"):
            assert sorted(report[key], key=int) == [str(k) for k in range(9)]
            assert sum(report[key].values()) == 3  # 仅 parse-ok 进入
        assert report["n_gt_objects"]["1"] == 3
        assert report["n_gt_risk_factors"]["1"] == 1
        assert report["n_pred_risk_factors"]["0"] == 3
        # 对象侧：GT / pred 全非空 → 全落第四格，两率 0
        assert report["object_empty_direction_counts"]["gt_nonempty_pred_nonempty"] == 3
        assert report["object_empty_hallucination_rate"] == 0.0
        assert report["object_empty_miss_rate"] == 0.0
        # risk 侧：b GT 1 个 risk 预测空 → miss = 1/1；GT 空 anchor 预测空
        assert report["risk_empty_direction_counts"] == {
            "gt_empty_pred_empty": 2,
            "gt_empty_pred_nonempty": 0,
            "gt_nonempty_pred_empty": 1,
            "gt_nonempty_pred_nonempty": 0,
        }
        assert report["risk_empty_miss_rate"] == pytest.approx(1.0)
        assert report["risk_empty_hallucination_rate"] == 0.0

    def test_denominators_allow_rate_recomputation(self):
        report = aggregate_v2(
            self._scores(),
            n_anchors_expected=4,
            taxonomies=TAX,
            primary_iou_threshold=PRIMARY,
            iou_thresholds=IoU,
            max_items=MAX_ITEMS,
            risk_max_items=8,
        )
        den = report["denominators"]
        assert den["anchors_expected"] == 4
        assert den["parse_ok"] == 3
        assert den["matched_pairs"] == 3
        assert den["motion_pairs"] == 3
        assert den["yield_counts"] == {"tp": 0, "fp": 0, "fn": 1, "tn": 2}
        assert den["stop_counts"] == {
            "stop_fp": 1, "stop_gt_keep": 3, "stop_fn": 0, "stop_gt_stop": 0,
        }
        assert den["micro_counts"]["0.5"] == [3, 0, 0]
        # 报告内可复算全部方向率（denominators 重组的验收口径）
        y = den["yield_counts"]
        s = den["stop_counts"]
        assert y["fn"] / (y["tp"] + y["fn"]) == pytest.approx(report["yield_miss_rate"])
        assert y["fp"] / (y["fp"] + y["tn"]) == pytest.approx(
            report["yield_overconservative_rate"]
        )
        assert s["stop_fp"] / s["stop_gt_keep"] == pytest.approx(
            report["stop_overconservative_rate"]
        )
        assert s["stop_fn"] / s["stop_gt_stop"] if s["stop_gt_stop"] else True

    def test_per_class_block_shapes_are_fixed(self):
        report = aggregate_v2(
            self._scores(),
            n_anchors_expected=4,
            taxonomies=TAX,
            primary_iou_threshold=PRIMARY,
            iou_thresholds=IoU,
            max_items=MAX_ITEMS,
            risk_max_items=8,
        )
        assert sorted(report["motion_state_f1_per_class"]) == sorted(STATES)
        assert sorted(report["speed_action_per_class_f1"]) == sorted(ACTIONS)
        assert sorted(report["risk_factors_f1_per_class"]) == sorted(RISKS)
