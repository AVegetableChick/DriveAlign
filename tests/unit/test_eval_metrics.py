"""Unit tests for Eval v1 metrics (S09 plan section 2.4).

A four-anchor synthetic scenario with hand-computed expectations covers the
full aggregate report; focused tests cover parse-failure exclusion, the
over-conservative proxies, maxItems counting, and explicit denominators.

Run:
    ``conda activate autovla_codeclean && cd /root/autodl-tmp/drivealign_workspace && \
    PYTHONPATH=DriveAlign/src python -m pytest DriveAlign/tests/unit/test_eval_metrics.py -q``
"""

from __future__ import annotations

import pytest

from drivealign.evaluation.metrics import (
    aggregate,
    f1_from_counts,
    score_anchor,
)

TAXONOMIES = {
    "speed_actions": ["ACCELERATE", "KEEP_SPEED", "DECELERATE", "STOP"],
    "risk_factors": [
        "pedestrian_crossing",
        "vehicle_merging",
        "lead_vehicle_braking",
        "corridor_conflict",
        "stationary_obstacle",
        "congestion",
        "oncoming_traffic",
    ],
    "categories": ["car", "pedestrian", "truck"],
}
IoU_THRESHOLDS = [0.5]
PRIMARY = 0.5
MAX_ITEMS = 8


def anchor_output(
    objects=None, risks=(), *, yield_required=False, speed_action="KEEP_SPEED"
):
    return {
        "critical_objects": objects or [],
        "risk_factors": list(risks),
        "reasoning": "r",
        "yield_required": yield_required,
        "speed_action": speed_action,
    }


def scene_object(category, bbox, motion_state):
    return {"category": category, "bbox_2d": list(bbox), "motion_state": motion_state}


def pred_row(output, *, parse_ok=True, status="ok", errors=()):
    return {"sample_token": "t", "parse_ok": parse_ok, "status": status, "output": output, "errors": list(errors)}


def make_scores():
    """Four anchors with hand-computed metric expectations.

    A: perfect frame (2 objects matched at IoU 1.0, all fields correct).
    B: category error (car GT vs truck pred -> FP+FN), GT yield=true predicted
       false, GT KEEP_SPEED predicted STOP.
    C: empty GT, one hallucinated car, hallucinated congestion risk, false
       yield flag (over-conservative yield numerator).
    D: parse failure (excluded from all content metrics).
    """
    gt_a = anchor_output(
        [
            scene_object("car", (0, 0, 10, 10), "stationary"),
            scene_object("pedestrian", (20, 20, 30, 30), "crossing"),
        ],
        ["pedestrian_crossing"],
    )
    pred_a = gt_a
    gt_b = anchor_output(
        [scene_object("car", (0, 0, 10, 10), "same_direction")],
        ["corridor_conflict", "congestion"],
        yield_required=True,
        speed_action="KEEP_SPEED",
    )
    pred_b = anchor_output(
        [scene_object("truck", (0, 0, 10, 10), "same_direction")],
        ["corridor_conflict"],
        speed_action="STOP",
    )
    gt_c = anchor_output()
    pred_c = anchor_output(
        [scene_object("car", (0, 0, 10, 10), "stationary")],
        ["congestion"],
        yield_required=True,
    )
    gt_d = gt_a

    scored = []
    for token, gt, output, ok in [
        ("a", gt_a, pred_a, True),
        ("b", gt_b, pred_b, True),
        ("c", gt_c, pred_c, True),
        ("d", gt_d, None, False),
    ]:
        row = {"sample_token": token, "parse_ok": ok, "status": "ok" if ok else "schema_failure",
               "output": output, "errors": [] if ok else [{"category": "schema_failure"}]}
        scored.append(score_anchor(gt, row, iou_thresholds=IoU_THRESHOLDS,
                                   primary_iou_threshold=PRIMARY, max_items=MAX_ITEMS))
    return scored


def make_report():
    return aggregate(
        make_scores(),
        n_anchors_expected=4,
        taxonomies=TAXONOMIES,
        primary_iou_threshold=PRIMARY,
        iou_thresholds=IoU_THRESHOLDS,
    )


class TestF1FromCounts:
    def test_zero_denominators_are_zero(self):
        assert f1_from_counts(0, 0, 0) == 0.0
        assert f1_from_counts(0, 5, 0) == 0.0

    def test_known_value(self):
        assert f1_from_counts(2, 2, 1) == pytest.approx(4.0 / 7.0)


class TestScoreAnchor:
    def test_parse_failure_shape(self):
        score = score_anchor(
            anchor_output(),
            {"sample_token": "x", "parse_ok": False, "status": "schema_failure",
             "errors": [{"category": "schema_failure"}]},
            iou_thresholds=IoU_THRESHOLDS, primary_iou_threshold=PRIMARY,
            max_items=MAX_ITEMS,
        )
        assert score["parse_ok"] is False
        assert score["parse_error_categories"] == ["schema_failure"]
        assert "per_tau" not in score

    def test_matched_and_leftover_categories(self):
        gt = anchor_output([scene_object("car", (0, 0, 10, 10), "stationary")])
        pred = anchor_output(
            [
                scene_object("car", (0, 0, 10, 10), "stationary"),
                scene_object("truck", (50, 50, 60, 60), "stationary"),
            ]
        )
        score = score_anchor(gt, pred_row(pred), iou_thresholds=IoU_THRESHOLDS,
                             primary_iou_threshold=PRIMARY, max_items=MAX_ITEMS)
        assert score["matched_categories"] == [["car", "car"]]
        assert score["unmatched_pred_categories"] == ["truck"]
        assert score["unmatched_gt_categories"] == []


class TestAggregate:
    def test_parse_block(self):
        report = make_report()
        assert report["parse_rate"] == pytest.approx(3 / 4)
        assert report["parse_error_counts"] == {"schema_failure": 1}
        assert report["coverage"] == {
            "anchors_expected": 4, "anchors_scored": 4,
            "parse_ok": 3, "parse_failed": 1,
        }

    def test_micro_object_metrics(self):
        report = make_report()
        assert report["object_precision_micro@0.5"] == pytest.approx(2 / 4)
        assert report["object_recall_micro@0.5"] == pytest.approx(2 / 3)
        assert report["object_f1_micro@0.5"] == pytest.approx(4.0 / 7.0)

    def test_macro_object_metrics(self):
        report = make_report()
        per_category = report["object_f1_per_category@0.5"]
        # car: tp=1 (a), fp=1 (c hallucination), fn=1 (b category error) -> 0.5;
        # pedestrian: 1.0; truck has no GT -> excluded.
        assert per_category == {"car": pytest.approx(0.5), "pedestrian": 1.0}
        assert report["object_f1_macro@0.5"] == pytest.approx((0.5 + 1.0) / 2)
        assert report["macro_categories_counted"] == 2

    def test_matched_iou_distribution(self):
        report = make_report()
        assert report["matched_pair_count"] == 2
        assert report["matched_iou_mean"] == pytest.approx(1.0)
        assert report["matched_iou_p50"] == pytest.approx(1.0)
        assert report["matched_iou_p90"] == pytest.approx(1.0)

    def test_motion_state(self):
        report = make_report()
        assert report["motion_state_accuracy"] == pytest.approx(1.0)
        assert report["motion_state_confusion"] == {
            "stationary": {"stationary": 1},
            "crossing": {"crossing": 1},
        }

    def test_speed_action_macro_f1(self):
        report = make_report()
        assert report["speed_action_confusion"]["KEEP_SPEED"] == {
            "KEEP_SPEED": 2, "STOP": 1, "ACCELERATE": 0, "DECELERATE": 0,
        }
        # KEEP_SPEED: tp=2 fn=1 -> 4/5; others 0; macro over 4 classes.
        assert report["speed_action_per_class_f1"]["KEEP_SPEED"] == pytest.approx(0.8)
        assert report["speed_action_f1_macro"] == pytest.approx(0.2)

    def test_yield_required_f1(self):
        # fp=1 (c), fn=1 (b), tp=0 -> F1 = 0.
        assert make_report()["yield_required_f1"] == 0.0

    def test_risk_factor_f1(self):
        per_class = make_report()["risk_factors_f1_per_class"]
        assert per_class["pedestrian_crossing"] == pytest.approx(1.0)
        assert per_class["corridor_conflict"] == pytest.approx(1.0)
        assert per_class["congestion"] == 0.0  # tp=0 (fp on c, fn on b)
        assert per_class["oncoming_traffic"] == 0.0
        assert make_report()["risk_factors_f1_macro"] == pytest.approx(2.0 / 7.0)

    def test_overconservative_rates(self):
        report = make_report()
        # yield: GT false frames = {a, c}; false-yield on c -> 1/2.
        assert report["overconservative_yield_rate"] == pytest.approx(0.5)
        # stop: GT keep/accel frames = {a, b, c}; STOP only on b (GT keep) -> 1/3.
        assert report["overconservative_stop_rate"] == pytest.approx(1.0 / 3.0)

    def test_max_items_hit_rate(self):
        assert make_report()["max_items_hit_rate"] == pytest.approx(0.0)

    def test_denominators_explicit(self):
        denominators = make_report()["denominators"]
        assert denominators["parse_ok"] == 3
        assert denominators["motion_pairs"] == 2
        assert denominators["micro_counts"]["0.5"] == [2, 2, 1]
        assert denominators["overconservative"] == {
            "yield_fp": 1, "yield_gt_false": 2, "stop_fp": 1, "stop_gt_keep": 3,
        }


class TestPercentile:
    def test_p90_linear_interpolation(self):
        from drivealign.evaluation.metrics import _percentile

        values = sorted([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        assert _percentile(values, 90) == pytest.approx(0.91)
        assert _percentile([], 90) == 0.0
        assert _percentile([0.42], 90) == pytest.approx(0.42)
