"""Unit tests for M09 v1 class-wise greedy matching (S09 plan section 2.4).

Run:
    ``conda activate autovla_codeclean && cd /root/autodl-tmp/drivealign_workspace && \
    PYTHONPATH=DriveAlign/src python -m pytest DriveAlign/tests/unit/test_eval_matching.py -q``
"""

from __future__ import annotations

import pytest

from drivealign.evaluation.matching import bbox_iou, match_objects


def obj(category: str, bbox: tuple[float, float, float, float]) -> dict:
    return {"category": category, "bbox_2d": list(bbox), "motion_state": "stationary"}


class TestBboxIou:
    def test_identical_boxes(self):
        assert bbox_iou([0, 0, 10, 10], [0, 0, 10, 10]) == pytest.approx(1.0)

    def test_disjoint_boxes(self):
        assert bbox_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0

    def test_half_overlap(self):
        # [0,0,10,10] vs [5,0,15,10]: intersection 50, union 150.
        assert bbox_iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(1.0 / 3.0)

    def test_degenerate_box_has_zero_iou(self):
        assert bbox_iou([5, 5, 5, 5], [0, 0, 10, 10]) == 0.0


class TestMatchObjects:
    def test_empty_sides(self):
        outcome = match_objects([], [], iou_threshold=0.5)
        assert outcome.matched == ()
        outcome = match_objects([obj("car", (0, 0, 10, 10))], [], iou_threshold=0.5)
        assert outcome.unmatched_gt == (0,) and outcome.matched == ()
        outcome = match_objects([], [obj("car", (0, 0, 10, 10))], iou_threshold=0.5)
        assert outcome.unmatched_pred == (0,) and outcome.matched == ()

    def test_cross_category_never_pairs(self):
        gt = [obj("car", (0, 0, 10, 10))]
        pred = [obj("pedestrian", (0, 0, 10, 10))]
        outcome = match_objects(gt, pred, iou_threshold=0.3)
        assert outcome.matched == ()
        assert outcome.unmatched_gt == (0,) and outcome.unmatched_pred == (0,)

    def test_iou_threshold_boundary_inclusive(self):
        gt = [obj("car", (0, 0, 10, 10))]
        pred = [obj("car", (5, 0, 15, 10))]  # IoU = 1/3
        assert match_objects(gt, pred, iou_threshold=0.3).matched
        assert not match_objects(gt, pred, iou_threshold=0.34).matched

    def test_same_gt_matches_at_most_once(self):
        gt = [obj("car", (0, 0, 10, 10))]
        pred = [
            obj("car", (0, 0, 10, 10)),  # IoU 1.0
            obj("car", (1, 1, 11, 11)),  # IoU ~0.68
        ]
        outcome = match_objects(gt, pred, iou_threshold=0.5)
        assert len(outcome.matched) == 1
        assert outcome.matched[0][:2] == (0, 0)
        assert outcome.unmatched_pred == (1,) and outcome.unmatched_gt == ()

    def test_greedy_takes_best_iou_first(self):
        gt = [obj("car", (0, 0, 10, 10))]
        pred = [
            obj("car", (9, 9, 19, 19)),  # IoU 1/81 ~ 0.0123 -> below tau
            obj("car", (0, 0, 10, 10)),  # IoU 1.0
        ]
        outcome = match_objects(gt, pred, iou_threshold=0.5)
        assert outcome.matched[0][:2] == (0, 1)

    def test_iou_tie_break_is_bbox_lexicographic(self):
        # Two GT boxes tie at IoU 1.0 with two predictions; assignment must
        # follow bbox lexicographic order, independent of list order.
        gt_a = [obj("car", (0, 0, 10, 10)), obj("car", (100, 100, 110, 110))]
        pred_a = [obj("car", (100, 100, 110, 110)), obj("car", (0, 0, 10, 10))]
        gt_b = list(reversed(gt_a))
        pred_b = list(reversed(pred_a))
        first = match_objects(gt_a, pred_a, iou_threshold=0.5)
        second = match_objects(gt_b, pred_b, iou_threshold=0.5)
        # (0,0,10,10) sorts lexicographically first on both sides: the SET of
        # pairs must be identical regardless of input list order.
        assert {(g, p) for g, p, _ in first.matched} == {(0, 1), (1, 0)}
        assert {(g, p) for g, p, _ in second.matched} == {(0, 1), (1, 0)}

    def test_duplicate_predictions_match_independently(self):
        gt = [obj("car", (0, 0, 10, 10)), obj("car", (0, 0, 10, 10))]
        pred = [obj("car", (0, 0, 10, 10)), obj("car", (0, 0, 10, 10))]
        outcome = match_objects(gt, pred, iou_threshold=0.5)
        assert len(outcome.matched) == 2
        assert {pair[0] for pair in outcome.matched} == {0, 1}
        assert {pair[1] for pair in outcome.matched} == {0, 1}

    def test_result_independent_of_prediction_order(self):
        gt = [obj("car", (0, 0, 10, 10)), obj("car", (50, 50, 60, 60))]
        preds = [obj("car", (51, 51, 61, 61)), obj("car", (0, 0, 10, 10))]
        backward_preds = list(reversed(preds))

        def content_with(outcome, pred_list):
            return {
                (tuple(gt[g]["bbox_2d"]), tuple(pred_list[p]["bbox_2d"]))
                for g, p, _ in outcome.matched
            }

        forward = match_objects(gt, preds, iou_threshold=0.5)
        backward = match_objects(gt, backward_preds, iou_threshold=0.5)
        expected = {
            (tuple(gt[0]["bbox_2d"]), (0.0, 0.0, 10.0, 10.0)),
            (tuple(gt[1]["bbox_2d"]), (51.0, 51.0, 61.0, 61.0)),
        }
        assert content_with(forward, preds) == expected
        assert content_with(backward, backward_preds) == expected

    def test_category_error_counts_as_fp_and_fn(self):
        gt = [obj("car", (0, 0, 10, 10))]
        pred = [obj("truck", (0, 0, 10, 10))]
        outcome = match_objects(gt, pred, iou_threshold=0.5)
        assert outcome.unmatched_gt == (0,) and outcome.unmatched_pred == (0,)
