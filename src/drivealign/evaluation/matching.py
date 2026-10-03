"""M09 v1 object matching: class-wise greedy IoU assignment (S09).

Purpose:
    Pair predicted critical objects against GT critical objects for the
    Stage 09 Base benchmark. Ruling (2026-10-03, user): class-wise GREEDY
    matching replaces the Stage 08 draft's Hungarian assignment — at the
    <=8x8 scale the numerical gap to the global optimum is negligible,
    class-wise matching makes "category error = FP + FN" structural (a
    prediction never pairs across categories), the delta-category parameter
    disappears, and no scipy dependency is needed. Greedy order: IoU
    descending, bbox lexicographic tie-break; the order is independent of
    the prediction list order, so results are deterministic and
    reproducible. A pair is only assignable when IoU >= threshold (boundary
    inclusive); each GT matches at most once; leftover predictions are FP
    and leftover GT are FN.

Example launch command:
    ``conda activate autovla_codeclean && PYTHONPATH=DriveAlign/src python -c \
    "from drivealign.evaluation.matching import match_objects; \
    gt = [{'category': 'car', 'bbox_2d': [0, 0, 10, 10]}]; \
    pred = [{'category': 'car', 'bbox_2d': [1, 1, 11, 11]}, \
    {'category': 'pedestrian', 'bbox_2d': [0, 0, 10, 10]}]; \
    m = match_objects(gt, pred, iou_threshold=0.5); \
    print(m.matched, m.unmatched_pred, m.unmatched_gt)"``
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class MatchOutcome:
    """Result of one class-wise greedy matching pass.

    ``matched`` holds ``(gt_index, pred_index, iou)`` triples; the index
    tuples refer to the input lists. ``unmatched_pred`` are FP indices and
    ``unmatched_gt`` are FN indices.
    """

    matched: tuple[tuple[int, int, float], ...]
    unmatched_pred: tuple[int, ...]
    unmatched_gt: tuple[int, ...]


def _normalized_category(category: str) -> str:
    """Canonical closed-vocab categories arrive identical on both sides;
    lowercasing/stripping is a defensive no-op that keeps matching total."""
    return category.strip().lower()


def bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    """IoU of two ``[x1, y1, x2, y2]`` boxes; 0.0 on degenerate unions."""
    ax1, ay1, ax2, ay2 = (float(v) for v in a[:4])
    bx1, by1, bx2, by2 = (float(v) for v in b[:4])
    inter_w = min(ax2, bx2) - max(ax1, bx1)
    inter_h = min(ay2, by2) - max(ay1, by1)
    if inter_w <= 0.0 or inter_h <= 0.0:
        return 0.0
    intersection = inter_w * inter_h
    area_a = max(ax2 - ax1, 0.0) * max(ay2 - ay1, 0.0)
    area_b = max(bx2 - bx1, 0.0) * max(by2 - by1, 0.0)
    union = area_a + area_b - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def match_objects(
    gt_objects: Sequence[Mapping],
    pred_objects: Sequence[Mapping],
    *,
    iou_threshold: float,
) -> MatchOutcome:
    """Class-wise greedy IoU matching between GT and predicted objects.

    Candidate pairs form only within one normalized category and only when
    IoU >= ``iou_threshold``; they are consumed in IoU-descending order with
    a bbox lexicographic tie-break (then stable index order), each GT and
    each prediction used at most once.
    """
    candidates: list[tuple[float, tuple, tuple, int, int]] = []
    for gt_index, gt in enumerate(gt_objects):
        gt_category = _normalized_category(str(gt["category"]))
        gt_box = tuple(float(v) for v in gt["bbox_2d"][:4])
        for pred_index, pred in enumerate(pred_objects):
            if _normalized_category(str(pred["category"])) != gt_category:
                continue
            pred_box = tuple(float(v) for v in pred["bbox_2d"][:4])
            iou = bbox_iou(gt_box, pred_box)
            if iou >= iou_threshold:
                candidates.append((iou, gt_box, pred_box, gt_index, pred_index))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2], item[3], item[4]))

    gt_taken: set[int] = set()
    pred_taken: set[int] = set()
    matched: list[tuple[int, int, float]] = []
    for iou, _, _, gt_index, pred_index in candidates:
        if gt_index in gt_taken or pred_index in pred_taken:
            continue
        gt_taken.add(gt_index)
        pred_taken.add(pred_index)
        matched.append((gt_index, pred_index, iou))

    return MatchOutcome(
        matched=tuple(matched),
        unmatched_pred=tuple(
            i for i in range(len(pred_objects)) if i not in pred_taken
        ),
        unmatched_gt=tuple(i for i in range(len(gt_objects)) if i not in gt_taken),
    )
