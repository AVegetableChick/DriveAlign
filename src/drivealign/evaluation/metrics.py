"""M09 v1 metrics: per-anchor scoring and aggregate report content (S09).

Purpose:
    Implement every metric key of the S09 plan section 3 table. GT comes from
    the frozen dataset records (``training_targets.expected_output``);
    predictions come from the Base benchmark JSONL. Parse-failed anchors are
    excluded from all content metrics and counted only in the parse block;
    every denominator is written explicitly into the report under
    ``denominators``. The counting helpers are exported so the scene-cluster
    bootstrap can recompute the same statistics on resampled anchor subsets
    (one definition, two consumers).

Report key naming: ``<field>_<statistic>[_<aggregate>]`` with the IoU
threshold embedded after ``@`` (e.g. ``object_f1_micro@0.5``); names and
implementation correspond one-to-one.

Example launch command:
    ``conda activate autovla_codeclean && PYTHONPATH=DriveAlign/src python -c \
    "from drivealign.evaluation.metrics import score_anchor, aggregate; \
    gt = {'critical_objects': [], 'risk_factors': [], 'reasoning': 'r', \
    'yield_required': False, 'speed_action': 'KEEP_SPEED'}; \
    row = {'parse_ok': True, 'output': gt}; \
    s = score_anchor(gt, row, iou_thresholds=[0.5], \
    primary_iou_threshold=0.5, max_items=8); \
    print(aggregate([s], n_anchors_total=1, taxonomies={}, \
    primary_iou_threshold=0.5, iou_thresholds=[0.5])['parse_rate'])"``
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from statistics import mean, median
from typing import Any

from drivealign.evaluation.matching import match_objects

#: Over-conservative stop definition: GT keep/accelerate but prediction STOP.
_OC_STOP_GT = ("KEEP_SPEED", "ACCELERATE")


def _tau_key(tau: float) -> str:
    return f"{tau:g}"


# ---------------------------------------------------------------------------
# per-anchor scoring
# ---------------------------------------------------------------------------


def score_anchor(
    gt_output: Mapping[str, Any],
    pred_row: Mapping[str, Any],
    *,
    iou_thresholds: Sequence[float],
    primary_iou_threshold: float,
    max_items: int,
) -> dict[str, Any]:
    """Score one anchor against one prediction row.

    ``gt_output`` is the record's ``expected_output`` dict; ``pred_row`` is
    one predictions-JSONL row (``parse_ok``, ``output``, ``status``,
    ``errors``). Parse failures return a content-free score dict that only
    feeds the parse block.
    """
    token = str(pred_row.get("sample_token", pred_row.get("sample_id", "")))
    if not pred_row.get("parse_ok", False) or pred_row.get("output") is None:
        categories = sorted(
            {
                str(error.get("category"))
                for error in pred_row.get("errors", [])
                if error.get("category")
            }
        )
        return {
            "sample_token": token,
            "parse_ok": False,
            "status": pred_row.get("status"),
            "parse_error_categories": categories,
        }

    pred = pred_row["output"]
    gt_objects = list(gt_output.get("critical_objects", []))
    pred_objects = list(pred.get("critical_objects", []))
    per_tau: dict[str, dict[str, int]] = {}
    for tau in iou_thresholds:
        outcome = match_objects(gt_objects, pred_objects, iou_threshold=float(tau))
        per_tau[_tau_key(float(tau))] = {
            "tp": len(outcome.matched),
            "fp": len(outcome.unmatched_pred),
            "fn": len(outcome.unmatched_gt),
        }
    primary = match_objects(
        gt_objects, pred_objects, iou_threshold=primary_iou_threshold
    )
    return {
        "sample_token": token,
        "parse_ok": True,
        "status": pred_row.get("status"),
        "per_tau": per_tau,
        "matched_ious": [iou for _, _, iou in primary.matched],
        "matched_categories": [
            [gt_objects[g]["category"], pred_objects[p]["category"]]
            for g, p, _ in primary.matched
        ],
        "unmatched_gt_categories": [
            gt_objects[i]["category"] for i in primary.unmatched_gt
        ],
        "unmatched_pred_categories": [
            pred_objects[i]["category"] for i in primary.unmatched_pred
        ],
        "motion_pairs": [
            [gt_objects[g].get("motion_state"), pred_objects[p].get("motion_state")]
            for g, p, _ in primary.matched
        ],
        "gt_yield_required": bool(gt_output.get("yield_required")),
        "pred_yield_required": bool(pred.get("yield_required")),
        "gt_speed_action": str(gt_output.get("speed_action")),
        "pred_speed_action": str(pred.get("speed_action")),
        "gt_risk_factors": sorted(set(gt_output.get("risk_factors", []))),
        "pred_risk_factors": sorted(set(pred.get("risk_factors", []))),
        "n_pred_objects": len(pred_objects),
        "n_gt_objects": len(gt_objects),
        "max_items_hit": len(pred_objects) == max_items,
    }


# ---------------------------------------------------------------------------
# counting helpers (shared with the bootstrap resampler)
# ---------------------------------------------------------------------------


def parse_ok_scores(scores: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Content metrics use only parse-successful anchors."""
    return [score for score in scores if score.get("parse_ok")]


def f1_from_counts(tp: int, fp: int, fn: int) -> float:
    """F1 with the frozen zero-denominator convention (0.0 when undefined)."""
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def micro_object_counts(
    scores: Sequence[Mapping[str, Any]], tau: float
) -> tuple[int, int, int]:
    """Pooled (tp, fp, fn) over parse-ok anchors at one IoU threshold."""
    key = _tau_key(tau)
    ok = parse_ok_scores(scores)
    tp = sum(int(s["per_tau"][key]["tp"]) for s in ok)
    fp = sum(int(s["per_tau"][key]["fp"]) for s in ok)
    fn = sum(int(s["per_tau"][key]["fn"]) for s in ok)
    return tp, fp, fn


def per_category_counts(
    scores: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[int, int, int]]:
    """Per-category (tp, fp, fn) at the primary threshold.

    tp = matched pairs of that category; fp = unmatched predictions of that
    category; fn = unmatched GT of that category. Cross-category matches
    cannot exist by construction (class-wise matching), so a category error
    surfaces as one fp (prediction side) plus one fn (GT side).
    """
    counts: dict[str, list[int]] = {}
    for s in parse_ok_scores(scores):
        for gt_cat, pred_cat in s.get("matched_categories", []):
            counts.setdefault(gt_cat, [0, 0, 0])[0] += 1
            if pred_cat != gt_cat:  # defensive; unreachable by construction
                counts.setdefault(gt_cat, [0, 0, 0])[2] += 1
        for cat in s.get("unmatched_pred_categories", []):
            counts.setdefault(cat, [0, 0, 0])[1] += 1
        for cat in s.get("unmatched_gt_categories", []):
            counts.setdefault(cat, [0, 0, 0])[2] += 1
    return {cat: tuple(values) for cat, values in counts.items()}


def motion_state_counts(
    scores: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, dict[str, int]], int, int]:
    """Confusion over matched pairs plus (correct, total) for accuracy."""
    confusion: dict[str, dict[str, int]] = {}
    correct = total = 0
    for s in parse_ok_scores(scores):
        for gt_state, pred_state in s.get("motion_pairs", []):
            row = confusion.setdefault(str(gt_state), {})
            row[str(pred_state)] = row.get(str(pred_state), 0) + 1
            total += 1
            correct += int(gt_state == pred_state)
    return confusion, correct, total


def speed_action_counts(
    scores: Sequence[Mapping[str, Any]],
    speed_actions: Sequence[str],
) -> dict[str, dict[str, int]]:
    """GT x prediction confusion over parse-ok anchors (rows GT, cols pred)."""
    confusion: dict[str, dict[str, int]] = {}
    for s in parse_ok_scores(scores):
        gt = str(s["gt_speed_action"])
        pred = str(s["pred_speed_action"])
        confusion.setdefault(gt, {})
        confusion[gt][pred] = confusion[gt].get(pred, 0) + 1
    for gt in speed_actions:  # keep the full frozen 4x4 shape
        confusion.setdefault(gt, {})
        for pred in speed_actions:
            confusion[gt].setdefault(pred, 0)
    return confusion


def speed_action_per_class_f1(
    confusion: Mapping[str, Mapping[str, int]],
    speed_actions: Sequence[str],
) -> dict[str, float]:
    """Per-class F1 from a GT-x-pred confusion (one-vs-rest per class)."""
    result: dict[str, float] = {}
    for action in speed_actions:
        tp = int(confusion.get(action, {}).get(action, 0))
        fp = sum(int(confusion.get(gt, {}).get(action, 0)) for gt in confusion if gt != action)
        fn = sum(int(confusion.get(action, {}).get(pred, 0)) for pred in confusion.get(action, {}) if pred != action)
        result[action] = f1_from_counts(tp, fp, fn)
    return result


def yield_required_counts(scores: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Binary counts with positive class ``true``: tp/fp/fn/tn."""
    tp = fp = fn = tn = 0
    for s in parse_ok_scores(scores):
        gt, pred = s["gt_yield_required"], s["pred_yield_required"]
        tp += int(gt and pred)
        fp += int(not gt and pred)
        fn += int(gt and not pred)
        tn += int(not gt and not pred)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def risk_factor_counts(
    scores: Sequence[Mapping[str, Any]],
    risk_factors: Sequence[str],
) -> dict[str, tuple[int, int, int]]:
    """One-vs-rest per-frame set counts (tp, fp, fn) per risk factor."""
    counts = {risk: [0, 0, 0] for risk in risk_factors}
    for s in parse_ok_scores(scores):
        gt, pred = set(s["gt_risk_factors"]), set(s["pred_risk_factors"])
        for risk in risk_factors:
            counts[risk][0] += int(risk in gt and risk in pred)
            counts[risk][1] += int(risk not in gt and risk in pred)
            counts[risk][2] += int(risk in gt and risk not in pred)
    return {risk: tuple(values) for risk, values in counts.items()}


def overconservative_counts(scores: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Numerator/denominator pairs for the two over-conservative proxies."""
    yield_fp = yield_gt_false = stop_fp = stop_gt_keep = 0
    for s in parse_ok_scores(scores):
        yield_gt_false += int(not s["gt_yield_required"])
        yield_fp += int(not s["gt_yield_required"] and s["pred_yield_required"])
        stop_gt_keep += int(s["gt_speed_action"] in _OC_STOP_GT)
        stop_fp += int(
            s["gt_speed_action"] in _OC_STOP_GT and s["pred_speed_action"] == "STOP"
        )
    return {
        "yield_fp": yield_fp,
        "yield_gt_false": yield_gt_false,
        "stop_fp": stop_fp,
        "stop_gt_keep": stop_gt_keep,
    }


def parse_error_counts(scores: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Parse-failure counts by frozen S03 error category (sorted keys)."""
    counts: dict[str, int] = {}
    for s in scores:
        if s.get("parse_ok"):
            continue
        for category in s.get("parse_error_categories", ["unknown"]):
            counts[category] = counts.get(category, 0) + 1
    return dict(sorted(counts.items()))


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------


def aggregate(
    scores: Sequence[Mapping[str, Any]],
    *,
    n_anchors_expected: int,
    taxonomies: Mapping[str, Sequence[str]],
    primary_iou_threshold: float,
    iou_thresholds: Sequence[float],
) -> dict[str, Any]:
    """Assemble the m09_report content dict (plan section 3 keys)."""
    ok = parse_ok_scores(scores)
    n_scored = len(scores)
    n_ok = len(ok)
    report: dict[str, Any] = {
        "coverage": {
            "anchors_expected": n_anchors_expected,
            "anchors_scored": n_scored,
            "parse_ok": n_ok,
            "parse_failed": n_scored - n_ok,
        },
        "parse_rate": n_ok / n_scored if n_scored else 0.0,
        "parse_error_counts": parse_error_counts(scores),
    }

    for tau in iou_thresholds:
        tag = _tau_key(float(tau))
        tp, fp, fn = micro_object_counts(scores, float(tau))
        report[f"object_precision_micro@{tag}"] = tp / (tp + fp) if tp + fp else 0.0
        report[f"object_recall_micro@{tag}"] = tp / (tp + fn) if tp + fn else 0.0
        report[f"object_f1_micro@{tag}"] = f1_from_counts(tp, fp, fn)

    per_category = per_category_counts(scores)
    per_category_f1 = {
        cat: f1_from_counts(*counts)
        for cat, counts in sorted(per_category.items())
        if counts[0] + counts[2] > 0  # only categories with non-empty GT
    }
    tag = _tau_key(primary_iou_threshold)
    report[f"object_f1_per_category@{tag}"] = per_category_f1
    report[f"object_f1_macro@{tag}"] = (
        mean(per_category_f1.values()) if per_category_f1 else 0.0
    )
    report["macro_categories_counted"] = len(per_category_f1)

    matched_ious = sorted(iou for s in ok for iou in s.get("matched_ious", []))
    report["matched_pair_count"] = len(matched_ious)
    report["matched_iou_mean"] = mean(matched_ious) if matched_ious else 0.0
    report["matched_iou_p50"] = (
        median(matched_ious) if matched_ious else 0.0
    )
    report["matched_iou_p90"] = _percentile(matched_ious, 90)

    confusion, correct, total = motion_state_counts(scores)
    report["motion_state_confusion"] = confusion
    report["motion_state_accuracy"] = correct / total if total else 0.0

    speed_actions = list(taxonomies.get("speed_actions", []))
    speed_confusion = speed_action_counts(scores, speed_actions)
    report["speed_action_confusion"] = speed_confusion
    per_class = speed_action_per_class_f1(speed_confusion, speed_actions)
    report["speed_action_per_class_f1"] = per_class
    report["speed_action_f1_macro"] = (
        mean(per_class.values()) if per_class else 0.0
    )

    yield_counts = yield_required_counts(scores)
    report["yield_required_f1"] = f1_from_counts(
        yield_counts["tp"], yield_counts["fp"], yield_counts["fn"]
    )

    risks = list(taxonomies.get("risk_factors", []))
    risk_counts = risk_factor_counts(scores, risks)
    report["risk_factors_f1_per_class"] = {
        risk: f1_from_counts(*risk_counts[risk]) for risk in risks
    }
    report["risk_factors_f1_macro"] = (
        mean(report["risk_factors_f1_per_class"].values()) if risks else 0.0
    )

    report["max_items_hit_rate"] = (
        sum(1 for s in ok if s.get("max_items_hit")) / n_ok if n_ok else 0.0
    )

    oc = overconservative_counts(scores)
    report["overconservative_yield_rate"] = (
        oc["yield_fp"] / oc["yield_gt_false"] if oc["yield_gt_false"] else 0.0
    )
    report["overconservative_stop_rate"] = (
        oc["stop_fp"] / oc["stop_gt_keep"] if oc["stop_gt_keep"] else 0.0
    )

    report["denominators"] = {
        "anchors_expected": n_anchors_expected,
        "anchors_scored": n_scored,
        "parse_ok": n_ok,
        "matched_pairs": len(matched_ious),
        "motion_pairs": total,
        "yield_counts": yield_counts,
        "overconservative": oc,
        "micro_counts": {
            _tau_key(float(tau)): list(micro_object_counts(scores, float(tau)))
            for tau in iou_thresholds
        },
    }
    return report


def _percentile(sorted_values: Sequence[float], pct: float) -> float:
    """Linear-interpolation percentile over an already-sorted list."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = (len(sorted_values) - 1) * pct / 100.0
    low = int(rank)
    high = min(low + 1, len(sorted_values) - 1)
    fraction = rank - low
    return float(sorted_values[low] * (1.0 - fraction) + sorted_values[high] * fraction)
