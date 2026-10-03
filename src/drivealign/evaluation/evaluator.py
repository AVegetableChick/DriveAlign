"""Eval v1 evaluator: predictions + frozen GT -> eval_report (S09).

Purpose:
    Read the Base benchmark predictions JSONL, locate every anchor's frozen
    record in the dataset shards through the anchor manifest, verify the
    record hash, score every anchor (:func:`score_anchor`), aggregate the
    full plan-section-3 report, attach scene-cluster bootstrap CIs and the
    cost summary, and write ``eval_report.json`` / ``eval_report.md`` /
    ``anchor_scores.jsonl``. The evaluator is strictly read-only on inputs.
    Determinism: with identical predictions and config the report JSON is
    byte-identical across reruns (sorted keys, no timestamps, no git state).

    Optional counterfactual comparison: when counterfactual prediction files
    (blank/shuffled) are supplied, the visual-dependence diagnostics
    (``visual_dependence_action_flip_*`` / ``visual_dependence_output_change_*``)
    are computed over anchors present in both the main and the counterfactual
    file; the compared anchor count is reported explicitly.

Example launch command:
    ``conda activate autovla_codeclean && cd /root/autodl-tmp/drivealign_workspace && \
    PYTHONPATH=DriveAlign/src python -m drivealign.cli.evaluate \
    --predictions runs/S09_base_benchmark/predictions.jsonl \
    --anchor-manifest data/dataset_v4/anchor_policy_manifest.json \
    --dataset-root data/dataset_v4 \
    --config DriveAlign/configs/evaluation/eval_v1.yaml \
    --out runs/S09_base_benchmark``
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import mean
from typing import Any

import yaml

from drivealign.evaluation.bootstrap import bootstrap_statistics
from drivealign.evaluation.metrics import (
    aggregate,
    f1_from_counts,
    micro_object_counts,
    motion_state_counts,
    overconservative_counts,
    parse_ok_scores,
    per_category_counts,
    risk_factor_counts,
    speed_action_counts,
    speed_action_per_class_f1,
    yield_required_counts,
    score_anchor,
)
from drivealign.records.record import DriveAlignRecord


# ---------------------------------------------------------------------------
# input loading
# ---------------------------------------------------------------------------


def load_predictions(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load the predictions JSONL keyed by sample token (duplicates fail)."""
    predictions: dict[str, dict[str, Any]] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle):
            raw = raw.strip()
            if not raw:
                continue
            row = json.loads(raw)
            token = str(row.get("sample_token", row.get("sample_id", "")))
            if not token:
                raise ValueError(f"predictions row {line_number} has no sample token")
            if token in predictions:
                raise ValueError(f"duplicate prediction for anchor {token}")
            predictions[token] = row
    return predictions


def load_gt_outputs(
    dataset_root: str | Path,
    manifest: Mapping[str, Any],
    tokens: Sequence[str],
) -> dict[str, dict[str, Any]]:
    """Read ``expected_output`` for each token; verify every record hash."""
    anchors = manifest["anchors"]
    by_shard: dict[str, list[tuple[int, str]]] = {}
    for token in tokens:
        entry = anchors[token]
        by_shard.setdefault(str(entry["shard"]), []).append(
            (int(entry["line"]), token)
        )
    gt: dict[str, dict[str, Any]] = {}
    for shard_rel, wanted in sorted(by_shard.items()):
        shard_path = Path(dataset_root) / shard_rel
        with shard_path.open("r", encoding="utf-8") as handle:
            lines = handle.readlines()
        for line_index, token in sorted(wanted):
            row = json.loads(lines[line_index])
            record = DriveAlignRecord.from_dict(row)
            expected_hash = str(anchors[token]["record_hash"])
            if record.canonical_hash() != expected_hash:
                raise ValueError(
                    f"record hash mismatch for {token} in {shard_rel}"
                    f" line {line_index}: frozen manifest says {expected_hash}"
                )
            expected = record.training_targets.expected_output
            if expected is None:
                raise ValueError(f"anchor {token} has no backfilled expected_output")
            gt[token] = expected
    return gt


# ---------------------------------------------------------------------------
# derived report blocks
# ---------------------------------------------------------------------------

#: Frozen Eval v1 enums (mirrored in eval_v1.yaml; pinned by unit tests).
_SPEED_ACTIONS = ("ACCELERATE", "KEEP_SPEED", "DECELERATE", "STOP")
_RISK_FACTORS = (
    "pedestrian_crossing",
    "vehicle_merging",
    "lead_vehicle_braking",
    "corridor_conflict",
    "stationary_obstacle",
    "congestion",
    "oncoming_traffic",
)


def subset_statistics(
    subset_scores: Sequence[Mapping[str, Any]],
    *,
    speed_actions: Sequence[str] = _SPEED_ACTIONS,
    risk_factors: Sequence[str] = _RISK_FACTORS,
    primary_iou_threshold: float = 0.5,
) -> dict[str, float]:
    """All bootstrapped statistics recomputed on one anchor subset.

    Uses exactly the same counting helpers as :func:`aggregate`, so point
    estimates and CI replicates share one definition. ``parse_rate`` uses the
    subset size as its denominator (the subset plays the role of the scored
    set during resampling). The enum defaults mirror eval_v1.yaml; a pin test
    keeps config and constants identical.
    """
    ok = parse_ok_scores(subset_scores)
    n = len(subset_scores)
    stats: dict[str, float] = {"parse_rate": len(ok) / n if n else 0.0}
    tp, fp, fn = micro_object_counts(subset_scores, primary_iou_threshold)
    stats[f"object_f1_micro@{primary_iou_threshold:g}"] = f1_from_counts(tp, fp, fn)
    yield_counts = yield_required_counts(subset_scores)
    stats["yield_required_f1"] = f1_from_counts(
        yield_counts["tp"], yield_counts["fp"], yield_counts["fn"]
    )
    per_category = {
        cat: f1_from_counts(*counts)
        for cat, counts in per_category_counts(subset_scores).items()
        if counts[0] + counts[2] > 0
    }
    stats[f"object_f1_macro@{primary_iou_threshold:g}"] = (
        mean(per_category.values()) if per_category else 0.0
    )
    _, correct, total = motion_state_counts(subset_scores)
    stats["motion_state_accuracy"] = correct / total if total else 0.0
    speed_confusion = speed_action_counts(subset_scores, speed_actions)
    per_class = speed_action_per_class_f1(speed_confusion, speed_actions)
    stats["speed_action_f1_macro"] = mean(per_class.values()) if per_class else 0.0
    risk_counts = risk_factor_counts(subset_scores, risk_factors)
    stats["risk_factors_f1_macro"] = (
        mean(f1_from_counts(*risk_counts[risk]) for risk in risk_factors)
        if risk_factors
        else 0.0
    )
    oc = overconservative_counts(subset_scores)
    stats["overconservative_yield_rate"] = (
        oc["yield_fp"] / oc["yield_gt_false"] if oc["yield_gt_false"] else 0.0
    )
    stats["overconservative_stop_rate"] = (
        oc["stop_fp"] / oc["stop_gt_keep"] if oc["stop_gt_keep"] else 0.0
    )
    stats["max_items_hit_rate"] = (
        sum(1 for s in ok if s.get("max_items_hit")) / len(ok) if ok else 0.0
    )
    return stats


def cost_summary(prediction_rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Cost block from per-record telemetry (plan section 3, last row)."""
    latencies = sorted(
        float(row["telemetry"]["generation_seconds"])
        for row in prediction_rows
        if row.get("telemetry")
    )
    tokens_in = sorted(
        float(row["telemetry"]["input_tokens"])
        for row in prediction_rows
        if row.get("telemetry")
    )
    peaks = [
        float(row["telemetry"]["peak_cuda_memory_bytes"])
        for row in prediction_rows
        if row.get("telemetry")
    ]
    total_generation_seconds = sum(latencies)
    return {
        "latency_p50": _percentile(latencies, 50),
        "latency_p90": _percentile(latencies, 90),
        "throughput_anchors_per_s": (
            len(latencies) / total_generation_seconds if total_generation_seconds else 0.0
        ),
        # NOTE: the runner telemetry records TOTAL prompt input tokens
        # (text prompt + image tokens); a visual-token-only split is not
        # available without touching the frozen S02 loader, so the plan's
        # ``visual_tokens_p50`` is reported as input_tokens_p50.
        "input_tokens_p50": _percentile(tokens_in, 50),
        "peak_cuda_mem_bytes": max(peaks) if peaks else 0.0,
        "gpu_hours": total_generation_seconds / 3600.0,
    }


def counterfactual_comparison(
    main_predictions: Mapping[str, Mapping[str, Any]],
    counterfactual_predictions: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Visual-dependence diagnostics over the anchors present in both files."""
    shared = sorted(set(main_predictions) & set(counterfactual_predictions))
    flips = output_changes = 0
    for token in shared:
        main = main_predictions[token]
        other = counterfactual_predictions[token]
        main_ok = bool(main.get("parse_ok")) and main.get("output") is not None
        other_ok = bool(other.get("parse_ok")) and other.get("output") is not None
        if not (main_ok and other_ok):
            continue  # flips are defined on parse-ok pairs only
        if main["output"]["speed_action"] != other["output"]["speed_action"]:
            flips += 1
        if main["output"] != other["output"]:
            output_changes += 1
    return {
        "n_compared": len(shared),
        "action_flip_rate": flips / len(shared) if shared else 0.0,
        "output_change_rate": output_changes / len(shared) if shared else 0.0,
    }


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------


def run_evaluation(
    predictions_path: str | Path,
    anchor_manifest_path: str | Path,
    dataset_root: str | Path,
    config_path: str | Path,
    out_dir: str | Path,
    counterfactual_paths: Mapping[str, str | Path] | None = None,
) -> dict[str, Any]:
    """Run the full Eval v1 evaluation and write the report artifacts."""
    config_bytes = Path(config_path).read_bytes()
    config = yaml.safe_load(config_bytes)
    manifest = json.loads(Path(anchor_manifest_path).read_text(encoding="utf-8"))
    split = str(config["evaluation_split"])

    predictions = load_predictions(predictions_path)
    expected_tokens = sorted(
        token
        for token, entry in manifest["anchors"].items()
        if entry["split"] == split
    )
    predicted_tokens = sorted(predictions)
    missing = sorted(set(expected_tokens) - set(predicted_tokens))
    extra = sorted(set(predicted_tokens) - set(expected_tokens))

    gt_outputs = load_gt_outputs(dataset_root, manifest, predicted_tokens)
    scores = [
        score_anchor(
            gt_outputs[token],
            predictions[token],
            iou_thresholds=[float(t) for t in config["matching"]["iou_thresholds"]],
            primary_iou_threshold=float(config["matching"]["primary_iou_threshold"]),
            max_items=int(config["max_items"]),
        )
        for token in predicted_tokens
    ]
    report: dict[str, Any] = aggregate(
        scores,
        n_anchors_expected=len(expected_tokens),
        taxonomies=config.get("taxonomies", {}),
        primary_iou_threshold=float(config["matching"]["primary_iou_threshold"]),
        iou_thresholds=[float(t) for t in config["matching"]["iou_thresholds"]],
    )
    report["coverage"]["missing_predictions"] = len(missing)
    report["coverage"]["extra_predictions"] = len(extra)
    if missing:
        report["coverage"]["missing_sample"] = missing[:5]

    boot = config["bootstrap"]
    clusters = {
        token: str(manifest["anchors"][token]["scene_token"])
        for token in predicted_tokens
    }
    scores_by_token = {s["sample_token"]: s for s in scores}
    report["bootstrap_ci"] = bootstrap_statistics(
        clusters,
        lambda tokens: subset_statistics(
            [scores_by_token[t] for t in tokens],
            speed_actions=list(config.get("taxonomies", {}).get("speed_actions", _SPEED_ACTIONS)),
            risk_factors=list(config.get("taxonomies", {}).get("risk_factors", _RISK_FACTORS)),
            primary_iou_threshold=float(config["matching"]["primary_iou_threshold"]),
        ),
        n_resamples=int(boot["n_resamples"]),
        seed=int(boot["seed"]),
        confidence_level=float(boot["confidence_level"]),
    )
    report["cost_summary"] = cost_summary(list(predictions.values()))

    if counterfactual_paths:
        counterfactual_block: dict[str, Any] = {}
        for transform, path in sorted(counterfactual_paths.items()):
            comparison = counterfactual_comparison(
                predictions, load_predictions(path)
            )
            counterfactual_block[transform] = comparison
            report[f"visual_dependence_action_flip_{transform}"] = comparison[
                "action_flip_rate"
            ]
            report[f"visual_dependence_output_change_{transform}"] = comparison[
                "output_change_rate"
            ]
        report["counterfactual"] = counterfactual_block

    report["meta"] = {
        "eval_version": str(config["eval_version"]),
        "config_status": str(config.get("status", "unknown")),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "contract_version": str(manifest.get("contract_version", "unknown")),
        "anchor_manifest_sha256": hashlib.sha256(
            Path(anchor_manifest_path).read_bytes()
        ).hexdigest(),
        "evaluation_split": split,
    }

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report_json = out / "eval_report.json"
    report_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (out / "anchor_scores.jsonl").open("w", encoding="utf-8") as handle:
        for score in scores:
            handle.write(json.dumps(score, sort_keys=True) + "\n")
    (out / "eval_report.md").write_text(_render_markdown(report), encoding="utf-8")
    return report


def _percentile(sorted_values: Sequence[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = (len(sorted_values) - 1) * pct / 100.0
    low = int(rank)
    high = min(low + 1, len(sorted_values) - 1)
    fraction = rank - low
    return float(sorted_values[low] * (1.0 - fraction) + sorted_values[high] * fraction)


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _render_markdown(report: Mapping[str, Any]) -> str:
    """Deterministic human rendering of the report (no locale/time state)."""
    lines: list[str] = ["# Eval v1 report", ""]
    meta = report["meta"]
    lines += [
        f"- eval_version: {meta['eval_version']} (config status: {meta['config_status']})",
        f"- contract_version: {meta['contract_version']}",
        f"- evaluation_split: {meta['evaluation_split']}",
        f"- config_sha256: `{meta['config_sha256']}`",
        "",
    ]
    lines += ["## Coverage & parse", ""]
    for key in ("anchors_expected", "anchors_scored", "parse_ok", "parse_failed",
                "missing_predictions", "extra_predictions"):
        lines.append(f"- {key}: {report['coverage'].get(key, 0)}")
    lines += ["", f"- parse_rate: {_fmt(report['parse_rate'])}", ""]
    if report["parse_error_counts"]:
        lines += ["Parse errors:", ""]
        lines += [f"- {k}: {v}" for k, v in report["parse_error_counts"].items()]
        lines.append("")
    lines += ["## Object metrics", ""]
    lines += ["| metric | value |", "|---|---|"]
    for key in sorted(report):
        if key.startswith("object_") or key.startswith("matched_iou"):
            lines.append(f"| {key} | {_fmt(report[key])} |")
    lines.append("")
    lines += ["## Field metrics", ""]
    lines += ["| metric | value |", "|---|---|"]
    for key in (
        "motion_state_accuracy", "speed_action_f1_macro", "yield_required_f1",
        "risk_factors_f1_macro", "max_items_hit_rate",
        "overconservative_yield_rate", "overconservative_stop_rate",
    ):
        lines.append(f"| {key} | {_fmt(report[key])} |")
    lines.append("")
    lines += ["## 95% CI (scene-cluster bootstrap)", ""]
    lines += ["| metric | point | ci_low | ci_high |", "|---|---|---|---|"]
    for key in sorted(report["bootstrap_ci"]):
        cell = report["bootstrap_ci"][key]
        lines.append(
            f"| {key} | {_fmt(cell['point_estimate'])} "
            f"| {_fmt(cell['ci_low'])} | {_fmt(cell['ci_high'])} |"
        )
    lines.append("")
    lines += ["## Cost summary", ""]
    lines += [f"- {k}: {_fmt(v)}" for k, v in sorted(report["cost_summary"].items())]
    if "counterfactual" in report:
        lines += ["", "## Visual dependence (counterfactual subset)", ""]
        for transform, block in sorted(report["counterfactual"].items()):
            lines.append(
                f"- {transform}: n_compared={block['n_compared']}, "
                f"action_flip={_fmt(block['action_flip_rate'])}, "
                f"output_change={_fmt(block['output_change_rate'])}"
            )
    lines.append("")
    return "\n".join(lines)
