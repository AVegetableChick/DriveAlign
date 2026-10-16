"""Eval v3 evaluator: predictions + frozen GT -> eval_report (S13 eval v3).

Purpose:
    Eval v3 编排层（S13 eval v3 指标新增草案，拍板定稿 2026-10-14）：
    eval v3 = eval v2 + BEV 误差族（:mod:`drivealign.evaluation.metrics_bev`）
    + risk 归因族（:mod:`drivealign.evaluation.metrics_attribution`）
    + risk 词表 v7 化（6 枚举）。matching 网格 / bootstrap 三参数
    （n=2000 / seed=20261003 / 0.95）/ counterfactual battery /
    max_items 全部沿用 eval_v2 冻结值（草案 §1 不变量表）。

    对 v2 路径的红线：**eval v2 报告行为不得改变**——本模块不复刻、
    不修改 v1/v2 任何函数；打分层 :func:`score_anchor_v3` 以 v1
    :func:`score_anchor` 的返回为基底**追加** v3 字段（``bev_pairs`` /
    ``risk_attribution_counts``），v2 消费者读不到新字段、行为逐字节
    不变；聚合层 :func:`aggregate_v3` = :func:`aggregate_v2` 全量键 +
    v3 新增键（51 标量 + 1 嵌套块，2026-10-14 用户裁定按草案 §5 清单
    枚举钉死，文档「50」为算术勘误）；bootstrap = v2 全部键 + 5 新键
    （bev 三 mean + 归因 micro/macro，草案 §2.5/§3.5 的「10→15」为
    增量记法，v2 实际标量基数 20 → v3 25）。

    counterfactual battery 不加 BEV/归因读数（草案 §6-5 维持默认）：
    对比块结构与 v2 逐字节相同（action flip / output change 两率）。

Example launch command:
    ``conda activate autovla_codeclean && cd /root/autodl-tmp/drivealign_workspace && \
    PYTHONPATH=DriveAlign/src python -c \
    "from drivealign.evaluation.evaluator_v3 import run_evaluation_v3; \
    run_evaluation_v3('runs/<run>/predictions.jsonl', \
    'data/face_manifests/v7/anchor_policy_manifest.json', 'data/dataset_v7', \
    'DriveAlign/configs/evaluation/eval_v3.yaml', 'runs/<run>/eval_v3')" ``
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from drivealign.evaluation.bootstrap import bootstrap_statistics_nested
from drivealign.evaluation.evaluator import (
    _fmt,
    cost_summary,
    counterfactual_comparison,
    load_gt_outputs,
    load_predictions,
)
from drivealign.evaluation.evaluator_v2 import subset_statistics_v2
from drivealign.evaluation.matching import match_objects
from drivealign.evaluation.metrics import per_category_counts, score_anchor
from drivealign.evaluation.metrics_attribution import (
    aggregate_attribution,
    anchor_attribution_counts,
    attribution_totals,
)
from drivealign.evaluation.metrics_bev import bev_statistics, collect_bev_pairs
from drivealign.evaluation.metrics_v2 import aggregate_v2

# ---------------------------------------------------------------------------
# 冻结枚举（risk 词表 = contract v7 六件套 risk_taxonomy canonical 顺序，
# 与 eval_v3.yaml taxonomies.risk_factors 逐值同源，pin 测试 import 契约
# 资产钉死防漂移；motion/action 枚举与 v2 逐值相同）
# ---------------------------------------------------------------------------

_MOTION_STATES = ("stationary", "same_direction", "oncoming", "crossing")
_SPEED_ACTIONS = ("ACCELERATE", "KEEP_SPEED", "DECELERATE", "STOP")
_RISK_FACTORS_V7 = (
    "vru_crossing",
    "vehicle_crossing",
    "lead_vehicle_braking",
    "stationary_obstacle",
    "object_near",
    "oncoming_traffic",
)


# ---------------------------------------------------------------------------
# per-anchor scoring（v1 score_anchor 追加 v3 字段，v2 消费者不受影响）
# ---------------------------------------------------------------------------


def score_anchor_v3(
    gt_output: Mapping[str, Any],
    pred_row: Mapping[str, Any],
    *,
    iou_thresholds: Sequence[float],
    primary_iou_threshold: float,
    max_items: int,
) -> dict[str, Any]:
    """Eval v3 逐 anchor 打分：v1 全部字段 + ``bev_pairs`` + 归因计数。

    v3 专属字段在 primary（主阈值）匹配结果上派生：

    - ``bev_pairs``：匹配对 ``(x_p, y_p, x_g, y_g)`` 样本（几何在此
      一次解析，聚合与 bootstrap 只消费样本列表）；
    - ``risk_attribution_counts``：conditional 双向归因的逐因子
      ``(tp, fp, fn)``（id 桥接复用 primary 匹配——与 object_f1 族
      同一份匹配结果，一次计算两处消费）。

    primary 匹配在本函数内独立重跑（与 v1 ``score_anchor`` 内部同参数
    同确定性结果，``match_objects`` 为纯函数）；不改动 v1/v2 函数签名
    与行为。
    """
    score = score_anchor(
        gt_output,
        pred_row,
        iou_thresholds=iou_thresholds,
        primary_iou_threshold=primary_iou_threshold,
        max_items=max_items,
    )
    if not score.get("parse_ok"):
        return score  # parse 失败：content-free，v3 字段同样缺席
    pred_objects = list(pred_row["output"].get("critical_objects", []))
    gt_objects = list(gt_output.get("critical_objects", []))
    primary = match_objects(
        gt_objects, pred_objects, iou_threshold=primary_iou_threshold
    )
    score["bev_pairs"] = collect_bev_pairs(gt_objects, pred_objects, primary)
    score["risk_attribution_counts"] = anchor_attribution_counts(
        gt_output, pred_row["output"], primary
    )
    return score


# ---------------------------------------------------------------------------
# aggregation（v2 全量键 + v3 新增：BEV 49 标量 + 归因 2 标量 + 嵌套块）
# ---------------------------------------------------------------------------


def aggregate_v3(
    scores: Sequence[Mapping[str, Any]],
    *,
    n_anchors_expected: int,
    taxonomies: Mapping[str, Sequence[str]],
    primary_iou_threshold: float,
    iou_thresholds: Sequence[float],
    max_items: int,
    risk_max_items: int,
) -> dict[str, Any]:
    """组装 eval_v3 报告内容 dict：``aggregate_v2`` 全量键 + v3 新增键。

    v3 新增（与草案 §5 清单逐键一致，51 标量 + 1 嵌套块）：
    ``bev_{dx,dy,dxy}_{mean,p50,p90}``（9）、
    ``bev_{dx,dy,dxy}_{mean,p50,p90}_bucket_{0_10,10_25,25_40,40p}``（36）、
    ``bev_signed_{dx,dy}_{pos,neg}_mean``（4）、
    ``risk_attribution_f1_micro`` / ``risk_attribution_f1_macro``（2）、
    ``risk_attribution_per_factor``（嵌套块）。
    """
    report: dict[str, Any] = aggregate_v2(
        scores,
        n_anchors_expected=n_anchors_expected,
        taxonomies=taxonomies,
        primary_iou_threshold=primary_iou_threshold,
        iou_thresholds=iou_thresholds,
        max_items=max_items,
        risk_max_items=risk_max_items,
    )

    ok = [score for score in scores if score.get("parse_ok")]
    bev_pairs = [pair for score in ok for pair in score.get("bev_pairs", [])]
    report.update(bev_statistics(bev_pairs))
    report.update(aggregate_attribution(attribution_totals(scores)))
    report["denominators"]["bev_pairs"] = len(bev_pairs)
    return report


# ---------------------------------------------------------------------------
# bootstrap 统计量（v2 全部键 + 5 新键：bev 三 mean + 归因 micro/macro）
# ---------------------------------------------------------------------------


def subset_statistics_v3(
    subset_scores: Sequence[Mapping[str, Any]],
    *,
    motion_states: Sequence[str] = _MOTION_STATES,
    speed_actions: Sequence[str] = _SPEED_ACTIONS,
    risk_factors: Sequence[str] = _RISK_FACTORS_V7,
    primary_iou_threshold: float = 0.5,
    iou_thresholds: Sequence[float] = (0.3, 0.5, 0.7),
    object_categories_full: Sequence[str] = (),
) -> dict[str, Any]:
    """eval_v3 全部被 bootstrap 统计量在一个 anchor 子集上的重算。

    与 :func:`subset_statistics_v2` 同构（点估计与重采样子集共用本函数，
    CI 与点估计同一定义）：先原样调用 v2 统计量（词表由调用方传 v7 六
    枚举），再追加 5 个新标量键。BEV p50/p90、分桶键、signed 键、
    per_factor 嵌套块不入 bootstrap（草案 §2.5/§3.5：首版只对 mean 出
    CI，嵌套块是诊断读数）。
    """
    stats: dict[str, Any] = subset_statistics_v2(
        subset_scores,
        motion_states=motion_states,
        speed_actions=speed_actions,
        risk_factors=risk_factors,
        primary_iou_threshold=primary_iou_threshold,
        iou_thresholds=iou_thresholds,
        object_categories_full=object_categories_full,
    )
    ok = [score for score in subset_scores if score.get("parse_ok")]
    bev_pairs = [pair for score in ok for pair in score.get("bev_pairs", [])]
    bev = bev_statistics(bev_pairs)
    for family in ("dx", "dy", "dxy"):
        stats[f"bev_{family}_mean"] = bev[f"bev_{family}_mean"]
    attribution = aggregate_attribution(attribution_totals(subset_scores))
    stats["risk_attribution_f1_micro"] = attribution["risk_attribution_f1_micro"]
    stats["risk_attribution_f1_macro"] = attribution["risk_attribution_f1_macro"]
    return stats


# ---------------------------------------------------------------------------
# orchestration（T6 硬前置 → 打分 → aggregate_v3 → bootstrap → 产物写出）
# ---------------------------------------------------------------------------


def run_evaluation_v3(
    predictions_path: str | Path,
    anchor_manifest_path: str | Path,
    dataset_root: str | Path,
    config_path: str | Path,
    out_dir: str | Path,
    counterfactual_paths: Mapping[str, str | Path] | None = None,
) -> dict[str, Any]:
    """运行完整 Eval v3 评测并写出报告产物（签名与 v2 run_evaluation_v2 一致）。"""
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

    # T6 完整性硬前置（v2 同款）：missing / extra 非空即在任何文件写出
    # 之前 fail-fast，残缺批次不允许产出任何产物。
    if missing or extra:
        raise ValueError(
            "eval_v3 completeness precondition violated: "
            f"{len(missing)} missing predictions (first 5: {missing[:5]}), "
            f"{len(extra)} extra predictions (first 5: {extra[:5]})"
        )

    gt_outputs = load_gt_outputs(dataset_root, manifest, predicted_tokens)
    scores = [
        score_anchor_v3(
            gt_outputs[token],
            predictions[token],
            iou_thresholds=[float(t) for t in config["matching"]["iou_thresholds"]],
            primary_iou_threshold=float(config["matching"]["primary_iou_threshold"]),
            max_items=int(config["max_items"]),
        )
        for token in predicted_tokens
    ]
    report: dict[str, Any] = aggregate_v3(
        scores,
        n_anchors_expected=len(expected_tokens),
        taxonomies=config.get("taxonomies", {}),
        primary_iou_threshold=float(config["matching"]["primary_iou_threshold"]),
        iou_thresholds=[float(t) for t in config["matching"]["iou_thresholds"]],
        max_items=int(config["max_items"]),
        risk_max_items=int(config["risk_max_items"]),
    )

    # bootstrap 逐类 F1 的固定类别集（v2 同规则：全量 GT 非空类）。
    per_category = per_category_counts(scores)
    object_categories_full = sorted(
        cat for cat, counts in per_category.items() if counts[0] + counts[2] > 0
    )

    boot = config["bootstrap"]
    taxonomies = config.get("taxonomies", {})
    clusters = {
        token: str(manifest["anchors"][token]["scene_token"])
        for token in predicted_tokens
    }
    scores_by_token = {s["sample_token"]: s for s in scores}
    report["bootstrap_ci"] = bootstrap_statistics_nested(
        clusters,
        lambda tokens: subset_statistics_v3(
            [scores_by_token[t] for t in tokens],
            motion_states=list(taxonomies.get("motion_states", _MOTION_STATES)),
            speed_actions=list(taxonomies.get("speed_actions", _SPEED_ACTIONS)),
            risk_factors=list(taxonomies.get("risk_factors", _RISK_FACTORS_V7)),
            primary_iou_threshold=float(config["matching"]["primary_iou_threshold"]),
            iou_thresholds=[float(t) for t in config["matching"]["iou_thresholds"]],
            object_categories_full=tuple(object_categories_full),
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
    (out / "eval_report.md").write_text(_render_markdown_v3(report), encoding="utf-8")
    return report


# ---------------------------------------------------------------------------
# report rendering（确定性：固定键序、无 locale / 时间态）
# ---------------------------------------------------------------------------


def _render_markdown_v3(report: Mapping[str, Any]) -> str:
    """eval_v3 报告的人类可读渲染；相同报告 dict 输出逐字节相同。

    结构与 v2 渲染同构（coverage / object / 直方图 / 空集 / field /
    per-class / bootstrap / denominators / cost / counterfactual），另加
    BEV 与 risk attribution 两段；v2 各段键遍历逻辑不捕获 bev_/归因键
    （前缀过滤 + per_class 块名过滤天然排除），因此逐段复刻而非调用
    v2 渲染拼接（标题与 eval_version 行需为 v3）。
    """
    lines: list[str] = ["# Eval v3 report", ""]
    meta = report["meta"]
    lines += [
        f"- eval_version: {meta['eval_version']} (config status: {meta['config_status']})",
        f"- contract_version: {meta['contract_version']}",
        f"- evaluation_split: {meta['evaluation_split']}",
        f"- config_sha256: `{meta['config_sha256']}`",
        "",
    ]

    lines += ["## Coverage & parse", ""]
    for key in ("anchors_expected", "parse_ok", "parse_failed"):
        lines.append(f"- {key}: {report['coverage'][key]}")
    lines += ["", f"- parse_rate: {_fmt(report['parse_rate'])}", ""]
    if report["parse_error_counts"]:
        lines += ["Parse errors:", ""]
        lines += [
            f"- {k}: {v}" for k, v in sorted(report["parse_error_counts"].items())
        ]
        lines.append("")

    lines += ["## Object metrics", ""]
    lines += ["| metric | value |", "|---|---|"]
    for key in sorted(report):
        value = report[key]
        if isinstance(value, (int, float)) and (
            key.startswith("object_") or key.startswith("matched_")
        ):
            lines.append(f"| {key} | {_fmt(value)} |")
    lines.append("")

    # v3 新增：BEV 误差族（顶层 + 分桶 + signed，全部 bev_ 前缀标量）
    lines += ["## BEV localization errors (matched pairs)", ""]
    lines += ["| metric | value |", "|---|---|"]
    for key in sorted(k for k, v in report.items() if k.startswith("bev_")):
        lines.append(f"| {key} | {_fmt(report[key])} |")
    lines.append("")

    lines += ["## Cardinality histograms (parse-ok anchors)", ""]
    for key in (
        "n_gt_objects", "n_pred_objects", "n_gt_risk_factors", "n_pred_risk_factors",
    ):
        hist = report[key]
        cells = ", ".join(f"{k}:{hist[k]}" for k in sorted(hist, key=int))
        lines.append(f"- {key}: {cells}")
    lines.append("")

    lines += ["## Empty-direction counts (parse-ok anchors)", ""]
    for prefix in ("object", "risk"):
        counts = report[f"{prefix}_empty_direction_counts"]
        cells = ", ".join(f"{k}={counts[k]}" for k in sorted(counts))
        lines.append(f"- {prefix}: {cells}")
    lines.append("")

    lines += ["## Field metrics", ""]
    lines += ["| metric | value |", "|---|---|"]
    for key in (
        "motion_state_f1_macro", "speed_action_f1_macro", "yield_required_f1",
        "yield_miss_rate", "yield_overconservative_rate", "risk_factors_f1_macro",
        "stop_overconservative_rate", "stop_miss_rate", "max_items_hit_rate",
    ):
        lines.append(f"| {key} | {_fmt(report[key])} |")
    lines.append("")

    # v3 新增：risk 归因族（micro/macro 标量 + per_factor 嵌套块）
    lines += ["## Risk attribution (conditional, matched-bridge)", ""]
    lines += ["| metric | value |", "|---|---|"]
    for key in ("risk_attribution_f1_micro", "risk_attribution_f1_macro"):
        lines.append(f"| {key} | {_fmt(report[key])} |")
    lines.append("")
    lines += ["| factor | tp | fp | fn | precision | recall | f1 |", "|---|---|---|---|---|---|---|"]
    for factor in sorted(report["risk_attribution_per_factor"]):
        block = report["risk_attribution_per_factor"][factor]
        lines.append(
            f"| {factor} | {block['tp']} | {block['fp']} | {block['fn']} "
            f"| {_fmt(block['precision'])} | {_fmt(block['recall'])} "
            f"| {_fmt(block['f1'])} |"
        )
    lines.append("")

    lines += ["## Per-class F1", ""]
    lines += ["| block | class | f1 |", "|---|---|---|"]
    per_class_blocks = sorted(
        key
        for key, value in report.items()
        if isinstance(value, Mapping)
        and value
        and all(isinstance(v, (int, float)) for v in value.values())
        and ("_per_class" in key or "_per_category" in key)
    )
    for block_key in per_class_blocks:
        for class_name in sorted(report[block_key]):
            lines.append(
                f"| {block_key} | {class_name} | {_fmt(report[block_key][class_name])} |"
            )
    lines.append("")

    boot = report["bootstrap_ci"]
    lines += ["## 95% CI (scene-cluster bootstrap)", ""]
    lines += ["| metric | point | ci_low | ci_high |", "|---|---|---|---|"]
    for key in sorted(
        k for k, v in boot.items()
        if isinstance(v, Mapping) and "point_estimate" in v
    ):
        cell = boot[key]
        lines.append(
            f"| {key} | {_fmt(cell['point_estimate'])} "
            f"| {_fmt(cell['ci_low'])} | {_fmt(cell['ci_high'])} |"
        )
    lines.append("")
    nested_blocks = sorted(
        k for k, v in boot.items()
        if isinstance(v, Mapping) and "point_estimate" not in v
    )
    if nested_blocks:
        lines += ["Per-class CI blocks:", ""]
        lines += [
            "| block | class | point | ci_low | ci_high |",
            "|---|---|---|---|---|",
        ]
        for block_key in nested_blocks:
            for class_name in sorted(boot[block_key]):
                cell = boot[block_key][class_name]
                lines.append(
                    f"| {block_key} | {class_name} | {_fmt(cell['point_estimate'])} "
                    f"| {_fmt(cell['ci_low'])} | {_fmt(cell['ci_high'])} |"
                )
        lines.append("")

    den = report["denominators"]
    lines += ["## Denominators", ""]
    for key in ("anchors_expected", "parse_ok", "matched_pairs", "motion_pairs"):
        lines.append(f"- {key}: {den[key]}")
    lines += [
        f"- bev_pairs: {den['bev_pairs']}",
        f"- yield_counts: {json.dumps(den['yield_counts'], sort_keys=True)}",
        f"- stop_counts: {json.dumps(den['stop_counts'], sort_keys=True)}",
        f"- micro_counts: {json.dumps(den['micro_counts'], sort_keys=True)}",
    ]
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
