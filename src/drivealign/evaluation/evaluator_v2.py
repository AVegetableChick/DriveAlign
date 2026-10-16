"""Eval v2 evaluator: predictions + frozen GT -> eval_report (S13 口径修订).

Purpose:
    Eval v2 的编排层：读取 Base benchmark predictions JSONL，经 anchor
    manifest 在数据集 shard 中定位每条冻结 GT 记录并校验记录哈希，逐
    anchor 打分（:func:`score_anchor`，与 v1 打分层逐字节同源），用
    :func:`aggregate_v2` 组装 S13 台账定稿的完整报告，附加 scene-cluster
    bootstrap CI（:func:`bootstrap_statistics_nested`，标量键 + 逐类嵌套
    键）、cost summary 与可选 counterfactual 对比，最后写出
    ``eval_report.json`` / ``eval_report.md`` / ``anchor_scores.jsonl``。
    对输入严格只读；确定性：相同 predictions + 相同 config 下报告 JSON
    逐字节相同（sort_keys、无时间戳、无 git 状态）。

    相对 v1 evaluator 的编排差异（全部有 S13 台账登记）：

    1. T6 完整性硬前置：missing / extra predictions 非空即在**任何文件
       写出之前** raise ``ValueError``（v1 是降级计数写进 coverage）；
    2. bootstrap 换用嵌套版（T4）：22 个标量键 + 4 个固定类别集逐类键，
       CI 块内为 ``{类别: 八字段}``；
    3. ``risk_max_items`` 从 config 读入并下传 :func:`aggregate_v2`
       （risk 基数直方图封顶出处 = contract v6 risk_factors maxItems）；
    4. 报告渲染 :func:`_render_markdown_v2` 覆盖 v2 新块（三级漏斗 /
       基数直方图 / 空集方向四格 / 逐类 CI 表）。

    输入加载（predictions/GT）/ cost summary / counterfactual 对比三块
    直接复用 v1 evaluator 的同一份实现（一份定义、两个消费者，v2 不复
    刻、不修改 v1 任何函数）。

Example launch command:
    ``conda activate autovla_codeclean && cd /root/autodl-tmp/drivealign_workspace && \
    PYTHONPATH=DriveAlign/src python -c \
    "from drivealign.evaluation.evaluator_v2 import run_evaluation_v2; \
    run_evaluation_v2('runs/S12_sft_1f/eval_v6_batched/predictions_bs16.jsonl', \
    'data/face_manifests/v6/anchor_policy_manifest.json', 'data/dataset_v4', \
    'DriveAlign/configs/evaluation/eval_v2.yaml', 'runs/S13_eval_v2_check')"``
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import mean
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
from drivealign.evaluation.metrics import (
    _tau_key,
    f1_from_counts,
    micro_object_counts,
    motion_state_counts,
    parse_ok_scores,
    per_category_counts,
    risk_factor_counts,
    score_anchor,
    speed_action_counts,
    speed_action_per_class_f1,
    yield_required_counts,
)
from drivealign.evaluation.metrics_v2 import (
    aggregate_v2,
    empty_direction_block,
    motion_state_per_class_f1,
    stop_counts,
    stop_direction_rates,
    yield_direction_rates,
)

# ---------------------------------------------------------------------------
# 冻结枚举（与 eval_v1.yaml / eval_v2.yaml taxonomies 逐值相同，pin 测试钉死）
# ---------------------------------------------------------------------------

_MOTION_STATES = ("stationary", "same_direction", "oncoming", "crossing")
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


# ---------------------------------------------------------------------------
# bootstrap 统计量（与 aggregate_v2 共享同一份 counting helper 定义）
# ---------------------------------------------------------------------------


def subset_statistics_v2(
    subset_scores: Sequence[Mapping[str, Any]],
    *,
    motion_states: Sequence[str] = _MOTION_STATES,
    speed_actions: Sequence[str] = _SPEED_ACTIONS,
    risk_factors: Sequence[str] = _RISK_FACTORS,
    primary_iou_threshold: float = 0.5,
    iou_thresholds: Sequence[float] = (0.3, 0.5, 0.7),
    object_categories_full: Sequence[str] = (),
) -> dict[str, Any]:
    """eval_v2 全部被 bootstrap 统计量在一个 anchor 子集上的重算。

    与 v1 :func:`subset_statistics` 同构：点估计与每个重采样子集共用本
    函数，保证 CI 与点估计同一定义。两层返回结构（T4）：22 个标量键 +
    4 个逐类嵌套键；嵌套键的类别集**固定**（object =
    ``object_categories_full``，其余 = 冻结枚举），子集缺席的类别按全零
    计数 (0, 0, 0) → F1 = 0.0，保证 bootstrap 嵌套版的全量/重采样类别集
    一致契约天然成立（缺席即 0 值而非缺键）。

    键集与 v1 的差异：``parse_rate`` 分母 = 子集大小（与 v1 相同）；
    ``motion_state_accuracy`` 退役 → ``motion_state_f1_macro``（T1）；
    ``overconservative_*`` 改名 ``yield/stop_overconservative_rate`` 并
    新增 ``yield_miss_rate`` / ``stop_miss_rate`` 方向对（T2/T3）；新增
    对象 / risk 空集方向两率（表 2.4/3.1）；object micro 增加
    precision/recall 全网格（F1 仅在 primary 阈值，避免键爆炸）。
    """
    ok = parse_ok_scores(subset_scores)
    n = len(subset_scores)
    stats: dict[str, Any] = {"parse_rate": len(ok) / n if n else 0.0}

    # ---- 对象 micro：f1@primary + precision/recall 全网格（台账 2.1）----
    # 每个 tau 只算一次计数（primary 可能与网格重合，去重防重复计算）。
    counts_by_tau: dict[float, tuple[int, int, int]] = {
        float(primary_iou_threshold): micro_object_counts(
            subset_scores, float(primary_iou_threshold)
        )
    }
    for tau in iou_thresholds:
        tau_f = float(tau)
        if tau_f not in counts_by_tau:
            counts_by_tau[tau_f] = micro_object_counts(subset_scores, tau_f)
    for tau_f in sorted(counts_by_tau):
        tag = _tau_key(tau_f)
        tp, fp, fn = counts_by_tau[tau_f]
        stats[f"object_precision_micro@{tag}"] = tp / (tp + fp) if tp + fp else 0.0
        stats[f"object_recall_micro@{tag}"] = tp / (tp + fn) if tp + fn else 0.0
    tp, fp, fn = counts_by_tau[float(primary_iou_threshold)]
    stats[f"object_f1_micro@{_tau_key(float(primary_iou_threshold))}"] = f1_from_counts(
        tp, fp, fn
    )

    # ---- 对象 macro@primary：逐子集 GT 非空类过滤（与 v1 同规则）----
    # 嵌套键：类别集固定 = object_categories_full，缺席类 F1 计 0.0。
    full_counts = per_category_counts(subset_scores)
    per_category = {
        cat: f1_from_counts(*counts)
        for cat, counts in full_counts.items()
        if counts[0] + counts[2] > 0
    }
    stats[f"object_f1_macro@{_tau_key(float(primary_iou_threshold))}"] = (
        mean(per_category.values()) if per_category else 0.0
    )
    stats[f"object_f1_per_category@{_tau_key(float(primary_iou_threshold))}"] = {
        cat: f1_from_counts(*full_counts.get(cat, (0, 0, 0)))
        for cat in object_categories_full
    }

    # ---- motion_state：T1 换统计量（accuracy 退役 → 4 态逐类 F1 + macro）----
    motion_confusion, _correct, _total = motion_state_counts(subset_scores)
    motion_per_class = motion_state_per_class_f1(motion_confusion, motion_states)
    stats["motion_state_f1_per_class"] = motion_per_class
    stats["motion_state_f1_macro"] = (
        mean(motion_per_class.values()) if motion_states else 0.0
    )

    # ---- speed_action：4 动作逐类 F1 + macro（保留，台账 5）----
    speed_confusion = speed_action_counts(subset_scores, speed_actions)
    speed_per_class = speed_action_per_class_f1(speed_confusion, speed_actions)
    stats["speed_action_per_class_f1"] = speed_per_class
    stats["speed_action_f1_macro"] = (
        mean(speed_per_class.values()) if speed_actions else 0.0
    )

    # ---- yield：T2 方向对（改名沿用 + miss 新增，台账 4）----
    yield_counts = yield_required_counts(subset_scores)
    stats["yield_required_f1"] = f1_from_counts(
        yield_counts["tp"], yield_counts["fp"], yield_counts["fn"]
    )
    stats["yield_miss_rate"], stats["yield_overconservative_rate"] = (
        yield_direction_rates(yield_counts)
    )

    # ---- risk_factors：7 类逐类 F1 + macro（保留，台账 3.2/3.3）----
    risk_per_class = {
        risk: f1_from_counts(*risk_factor_counts(subset_scores, risk_factors)[risk])
        for risk in risk_factors
    }
    stats["risk_factors_f1_per_class"] = risk_per_class
    stats["risk_factors_f1_macro"] = (
        mean(risk_per_class.values()) if risk_factors else 0.0
    )

    # ---- stop：T3 方向对（改名沿用 + miss 新增，台账 5）----
    stats["stop_overconservative_rate"], stats["stop_miss_rate"] = (
        stop_direction_rates(stop_counts(subset_scores))
    )

    # ---- 空集方向两率（新增，台账 2.4 / 3.1；四格计数不进 bootstrap）----
    _, obj_hall, obj_miss = empty_direction_block(
        [(int(s["n_gt_objects"]), int(s["n_pred_objects"])) for s in ok]
    )
    stats["object_empty_hallucination_rate"] = obj_hall
    stats["object_empty_miss_rate"] = obj_miss
    _, risk_hall, risk_miss = empty_direction_block(
        [(len(s["gt_risk_factors"]), len(s["pred_risk_factors"])) for s in ok]
    )
    stats["risk_empty_hallucination_rate"] = risk_hall
    stats["risk_empty_miss_rate"] = risk_miss

    # ---- 工作点（保留，台账 2.4）----
    stats["max_items_hit_rate"] = (
        sum(1 for s in ok if s.get("max_items_hit")) / len(ok) if ok else 0.0
    )
    return stats


# ---------------------------------------------------------------------------
# orchestration（T6 硬前置 → 打分 → aggregate_v2 → bootstrap → 产物写出）
# ---------------------------------------------------------------------------


def run_evaluation_v2(
    predictions_path: str | Path,
    anchor_manifest_path: str | Path,
    dataset_root: str | Path,
    config_path: str | Path,
    out_dir: str | Path,
    counterfactual_paths: Mapping[str, str | Path] | None = None,
) -> dict[str, Any]:
    """运行完整 Eval v2 评测并写出报告产物（签名与 v1 run_evaluation 一致）。"""
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

    # T6 完整性硬前置（台账 1）：missing / extra 非空即 fail-fast，且必须
    # 发生在任何文件写出之前——v2 报告不允许在残缺批次上产出任何产物。
    if missing or extra:
        raise ValueError(
            "eval_v2 completeness precondition violated: "
            f"{len(missing)} missing predictions (first 5: {missing[:5]}), "
            f"{len(extra)} extra predictions (first 5: {extra[:5]})"
        )

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
    report: dict[str, Any] = aggregate_v2(
        scores,
        n_anchors_expected=len(expected_tokens),
        taxonomies=config.get("taxonomies", {}),
        primary_iou_threshold=float(config["matching"]["primary_iou_threshold"]),
        iou_thresholds=[float(t) for t in config["matching"]["iou_thresholds"]],
        max_items=int(config["max_items"]),
        risk_max_items=int(config["risk_max_items"]),
    )

    # bootstrap 逐类 F1 的固定类别集：全量 scores 的 GT 非空类（与
    # aggregate_v2 的 macro 过滤同规则；重采样子集缺席类 F1 计 0.0）。
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
        lambda tokens: subset_statistics_v2(
            [scores_by_token[t] for t in tokens],
            motion_states=list(taxonomies.get("motion_states", _MOTION_STATES)),
            speed_actions=list(taxonomies.get("speed_actions", _SPEED_ACTIONS)),
            risk_factors=list(taxonomies.get("risk_factors", _RISK_FACTORS)),
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
    (out / "eval_report.md").write_text(_render_markdown_v2(report), encoding="utf-8")
    return report


# ---------------------------------------------------------------------------
# report rendering（确定性：固定键序、无 locale / 时间态）
# ---------------------------------------------------------------------------


def _render_markdown_v2(report: Mapping[str, Any]) -> str:
    """eval_v2 报告的人类可读渲染；相同报告 dict 输出逐字节相同。"""
    lines: list[str] = ["# Eval v2 report", ""]
    meta = report["meta"]
    lines += [
        f"- eval_version: {meta['eval_version']} (config status: {meta['config_status']})",
        f"- contract_version: {meta['contract_version']}",
        f"- evaluation_split: {meta['evaluation_split']}",
        f"- config_sha256: `{meta['config_sha256']}`",
        "",
    ]

    # T6 三级漏斗（anchors_scored 已退役）
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

    # 对象侧标量指标（micro 全网格 + macro + 空集两率 + 匹配质量）
    lines += ["## Object metrics", ""]
    lines += ["| metric | value |", "|---|---|"]
    for key in sorted(report):
        value = report[key]
        if isinstance(value, (int, float)) and (
            key.startswith("object_") or key.startswith("matched_")
        ):
            lines.append(f"| {key} | {_fmt(value)} |")
    lines.append("")

    # 基数直方图（k = 0..max_items / risk_max_items 全桶）
    lines += ["## Cardinality histograms (parse-ok anchors)", ""]
    for key in (
        "n_gt_objects", "n_pred_objects", "n_gt_risk_factors", "n_pred_risk_factors",
    ):
        hist = report[key]
        cells = ", ".join(f"{k}:{hist[k]}" for k in sorted(hist, key=int))
        lines.append(f"- {key}: {cells}")
    lines.append("")

    # 空集方向四格计数（两率由报告内对象 / risk 标量行承载）
    lines += ["## Empty-direction counts (parse-ok anchors)", ""]
    for prefix in ("object", "risk"):
        counts = report[f"{prefix}_empty_direction_counts"]
        cells = ", ".join(f"{k}={counts[k]}" for k in sorted(counts))
        lines.append(f"- {prefix}: {cells}")
    lines.append("")

    # 场级标量指标（T1/T2/T3 换名后的键集）
    lines += ["## Field metrics", ""]
    lines += ["| metric | value |", "|---|---|"]
    for key in (
        "motion_state_f1_macro", "speed_action_f1_macro", "yield_required_f1",
        "yield_miss_rate", "yield_overconservative_rate", "risk_factors_f1_macro",
        "stop_overconservative_rate", "stop_miss_rate", "max_items_hit_rate",
    ):
        lines.append(f"| {key} | {_fmt(report[key])} |")
    lines.append("")

    # 逐类 F1（4 个固定类别集块；排除混淆矩阵等嵌套 dict）
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

    # bootstrap CI：标量键一张表 + 逐类嵌套键一张表
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

    # denominators（T3/T6 重组块，报告内可复算全部率值）
    den = report["denominators"]
    lines += ["## Denominators", ""]
    for key in ("anchors_expected", "parse_ok", "matched_pairs", "motion_pairs"):
        lines.append(f"- {key}: {den[key]}")
    lines += [
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
