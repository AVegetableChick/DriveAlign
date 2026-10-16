"""Eval v3 risk-factor attribution metrics (S13 eval v3 additions, §3).

Purpose:
    Implement the frozen conditional bidirectional attribution family of
    ``docs/stages_record/S13_contract_and_eval_revision/S13_eval_v3_metric_additions_draft.md``
    (拍板 2026-10-14). The atomic unit is a ``(factor, object)`` pair: one
    object reference inside one factor entry. Prediction and GT object ids
    live in two independent local numbering spaces — comparing ids directly
    is FORBIDDEN (draft §3.2); bridging goes through the primary-threshold
    class-wise greedy IoU matching (the SAME MatchOutcome the object_f1
    family consumed, computed once and reused): a referenced prediction
    object that matched GT object ``g`` lands on ``g``; an unmatched
    (hallucinated / badly-localized) reference has no GT landing.

    Conditional bidirectional gate (2026-10-14 拍板): only factors in the
    INTERSECTION of both factor-name sets produce atoms. A prediction-side
    factor absent from GT (over-reported) contributes nothing here (its FP
    is already counted by ``risk_factors_f1_macro``); a GT factor the
    prediction never reported contributes nothing either (its FN is
    already counted at factor level). This keeps attribution readings
    decoupled: "given the factor was correctly detected, how precisely
    were its objects identified?" — unconditional counting would double-
    count every factor-level error once per object reference.

    Boundary rules (draft §3.3 table): predicted factor with empty
    objects → all of ``O_gt(f)`` FN; hallucinated / IoU<0.5 reference →
    FP; GT object no prediction reference covers → FN (missed detection
    and detected-but-not-referenced are indistinguishable through the
    matching bridge — the information boundary); GT objects empty → all
    references FP (defensive); both sides empty → factor skipped.

    Aggregation: micro pools all atoms; macro averages per-factor F1 over
    factors WITH atoms only (empty-atom factors are skipped, never null);
    ``risk_attribution_per_factor`` exposes per-factor
    ``{tp, fp, fn, precision, recall, f1}``. The nested block is a
    diagnostic reading and never enters bootstrap (draft §3.5).

Example launch command:
    ``conda activate autovla_codeclean && PYTHONPATH=DriveAlign/src python -c \
    "from drivealign.evaluation.metrics_attribution import aggregate_attribution; \
    print(aggregate_attribution({'vru_crossing': (2, 1, 1)}))"``
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from statistics import mean
from typing import Any

from drivealign.evaluation.matching import MatchOutcome
from drivealign.evaluation.metrics import f1_from_counts, parse_ok_scores

#: per_factor 嵌套块内每因子的六字段（冻结键序，报告逐键 pin）。
_PER_FACTOR_FIELDS = ("tp", "fp", "fn", "precision", "recall", "f1")


def _factor_object_map(entries: Any) -> dict[str, set[int]]:
    """risk_factors 条目列表 → ``{factor: {object id 集}}``。

    v7 面（GT 与模型同构）条目 = ``{factor, objects: [id]}``；v6- 裸名
    字符串防御性按 objects 空处理（报了因子没指对象 = 全漏归因口径，
    草案 §3.3 边界表）。objects 内重复 id（契约层已被 parser/GT 生成
    拒绝）按集合去重，不放大原子数。
    """
    factor_map: dict[str, set[int]] = {}
    for entry in entries or []:
        if isinstance(entry, Mapping):
            refs = entry.get("objects") or ()
            factor_map.setdefault(str(entry["factor"]), set()).update(
                int(object_id) for object_id in refs
            )
        else:
            factor_map.setdefault(str(entry), set())
    return factor_map


def _count_one_anchor(
    gt_factor_map: Mapping[str, set[int]],
    pred_factor_map: Mapping[str, set[int]],
    bridge: Mapping[int, int | None],
) -> dict[str, tuple[int, int, int]]:
    """conditional 双向原子计数核心（桥接表已就绪）。

    对交集中每个因子 ``f``：预测引用逐个经桥接落点，落点 ∈ ``O_gt(f)``
    计 TP 并覆盖该 GT 对象，否则计 FP；``O_gt(f)`` 中未被覆盖者计 FN。
    """
    counts: dict[str, tuple[int, int, int]] = {}
    for factor in sorted(set(gt_factor_map) & set(pred_factor_map)):
        gt_refs = gt_factor_map[factor]
        pred_refs = pred_factor_map[factor]
        if not gt_refs and not pred_refs:
            continue  # 两侧 objects 均空：跳过该因子（边界表末行）
        tp = fp = 0
        covered: set[int] = set()
        for pred_id in pred_refs:
            gt_id = bridge.get(pred_id)
            if gt_id is not None and gt_id in gt_refs:
                tp += 1
                covered.add(gt_id)
            else:
                fp += 1
        fn = len(gt_refs - covered)
        counts[factor] = (tp, fp, fn)
    return counts


def anchor_attribution_counts(
    gt_output: Mapping[str, Any],
    pred_output: Mapping[str, Any],
    primary: MatchOutcome,
) -> dict[str, tuple[int, int, int]]:
    """一帧归因计数入口：GT/pred 输出 dict + primary 匹配 → 因子计数。

    桥接表由 primary 匹配结果构造：``pred 对象 id → 落点 GT 对象 id``
    （未匹配预测对象不进表 → 其引用按无落点 FP 处理）。id 取对象条目
    的 ``id`` 字段（v7 双侧独立编号空间，严禁列表下标直比——GT id 恰
    等于下标是构造巧合，pred id 不是）。id 字段缺失（v1–v6 面混入）
    的对象不参与桥接，其引用按无落点处理。
    """
    gt_factor_map = _factor_object_map(gt_output.get("risk_factors"))
    pred_factor_map = _factor_object_map(pred_output.get("risk_factors"))
    bridge: dict[int, int | None] = {}
    for gt_index, pred_index, _iou in primary.matched:
        pred_id = pred_output["critical_objects"][pred_index].get("id")
        gt_id = gt_output["critical_objects"][gt_index].get("id")
        if pred_id is None:
            continue
        bridge[int(pred_id)] = None if gt_id is None else int(gt_id)
    return _count_one_anchor(gt_factor_map, pred_factor_map, bridge)


def attribution_totals(
    scores: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[int, int, int]]:
    """逐 anchor 因子计数 → 全量合计（aggregate 与 bootstrap 共用同一份定义）。

    与 :func:`micro_object_counts` 同款「一份定义、两个消费者」模式：
    score dict 携带打分层算好的 ``risk_attribution_counts``，此处仅
    池化求和；parse 失败 anchors 不进任何格子。
    """
    totals: dict[str, list[int]] = {}
    for score in parse_ok_scores(scores):
        for factor, (tp, fp, fn) in score.get("risk_attribution_counts", {}).items():
            cell = totals.setdefault(str(factor), [0, 0, 0])
            cell[0] += int(tp)
            cell[1] += int(fp)
            cell[2] += int(fn)
    return {factor: tuple(values) for factor, values in totals.items()}


def aggregate_attribution(
    totals: Mapping[str, tuple[int, int, int]],
) -> dict[str, Any]:
    """合计因子计数 → 冻结键集（2 标量 + 1 嵌套块，草案 §3.4）。

    - ``risk_attribution_f1_micro``：全原子池直接计 F1（主读数）；
    - ``risk_attribution_f1_macro``：有原子因子的 F1 均值（无原子因子
      跳过；全空计 0.0）；
    - ``risk_attribution_per_factor``：有原子因子各自的
      ``{tp, fp, fn, precision, recall, f1}``（空因子跳过不置 null）。
    """
    tp = sum(int(values[0]) for values in totals.values())
    fp = sum(int(values[1]) for values in totals.values())
    fn = sum(int(values[2]) for values in totals.values())
    per_factor: dict[str, dict[str, Any]] = {}
    for factor in sorted(totals):
        factor_tp, factor_fp, factor_fn = (int(v) for v in totals[factor])
        if factor_tp + factor_fp + factor_fn == 0:
            continue  # 无原子因子：per_factor / macro 一并跳过
        per_factor[factor] = {
            "tp": factor_tp,
            "fp": factor_fp,
            "fn": factor_fn,
            "precision": (
                factor_tp / (factor_tp + factor_fp) if factor_tp + factor_fp else 0.0
            ),
            "recall": (
                factor_tp / (factor_tp + factor_fn) if factor_tp + factor_fn else 0.0
            ),
            "f1": f1_from_counts(factor_tp, factor_fp, factor_fn),
        }
    return {
        "risk_attribution_f1_micro": f1_from_counts(tp, fp, fn),
        "risk_attribution_f1_macro": (
            mean(per_factor[f]["f1"] for f in sorted(per_factor))
            if per_factor
            else 0.0
        ),
        "risk_attribution_per_factor": per_factor,
    }
