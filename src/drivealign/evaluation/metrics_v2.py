"""Eval v2 metrics: 复用 v1 打分层 + v2 聚合层（S13 口径修订）。

Purpose:
    实现 S13 台账
    (``docs/stages_record/S13_contract_and_eval_revision/S13_eval_v1_to_v2_metric_changes.md``)
    定稿的 eval_v2 全部聚合指标。打分层 ``score_anchor`` 与冻结时的
    eval_v1 逐字节相同（v2 所需字段——``n_pred_objects`` /
    ``n_gt_objects`` / ``gt/pred_risk_factors`` / ``motion_pairs`` /
    ``per_tau`` 等——v1 打分层已全部产出），本模块只做三件事：

    1. 新增 counting helper：motion_state 逐类 F1（T1）、stop 方向对
       计数（T3）、空集方向四格（对象 / risk 共用一份实现）、基数
       直方图（对象 / risk 共用一份实现）；
    2. 方向对取值函数（T2/T3）：0.0 守卫下塌缩不变性自动成立
       （纯规范条款、零新增计算），由单测钉死三种塌缩情形；
    3. ``aggregate_v2``：T6 三级漏斗、T1/T2/T3 换统计量与改名、
       denominators 重组（``stop_counts`` 进、``overconservative``
       块撤、``anchors_scored`` 退役）、直方图与方向率块落位。

    与 eval_v1 的关系：本模块不复刻、不修改 v1 任何函数；v1 报告是
    历史基线，其数值等值性由回归证据（同批 predictions 对照）承载。
    全部新键名与台账条目一一对应；risk 基数直方图键名
    ``n_gt_risk_factors`` / ``n_pred_risk_factors`` 为台账 3.1 未显式
    定名处的实现定名（沿用对象侧 ``n_gt_objects`` / ``n_pred_objects``
    的命名模式）。

Example launch command:
    ``conda activate autovla_codeclean && PYTHONPATH=DriveAlign/src python -c \
    "from drivealign.evaluation.metrics_v2 import aggregate_v2; \
    from drivealign.evaluation.metrics import score_anchor; \
    gt = {'critical_objects': [], 'risk_factors': [], \
    'yield_required': False, 'speed_action': 'KEEP_SPEED'}; \
    row = {'parse_ok': True, 'output': gt}; \
    s = score_anchor(gt, row, iou_thresholds=[0.5], \
    primary_iou_threshold=0.5, max_items=8); \
    print(sorted(aggregate_v2([s], n_anchors_expected=1, taxonomies={}, \
    primary_iou_threshold=0.5, iou_thresholds=[0.5], \
    max_items=8, risk_max_items=8))[:6])"``
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from statistics import mean, median
from typing import Any

from drivealign.evaluation.metrics import (
    _percentile,
    _tau_key,
    f1_from_counts,
    micro_object_counts,
    motion_state_counts,
    parse_error_counts,
    parse_ok_scores,
    per_category_counts,
    risk_factor_counts,
    speed_action_counts,
    speed_action_per_class_f1,
    yield_required_counts,
)

#: T3「不该停却停」分子/分母侧：GT ∈ {KEEP_SPEED, ACCELERATE} 预测 STOP。
_STOP_GT_KEEP = ("KEEP_SPEED", "ACCELERATE")
#: T3「该停不停」分母侧：GT ∈ {DECELERATE, STOP}（预测 ∉ STOP 计 fn）。
_STOP_GT_STOP = ("DECELERATE", "STOP")

#: 空集方向四格的固定键序（对象 / risk 两块共用，报告内可复算两率）。
_EMPTY_DIRECTION_KEYS = (
    "gt_empty_pred_empty",
    "gt_empty_pred_nonempty",
    "gt_nonempty_pred_empty",
    "gt_nonempty_pred_nonempty",
)


# ---------------------------------------------------------------------------
# counting helpers（与 bootstrap 重采样共享：一份定义、两个消费者）
# ---------------------------------------------------------------------------


def motion_state_per_class_f1(
    confusion: Mapping[str, Mapping[str, int]],
    motion_states: Sequence[str],
) -> dict[str, float]:
    """t0 配对对象混淆矩阵 → 4 态固定形状 one-vs-rest F1（T1）。

    与 speed_action 逐类 F1 是同一份 one-vs-rest 实现（T4 同构原则：
    一份 helper、两个消费者），仅在 motion_state 的 4 态枚举上调用。
    混淆矩阵行取 ``confusion.get(state, {})``，未出现的 GT 态按全零行
    处理，保证 4 态固定形状。
    """
    return speed_action_per_class_f1(confusion, motion_states)


def stop_counts(scores: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """stop 方向对分子/分母（T3，表 5）。

    返回四键：``stop_fp``（GT keep/accel 且预测 STOP）、
    ``stop_gt_keep``（GT keep/accel 总数）、``stop_fn``（GT
    decel/stop 且预测非 STOP）、``stop_gt_stop``（GT decel/stop
    总数）。仅统计 parse-ok anchors。
    """
    stop_fp = stop_gt_keep = stop_fn = stop_gt_stop = 0
    for s in parse_ok_scores(scores):
        gt = str(s["gt_speed_action"])
        pred = str(s["pred_speed_action"])
        stop_gt_keep += int(gt in _STOP_GT_KEEP)
        stop_fp += int(gt in _STOP_GT_KEEP and pred == "STOP")
        stop_gt_stop += int(gt in _STOP_GT_STOP)
        stop_fn += int(gt in _STOP_GT_STOP and pred != "STOP")
    return {
        "stop_fp": stop_fp,
        "stop_gt_keep": stop_gt_keep,
        "stop_fn": stop_fn,
        "stop_gt_stop": stop_gt_stop,
    }


def empty_direction_block(
    pairs: Sequence[tuple[int, int]],
) -> tuple[dict[str, int], float, float]:
    """空集方向四格计数 + 两个条件率（表 2.4 / 3.1，对象与 risk 共用）。

    ``pairs`` 为逐 anchor 的 ``(GT 基数, pred 基数)``。返回
    ``(四格计数 dict, hallucination_rate, miss_rate)``：

    - hallucination = GT∅ 且 pred 非空 ÷ GT∅ 数（「该空不空」）；
    - miss = GT 非空且 pred 空 ÷ GT 非空数（「该有不有」）；
    - 分母为 0 时取 0.0（与既有实现守卫一致）。
    """
    counts = {key: 0 for key in _EMPTY_DIRECTION_KEYS}
    for gt_size, pred_size in pairs:
        if gt_size == 0 and pred_size == 0:
            counts["gt_empty_pred_empty"] += 1
        elif gt_size == 0:
            counts["gt_empty_pred_nonempty"] += 1
        elif pred_size == 0:
            counts["gt_nonempty_pred_empty"] += 1
        else:
            counts["gt_nonempty_pred_nonempty"] += 1
    gt_empty = counts["gt_empty_pred_empty"] + counts["gt_empty_pred_nonempty"]
    gt_nonempty = counts["gt_nonempty_pred_empty"] + counts["gt_nonempty_pred_nonempty"]
    hallucination = counts["gt_empty_pred_nonempty"] / gt_empty if gt_empty else 0.0
    miss = counts["gt_nonempty_pred_empty"] / gt_nonempty if gt_nonempty else 0.0
    return counts, hallucination, miss


def cardinality_histogram(values: Sequence[int], max_k: int) -> dict[str, int]:
    """逐 anchor 基数 → 整数直方图 ``{k: anchor_count}``（表 2.4 / 3.1）。

    k = 0..max_k 全桶覆盖（含 0 计数桶）；0 桶即空集方向四格的
    GT∅ / Pred∅ 边界。max_k 为封顶（对象 = ``max_items``、risk =
    contract v6 ``risk_factors`` maxItems），契约 maxItems 保证基数
    不超封顶，min 截断仅为防御性兜底（理论不可达）。
    """
    counts = {str(k): 0 for k in range(max_k + 1)}
    for value in values:
        counts[str(min(int(value), max_k))] += 1
    return counts


def yield_direction_rates(
    yield_counts: Mapping[str, int],
) -> tuple[float, float]:
    """T2 方向对取值：``(yield_miss_rate, yield_overconservative_rate)``。

    条件分母由 ``yield_counts``（tp/fp/fn/tn）派生：
    ``yield_gt_true = tp + fn``、``yield_gt_false = fp + tn``；分母为 0
    取 0.0。塌缩不变性由该守卫自动成立（纯定义条款）：恒定 false ⇒
    miss = 1.0 / overcons = 0.0；恒定 true ⇒ miss = 0.0 / overcons =
    1.0；恒定输出恰为 GT 常量时两率同 0.0（真实全对）。
    """
    gt_true = int(yield_counts["tp"]) + int(yield_counts["fn"])
    gt_false = int(yield_counts["fp"]) + int(yield_counts["tn"])
    miss = int(yield_counts["fn"]) / gt_true if gt_true else 0.0
    overcons = int(yield_counts["fp"]) / gt_false if gt_false else 0.0
    return miss, overcons


def stop_direction_rates(stop_counts_map: Mapping[str, int]) -> tuple[float, float]:
    """T3 方向对取值：``(stop_overconservative_rate, stop_miss_rate)``。

    分母为 0 取 0.0；塌缩不变性与 :func:`yield_direction_rates` 同理：
    恒定 STOP ⇒ overcons = 1.0 / miss = 0.0；恒定不输出 STOP ⇒
    overcons = 0.0 / miss = 1.0；恒定输出恰为 GT 常量时两率同 0.0。
    """
    overcons = (
        int(stop_counts_map["stop_fp"]) / int(stop_counts_map["stop_gt_keep"])
        if int(stop_counts_map["stop_gt_keep"])
        else 0.0
    )
    miss = (
        int(stop_counts_map["stop_fn"]) / int(stop_counts_map["stop_gt_stop"])
        if int(stop_counts_map["stop_gt_stop"])
        else 0.0
    )
    return overcons, miss


# ---------------------------------------------------------------------------
# aggregation（v2 报告键集，与台账各表一一对应）
# ---------------------------------------------------------------------------


def aggregate_v2(
    scores: Sequence[Mapping[str, Any]],
    *,
    n_anchors_expected: int,
    taxonomies: Mapping[str, Sequence[str]],
    primary_iou_threshold: float,
    iou_thresholds: Sequence[float],
    max_items: int,
    risk_max_items: int,
) -> dict[str, Any]:
    """组装 eval_v2 报告内容 dict（台账表 1–5）。

    键集相对 v1 的差异（全部有台账登记）：``coverage`` 三级漏斗、
    ``motion_state_accuracy`` 退役（T1）、``overconservative_*`` 改名
    ``yield/stop_overconservative_rate``（T2/T3）、新增
    ``yield_miss_rate`` / ``stop_miss_rate`` 方向对、新增直方图与空集
    方向块（表 2.4 / 3.1）、``denominators`` 重组（T3/T6）。
    """
    ok = parse_ok_scores(scores)
    n_ok = len(ok)
    report: dict[str, Any] = {
        # T6：三级漏斗；anchors_scored 退役，唯一锚点分母 = anchors_expected。
        "coverage": {
            "anchors_expected": n_anchors_expected,
            "parse_ok": n_ok,
            "parse_failed": len(scores) - n_ok,
        },
        # T6：parse_rate 分母改写为 anchors_expected（不变性下与 v1 数值等价）。
        "parse_rate": n_ok / n_anchors_expected if n_anchors_expected else 0.0,
        "parse_error_counts": parse_error_counts(scores),
    }

    # ---- 对象检测：micro 全网格（保留，台账 2.1）----
    for tau in iou_thresholds:
        tag = _tau_key(float(tau))
        tp, fp, fn = micro_object_counts(scores, float(tau))
        report[f"object_precision_micro@{tag}"] = tp / (tp + fp) if tp + fp else 0.0
        report[f"object_recall_micro@{tag}"] = tp / (tp + fn) if tp + fn else 0.0
        report[f"object_f1_micro@{tag}"] = f1_from_counts(tp, fp, fn)

    # T5：per-category 仅统计 GT 非空类（实现隐含 → 规范显式）。
    per_category = per_category_counts(scores)
    per_category_f1 = {
        cat: f1_from_counts(*counts)
        for cat, counts in sorted(per_category.items())
        if counts[0] + counts[2] > 0
    }
    tag = _tau_key(primary_iou_threshold)
    report[f"object_f1_per_category@{tag}"] = per_category_f1
    report[f"object_f1_macro@{tag}"] = (
        mean(per_category_f1.values()) if per_category_f1 else 0.0
    )
    report["macro_categories_counted"] = len(per_category_f1)

    # ---- 匹配质量（保留，台账 2.2）----
    matched_ious = sorted(iou for s in ok for iou in s.get("matched_ious", []))
    report["matched_pair_count"] = len(matched_ious)
    report["matched_iou_mean"] = mean(matched_ious) if matched_ious else 0.0
    report["matched_iou_p50"] = median(matched_ious) if matched_ious else 0.0
    report["matched_iou_p90"] = _percentile(matched_ious, 90)

    # ---- motion_state：T1 换统计量（accuracy 退役 → macro-F1）----
    motion_confusion, _correct, motion_pair_total = motion_state_counts(scores)
    report["motion_state_confusion"] = motion_confusion
    motion_states = list(taxonomies.get("motion_states", []))
    motion_per_class = motion_state_per_class_f1(motion_confusion, motion_states)
    report["motion_state_f1_per_class"] = motion_per_class
    report["motion_state_f1_macro"] = (
        mean(motion_per_class.values()) if motion_states else 0.0
    )

    # ---- speed_action（保留，台账 5）----
    speed_actions = list(taxonomies.get("speed_actions", []))
    speed_confusion = speed_action_counts(scores, speed_actions)
    report["speed_action_confusion"] = speed_confusion
    speed_per_class = speed_action_per_class_f1(speed_confusion, speed_actions)
    report["speed_action_per_class_f1"] = speed_per_class
    report["speed_action_f1_macro"] = (
        mean(speed_per_class.values()) if speed_per_class else 0.0
    )

    # ---- yield：T2 方向对（改名沿用 + miss 新增，台账 4）----
    yield_counts = yield_required_counts(scores)
    report["yield_required_f1"] = f1_from_counts(
        yield_counts["tp"], yield_counts["fp"], yield_counts["fn"]
    )
    yield_miss, yield_overcons = yield_direction_rates(yield_counts)
    report["yield_miss_rate"] = yield_miss
    report["yield_overconservative_rate"] = yield_overcons

    # ---- risk_factors（保留，台账 3.2/3.3）----
    risks = list(taxonomies.get("risk_factors", []))
    risk_counts = risk_factor_counts(scores, risks)
    report["risk_factors_f1_per_class"] = {
        risk: f1_from_counts(*risk_counts[risk]) for risk in risks
    }
    report["risk_factors_f1_macro"] = (
        mean(report["risk_factors_f1_per_class"].values()) if risks else 0.0
    )

    # ---- stop：T3 方向对（改名沿用 + miss 新增，台账 5）----
    stop = stop_counts(scores)
    stop_overcons, stop_miss = stop_direction_rates(stop)
    report["stop_overconservative_rate"] = stop_overcons
    report["stop_miss_rate"] = stop_miss

    # ---- 计数与工作点（新增，台账 2.4）----
    report["max_items_hit_rate"] = (
        sum(1 for s in ok if s.get("max_items_hit")) / n_ok if n_ok else 0.0
    )
    report["n_pred_objects"] = cardinality_histogram(
        [int(s["n_pred_objects"]) for s in ok], max_items
    )
    report["n_gt_objects"] = cardinality_histogram(
        [int(s["n_gt_objects"]) for s in ok], max_items
    )
    obj_dir, obj_hall, obj_miss = empty_direction_block(
        [(int(s["n_gt_objects"]), int(s["n_pred_objects"])) for s in ok]
    )
    report["object_empty_direction_counts"] = obj_dir
    report["object_empty_hallucination_rate"] = obj_hall
    report["object_empty_miss_rate"] = obj_miss

    # ---- risk 基数与空集（新增，台账 3.1）----
    report["n_gt_risk_factors"] = cardinality_histogram(
        [len(s["gt_risk_factors"]) for s in ok], risk_max_items
    )
    report["n_pred_risk_factors"] = cardinality_histogram(
        [len(s["pred_risk_factors"]) for s in ok], risk_max_items
    )
    risk_dir, risk_hall, risk_miss = empty_direction_block(
        [(len(s["gt_risk_factors"]), len(s["pred_risk_factors"])) for s in ok]
    )
    report["risk_empty_direction_counts"] = risk_dir
    report["risk_empty_hallucination_rate"] = risk_hall
    report["risk_empty_miss_rate"] = risk_miss

    # ---- denominators（T3/T6 重组，台账 1/4/5）----
    report["denominators"] = {
        "anchors_expected": n_anchors_expected,
        "parse_ok": n_ok,
        "matched_pairs": len(matched_ious),
        "motion_pairs": motion_pair_total,
        "yield_counts": yield_counts,
        "stop_counts": stop,
        "micro_counts": {
            _tau_key(float(tau)): list(micro_object_counts(scores, float(tau)))
            for tau in iou_thresholds
        },
    }
    return report
