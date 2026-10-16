"""Eval v3 BEV localization-error metrics (S13 eval v3 additions, §2).

Purpose:
    Implement the frozen BEV error family of
    ``docs/stages_record/S13_contract_and_eval_revision/S13_eval_v3_metric_additions_draft.md``
    (拍板定稿 2026-10-14). Samples are matched pairs only (primary IoU
    threshold): a pair that failed 2D-IoU matching contributes no BEV
    sample — misses/hallucinations are measured by the object_f1 family,
    never leaked into BEV. Both sides live on the same 0.1 m grid
    (contract v7 parse-time rounding), so errors are quantization
    symmetric. Three families use the all-non-negative error semantics
    (2026-10-13 拍板): ``bev_dx = abs(Δx)`` (longitudinal),
    ``bev_dy = abs(Δy)`` (lateral), ``bev_dxy = hypot(Δx, Δy)`` (planar
    distance). Distance-conditioned buckets key on the GT nearest-point
    range ``d_gt = hypot(x_g, y_g)`` — the condition quantity is
    prediction-independent, keeping bucketed readings comparable across
    models. Four buckets (2026-10-14 拍板, user ruling): [0,10) /
    [10,25) / [25,40) / [40, ∞) open bucket. The signed-bias diagnostics
    ``bev_signed_{dx,dy}_{pos,neg}_mean`` (2026-10-14 拍板) MUST average
    the positive and negative signed-error subsets separately — an
    algebraic mean over both subsets cancels opposing biases (+2 m / −2 m
    at parity averages to 0 and loses the signal entirely); zero errors
    enter neither subset; an empty subset is 0.0.

    p50/p90 use the same linear-interpolation ``_percentile`` as
    ``matched_iou_p90`` (draft §2.3); the empty-set convention places
    0.0 in every key for slices without matched pairs (aggregate_v2
    convention, never null).

Example launch command:
    ``conda activate autovla_codeclean && PYTHONPATH=DriveAlign/src python -c \
    "from drivealign.evaluation.metrics_bev import bev_statistics; \
    pairs = [(17.75, 0.0, 17.75, 0.0), (11.0, 1.0, 10.0, 0.0)]; \
    print(sorted(bev_statistics(pairs))[:4])"``
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from statistics import mean
from typing import Any

from drivealign.evaluation.matching import MatchOutcome
from drivealign.evaluation.metrics import _percentile

#: BEV 误差三族（全非负语义，2026-10-13 拍板）：键族名固定顺序。
_BEV_FAMILIES = ("dx", "dy", "dxy")

#: d_gt 四桶（2026-10-14 拍板）：[lo, hi) 左闭右开；末桶开放（hi=inf）。
#: 键后缀与草案 §2.3 表逐字对应（bucket_0_10 / 10_25 / 25_40 / 40p）。
_BEV_BUCKETS: tuple[tuple[float, float, str], ...] = (
    (0.0, 10.0, "0_10"),
    (10.0, 25.0, "10_25"),
    (25.0, 40.0, "25_40"),
    (40.0, math.inf, "40p"),
)


def collect_bev_pairs(
    gt_objects: Sequence[Mapping[str, Any]],
    pred_objects: Sequence[Mapping[str, Any]],
    primary: MatchOutcome,
) -> list[tuple[float, float, float, float]]:
    """Primary-threshold 匹配对 → ``(x_p, y_p, x_g, y_g)`` BEV 样本列表。

    匹配对几何在打分层一次性解析（score_anchor_v3 落盘
    ``bev_pairs``，聚合与 bootstrap 只消费解析后的四元组）。任一侧
    bev 字段缺失（v1–v6 面、残缺记录）即跳过该对——不产生 BEV 样本，
    也不置 0 误差（0 是合法误差值，缺样本与零误差语义必须分离）。
    """
    pairs: list[tuple[float, float, float, float]] = []
    for gt_index, pred_index, _iou in primary.matched:
        gt_x = gt_objects[gt_index].get("bev_x")
        gt_y = gt_objects[gt_index].get("bev_y")
        pred_x = pred_objects[pred_index].get("bev_x")
        pred_y = pred_objects[pred_index].get("bev_y")
        if None in (gt_x, gt_y, pred_x, pred_y):
            continue
        pairs.append((float(pred_x), float(pred_y), float(gt_x), float(gt_y)))
    return pairs


def bev_statistics(
    pairs: Sequence[tuple[float, float, float, float]],
) -> dict[str, float]:
    """BEV 匹配对样本 → 冻结键集全部 49 个标量（§2.3 顶层 9 + 分桶 36 + §6-3 signed 4）。

    分桶条件量 = GT 最近点距 ``d_gt = hypot(x_g, y_g)``（与预测无关，
    模型间可比）；signed 偏置按正/负子集分别求均值（方向分解口径，
    严禁代数均值）；空样本集 / 空子集一律 0.0（aggregate_v2 空集口径）。
    """
    errors: dict[str, list[float]] = {"dx": [], "dy": [], "dxy": []}
    # signed 偏置诊断：保留符号的逐对误差（x_p - x_g / y_p - y_g）。
    signed: dict[str, list[float]] = {"dx": [], "dy": []}
    # 分桶样本：桶 tag -> 三族误差列表（d_gt 归桶后逐族收集）。
    bucket_errors: dict[str, dict[str, list[float]]] = {
        tag: {family: [] for family in _BEV_FAMILIES} for _, _, tag in _BEV_BUCKETS
    }

    for pred_x, pred_y, gt_x, gt_y in pairs:
        delta_x = pred_x - gt_x
        delta_y = pred_y - gt_y
        errors["dx"].append(abs(delta_x))
        errors["dy"].append(abs(delta_y))
        errors["dxy"].append(math.hypot(delta_x, delta_y))
        signed["dx"].append(delta_x)
        signed["dy"].append(delta_y)
        d_gt = math.hypot(gt_x, gt_y)
        for lo, hi, tag in _BEV_BUCKETS:
            if lo <= d_gt < hi:
                bucket_errors[tag]["dx"].append(abs(delta_x))
                bucket_errors[tag]["dy"].append(abs(delta_y))
                bucket_errors[tag]["dxy"].append(math.hypot(delta_x, delta_y))
                break

    stats: dict[str, float] = {}
    # ---- 顶层 9 键（全量匹配对；p50/p90 与 matched_iou_p90 同一线性插值口径）----
    for family in _BEV_FAMILIES:
        values = sorted(errors[family])
        stats[f"bev_{family}_mean"] = mean(values) if values else 0.0
        stats[f"bev_{family}_p50"] = _percentile(values, 50)
        stats[f"bev_{family}_p90"] = _percentile(values, 90)

    # ---- 分桶 36 键（桶内三族 × mean/p50/p90；空桶 0.0）----
    for _lo, _hi, tag in _BEV_BUCKETS:
        for family in _BEV_FAMILIES:
            values = sorted(bucket_errors[tag][family])
            stats[f"bev_{family}_mean_bucket_{tag}"] = (
                mean(values) if values else 0.0
            )
            stats[f"bev_{family}_p50_bucket_{tag}"] = _percentile(values, 50)
            stats[f"bev_{family}_p90_bucket_{tag}"] = _percentile(values, 90)

    # ---- signed 方向分解 4 键（正/负子集分别均值：pos 恒 >0 = 推远量，
    # neg 恒 <0 = 推近量；零误差两侧都不进；空子集 0.0）----
    for axis in ("dx", "dy"):
        positives = [value for value in signed[axis] if value > 0.0]
        negatives = [value for value in signed[axis] if value < 0.0]
        stats[f"bev_signed_{axis}_pos_mean"] = mean(positives) if positives else 0.0
        stats[f"bev_signed_{axis}_neg_mean"] = mean(negatives) if negatives else 0.0
    return stats
