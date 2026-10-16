"""Scene-cluster bootstrap for Eval v1 metrics (S09 plan section 2.1/3).

Purpose:
    Percentile confidence intervals for Eval v1 statistics via scene-cluster
    resampling: scenes (not anchors) are drawn with replacement so the
    within-scene correlation between anchors is preserved. One shared
    resample index matrix drives every statistic per replicate, which makes
    the CIs mutually consistent and enables the paired-delta variant: two
    systems evaluated with the same cluster map and the same seed are
    resampled identically, so ``delta = system_a - system_b`` has a paired
    CI (the S12 comparison reuses exactly this path). All randomness flows
    through ``numpy.random.RandomState(seed)`` for bit-level determinism.

Example launch command:
    ``conda activate autovla_codeclean && PYTHONPATH=DriveAlign/src python -c \
    "from drivealign.evaluation.bootstrap import bootstrap_statistics; \
    clusters = {'a1': 's1', 'a2': 's1', 'b1': 's2'}; \
    values = {'a1': 1.0, 'a2': 0.0, 'b1': 1.0}; \
    stat = lambda tokens: {'acc': sum(values[t] for t in tokens) / len(tokens)}; \
    out = bootstrap_statistics(clusters, stat, n_resamples=100, seed=7); \
    print(out['acc']['point_estimate'], out['acc']['ci_low'], out['acc']['ci_high'])"``
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

Statistic = Callable[[Sequence[str]], Mapping[str, float]]

#: Eval v2 statistic：返回值可为「标量键 + 逐类嵌套键」两层结构
#: （标量 -> float，逐类 -> {类别: float}），见 bootstrap_statistics_nested。
NestedStatistic = Callable[[Sequence[str]], Mapping[str, Any]]


def _cluster_map(
    clusters: Mapping[str, str],
) -> tuple[list[str], dict[str, list[str]]]:
    """Sorted (tokens, scene -> tokens) structure for deterministic order."""
    tokens = sorted(clusters)
    members: dict[str, list[str]] = {}
    for token in tokens:
        members.setdefault(str(clusters[token]), []).append(token)
    return tokens, members


def _resample_matrix(
    scene_names: Sequence[str], n_resamples: int, seed: int
) -> np.ndarray:
    """Shared replicate matrix: one row of scene indices per resample."""
    rng = np.random.RandomState(seed)
    return rng.randint(0, len(scene_names), size=(n_resamples, len(scene_names)))


def _subset_tokens(
    scene_names: Sequence[str], members: Mapping[str, list[str]], row: np.ndarray
) -> list[str]:
    subset: list[str] = []
    for scene_index in row:
        subset.extend(members[scene_names[int(scene_index)]])
    return subset


def bootstrap_statistics(
    clusters: Mapping[str, str],
    statistic: Statistic,
    *,
    n_resamples: int,
    seed: int,
    confidence_level: float = 0.95,
) -> dict[str, dict[str, Any]]:
    """Point estimate + percentile CI for every key ``statistic`` returns.

    ``statistic`` receives the resampled anchor-token subset and returns a
    dict of scalar values; it is called once per replicate plus once on the
    full token set for the point estimate.
    """
    if not clusters:
        raise ValueError("bootstrap requires at least one anchor")
    tokens, members = _cluster_map(clusters)
    scene_names = sorted(members)
    point = dict(statistic(tokens))
    matrix = _resample_matrix(scene_names, n_resamples, seed)
    replicate: dict[str, list[float]] = {key: [] for key in point}
    for row in matrix:
        subset = _subset_tokens(scene_names, members, row)
        for key, value in dict(statistic(subset)).items():
            replicate[key].append(float(value))
    alpha = (1.0 - confidence_level) / 2.0 * 100.0
    return {
        key: {
            "point_estimate": float(point[key]),
            "ci_low": float(np.percentile(replicate[key], alpha)),
            "ci_high": float(np.percentile(replicate[key], 100.0 - alpha)),
            "n_scenes": len(scene_names),
            "n_values": len(tokens),
            "n_resamples": int(n_resamples),
            "seed": int(seed),
            "confidence_level": float(confidence_level),
        }
        for key in point
    }


def bootstrap_statistics_nested(
    clusters: Mapping[str, str],
    statistic: NestedStatistic,
    *,
    n_resamples: int,
    seed: int,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Point estimate + percentile CI，支持「标量键 + 逐类嵌套键」两层结构（Eval v2，T4）。

    重采样机制与 :func:`bootstrap_statistics` 完全一致（同一 cluster
    map、同一共享重采样矩阵、同一 seed），唯一扩展：当 ``statistic``
    对某键返回映射（而非标量）时，该键是逐类条目，CI 块展开为嵌套
    ``{<类别>: 八字段}``，与点估计键的 ``{类别: 值}`` 结构对称（S13
    台账 2026-10-09 定稿的块内表示）；标量键的八字段条目与 v1 逐字段
    相同。v1 的 :func:`bootstrap_statistics` 保持不动（历史基线由单测
    钉死），eval_v2 专用本函数。

    契约：逐类键的类别集由 ``statistic`` 在全量与每个重采样子集上保持
    一致（subset_statistics_v2 对固定类别列表逐类别计算、缺席 F1 计
    0.0），全量出现而重采样缺席的类别 fail-fast 报错而非静默补值。
    """
    if not clusters:
        raise ValueError("bootstrap requires at least one anchor")
    tokens, members = _cluster_map(clusters)
    scene_names = sorted(members)
    point = dict(statistic(tokens))
    scalar_keys = sorted(k for k, v in point.items() if not isinstance(v, Mapping))
    nested_keys = sorted(k for k, v in point.items() if isinstance(v, Mapping))
    matrix = _resample_matrix(scene_names, n_resamples, seed)
    scalar_replicates: dict[str, list[float]] = {key: [] for key in scalar_keys}
    nested_replicates: dict[str, dict[str, list[float]]] = {
        key: {} for key in nested_keys
    }
    for row in matrix:
        subset = _subset_tokens(scene_names, members, row)
        values = dict(statistic(subset))
        for key in scalar_keys:
            scalar_replicates[key].append(float(values[key]))
        for key in nested_keys:
            block = dict(values[key])
            # 固定类别集契约：每个重采样子集的逐类类别集必须与全量点估计
            # 完全一致（subset_statistics_v2 对固定类别列表逐类别计算、缺席
            # F1 计 0.0，因此正常用法恒成立）；出现「全量有而某次重采样缺
            # 席」说明 statistic 的类别集是数据派生的，fail-fast 报错而非
            # 静默产生样本数不足的 CI。
            if set(block) != set(point[key]):
                raise ValueError(
                    f"bootstrap nested key {key!r} category set differs between "
                    f"point estimate and a resample (missing: "
                    f"{sorted(set(point[key]) - set(block))}); per-category "
                    "statistics must use a fixed category list"
                )
            for category, value in block.items():
                nested_replicates[key].setdefault(str(category), []).append(
                    float(value)
                )
    alpha = (1.0 - confidence_level) / 2.0 * 100.0

    def _entry(replicates: list[float], point_value: float) -> dict[str, Any]:
        return {
            "point_estimate": float(point_value),
            "ci_low": float(np.percentile(replicates, alpha)),
            "ci_high": float(np.percentile(replicates, 100.0 - alpha)),
            "n_scenes": len(scene_names),
            "n_values": len(tokens),
            "n_resamples": int(n_resamples),
            "seed": int(seed),
            "confidence_level": float(confidence_level),
        }

    result: dict[str, Any] = {}
    for key in scalar_keys:
        result[key] = _entry(scalar_replicates[key], float(point[key]))
    for key in nested_keys:
        point_block = dict(point[key])
        block: dict[str, Any] = {}
        for category in sorted(point_block):
            replicates = nested_replicates[key].get(str(category))
            if not replicates:
                raise ValueError(
                    f"bootstrap nested key {key!r} category {category!r} "
                    "missing in resample replicates; per-category statistics "
                    "must use a fixed category list"
                )
            block[str(category)] = _entry(replicates, float(point_block[category]))
        result[key] = block
    return result


def bootstrap_paired_delta(
    clusters: Mapping[str, str],
    statistic_a: Statistic,
    statistic_b: Statistic,
    *,
    n_resamples: int,
    seed: int,
    confidence_level: float = 0.95,
) -> dict[str, dict[str, Any]]:
    """Paired per-key CI for ``a - b`` under identical cluster resamples.

    Both statistics see the exact same anchor subsets every replicate (same
    matrix, same seed), so the delta distribution reflects the within-scene
    pairing. Returns per key: point estimates of both systems, the delta,
    and the delta CI.
    """
    if not clusters:
        raise ValueError("bootstrap requires at least one anchor")
    tokens, members = _cluster_map(clusters)
    scene_names = sorted(members)
    point_a = dict(statistic_a(tokens))
    point_b = dict(statistic_b(tokens))
    if set(point_a) != set(point_b):
        raise ValueError("paired statistics must return identical key sets")
    matrix = _resample_matrix(scene_names, n_resamples, seed)
    replicate_delta: dict[str, list[float]] = {key: [] for key in point_a}
    for row in matrix:
        subset = _subset_tokens(scene_names, members, row)
        values_a = dict(statistic_a(subset))
        values_b = dict(statistic_b(subset))
        for key in point_a:
            replicate_delta[key].append(float(values_a[key]) - float(values_b[key]))
    alpha = (1.0 - confidence_level) / 2.0 * 100.0
    return {
        key: {
            "point_a": float(point_a[key]),
            "point_b": float(point_b[key]),
            "delta": float(point_a[key]) - float(point_b[key]),
            "delta_ci_low": float(np.percentile(replicate_delta[key], alpha)),
            "delta_ci_high": float(np.percentile(replicate_delta[key], 100.0 - alpha)),
            "n_scenes": len(scene_names),
            "n_values": len(tokens),
            "n_resamples": int(n_resamples),
            "seed": int(seed),
            "confidence_level": float(confidence_level),
        }
        for key in point_a
    }
