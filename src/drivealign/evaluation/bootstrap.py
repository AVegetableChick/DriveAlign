"""Scene-cluster bootstrap for M09 metrics (S09 plan section 2.1/3).

Purpose:
    Percentile confidence intervals for M09 statistics via scene-cluster
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
