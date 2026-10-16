"""Unit tests for the nested scene-cluster bootstrap (Eval v2, S13 T4).

覆盖：标量键与 v1 ``bootstrap_statistics`` 完全一致（共享重采样矩阵的
证据）、嵌套键的 ``{类别: 八字段}`` 展开、固定类别集契约的全量/重采样
fail-fast、以及 seed 确定性。

Run:
    ``conda activate autovla_codeclean && cd /root/autodl-tmp/drivealign_workspace && \
    PYTHONPATH=DriveAlign/src python -m pytest DriveAlign/tests/unit/test_eval_bootstrap_nested.py -q``
"""

from __future__ import annotations

import pytest

from drivealign.evaluation.bootstrap import (
    bootstrap_statistics,
    bootstrap_statistics_nested,
)

CLUSTERS = {
    "a1": "s1", "a2": "s1",              # scene s1: two anchors
    "b1": "s2",                           # scene s2: one anchor
    "c1": "s3", "c2": "s3", "c3": "s3",  # scene s3: three anchors
}
VALUES = {"a1": 1.0, "a2": 0.0, "b1": 1.0, "c1": 0.0, "c2": 0.0, "c3": 1.0}
FIELDS = {
    "point_estimate", "ci_low", "ci_high", "n_scenes",
    "n_values", "n_resamples", "seed", "confidence_level",
}


def mean_stat(values):
    def statistic(tokens):
        return {"acc": sum(values[t] for t in tokens) / len(tokens)}

    return statistic


LABELS = {1.0: "one", 0.0: "zero"}


def mixed_statistic(tokens):
    """标量键 + 逐类嵌套键（类别集固定，缺席计 0）的两层返回结构。"""
    acc = sum(VALUES[t] for t in tokens) / len(tokens)
    per_cat = {
        LABELS[value]: sum(1.0 for t in tokens if VALUES[t] == value) / len(tokens)
        for value in (1.0, 0.0)
    }
    return {"acc": acc, "per_value": per_cat}


class TestScalarKeysMatchV1:
    def test_scalar_only_statistic_equals_v1_exactly(self):
        """仅标量键的 statistic：嵌套版输出与 v1 逐字节一致——两版共享
        同一 cluster map / 重采样矩阵 / seed 的直接证据。"""
        kwargs = dict(n_resamples=60, seed=13, confidence_level=0.9)
        assert bootstrap_statistics_nested(
            CLUSTERS, mean_stat(VALUES), **kwargs
        ) == bootstrap_statistics(CLUSTERS, mean_stat(VALUES), **kwargs)

    def test_scalar_entry_has_eight_fields(self):
        out = bootstrap_statistics_nested(CLUSTERS, mixed_statistic, n_resamples=40, seed=1)
        assert set(out["acc"]) == FIELDS
        assert out["acc"]["point_estimate"] == pytest.approx(3.0 / 6.0)


class TestNestedKeys:
    def test_nested_block_expansion(self):
        out = bootstrap_statistics_nested(CLUSTERS, mixed_statistic, n_resamples=40, seed=1)
        assert sorted(out["per_value"]) == ["one", "zero"]
        for entry in out["per_value"].values():
            assert set(entry) == FIELDS
        assert out["per_value"]["one"]["point_estimate"] == pytest.approx(0.5)
        assert out["per_value"]["zero"]["point_estimate"] == pytest.approx(0.5)
        assert out["per_value"]["one"]["n_values"] == 6
        assert out["per_value"]["one"]["n_scenes"] == 3

    def test_ci_brackets_point_estimate_for_nested(self):
        out = bootstrap_statistics_nested(CLUSTERS, mixed_statistic, n_resamples=200, seed=7)
        for entry in out["per_value"].values():
            assert entry["ci_low"] <= entry["point_estimate"] <= entry["ci_high"]

    def test_missing_category_in_resample_fails_fast(self):
        """全量含类别 b、部分重采样子集缺席 → fail-fast（不许静默补值）。"""
        def flaky_statistic(tokens):
            values = {"acc": 1.0}
            # 子集只含 s1 的 anchor 时缺席类别 "b"（违反固定类别集契约）
            if set(tokens) <= {"a1", "a2"}:
                values["per_cat"] = {"a": 1.0}
            else:
                values["per_cat"] = {"a": 1.0, "b": 0.0}
            return values

        # n=50 重采样下几乎必然出现「两次都抽 s1」的子集（P≈1-0.75^50），
        # 固定 seed 保证该测试确定性触发。
        with pytest.raises(ValueError, match="category set differs"):
            bootstrap_statistics_nested(
                CLUSTERS, flaky_statistic, n_resamples=50, seed=1
            )

    def test_deterministic_given_seed(self):
        kwargs = dict(n_resamples=40, seed=21)
        assert bootstrap_statistics_nested(
            CLUSTERS, mixed_statistic, **kwargs
        ) == bootstrap_statistics_nested(CLUSTERS, mixed_statistic, **kwargs)

    def test_empty_clusters_rejected(self):
        with pytest.raises(ValueError):
            bootstrap_statistics_nested({}, mixed_statistic, n_resamples=10, seed=1)
