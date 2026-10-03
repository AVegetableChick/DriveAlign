"""Unit tests for scene-cluster bootstrap (S09 plan section 2.4).

Run:
    ``conda activate autovla_codeclean && cd /root/autodl-tmp/drivealign_workspace && \
    PYTHONPATH=DriveAlign/src python -m pytest DriveAlign/tests/unit/test_eval_bootstrap.py -q``
"""

from __future__ import annotations

import pytest

from drivealign.evaluation.bootstrap import (
    bootstrap_paired_delta,
    bootstrap_statistics,
)

CLUSTERS = {
    "a1": "s1", "a2": "s1",          # scene s1: two anchors
    "b1": "s2",                       # scene s2: one anchor
    "c1": "s3", "c2": "s3", "c3": "s3",  # scene s3: three anchors
}
VALUES = {"a1": 1.0, "a2": 0.0, "b1": 1.0, "c1": 0.0, "c2": 0.0, "c3": 1.0}


def mean_stat(values):
    def statistic(tokens):
        return {"acc": sum(values[t] for t in tokens) / len(tokens)}

    return statistic


class TestBootstrapStatistics:
    def test_point_estimate_is_full_set_mean(self):
        out = bootstrap_statistics(
            CLUSTERS, mean_stat(VALUES), n_resamples=50, seed=1
        )
        assert out["acc"]["point_estimate"] == pytest.approx(3.0 / 6.0)
        assert out["acc"]["n_values"] == 6
        assert out["acc"]["n_scenes"] == 3

    def test_deterministic_given_seed(self):
        first = bootstrap_statistics(CLUSTERS, mean_stat(VALUES), n_resamples=50, seed=42)
        second = bootstrap_statistics(CLUSTERS, mean_stat(VALUES), n_resamples=50, seed=42)
        assert first == second

    def test_ci_brackets_point_estimate(self):
        out = bootstrap_statistics(
            CLUSTERS, mean_stat(VALUES), n_resamples=200, seed=7
        )
        assert out["acc"]["ci_low"] <= out["acc"]["point_estimate"] <= out["acc"]["ci_high"]
        # Scene means are s1=0.5, s2=1.0, s3=1/3; every resample mean lies in
        # the convex hull of the scene means.
        assert 0.0 <= out["acc"]["ci_low"] and out["acc"]["ci_high"] <= 1.0

    def test_resampling_is_scene_clustered(self):
        """Within any replicate subset, a scene's anchors move together."""
        seen_subsets: list[list[str]] = []

        def recording_statistic(tokens):
            seen_subsets.append(list(tokens))
            return {"acc": 0.0}

        bootstrap_statistics(CLUSTERS, recording_statistic, n_resamples=20, seed=3)
        assert len(seen_subsets) == 21  # 20 replicates + 1 point estimate
        scene_sizes = {"s1": 2, "s2": 1, "s3": 3}
        for subset in seen_subsets:
            scenes_in_subset = {CLUSTERS[token] for token in subset}
            for scene in scenes_in_subset:
                members = [t for t in subset if CLUSTERS[t] == scene]
                # A drawn scene contributes its whole anchor set; a scene may
                # be drawn multiple times per replicate (with replacement).
                assert len(members) % scene_sizes[scene] == 0
                assert len(members) >= scene_sizes[scene]

    def test_empty_clusters_rejected(self):
        with pytest.raises(ValueError):
            bootstrap_statistics({}, mean_stat(VALUES), n_resamples=10, seed=1)


class TestPairedDelta:
    def test_paired_statistics_see_identical_subsets(self):
        seen_a: list[frozenset] = []
        seen_b: list[frozenset] = []

        def stat_a(tokens):
            seen_a.append(frozenset(tokens))
            return {"acc": 1.0}

        def stat_b(tokens):
            seen_b.append(frozenset(tokens))
            return {"acc": 0.0}

        out = bootstrap_paired_delta(
            CLUSTERS, stat_a, stat_b, n_resamples=30, seed=11
        )
        assert seen_a == seen_b
        assert out["acc"]["point_a"] == 1.0
        assert out["acc"]["point_b"] == 0.0
        assert out["acc"]["delta"] == pytest.approx(1.0)
        assert out["acc"]["delta_ci_low"] == pytest.approx(1.0)
        assert out["acc"]["delta_ci_high"] == pytest.approx(1.0)

    def test_deterministic_given_seed(self):
        kwargs = dict(n_resamples=40, seed=5)
        first = bootstrap_paired_delta(CLUSTERS, mean_stat(VALUES), mean_stat(VALUES), **kwargs)
        second = bootstrap_paired_delta(CLUSTERS, mean_stat(VALUES), mean_stat(VALUES), **kwargs)
        assert first == second
        assert first["acc"]["delta"] == pytest.approx(0.0)

    def test_key_set_mismatch_rejected(self):
        def stat_a(tokens):
            return {"acc": 1.0}

        def stat_b(tokens):
            return {"different_key": 0.0}

        with pytest.raises(ValueError):
            bootstrap_paired_delta(CLUSTERS, stat_a, stat_b, n_resamples=10, seed=1)
