"""Unit tests for the counterfactual subset sampler (S09 plan section 2.4).

Run:
    ``conda activate autovla_codeclean && cd /root/autodl-tmp/drivealign_workspace && \
    PYTHONPATH=DriveAlign/src python -m pytest DriveAlign/tests/unit/test_counterfactual_subset.py -q``
"""

from __future__ import annotations

from drivealign.cli.sample_counterfactual_subset import sample_subset


def make_manifest():
    anchors = {}
    for scene_index in range(3):
        for member in range(2):
            token = f"t{scene_index}{member}"
            anchors[token] = {
                "split": "test" if scene_index < 2 else "train",
                "scene_token": f"scene_{scene_index}",
            }
    return {"anchors": anchors}


class TestSampleSubset:
    def test_deterministic_given_seed(self):
        manifest = make_manifest()
        assert sample_subset(manifest, split="test", n=3, seed=7) == sample_subset(
            manifest, split="test", n=3, seed=7
        )

    def test_respects_split_and_n(self):
        manifest = make_manifest()
        tokens = sample_subset(manifest, split="test", n=3, seed=1)
        assert len(tokens) == 3
        assert all(token in ("t00", "t01", "t10", "t11") for token in tokens)
        train_tokens = sample_subset(manifest, split="train", n=5, seed=1)
        assert set(train_tokens) == {"t20", "t21"}  # split exhausted

    def test_scene_diversity(self):
        manifest = make_manifest()
        tokens = sample_subset(manifest, split="test", n=2, seed=3)
        scenes = {manifest["anchors"][t]["scene_token"] for t in tokens}
        assert len(scenes) == 2  # round-robin spreads across scenes first

    def test_sorted_output(self):
        manifest = make_manifest()
        tokens = sample_subset(manifest, split="test", n=4, seed=9)
        assert tokens == sorted(tokens)
