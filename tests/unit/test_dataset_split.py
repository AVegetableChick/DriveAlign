"""Unit tests for the Stage 07 scene split (stratified dictionary order).

Purpose:
    Freeze the Step 1 rule mechanics on synthetic NuScenes stand-ins (no
    trainval dependency): the D4 half-up 10% val share is taken from the
    lexicographic tail of every (location x time_of_day) bucket, the split
    is deterministic across calls, test scenes pass through unfiltered,
    distribution stats count locations / time_of_day / keyframes /
    description keywords, and scene manifests hash deterministically.
    One integration point reuses the real v1.0-mini metadata with a
    monkeypatched devkit name list (skipped when mini is absent).

Startup command:
    ``conda activate autovla_codeclean && PYTHONPATH=DriveAlign/src python -m \
    pytest DriveAlign/tests/unit/test_dataset_split.py -q``
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from drivealign.dataset import split as split_mod
from drivealign.dataset.split import (
    build_scene_manifest,
    distribution_stats,
    stratified_split,
)

MINI_ROOT = Path(__file__).resolve().parents[3] / "data" / "nuscenes" / "mini"
_BASE_US = int(datetime(2023, 1, 1, tzinfo=timezone.utc).timestamp() * 1e6)


class FakeNusc:
    """Minimal NuScenes stand-in serving scene/sample/log lookups."""

    def __init__(self, scene_specs):
        self.scene = []
        self._samples = {}
        self._logs = {}
        for spec in scene_specs:
            token = spec["name"]  # token == name so ordering is readable
            log_token = f"log-{spec['name']}"
            self._logs[log_token] = {"location": spec["location"]}
            n = spec["n_keyframes"]
            first = f"{token}-kf000"
            for k in range(n):
                self._samples[f"{token}-kf{k:03d}"] = {
                    "prev": f"{token}-kf{k - 1:03d}" if k else "",
                    "next": f"{token}-kf{k + 1:03d}" if k + 1 < n else "",
                    "timestamp": _BASE_US + spec["hour"] * 3600 * 1_000_000
                    + k * 500_000,
                }
            self.scene.append(
                {
                    "name": spec["name"],
                    "token": token,
                    "log_token": log_token,
                    "first_sample_token": first,
                    "description": spec.get("description", ""),
                }
            )

    def get(self, table, token):
        if table == "scene":
            return next(s for s in self.scene if s["token"] == token)
        if table == "sample":
            return self._samples[token]
        if table == "log":
            return self._logs[token]
        raise KeyError(table)


def _specs():
    """Buckets: boston/day 21, boston/night 10, singapore/day 3 (nightless)."""
    specs = []
    for i in range(21):
        specs.append(
            {"name": f"b-day-{i:02d}", "location": "boston", "hour": 12,
             "n_keyframes": 6,
             "description": "Parked cars, busy intersection with many peds"
             if i == 0 else ""}
        )
    for i in range(10):
        specs.append(
            {"name": f"b-night-{i:02d}", "location": "boston", "hour": 23, "n_keyframes": 4}
        )
    for i in range(3):
        specs.append(
            {"name": f"s-day-{i:02d}", "location": "singapore", "hour": 7, "n_keyframes": 5}
        )
    return specs


@pytest.fixture()
def fake(monkeypatch):
    nusc = FakeNusc(_specs())
    all_names = [s["name"] for s in nusc.scene]
    monkeypatch.setattr(split_mod, "trainval_train_names", set(all_names))
    monkeypatch.setattr(split_mod, "trainval_val_names", {"test-0", "test-1"})
    for name in ("test-0", "test-1"):
        nusc.scene.append(
            {"name": name, "token": name, "log_token": f"log-{name}",
             "first_sample_token": f"{name}-kf000", "description": ""}
        )
        nusc._samples[f"{name}-kf000"] = {"prev": "", "next": "", "timestamp": _BASE_US}
        nusc._logs[f"log-{name}"] = {"location": "elsewhere"}
    return nusc


def test_split_deterministic_across_calls(fake):
    first = stratified_split(fake)
    second = stratified_split(fake)
    assert first[0] == second[0] and first[1] == second[1]


def test_half_up_val_share_from_bucket_tail(fake):
    tokens, _ = stratified_split(fake)
    by_split = {t: split for split, toks in tokens.items() for t in toks}
    # boston/day: 21 scenes -> half-up(0.1*21)=2 val; tail in dictionary order.
    day_train = [t for t in tokens["train"] if by_split[t] and t.startswith("b-day")]
    day_val = [t for t in tokens["val"] if t.startswith("b-day")]
    assert len(day_val) == 2
    assert day_val == [f"b-day-{i:02d}" for i in (20, 19)] or day_val == sorted(
        set(f"b-day-{i:02d}" for i in range(21)) - set(day_train)
    )
    assert set(day_val) == {f"b-day-{i:02d}" for i in range(19, 21)}
    # boston/night: 10 scenes -> 1 val (the last one); singapore/day: 3 -> 0 val.
    assert [t for t in tokens["val"] if t.startswith("b-night")] == ["b-night-09"]
    assert not [t for t in tokens["val"] if t.startswith("s-day")]


def test_bucket_table_counts_match_assignment(fake):
    tokens, buckets = stratified_split(fake)
    rows = {(b["location"], b["time_of_day"]): b for b in buckets}
    assert rows[("boston", "day")]["val_count"] == 2
    assert rows[("boston", "night")]["val_count"] == 1
    assert rows[("singapore", "day")]["val_count"] == 0
    total = sum(b["train_count"] + b["val_count"] for b in buckets)
    assert total == len(tokens["train"]) + len(tokens["val"])


def test_test_split_unfiltered_and_sorted(fake):
    tokens, _ = stratified_split(fake)
    assert tokens["test"] == ["test-0", "test-1"]


def test_distribution_stats_counts(fake):
    tokens, _ = stratified_split(fake)
    stats = distribution_stats(fake, tokens)
    train = stats["per_split"]["train"]
    assert train["scene_count"] == len(tokens["train"])
    assert train["time_of_day"] == {"day": 22, "night": 9}  # b-day 21-2, s-day 3-0, b-night 10-1
    assert train["keyframe_count"] == 19 * 6 + 9 * 4 + 3 * 5
    kw = {row["keyword"]: row for row in stats["keywords"]}
    assert kw["parked"]["total_count"] == 1  # only b-day-00 carries a description


def test_scene_manifest_sha256_deterministic():
    rule = {"val_fraction": 0.1}
    a = build_scene_manifest("train", ["s-b", "s-a"], [], rule, "deadbeef")
    b = build_scene_manifest("train", ["s-b", "s-a"], [], rule, "deadbeef")
    assert a["manifest_sha256"] == b["manifest_sha256"]
    c = build_scene_manifest("train", ["s-a", "s-b"], [], rule, "deadbeef")
    assert c["manifest_sha256"] != a["manifest_sha256"]  # order matters
    payload = {k: v for k, v in a.items() if k != "manifest_sha256"}
    expected = json.dumps(payload, indent=2, sort_keys=True)
    import hashlib

    assert (
        hashlib.sha256(expected.encode("utf-8")).hexdigest()
        == a["manifest_sha256"]
    )


def test_split_assignment_matches_devkit_names_real_mini(monkeypatch):
    if not MINI_ROOT.is_dir():
        pytest.skip("mini dataroot not present")
    from drivealign.data.nuscenes_io import load_nuscenes
    from nuscenes.utils.splits import mini_train

    nusc = load_nuscenes(MINI_ROOT)
    # v1.0-mini holds 10 scenes but devkit mini_train names only 8; the other
    # two belong to the mini val blind set, so the val name list stays empty.
    monkeypatch.setattr(split_mod, "trainval_train_names", set(mini_train))
    monkeypatch.setattr(split_mod, "trainval_val_names", set())
    tokens, _ = stratified_split(nusc)
    assert sorted(tokens["train"] + tokens["val"]) == sorted(
        s["token"] for s in nusc.scene if s["name"] in set(mini_train)
    )
    assert not (set(tokens["train"]) & set(tokens["val"]))
