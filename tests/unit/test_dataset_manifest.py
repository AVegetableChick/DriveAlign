"""Unit tests for the Stage 07 anchor-policy manifest (Step 3).

Purpose:
    Freeze the Step 3 mechanics without NuScenes: asset loading verifies the
    embedded sha256, the 1F/4F derivation of a synthetic record yields
    consistent fingerprints (prompt parity, 1F image == 4F newest frame,
    distinct policy hashes), the record-level distribution aggregates
    locations / time_of_day / keywords per split, and the anchor-policy
    manifest hashes deterministically over the anchor index and bindings.

Startup command:
    ``conda activate autovla_codeclean && PYTHONPATH=DriveAlign/src python -m \
    pytest DriveAlign/tests/unit/test_dataset_manifest.py -q``
"""

from __future__ import annotations

import hashlib
import json

import pytest

from drivealign.dataset.manifest import (
    build_anchor_policy_manifest,
    record_distribution,
    _derive_anchor,
    _pct,
    _scene_meta,
    _verified_payload,
)
from drivealign.records.record import DriveAlignRecord


def _record(token: str) -> DriveAlignRecord:
    return DriveAlignRecord.from_dict(
        {
            "record_version": "v1",
            "sample_token": token,
            "scene_token": "scene-0",
            "model_inputs": {
                "frames": [
                    {
                        "image_relpath": f"samples/CAM_FRONT/{token}-{i}.jpg",
                        "frame_token": token if i == 3 else f"{token}-f{i}",
                        "timestamp_us": 1_510_000_000_000_000 + i * 500_000,
                        "time_offset_s": (i - 3) * 0.5,
                    }
                    for i in range(4)
                ],
                "ego_speed_mps": 5.2,
            },
            "training_targets": {"expected_output": None, "language_reference": None},
            "oracle_only": {
                "future_ego_poses": [
                    {
                        "sample_token": f"{token}-fu{k}",
                        "timestamp_us": 1_510_000_000_002_000_000 + k * 500_000,
                        "x": 1.0,
                        "y": 2.0,
                        "yaw": 0.1,
                    }
                    for k in range(2)
                ]
            },
            "provenance": {
                "contract_version": "v3",
                "temporal_policy_version": "v1",
                "builder_commit": "deadbeef",
                "nuscenes_version": "v1.0-mini",
            },
        }
    )


# --- verified payload loading ------------------------------------------------


def _write_asset(tmp_path, payload, name="asset.json"):
    digest = hashlib.sha256(
        json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    ).hexdigest()
    path = tmp_path / name
    path.write_text(
        json.dumps({**payload, "manifest_sha256": digest}, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return path, digest


def test_verified_payload_ok(tmp_path):
    path, digest = _write_asset(tmp_path, {"scene_tokens": ["a"], "split": "train"})
    payload, fingerprint = _verified_payload(path)
    assert payload["scene_tokens"] == ["a"] and fingerprint == digest


def test_verified_payload_rejects_tamper(tmp_path):
    path, _ = _write_asset(tmp_path, {"scene_tokens": ["a"], "split": "train"})
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["split"] = "val"  # tamper without rehashing
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="sha256 mismatch"):
        _verified_payload(path)


# --- 1F/4F derivation --------------------------------------------------------


def test_derive_anchor_happy_and_hashed():
    record = _record("tok-a")
    fields, problem = _derive_anchor(record, "tok-a")
    assert problem is None
    assert fields["anchor_image_relpath"] == "samples/CAM_FRONT/tok-a-3.jpg"
    assert fields["request_hash_1f"] != fields["request_hash_4f"]
    assert len(fields["request_hash_1f"]) == 64


def test_derive_anchor_rejects_wrong_token():
    fields, problem = _derive_anchor(_record("tok-a"), "tok-other")
    assert fields is None and "anchor frame token mismatch" in problem


# --- scene meta and record-level distribution --------------------------------


class FakeNusc:
    def __init__(self, scenes):
        self.scene = scenes
        self._samples = {
            s["first_sample_token"]: {"timestamp": s["_timestamp"]} for s in scenes
        }
        self._logs = {s["log_token"]: {"location": s["_location"]} for s in scenes}

    def get(self, table, token):
        if table == "scene":
            return next(s for s in self.scene if s["token"] == token)
        if table == "sample":
            return self._samples[token]
        if table == "log":
            return self._logs[token]
        raise KeyError(table)


def _nusc():
    # hour 12 -> day; hour 23 -> night (UTC day window is [06, 18)).
    return FakeNusc(
        [
            {"token": "s1", "log_token": "l1", "first_sample_token": "s1-kf",
             "description": "Parked cars, busy intersection", "_location": "boston",
             "_timestamp": 1_700_000_000 * 1_000_000 + 12 * 3_600 * 1_000_000},
            {"token": "s2", "log_token": "l2", "first_sample_token": "s2-kf",
             "description": "A bus and a bicycle at night", "_location": "singapore",
             "_timestamp": 1_700_000_000 * 1_000_000 + 23 * 3_600 * 1_000_000},
        ]
    )


def test_scene_meta_day_night_and_location():
    meta = _scene_meta(_nusc())
    assert meta["s1"] == {
        "description": "Parked cars, busy intersection",
        "location": "boston",
        "time_of_day": "day",
    }
    assert meta["s2"]["time_of_day"] == "night"


def test_record_distribution_aggregates():
    anchors = {
        "t1": {"split": "train", "scene_token": "s1"},
        "t2": {"split": "train", "scene_token": "s1"},
        "t3": {"split": "val", "scene_token": "s2"},
    }
    stats = record_distribution(anchors, _scene_meta(_nusc()))
    assert stats["train"]["record_count"] == 2
    assert stats["train"]["locations"] == {"boston": 2}
    assert stats["train"]["time_of_day"] == {"day": 2}
    kw = {row["keyword"]: row for row in stats["train"]["keywords"]}
    assert kw["parked"]["count"] == 2 and kw["parked"]["per_1k_records"] == 1000.0
    assert "parked" not in {row["keyword"] for row in stats["val"]["keywords"]}


def test_pct_zero_total_is_zero():
    assert _pct(1, 0) == 0.0 and _pct(1, 4) == 25.0


# --- anchor-policy manifest --------------------------------------------------


def _anchors():
    return {
        "t1": {"anchor_image_relpath": "a.jpg", "line": 0, "record_hash": "h1",
               "request_hash_1f": "r1a", "request_hash_4f": "r4a",
               "scene_token": "s1", "shard": "train/shard-0000.jsonl", "split": "train"},
        "t2": {"anchor_image_relpath": "b.jpg", "line": 1, "record_hash": "h2",
               "request_hash_1f": "r1b", "request_hash_4f": "r4b",
               "scene_token": "s1", "shard": "train/shard-0000.jsonl", "split": "train"},
    }


def test_anchor_manifest_counts_and_determinism():
    config = {"builder_commit": "c0", "mode": "full"}
    scene_fp = {"train": "f" * 64, "val": "e" * 64, "test": "d" * 64}
    a = build_anchor_policy_manifest("dataset-digest", scene_fp, _anchors(), config)
    b = build_anchor_policy_manifest("dataset-digest", scene_fp, _anchors(), config)
    assert a == b and a["manifest_sha256"] == b["manifest_sha256"]
    assert a["counts"] == {"train": 2, "val": 0, "test": 0, "total": 2}
    assert a["dataset_manifest_sha256"] == "dataset-digest"
    c = build_anchor_policy_manifest(
        "dataset-digest-2", scene_fp, _anchors(), config
    )
    assert c["manifest_sha256"] != a["manifest_sha256"]
    # self-hash reproducible exactly like the verifier does
    payload = {k: v for k, v in a.items() if k != "manifest_sha256"}
    assert (
        hashlib.sha256(
            json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
        ).hexdigest()
        == a["manifest_sha256"]
    )
