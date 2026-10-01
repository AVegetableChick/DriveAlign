"""Unit tests for the Stage 07 dataset build (shards, roundtrip, quarantine).

Purpose:
    Freeze the Step 2 mechanics without any NuScenes dependency: the shard
    writer rolls over at exactly 1000 canonical lines whose sha256 equals
    the record's canonical hash, the roundtrip verifier detects tampered
    lines / misplaced index entries / missing tokens, scene-manifest
    loading enforces the embedded sha256 and filename agreement, the
    quarantine taxonomy stays the frozen 8 codes, and the root manifest
    hashes deterministically over the record index.

Startup command:
    ``conda activate autovla_codeclean && PYTHONPATH=DriveAlign/src python -m \
    pytest DriveAlign/tests/unit/test_dataset_build.py -q``
"""

from __future__ import annotations

import hashlib
import json

import pytest

from drivealign.dataset.build_dataset import (
    QUARANTINE_CODES,
    SHARD_SIZE,
    _ShardWriter,
    build_dataset_manifest,
    load_scene_manifest,
    verify_shard_roundtrip,
)
from drivealign.records.record import DriveAlignRecord


def _record(token: str, speed: float = 5.2) -> DriveAlignRecord:
    return DriveAlignRecord.from_dict(
        {
            "record_version": "v1",
            "sample_token": token,
            "scene_token": "scene-0",
            "model_inputs": {
                "frames": [
                    {
                        "image_relpath": f"samples/CAM_FRONT/{token}-{i}.jpg",
                        # validate_record contract: newest frame token == sample_token
                        "frame_token": token if i == 3 else f"{token}-f{i}",
                        "timestamp_us": 1_510_000_000_000_000 + i * 500_000,
                        "time_offset_s": (i - 3) * 0.5,
                    }
                    for i in range(4)
                ],
                "ego_speed_mps": speed,
            },
            "training_targets": {"expected_output": None, "language_reference": None},
            "oracle_only": {
                "future_ego_poses": [
                    {
                        "sample_token": f"{token}-fu{k}",
                        "timestamp_us": 1_510_000_000_002_000_000 + k * 500_000,
                        "x": 1.0 + k,
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


# --- shard writer ------------------------------------------------------------


def test_shard_writer_rolls_over_at_1000(tmp_path):
    writer = _ShardWriter(tmp_path, "train", SHARD_SIZE)
    placements = []
    for i in range(2500):
        record = _record(f"tok{i:05d}")
        placements.append((writer.write(record.to_dict()), record))
    writer.close()
    assert writer.shards == [f"train/shard-{k:04d}.jsonl" for k in range(3)]
    counts = [1000, 1000, 500]
    for shard, expected in zip(writer.shards, counts):
        lines = (tmp_path / shard).read_text(encoding="utf-8").splitlines()
        assert len(lines) == expected


def test_shard_line_is_canonical_hash(tmp_path):
    writer = _ShardWriter(tmp_path, "val", 10)
    record = _record("tok-a")
    shard, line_no = writer.write(record.to_dict())
    writer.close()
    line = (tmp_path / shard).read_text(encoding="utf-8").splitlines()[line_no]
    assert hashlib.sha256(line.encode("utf-8")).hexdigest() == record.canonical_hash()


# --- roundtrip verifier ------------------------------------------------------


def _write_and_index(tmp_path, n=3):
    writer = _ShardWriter(tmp_path, "train", 10)
    index = {}
    for i in range(n):
        record = _record(f"tok{i}")
        shard, line_no = writer.write(record.to_dict())
        index[f"tok{i}"] = {
            "line": line_no,
            "record_hash": record.canonical_hash(),
            "shard": shard,
            "split": "train",
        }
    writer.close()
    per_split = {"train": {"record_count": n, "shard_count": 1, "shards": writer.shards}}
    return per_split, index


def test_roundtrip_happy_path(tmp_path):
    per_split, index = _write_and_index(tmp_path)
    assert verify_shard_roundtrip(tmp_path, per_split, index) == []


def test_roundtrip_detects_tampered_line(tmp_path):
    per_split, index = _write_and_index(tmp_path)
    path = tmp_path / per_split["train"]["shards"][0]
    lines = path.read_text(encoding="utf-8").splitlines()
    tampered = json.loads(lines[1])
    tampered["model_inputs"]["ego_speed_mps"] = 99.0
    lines[1] = json.dumps(tampered, sort_keys=True, separators=(",", ":"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    problems = verify_shard_roundtrip(tmp_path, per_split, index)
    assert problems and "sha256(line)" in problems[0]


def test_roundtrip_detects_index_misplacement(tmp_path):
    per_split, index = _write_and_index(tmp_path)
    index["tok1"]["line"] = 0  # lie about the placement
    problems = verify_shard_roundtrip(tmp_path, per_split, index)
    assert problems and "placement mismatch" in problems[0]


def test_roundtrip_detects_missing_token(tmp_path):
    per_split, index = _write_and_index(tmp_path)
    index["tok-ghost"] = {
        "line": 0,
        "record_hash": "0" * 64,
        "shard": per_split["train"]["shards"][0],
        "split": "train",
    }
    problems = verify_shard_roundtrip(tmp_path, per_split, index)
    assert any("missing in shards" in p for p in problems)


# --- scene manifest loading --------------------------------------------------


def _manifest_file(tmp_path, split="train", tokens=None, tamper=False):
    from drivealign.dataset.split import build_scene_manifest

    payload = build_scene_manifest(split, tokens or ["s-a"], [], {"val_fraction": 0.1}, "c0")
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if tamper:
        payload2 = json.loads(text)
        payload2["scene_count"] = 999
        text = json.dumps(payload2, indent=2, sort_keys=True) + "\n"
    path = tmp_path / f"{split}_scene_manifest.json"
    path.write_text(text, encoding="utf-8")
    return path


def test_load_scene_manifest_ok(tmp_path):
    path = _manifest_file(tmp_path)
    tokens, fingerprint = load_scene_manifest(path)
    assert tokens == ["s-a"] and len(fingerprint) == 64


def test_load_scene_manifest_rejects_tamper(tmp_path):
    path = _manifest_file(tmp_path, tamper=True)
    with pytest.raises(ValueError, match="sha256 mismatch"):
        load_scene_manifest(path)


def test_load_scene_manifest_rejects_wrong_filename(tmp_path):
    path = _manifest_file(tmp_path, split="val")
    path = path.rename(tmp_path / "train_scene_manifest.json")  # content says val
    with pytest.raises(ValueError, match="split field"):
        load_scene_manifest(path)


# --- frozen taxonomy and root manifest ---------------------------------------


def test_quarantine_taxonomy_is_frozen_8_codes():
    assert QUARANTINE_CODES == {
        "INSUFFICIENT_HISTORY",
        "CROSS_SCENE",
        "NON_INCREASING_TIME",
        "MISSING_IMAGE",
        "BAD_GAP",
        "NO_CAM_FRONT",
        "MISSING_CALIBRATION",
        "NUMERIC_ANOMALY",
    }


def test_dataset_manifest_deterministic():
    config = {"builder_commit": "c0", "shard_size": 1000}
    index = {"tok": {"line": 0, "record_hash": "0" * 64, "shard": "train/shard-0000.jsonl", "split": "train"}}
    per_split = {"train": {"record_count": 1, "shard_count": 1, "shards": ["train/shard-0000.jsonl"]}}
    a = build_dataset_manifest(config, {"train": "f" * 64}, per_split, index, 0)
    b = build_dataset_manifest(config, {"train": "f" * 64}, per_split, index, 0)
    assert a["manifest_sha256"] == b["manifest_sha256"]
    c = build_dataset_manifest(config, {"train": "f" * 64}, per_split, index, 1)  # quarantine count
    assert c["manifest_sha256"] != a["manifest_sha256"]
