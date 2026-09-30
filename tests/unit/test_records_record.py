"""Unit tests for the Stage 06 DriveAlignRecord v1 contract.

Purpose:
    Freeze the Step 1 gate cases: roundtrip + deterministic canonical hash,
    construction-time float normalization, strict structural rejection in
    ``from_dict``, and semantic violations caught by ``validate_record``.

Startup command:
    ``conda activate autovla_codeclean && PYTHONPATH=DriveAlign/src python -m \
    pytest DriveAlign/tests/unit/test_records_record.py -q``
"""

from __future__ import annotations

import copy

import pytest

from drivealign.records.record import (
    DriveAlignRecord,
    validate_record,
)


def _base_dict() -> dict:
    frames = [
        {
            "image_relpath": f"samples/CAM_FRONT/n{i}.jpg",
            "frame_token": f"tok{i}",
            "timestamp_us": 1_510_000_000_000_000 + i * 500_000,
            "time_offset_s": (i - 3) * 0.5,
        }
        for i in range(4)
    ]
    return {
        "record_version": "v1",
        "sample_token": "tok3",
        "scene_token": "scene0",
        "model_inputs": {"frames": frames, "ego_speed_mps": 5.2},
        "training_targets": {"expected_output": None, "language_reference": None},
        "oracle_only": {
            "future_ego_poses": [
                {
                    "sample_token": "fut0",
                    "timestamp_us": 1_510_000_002_000_000,
                    "x": 1.0,
                    "y": 2.0,
                    "yaw": 0.1,
                }
            ]
        },
        "provenance": {
            "contract_version": "v3",
            "temporal_policy_version": "v1",
            "builder_commit": "deadbeef",
            "nuscenes_version": "v1.0-mini",
        },
    }


def _record() -> DriveAlignRecord:
    return DriveAlignRecord.from_dict(_base_dict())


def _variant(**changes) -> dict:
    data = copy.deepcopy(_base_dict())
    data.update(changes)
    return data


def test_roundtrip_identity_and_hash():
    rec = _record()
    rebuilt = DriveAlignRecord.from_dict(rec.to_dict())
    assert rebuilt == rec
    assert rebuilt.canonical_hash() == rec.canonical_hash()


def test_float_normalization_same_hash():
    noisy = _variant()
    noisy["model_inputs"]["ego_speed_mps"] = 5.199999999
    assert DriveAlignRecord.from_dict(noisy).canonical_hash() == _record().canonical_hash()


def test_different_speed_changes_hash():
    changed = _variant()
    changed["model_inputs"]["ego_speed_mps"] = 5.3
    assert DriveAlignRecord.from_dict(changed).canonical_hash() != _record().canonical_hash()


@pytest.mark.parametrize(
    "mutate, needle",
    [
        (lambda d: d["model_inputs"].update(extra=1), "model_inputs"),
        (lambda d: d.update(record_version="v2"), "record_version"),
        (lambda d: d["model_inputs"]["frames"].pop(0), "exactly 4"),
        (lambda d: d["oracle_only"].pop("future_ego_poses"), "key mismatch"),
        (lambda d: d["model_inputs"].update(ego_speed_mps="fast"), "must be a number"),
        (
            lambda d: d["model_inputs"]["frames"][0].update(timestamp_us=1.5),
            "must be an int",
        ),
        (lambda d: d["provenance"].pop("builder_commit"), "provenance"),
    ],
)
def test_from_dict_rejects_structural_errors(mutate, needle):
    data = _base_dict()
    mutate(data)
    with pytest.raises(ValueError, match=needle):
        DriveAlignRecord.from_dict(data)


def test_validate_flags_anchor_mismatch():
    rec = _record()
    bad = DriveAlignRecord.from_dict(_variant(sample_token="other"))
    assert any("sample_token" in p for p in validate_record(bad))
    assert validate_record(rec) == []


def test_validate_flags_offset_mismatch():
    data = _variant()
    data["model_inputs"]["frames"][2]["time_offset_s"] = -9.9
    problems = validate_record(DriveAlignRecord.from_dict(data))
    assert any("time_offset_s" in p for p in problems)


def test_validate_flags_negative_speed():
    data = _variant()
    data["model_inputs"]["ego_speed_mps"] = -0.1
    problems = validate_record(DriveAlignRecord.from_dict(data))
    assert any("non-negative" in p for p in problems)


def test_validate_flags_unknown_contract_version():
    data = _variant()
    data["provenance"]["contract_version"] = "v9"
    problems = validate_record(DriveAlignRecord.from_dict(data))
    assert any("contract_version" in p for p in problems)


def test_validate_flags_future_token_overlap():
    data = _variant()
    data["oracle_only"]["future_ego_poses"][0]["sample_token"] = "tok2"
    problems = validate_record(DriveAlignRecord.from_dict(data))
    assert any("overlap" in p for p in problems)
