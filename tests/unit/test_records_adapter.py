"""Unit tests for the Stage 06 record adapter.

Purpose:
    Freeze the Step 2 gate cases: real mini-data builds pass validation and
    are deterministic, quarantine codes are classified deterministically
    (chain codes via a synthetic ChainResult, adapter codes via injected
    pose resolution), scene-end truncation stays valid, and an empty
    builder commit is rejected.

Startup command:
    ``conda activate autovla_codeclean && PYTHONPATH=DriveAlign/src python -m \
    pytest DriveAlign/tests/unit/test_records_adapter.py -q``
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from drivealign.data.nuscenes_io import (
    BAD_GAP,
    ChainResult,
    INSUFFICIENT_HISTORY,
    load_nuscenes,
    read_keyframe_chain,
    resolve_anchor,
)
from drivealign.records import adapter as adapter_mod
from drivealign.records.adapter import (
    MISSING_CALIBRATION,
    NUMERIC_ANOMALY,
    NUM_FUTURE_POSES,
    build_record,
)
from drivealign.records.record import validate_record

MINI_ROOT = Path(__file__).resolve().parents[3] / "data" / "nuscenes" / "mini"
COMMIT = "deadbeef"

requires_mini = pytest.mark.skipif(
    not MINI_ROOT.is_dir(), reason="mini dataroot not present"
)


@pytest.fixture(scope="module")
def mini():
    if not MINI_ROOT.is_dir():
        pytest.skip("mini dataroot not present")
    return load_nuscenes(MINI_ROOT)


# --- synthetic chain passthrough (no data dependency) -----------------------


def _passthrough(monkeypatch, reason: str):
    seen = {}

    def fake_chain(_nusc, token):
        seen["token"] = token
        return ChainResult(
            frames=None, scene_token=None, scene_name=None, reason_code=reason
        )

    monkeypatch.setattr(adapter_mod, "read_keyframe_chain", fake_chain)
    result = build_record(object(), "tok-x", COMMIT)
    assert result.record is None and result.reason_code == reason
    assert seen["token"] == "tok-x"


@pytest.mark.parametrize(
    "reason",
    [
        INSUFFICIENT_HISTORY,
        "CROSS_SCENE",
        "NON_INCREASING_TIME",
        "MISSING_IMAGE",
        BAD_GAP,
        "NO_CAM_FRONT",
    ],
)
def test_chain_reason_codes_passthrough(monkeypatch, reason):
    _passthrough(monkeypatch, reason)


def test_missing_calibration_quarantine(mini, monkeypatch):
    token = resolve_anchor(mini)
    monkeypatch.setattr(
        adapter_mod, "_resolve_ego_pose", lambda nusc, sample: (None, MISSING_CALIBRATION)
    )
    result = build_record(mini, token, COMMIT)
    assert result.record is None and result.reason_code == MISSING_CALIBRATION


def test_numeric_anomaly_quarantine(mini, monkeypatch):
    token = resolve_anchor(mini)

    def fake_pose(_nusc, sample):
        # Keep real timestamps so the backward-difference dt stays positive;
        # the non-finite x must trip the speed check, not a zero dt.
        return (
            adapter_mod._EgoPose(
                timestamp_us=sample["timestamp"], x=math.nan, y=0.0, yaw=0.0
            ),
            None,
        )

    monkeypatch.setattr(adapter_mod, "_resolve_ego_pose", fake_pose)
    result = build_record(mini, token, COMMIT)
    assert result.record is None and result.reason_code == NUMERIC_ANOMALY


# --- real mini-data builds ---------------------------------------------------


@requires_mini
def test_happy_path_valid(mini):
    token = resolve_anchor(mini)
    result = build_record(mini, token, COMMIT)
    assert result.record is not None and result.reason_code is None
    rec = result.record
    assert validate_record(rec) == []
    assert len(rec.model_inputs.frames) == 4
    assert rec.model_inputs.frames[-1].frame_token == token
    assert 0 < len(rec.oracle_only.future_ego_poses) <= NUM_FUTURE_POSES
    assert 0.0 <= rec.model_inputs.ego_speed_mps < 40.0
    assert rec.provenance.builder_commit == COMMIT
    assert rec.provenance.nuscenes_version == "v1.0-mini"


@requires_mini
def test_deterministic_hash(mini):
    token = resolve_anchor(mini)
    first = build_record(mini, token, COMMIT)
    second = build_record(mini, token, COMMIT)
    assert first.record.canonical_hash() == second.record.canonical_hash()


@requires_mini
def test_insufficient_history_quarantine(mini):
    first_token = mini.scene[0]["first_sample_token"]
    result = build_record(mini, first_token, COMMIT)
    assert result.record is None
    assert result.reason_code == INSUFFICIENT_HISTORY


@requires_mini
def test_scene_end_truncation_is_valid(mini):
    checked = 0
    for scene in mini.scene:
        tok = scene["last_sample_token"]
        sample = mini.get("sample", tok)
        prevs = 0
        while sample.get("prev"):
            prevs += 1
            sample = mini.get("sample", sample["prev"])
        if prevs < 3:
            continue
        result = build_record(mini, tok, COMMIT)
        assert result.record is not None, (result.reason_code, result.detail)
        assert len(result.record.oracle_only.future_ego_poses) == 0
        assert validate_record(result.record) == []
        checked += 1
    assert checked > 0


@requires_mini
def test_empty_builder_commit_rejected(mini):
    token = resolve_anchor(mini)
    with pytest.raises(ValueError, match="builder_commit"):
        build_record(mini, token, "")


@requires_mini
def test_speed_matches_backward_difference(mini):
    token = resolve_anchor(mini)
    chain = read_keyframe_chain(mini, token)
    result = build_record(mini, token, COMMIT)
    assert chain.frames is not None and result.record is not None
    older, anchor = chain.frames[-2], chain.frames[-1]
    older_sample = mini.get("sample", older.sample_token)
    anchor_sample = mini.get("sample", anchor.sample_token)
    pose_a = mini.get(
        "ego_pose",
        mini.get("sample_data", anchor_sample["data"]["CAM_FRONT"])["ego_pose_token"],
    )
    pose_b = mini.get(
        "ego_pose",
        mini.get("sample_data", older_sample["data"]["CAM_FRONT"])["ego_pose_token"],
    )
    dt = (anchor_sample["timestamp"] - older_sample["timestamp"]) / 1e6
    expected = round(
        math.hypot(pose_a["translation"][0] - pose_b["translation"][0],
                   pose_a["translation"][1] - pose_b["translation"][1]) / dt, 3)
    assert result.record.model_inputs.ego_speed_mps == expected
