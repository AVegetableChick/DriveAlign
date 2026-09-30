"""Unit tests for the Stage 06 serializer and leak scanning.

Purpose:
    Freeze the Step 3 gate cases: the model_inputs-only type guard, the
    frozen v3 prompt with exactly one causal speed line, 1F/4F frame
    selection, request-hash determinism and speed sensitivity, and
    zero-hit future-fingerprint scanning with a positive control.

Startup command:
    ``conda activate autovla_codeclean && PYTHONPATH=DriveAlign/src python -m \
    pytest DriveAlign/tests/unit/test_records_serializer.py -q``
"""

from __future__ import annotations

import json

import pytest

from drivealign.contracts.prompt import build_structured_prompt
from drivealign.records.record import (
    DriveAlignRecord,
    FrameInput,
    ModelInputs,
)
from drivealign.records.serializer import (
    InputPolicy,
    collate,
    future_fingerprints,
    scan_future_fingerprints,
    serialize,
    to_processor_inputs,
)

SPEED_LINE_PREFIX = (
    "Currently available ego speed: 5.200 m/s "
    "(backward difference from the previous keyframe)"
)


def _model_inputs(speed: float = 5.2) -> ModelInputs:
    frames = [
        FrameInput(
            image_relpath=f"samples/CAM_FRONT/n{i}.jpg",
            frame_token=f"tok{i}",
            timestamp_us=1_510_000_000_000_000 + i * 500_000,
            time_offset_s=(i - 3) * 0.5,
        )
        for i in range(4)
    ]
    return ModelInputs(frames=frames, ego_speed_mps=speed)


def _record_dict() -> dict:
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
                    "sample_token": f"fut{i}",
                    "timestamp_us": 1_510_000_002_000_000 + i * 500_000,
                    "x": 1.0 + i,
                    "y": 2.0,
                    "yaw": 0.1,
                }
                for i in range(2)
            ]
        },
        "provenance": {
            "contract_version": "v3",
            "temporal_policy_version": "v1",
            "builder_commit": "deadbeef",
            "nuscenes_version": "v1.0-mini",
        },
    }


def test_serialize_rejects_non_model_inputs():
    with pytest.raises(TypeError, match="ModelInputs"):
        serialize("not model inputs", InputPolicy.ONE_FRAME)  # type: ignore[arg-type]


def test_prompt_is_frozen_plus_one_speed_line():
    frozen = build_structured_prompt()
    for policy in InputPolicy:
        req = serialize(_model_inputs(), policy)
        assert req.prompt.startswith(SPEED_LINE_PREFIX)
        assert req.prompt.count("Currently available ego speed") == 1
        assert req.prompt.split("\n\n", 1)[1] == frozen


def test_1f_and_4f_prompts_identical():
    mi = _model_inputs()
    assert serialize(mi, InputPolicy.ONE_FRAME).prompt == serialize(
        mi, InputPolicy.FOUR_FRAME
    ).prompt


def test_frame_selection():
    mi = _model_inputs()
    req1 = serialize(mi, InputPolicy.ONE_FRAME)
    req4 = serialize(mi, InputPolicy.FOUR_FRAME)
    assert req1.image_relpaths == ("samples/CAM_FRONT/n3.jpg",)
    assert req1.frame_tokens == ("tok3",)
    assert req1.time_offsets_s == (0.0,)
    assert req4.image_relpaths == tuple(f.image_relpath for f in mi.frames)
    assert req4.frame_tokens == ("tok0", "tok1", "tok2", "tok3")


def test_request_hash_determinism_and_sensitivity():
    req = serialize(_model_inputs(), InputPolicy.FOUR_FRAME)
    again = serialize(_model_inputs(), InputPolicy.FOUR_FRAME)
    assert again.canonical_hash() == req.canonical_hash()
    other = serialize(_model_inputs(speed=5.3), InputPolicy.FOUR_FRAME)
    assert other.canonical_hash() != req.canonical_hash()


def test_fingerprints_cover_tokens_and_timestamps():
    rec = DriveAlignRecord.from_dict(_record_dict())
    fps = future_fingerprints(rec)
    poses = rec.oracle_only.future_ego_poses
    assert len(fps) == 2 * len(poses)
    assert poses[0].sample_token in fps
    assert str(poses[0].timestamp_us) in fps


def test_scan_clean_and_positive_control():
    rec = DriveAlignRecord.from_dict(_record_dict())
    fps = future_fingerprints(rec)
    reqs = [serialize(rec.model_inputs, p) for p in InputPolicy]
    procs = [to_processor_inputs(r) for r in reqs]
    batch = collate(procs)
    payloads = {
        "prompt_1f": reqs[0].prompt,
        "prompt_4f": reqs[1].prompt,
        "processor": json.dumps([p.to_dict() for p in procs]),
        "batch": json.dumps(batch.to_dict()),
    }
    assert scan_future_fingerprints(fps, payloads) == []
    leaked = dict(payloads, injected=rec.oracle_only.future_ego_poses[0].sample_token)
    assert scan_future_fingerprints(fps, leaked) == ["injected"]


def test_1f_4f_share_anchor_prompt_speed():
    rec = DriveAlignRecord.from_dict(_record_dict())
    req1 = serialize(rec.model_inputs, InputPolicy.ONE_FRAME)
    req4 = serialize(rec.model_inputs, InputPolicy.FOUR_FRAME)
    assert req1.prompt == req4.prompt
    assert req1.frame_tokens[-1] == req4.frame_tokens[-1] == rec.sample_token
    assert req4.image_relpaths[-1] == req1.image_relpaths[0]
    assert req1.ego_speed_mps == req4.ego_speed_mps == rec.model_inputs.ego_speed_mps
