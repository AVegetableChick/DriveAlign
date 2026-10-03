"""Unit tests for the base benchmark request building (S09 plan section 2.4).

Covers the 1F input policy (anchor frame only), the exact ego-speed line
format shared with the serializer, request-hash recomputation against the
v5-face manifest values, counterfactual image routing, and the future-
fingerprint zero-hit requirement on the prompt.

Run:
    ``conda activate autovla_codeclean && cd /root/autodl-tmp/drivealign_workspace && \
    PYTHONPATH=DriveAlign/src python -m pytest DriveAlign/tests/unit/test_base_benchmark_requests.py -q``
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from drivealign.cli.base_benchmark import (
    build_benchmark_request,
    ego_speed_line,
    load_record,
    select_tokens,
)
from drivealign.records.record import DriveAlignRecord
from drivealign.records.serializer import (
    InputPolicy,
    future_fingerprints,
    scan_future_fingerprints,
    serialize,
)


def make_record(token: str, scene: str) -> dict:
    frames = [
        {
            "image_relpath": f"samples/CAM_FRONT/{token}_{i}.jpg",
            "frame_token": f"{token}_f{i}",
            "timestamp_us": 1_510_000_000_000_000 + i * 500_000,
            "time_offset_s": (i - 3) * 0.5,
        }
        for i in range(4)
    ]
    return {
        "record_version": "v1",
        "sample_token": token,
        "scene_token": scene,
        "model_inputs": {"frames": frames, "ego_speed_mps": 5.2},
        "training_targets": {"expected_output": None, "language_reference": None},
        "oracle_only": {
            "future_ego_poses": [
                {
                    "sample_token": f"{token}_FUTURE",
                    "timestamp_us": 1_510_000_003_000_000,
                    "x": 12.5,
                    "y": 0.25,
                    "yaw": 0.0,
                }
            ]
        },
        "provenance": {
            "contract_version": "v5",
            "temporal_policy_version": "window-v1",
            "builder_commit": "0" * 40,
            "nuscenes_version": "v1.0-trainval",
        },
    }


@pytest.fixture()
def world(tmp_path: Path):
    """One shard with one record, an image on disk, and a manifest entry."""
    record_dict = make_record("tok", "scene")
    record = DriveAlignRecord.from_dict(record_dict)
    shard_dir = tmp_path / "dataset" / "test"
    shard_dir.mkdir(parents=True)
    shard_path = shard_dir / "shard-0000.jsonl"
    shard_path.write_text(json.dumps(record_dict, sort_keys=True) + "\n", encoding="utf-8")

    dataroot = tmp_path / "nuscenes"
    image_path = dataroot / record_dict["model_inputs"]["frames"][3]["image_relpath"]
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (64, 64)).save(image_path)

    manifest = {
        "anchors": {
            "tok": {
                "split": "test",
                "shard": "test/shard-0000.jsonl",
                "line": 0,
                "record_hash": record.canonical_hash(),
                "request_hash_1f": serialize(
                    record.model_inputs, InputPolicy.ONE_FRAME
                ).canonical_hash(),
                "request_hash_4f": "1" * 64,
                "scene_token": "scene",
                "anchor_image_relpath": record_dict["model_inputs"]["frames"][3][
                    "image_relpath"
                ],
            }
        }
    }
    return tmp_path, record, manifest, image_path


class TestBuildBenchmarkRequest:
    def test_one_frame_policy_and_speed_format(self, world):
        tmp_path, record, manifest, image_path = world
        entry = manifest["anchors"]["tok"]
        request, model_request = build_benchmark_request(
            record, entry, dataroot=tmp_path / "nuscenes",
            policy=InputPolicy.ONE_FRAME, contract_version="v5",
        )
        assert request.image == image_path
        assert request.available_speed == (
            "5.200 m/s (backward difference from the previous keyframe)"
        )
        assert request.contract_version == "v5"
        # 1F: only the anchor (newest) frame is exposed.
        assert list(model_request.frame_tokens) == ["tok_f3"]
        assert model_request.image_relpaths == (
            record.model_inputs.frames[-1].image_relpath,
        )
        assert model_request.ego_speed_mps == pytest.approx(5.2)

    def test_request_hash_matches_manifest(self, world):
        tmp_path, record, manifest, _ = world
        _, model_request = build_benchmark_request(
            record, manifest["anchors"]["tok"], dataroot=tmp_path / "nuscenes",
            policy=InputPolicy.ONE_FRAME, contract_version="v5",
        )
        assert model_request.canonical_hash() == manifest["anchors"]["tok"]["request_hash_1f"]

    def test_prompt_has_zero_future_fingerprints(self, world):
        tmp_path, record, manifest, _ = world
        _, model_request = build_benchmark_request(
            record, manifest["anchors"]["tok"], dataroot=tmp_path / "nuscenes",
            policy=InputPolicy.ONE_FRAME, contract_version="v5",
        )
        fingerprints = future_fingerprints(record)
        hits = scan_future_fingerprints(
            fingerprints, {"prompt": model_request.prompt}
        )
        assert hits == []

    def test_hash_face_mismatch_raises(self, world):
        tmp_path, record, manifest, _ = world
        broken = {"anchors": {"tok": dict(manifest["anchors"]["tok"])}}
        broken["anchors"]["tok"]["request_hash_1f"] = "0" * 64
        with pytest.raises(ValueError, match="request hash mismatch"):
            build_benchmark_request(
                record, broken["anchors"]["tok"], dataroot=tmp_path / "nuscenes",
                policy=InputPolicy.ONE_FRAME, contract_version="v5",
            )

    def test_counterfactual_transform_creates_scratch_image(self, world):
        tmp_path, record, manifest, _ = world
        scratch = tmp_path / "scratch"
        for transform in ("blank", "shuffled"):
            request, _ = build_benchmark_request(
                record, manifest["anchors"]["tok"], dataroot=tmp_path / "nuscenes",
                policy=InputPolicy.ONE_FRAME, contract_version="v5",
                transform=transform, scratch_dir=scratch,
            )
            assert request.image.is_file()
            assert str(request.image).startswith(str(scratch))
        # Idempotent regeneration: same paths, no growth.
        blank_path = scratch / "counterfactual_images" / "blank" / "tok.jpg"
        assert blank_path.is_file()


class TestLoadRecordAndSelection:
    def test_load_record_verifies_hash(self, world):
        tmp_path, record, manifest, _ = world
        loaded = load_record(tmp_path / "dataset", manifest, "tok")
        assert loaded.sample_token == "tok"

    def test_load_record_hash_mismatch_raises(self, tmp_path, world):
        _, record, manifest, _ = world
        manifest["anchors"]["tok"]["record_hash"] = "f" * 64
        with pytest.raises(ValueError, match="record hash mismatch"):
            load_record(tmp_path / "dataset", manifest, "tok")

    def test_select_tokens_split_and_file(self, world):
        tmp_path, _, manifest, _ = world
        assert select_tokens(manifest, "test", None, None) == ["tok"]
        assert select_tokens(manifest, "train", None, None) == []
        anchors_file = tmp_path / "tokens.txt"
        anchors_file.write_text("# comment\ntok\n", encoding="utf-8")
        assert select_tokens(manifest, "test", str(anchors_file), None) == ["tok"]
        bad_file = tmp_path / "bad.txt"
        bad_file.write_text("ghost_token\n", encoding="utf-8")
        with pytest.raises(ValueError, match="missing from manifest"):
            select_tokens(manifest, "test", str(bad_file), None)


def test_ego_speed_line_exact_format():
    assert ego_speed_line(5.2) == (
        "5.200 m/s (backward difference from the previous keyframe)"
    )
    assert ego_speed_line(0.0) == (
        "0.000 m/s (backward difference from the previous keyframe)"
    )
