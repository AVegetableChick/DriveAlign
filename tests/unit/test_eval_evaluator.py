"""End-to-end unit tests for the Eval v1 evaluator (S09 plan section 2.4).

Builds a synthetic four-anchor dataset (two shards, two scenes), a matching
anchor manifest with true record hashes, and synthetic predictions; asserts
coverage semantics, hash verification, report content, and byte-level
determinism of a rerun.

Run:
    ``conda activate autovla_codeclean && cd /root/autodl-tmp/drivealign_workspace && \
    PYTHONPATH=DriveAlign/src python -m pytest DriveAlign/tests/unit/test_eval_evaluator.py -q``
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from drivealign.evaluation.evaluator import run_evaluation
from drivealign.records.record import DriveAlignRecord

SPLIT = "test"


def make_record(token: str, scene: str, expected: dict) -> dict:
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
        "training_targets": {"expected_output": expected, "language_reference": None},
        "oracle_only": {
            "future_ego_poses": [
                {
                    "sample_token": f"{token}_future",
                    "timestamp_us": 1_510_000_003_000_000,
                    "x": 10.0,
                    "y": 0.5,
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


def output(car_bbox, *, risks, yield_required, speed_action, n_objects=1):
    return {
        "critical_objects": [
            {"category": "car", "bbox_2d": list(car_bbox), "motion_state": "stationary"}
        ][:n_objects],
        "risk_factors": list(risks),
        "reasoning": "r",
        "yield_required": yield_required,
        "speed_action": speed_action,
    }


def build_world(tmp_path: Path):
    """Return (dataset_root, manifest_path, predictions_path, config_path)."""
    records = {
        "tok_a": make_record("tok_a", "scene_1", output((0, 0, 10, 10), risks=[], yield_required=False, speed_action="KEEP_SPEED")),
        "tok_b": make_record("tok_b", "scene_1", output((0, 0, 10, 10), risks=["corridor_conflict"], yield_required=True, speed_action="KEEP_SPEED")),
        "tok_c": make_record("tok_c", "scene_2", output((0, 0, 10, 10), risks=[], yield_required=False, speed_action="KEEP_SPEED")),
        "tok_d": make_record("tok_d", "scene_2", output((0, 0, 10, 10), risks=[], yield_required=False, speed_action="KEEP_SPEED")),
    }
    dataset_root = tmp_path / "dataset"
    manifest_anchors = {}
    shard_of = {"tok_a": "test/shard-0000.jsonl", "tok_b": "test/shard-0000.jsonl",
                "tok_c": "test/shard-0001.jsonl", "tok_d": "test/shard-0001.jsonl"}
    lines: dict[str, dict[str, int]] = {}
    for shard in sorted(set(shard_of.values())):
        shard_records = [t for t in sorted(records) if shard_of[t] == shard]
        (dataset_root / shard).parent.mkdir(parents=True, exist_ok=True)
        (dataset_root / shard).write_text(
            "".join(json.dumps(records[t], sort_keys=True) + "\n" for t in shard_records),
            encoding="utf-8",
        )
        for line_no, token in enumerate(shard_records):
            lines[token] = {"shard": shard, "line": line_no}
    for token, record in records.items():
        manifest_anchors[token] = {
            "split": SPLIT,
            "shard": lines[token]["shard"],
            "line": lines[token]["line"],
            "record_hash": DriveAlignRecord.from_dict(record).canonical_hash(),
            "request_hash_1f": "0" * 64,
            "request_hash_4f": "1" * 64,
            "scene_token": record["scene_token"],
            "anchor_image_relpath": f"samples/CAM_FRONT/{token}_3.jpg",
        }
    manifest_path = tmp_path / "anchor_policy_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "anchors": manifest_anchors,
                "counts": {SPLIT: len(records), "total": len(records)},
                "contract_version": "v5",
            }
        ),
        encoding="utf-8",
    )

    predictions_path = tmp_path / "predictions.jsonl"
    rows = [
        {"sample_token": "tok_a", "parse_ok": True, "status": "ok", "errors": [],
         "telemetry": {"generation_seconds": 1.0, "input_tokens": 900,
                       "peak_cuda_memory_bytes": 8_000_000_000},
         "output": output((0, 0, 10, 10), risks=[], yield_required=False, speed_action="KEEP_SPEED")},
        {"sample_token": "tok_b", "parse_ok": True, "status": "ok", "errors": [],
         "telemetry": {"generation_seconds": 2.0, "input_tokens": 950,
                       "peak_cuda_memory_bytes": 8_000_000_000},
         "output": output((0, 0, 10, 10), risks=[], yield_required=False, speed_action="STOP")},
        {"sample_token": "tok_c", "parse_ok": True, "status": "ok", "errors": [],
         "telemetry": {"generation_seconds": 3.0, "input_tokens": 1000,
                       "peak_cuda_memory_bytes": 8_000_000_000},
         "output": output((0, 0, 10, 10), risks=[], yield_required=False, speed_action="KEEP_SPEED")},
        {"sample_token": "tok_d", "parse_ok": False, "status": "schema_failure",
         "errors": [{"category": "schema_failure"}], "telemetry": None, "output": None},
    ]
    predictions_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )

    config_path = tmp_path / "eval_test.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "status": "provisional",
                "eval_version": "eval_v1",
                "evaluation_split": SPLIT,
                "matching": {"iou_thresholds": [0.5], "primary_iou_threshold": 0.5},
                "taxonomies": {
                    "motion_states": ["stationary", "same_direction", "oncoming", "crossing"],
                    "speed_actions": ["ACCELERATE", "KEEP_SPEED", "DECELERATE", "STOP"],
                    "risk_factors": ["corridor_conflict", "congestion", "pedestrian_crossing"],
                    "categories": ["car"],
                },
                "max_items": 8,
                "bootstrap": {"n_resamples": 50, "seed": 7, "confidence_level": 0.95},
                "counterfactual": {"subset_size": 200, "transforms": ["blank", "shuffled"]},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return dataset_root, manifest_path, predictions_path, config_path


class TestRunEvaluation:
    def test_report_content_and_artifacts(self, tmp_path):
        dataset_root, manifest_path, predictions_path, config_path = build_world(tmp_path)
        report = run_evaluation(
            predictions_path, manifest_path, dataset_root, config_path, tmp_path / "out"
        )
        assert (tmp_path / "out" / "eval_report.json").is_file()
        assert (tmp_path / "out" / "eval_report.md").is_file()
        assert (tmp_path / "out" / "anchor_scores.jsonl").is_file()
        assert report["coverage"] == {
            "anchors_expected": 4, "anchors_scored": 4,
            "parse_ok": 3, "parse_failed": 1,
            "missing_predictions": 0, "extra_predictions": 0,
        }
        assert report["parse_rate"] == pytest.approx(3 / 4)
        # tok_a/tok_c/tok_d GT vs prediction: three matched cars; tok_d excluded.
        assert report["object_f1_micro@0.5"] == pytest.approx(1.0)
        # tok_b: GT KEEP_SPEED vs pred STOP (over-conservative stop); yield
        # GT true predicted false. Oc-stop denominator = GT keep/accel frames
        # {a, b, c}; numerator = {b} -> 1/3.
        assert report["overconservative_stop_rate"] == pytest.approx(1.0 / 3.0)
        assert report["cost_summary"]["gpu_hours"] == pytest.approx(6.0 / 3600.0)
        assert report["meta"]["eval_version"] == "eval_v1"
        assert len(report["bootstrap_ci"]) > 0

    def test_rerun_is_byte_identical(self, tmp_path):
        dataset_root, manifest_path, predictions_path, config_path = build_world(tmp_path)
        run_evaluation(
            predictions_path, manifest_path, dataset_root, config_path, tmp_path / "out_a"
        )
        run_evaluation(
            predictions_path, manifest_path, dataset_root, config_path, tmp_path / "out_b"
        )
        assert (tmp_path / "out_a" / "eval_report.json").read_bytes() == (
            tmp_path / "out_b" / "eval_report.json"
        ).read_bytes()
        assert (tmp_path / "out_a" / "anchor_scores.jsonl").read_bytes() == (
            tmp_path / "out_b" / "anchor_scores.jsonl"
        ).read_bytes()

    def test_record_hash_mismatch_raises(self, tmp_path):
        dataset_root, manifest_path, predictions_path, config_path = build_world(tmp_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["anchors"]["tok_a"]["record_hash"] = "f" * 64
        broken = tmp_path / "broken_manifest.json"
        broken.write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(ValueError, match="record hash mismatch"):
            run_evaluation(
                predictions_path, broken, dataset_root, config_path, tmp_path / "out"
            )

    def test_duplicate_predictions_raise(self, tmp_path):
        dataset_root, manifest_path, predictions_path, config_path = build_world(tmp_path)
        duplicated = tmp_path / "duplicated.jsonl"
        text = predictions_path.read_text(encoding="utf-8")
        duplicated.write_text(text + text.splitlines(True)[0], encoding="utf-8")
        with pytest.raises(ValueError, match="duplicate prediction"):
            run_evaluation(
                duplicated, manifest_path, dataset_root, config_path, tmp_path / "out"
            )

    def test_counterfactual_block(self, tmp_path):
        dataset_root, manifest_path, predictions_path, config_path = build_world(tmp_path)
        counterfactual = tmp_path / "predictions_blank.jsonl"
        flipped = output((0, 0, 10, 10), risks=[], yield_required=False, speed_action="ACCELERATE")
        row = {"sample_token": "tok_a", "parse_ok": True, "status": "ok", "errors": [],
               "telemetry": None, "output": flipped}
        counterfactual.write_text(json.dumps(row) + "\n", encoding="utf-8")
        report = run_evaluation(
            predictions_path, manifest_path, dataset_root, config_path, tmp_path / "out",
            counterfactual_paths={"blank": counterfactual},
        )
        # Only tok_a exists in both files (parse-ok pairs required).
        assert report["visual_dependence_action_flip_blank"] == pytest.approx(1.0)
        assert report["visual_dependence_output_change_blank"] == pytest.approx(1.0)
        assert report["counterfactual"]["blank"]["n_compared"] == 1
