"""End-to-end unit tests for the Eval v2 evaluator + config pin (S13).

自包含合成世界（4 anchors / 2 shards / 2 scenes，与 v1 evaluator 测试同
构）：a/c 完美帧、b 过度保守帧（GT yield true 预测 false、GT KEEP_SPEED
预测 STOP、GT risk 预测缺席）、d parse 失败。覆盖：T6 完整性硬前置
（missing / extra 在任何产物写出之前 fail-fast）、T1/T2/T3 键集变化、
与 v1 evaluator 同世界输出的改名等值对照、直方图 / 空集块 /
denominators 复算、嵌套 bootstrap CI 形状、byte-identical 重跑，以及
仓库真实 eval_v1 / eval_v2 config 的逐值 pin。

Run:
    ``conda activate autovla_codeclean && cd /root/autodl-tmp/drivealign_workspace && \
    PYTHONPATH=DriveAlign/src python -m pytest DriveAlign/tests/unit/test_eval_evaluator_v2.py -q``
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from drivealign.evaluation.evaluator import run_evaluation
from drivealign.evaluation.evaluator_v2 import (
    _MOTION_STATES,
    _RISK_FACTORS,
    _SPEED_ACTIONS,
    run_evaluation_v2,
)
from drivealign.records.record import DriveAlignRecord

SPLIT = "test"
STATES = ["stationary", "same_direction", "oncoming", "crossing"]
ACTIONS = ["ACCELERATE", "KEEP_SPEED", "DECELERATE", "STOP"]
RISKS = [
    "pedestrian_crossing",
    "vehicle_merging",
    "lead_vehicle_braking",
    "corridor_conflict",
    "stationary_obstacle",
    "congestion",
    "oncoming_traffic",
]


def make_record(token: str, scene: str, expected: dict) -> dict:
    """最小可哈希 DriveAlignRecord（与 v1 evaluator 测试同构）。"""
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


def output(car_bbox, *, risks, yield_required, speed_action):
    return {
        "critical_objects": [
            {"category": "car", "bbox_2d": list(car_bbox), "motion_state": "stationary"}
        ],
        "risk_factors": list(risks),
        "reasoning": "r",
        "yield_required": yield_required,
        "speed_action": speed_action,
    }


def build_world(tmp_path: Path, eval_version: str):
    """4-anchor 世界：b 是过度保守帧，d parse 失败；config 按版本分发。"""
    records = {
        "tok_a": make_record("tok_a", "scene_1", output((0, 0, 10, 10), risks=[], yield_required=False, speed_action="KEEP_SPEED")),
        "tok_b": make_record("tok_b", "scene_1", output((0, 0, 10, 10), risks=["corridor_conflict"], yield_required=True, speed_action="KEEP_SPEED")),
        "tok_c": make_record("tok_c", "scene_2", output((0, 0, 10, 10), risks=[], yield_required=False, speed_action="KEEP_SPEED")),
        "tok_d": make_record("tok_d", "scene_2", output((0, 0, 10, 10), risks=[], yield_required=False, speed_action="KEEP_SPEED")),
    }
    dataset_root = tmp_path / "dataset"
    shard_of = {
        "tok_a": "test/shard-0000.jsonl", "tok_b": "test/shard-0000.jsonl",
        "tok_c": "test/shard-0001.jsonl", "tok_d": "test/shard-0001.jsonl",
    }
    manifest_anchors = {}
    for shard in sorted(set(shard_of.values())):
        shard_records = [t for t in sorted(records) if shard_of[t] == shard]
        (dataset_root / shard).parent.mkdir(parents=True, exist_ok=True)
        (dataset_root / shard).write_text(
            "".join(json.dumps(records[t], sort_keys=True) + "\n" for t in shard_records),
            encoding="utf-8",
        )
        for line_no, token in enumerate(shard_records):
            manifest_anchors[token] = {
                "split": SPLIT,
                "shard": shard,
                "line": line_no,
                "record_hash": DriveAlignRecord.from_dict(records[token]).canonical_hash(),
                "request_hash_1f": "0" * 64,
                "request_hash_4f": "1" * 64,
                "scene_token": records[token]["scene_token"],
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
        # 过度保守帧：GT yield true / KEEP_SPEED / corridor_conflict，
        # 预测 yield false / STOP / 无 risk
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

    config = {
        "status": "provisional",
        "eval_version": eval_version,
        "evaluation_split": SPLIT,
        "matching": {"iou_thresholds": [0.3, 0.5, 0.7], "primary_iou_threshold": 0.5},
        "taxonomies": {
            "motion_states": STATES,
            "speed_actions": ACTIONS,
            "risk_factors": RISKS,
            "categories": ["car"],
        },
        "max_items": 8,
        "bootstrap": {"n_resamples": 50, "seed": 7, "confidence_level": 0.95},
        "counterfactual": {"subset_size": 200, "transforms": ["blank", "shuffled"]},
    }
    if eval_version == "eval_v2":
        config["risk_max_items"] = 8
    config_path = tmp_path / "eval_test.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    return dataset_root, manifest_path, predictions_path, config_path


class TestRunEvaluationV2:
    def test_report_content_and_artifacts(self, tmp_path):
        dataset_root, manifest_path, predictions_path, config_path = build_world(
            tmp_path, "eval_v2"
        )
        report = run_evaluation_v2(
            predictions_path, manifest_path, dataset_root, config_path, tmp_path / "out"
        )
        assert (tmp_path / "out" / "eval_report.json").is_file()
        assert (tmp_path / "out" / "eval_report.md").is_file()
        assert (tmp_path / "out" / "anchor_scores.jsonl").is_file()
        # T6 漏斗：anchors_scored 退役
        assert report["coverage"] == {
            "anchors_expected": 4, "parse_ok": 3, "parse_failed": 1,
        }
        assert report["parse_rate"] == pytest.approx(3 / 4)
        # T1/T2/T3：旧键退役、新键在位（手算值）
        assert "motion_state_accuracy" not in report
        assert "overconservative_yield_rate" not in report
        assert "overconservative_stop_rate" not in report
        assert report["motion_state_f1_macro"] == pytest.approx(0.25)
        assert report["yield_miss_rate"] == pytest.approx(1.0)
        assert report["yield_overconservative_rate"] == 0.0
        assert report["stop_overconservative_rate"] == pytest.approx(1.0 / 3.0)
        assert report["stop_miss_rate"] == 0.0
        assert report["object_f1_micro@0.5"] == pytest.approx(1.0)
        assert report["meta"]["eval_version"] == "eval_v2"
        # 直方图全桶（对象与 risk 均 0..8）
        for key in ("n_pred_objects", "n_gt_objects", "n_gt_risk_factors",
                    "n_pred_risk_factors"):
            assert sorted(report[key], key=int) == [str(k) for k in range(9)]
        # 空集方向：对象全非空；risk 侧 b GT 非空预测空
        assert report["object_empty_direction_counts"]["gt_nonempty_pred_nonempty"] == 3
        assert report["risk_empty_direction_counts"]["gt_nonempty_pred_empty"] == 1
        assert report["risk_empty_miss_rate"] == pytest.approx(1.0)

    def test_t6_missing_predictions_fail_fast_without_artifacts(self, tmp_path):
        dataset_root, manifest_path, predictions_path, config_path = build_world(
            tmp_path, "eval_v2"
        )
        text = predictions_path.read_text(encoding="utf-8")
        truncated = tmp_path / "truncated.jsonl"
        truncated.write_text(
            "".join(line for line in text.splitlines(True) if "tok_a" not in line),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="missing predictions"):
            run_evaluation_v2(
                truncated, manifest_path, dataset_root, config_path, tmp_path / "out"
            )
        # 硬前置在任何文件写出之前：不得产生任何产物
        assert not (tmp_path / "out" / "eval_report.json").exists()
        assert not (tmp_path / "out" / "anchor_scores.jsonl").exists()
        assert not (tmp_path / "out" / "eval_report.md").exists()

    def test_t6_extra_predictions_fail_fast(self, tmp_path):
        dataset_root, manifest_path, predictions_path, config_path = build_world(
            tmp_path, "eval_v2"
        )
        text = predictions_path.read_text(encoding="utf-8")
        extra_row = json.dumps(
            {"sample_token": "tok_z", "parse_ok": True, "status": "ok", "errors": [],
             "telemetry": None, "output": output((0, 0, 10, 10), risks=[],
                                                 yield_required=False,
                                                 speed_action="KEEP_SPEED")}
        ) + "\n"
        extended = tmp_path / "extended.jsonl"
        extended.write_text(text + extra_row, encoding="utf-8")
        with pytest.raises(ValueError, match="extra predictions"):
            run_evaluation_v2(
                extended, manifest_path, dataset_root, config_path, tmp_path / "out"
            )
        assert not (tmp_path / "out" / "eval_report.json").exists()

    def test_renamed_keys_equal_v1_on_same_world(self, tmp_path):
        """改名等值对照：同世界、同 config 参数下，v2 的新名率值与 v1 的
        旧名率值逐值相同（T2/T3 只改名 + 补方向，数值定义不变）。"""
        dataset_root, manifest_path, predictions_path, config_path = build_world(
            tmp_path, "eval_v1"
        )
        v1_report = run_evaluation(
            predictions_path, manifest_path, dataset_root, config_path, tmp_path / "out_v1"
        )
        # 同一份世界改跑 v2：仅换 eval_version / risk_max_items 键
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        config["eval_version"] = "eval_v2"
        config["risk_max_items"] = 8
        config_path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
        v2_report = run_evaluation_v2(
            predictions_path, manifest_path, dataset_root, config_path, tmp_path / "out_v2"
        )
        assert v1_report["overconservative_yield_rate"] == pytest.approx(
            v2_report["yield_overconservative_rate"]
        )
        assert v1_report["overconservative_stop_rate"] == pytest.approx(
            v2_report["stop_overconservative_rate"]
        )
        for key in ("yield_required_f1", "speed_action_f1_macro",
                    "risk_factors_f1_macro", "max_items_hit_rate"):
            assert v1_report[key] == pytest.approx(v2_report[key]), key
        # parse_rate：本世界 anchors_scored == anchors_expected，两版数值等价
        assert v1_report["parse_rate"] == pytest.approx(v2_report["parse_rate"])

    def test_rerun_is_byte_identical(self, tmp_path):
        dataset_root, manifest_path, predictions_path, config_path = build_world(
            tmp_path, "eval_v2"
        )
        run_evaluation_v2(
            predictions_path, manifest_path, dataset_root, config_path, tmp_path / "out_a"
        )
        run_evaluation_v2(
            predictions_path, manifest_path, dataset_root, config_path, tmp_path / "out_b"
        )
        assert (tmp_path / "out_a" / "eval_report.json").read_bytes() == (
            tmp_path / "out_b" / "eval_report.json"
        ).read_bytes()
        assert (tmp_path / "out_a" / "anchor_scores.jsonl").read_bytes() == (
            tmp_path / "out_b" / "anchor_scores.jsonl"
        ).read_bytes()

    def test_bootstrap_ci_shape_scalar_and_nested(self, tmp_path):
        dataset_root, manifest_path, predictions_path, config_path = build_world(
            tmp_path, "eval_v2"
        )
        report = run_evaluation_v2(
            predictions_path, manifest_path, dataset_root, config_path, tmp_path / "out"
        )
        boot = report["bootstrap_ci"]
        scalars = {k: v for k, v in boot.items() if "point_estimate" in v}
        nested = {k: v for k, v in boot.items() if "point_estimate" not in v}
        # 22 个标量键 + 4 个固定类别集嵌套键
        assert len(scalars) == 22
        assert sorted(nested) == [
            "motion_state_f1_per_class",
            "object_f1_per_category@0.5",
            "risk_factors_f1_per_class",
            "speed_action_per_class_f1",
        ]
        # 标量点估计与报告同名指标一致（同一定义的重算）
        for key in ("yield_overconservative_rate", "stop_overconservative_rate",
                    "motion_state_f1_macro"):
            assert scalars[key]["point_estimate"] == pytest.approx(report[key]), key
        # 嵌套块：类别集固定 + 八字段
        assert sorted(nested["motion_state_f1_per_class"]) == sorted(STATES)
        assert sorted(nested["speed_action_per_class_f1"]) == sorted(ACTIONS)
        assert sorted(nested["risk_factors_f1_per_class"]) == sorted(RISKS)
        assert sorted(nested["object_f1_per_category@0.5"]) == ["car"]
        for block in nested.values():
            for entry in block.values():
                assert set(entry) == {
                    "point_estimate", "ci_low", "ci_high", "n_scenes", "n_values",
                    "n_resamples", "seed", "confidence_level",
                }

    def test_counterfactual_block(self, tmp_path):
        dataset_root, manifest_path, predictions_path, config_path = build_world(
            tmp_path, "eval_v2"
        )
        counterfactual = tmp_path / "predictions_blank.jsonl"
        flipped = output((0, 0, 10, 10), risks=[], yield_required=False,
                         speed_action="ACCELERATE")
        row = {"sample_token": "tok_a", "parse_ok": True, "status": "ok", "errors": [],
               "telemetry": None, "output": flipped}
        counterfactual.write_text(json.dumps(row) + "\n", encoding="utf-8")
        report = run_evaluation_v2(
            predictions_path, manifest_path, dataset_root, config_path, tmp_path / "out",
            counterfactual_paths={"blank": counterfactual},
        )
        assert report["visual_dependence_action_flip_blank"] == pytest.approx(1.0)
        assert report["counterfactual"]["blank"]["n_compared"] == 1


class TestConfigPin:
    """仓库真实 config pin：eval_v2 与 eval_v1 共享块逐值相同，且与
    evaluator 模块内冻结枚举一致（防 config / 代码漂移）。"""

    @staticmethod
    def _configs():
        config_dir = Path(__file__).resolve().parents[2] / "configs" / "evaluation"
        v1 = yaml.safe_load((config_dir / "eval_v1.yaml").read_text(encoding="utf-8"))
        v2 = yaml.safe_load((config_dir / "eval_v2.yaml").read_text(encoding="utf-8"))
        return v1, v2

    def test_v2_shared_blocks_identical_to_v1(self):
        v1, v2 = self._configs()
        for key in ("evaluation_split", "matching", "taxonomies", "max_items",
                    "bootstrap", "counterfactual"):
            assert v2[key] == v1[key], key

    def test_v2_specific_fields(self):
        v1, v2 = self._configs()
        assert v2["risk_max_items"] == 8  # contract v7 risk_factors maxItems
        assert "gate_thresholds" not in v2  # 2026-10-09 裁定：v2 取消 gate
        assert v1["status"] == "frozen"
        assert v2["status"] == "frozen"  # 2026-10-09 冻结（yaml 内注释）
        assert v1["eval_version"] == "eval_v1"
        assert v2["eval_version"] == "eval_v2"

    def test_module_enums_match_config_taxonomies(self):
        _, v2 = self._configs()
        assert tuple(v2["taxonomies"]["motion_states"]) == _MOTION_STATES
        assert tuple(v2["taxonomies"]["speed_actions"]) == _SPEED_ACTIONS
        assert tuple(v2["taxonomies"]["risk_factors"]) == _RISK_FACTORS

    def test_bootstrap_preregistered_values(self):
        _, v2 = self._configs()
        assert v2["bootstrap"] == {
            "n_resamples": 2000, "seed": 20261003, "confidence_level": 0.95,
        }
