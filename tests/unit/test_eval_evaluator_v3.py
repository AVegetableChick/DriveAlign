"""End-to-end unit tests for the Eval v3 evaluator + key-set/config pin (S13).

自包含合成世界（4 anchors / 2 shards / 2 scenes，与 v2 evaluator 测试
同构，记录为 v7 形状：critical_objects 带 bev_x/bev_y/id，risk_factors
对象化）：a/c 完美帧（含 BEV 配对与归因 TP）、b 部分帧（GT 对象漏检
→ FN、预测多报因子被 conditional 闸门挡下）、d parse 失败。覆盖：
G7 键集 pin（51 标量 + 1 嵌套块逐键 = 草案 §5 清单，v3 − v2 恰为其
差集）、词表同源 pin（eval_v3.yaml ↔ contract v7 risk_taxonomy.json
↔ 模块冻结枚举，import 契约资产防漂移）、config 共享块与 v2 逐值同、
T6 完整性硬前置、bootstrap 新键（+5：bev 三 mean + 归因 micro/macro）
与确定性（同 config 重跑逐字节相同）、空集口径、counterfactual 块
不加 BEV/归因（§6-5 维持默认）、v1 score_anchor 不带 v3 字段（v2
路径行为不变红线）。

Run:
    ``conda activate autovla_codeclean && cd /root/autodl-tmp/drivealign_workspace && \
    PYTHONPATH=DriveAlign/src python -m pytest DriveAlign/tests/unit/test_eval_evaluator_v3.py -q``
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from drivealign.evaluation.evaluator_v3 import (
    _RISK_FACTORS_V7,
    aggregate_v2,
    aggregate_v3,
    run_evaluation_v3,
    score_anchor_v3,
)
from drivealign.evaluation.metrics import score_anchor
from drivealign.records.record import DriveAlignRecord

SPLIT = "test"
STATES = ["stationary", "same_direction", "oncoming", "crossing"]
ACTIONS = ["ACCELERATE", "KEEP_SPEED", "DECELERATE", "STOP"]
RISKS_V7 = list(_RISK_FACTORS_V7)
IoU = [0.3, 0.5, 0.7]
PRIMARY = 0.5
MAX_ITEMS = 8

# ---------------------------------------------------------------------------
# 冻结新增键集（草案 §5 清单逐键枚举；2026-10-14 用户裁定 51 标量 + 1
# 嵌套块，文档「50」为算术勘误）
# ---------------------------------------------------------------------------

_FAMILIES = ("dx", "dy", "dxy")
_BEV_STATS = ("mean", "p50", "p90")
_BUCKETS = ("0_10", "10_25", "25_40", "40p")

NEW_SCALAR_KEYS = (
    {f"bev_{family}_{stat}" for family in _FAMILIES for stat in _BEV_STATS}  # 9
    | {
        f"bev_{family}_{stat}_bucket_{bucket}"
        for family in _FAMILIES
        for stat in _BEV_STATS
        for bucket in _BUCKETS
    }  # 36
    | {
        f"bev_signed_{axis}_{direction}_mean"
        for axis in ("dx", "dy")
        for direction in ("pos", "neg")
    }  # 4
    | {"risk_attribution_f1_micro", "risk_attribution_f1_macro"}  # 2
)
NEW_NESTED_KEY = "risk_attribution_per_factor"  # 嵌套块


# ---------------------------------------------------------------------------
# 合成世界（v7 形状：bev_x/bev_y/id + 对象化 risk_factors；expected_output
# 四字段无 reasoning，V7-C1）
# ---------------------------------------------------------------------------


def make_record(token: str, scene: str, expected: dict) -> dict:
    """最小可哈希 DriveAlignRecord（v7 provenance，与 v2 测试同构）。"""
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
            "contract_version": "v7",
            "temporal_policy_version": "window-v1",
            "builder_commit": "0" * 40,
            "nuscenes_version": "v1.0-trainval",
        },
    }


def v7_object(number, *, object_id, bev_x, bev_y):
    return {
        "category": "car",
        "bbox_2d": [number * 100, 0, number * 100 + 10, 10],
        "id": object_id,
        "motion_state": "stationary",
        "bev_x": bev_x,
        "bev_y": bev_y,
    }


def build_world(tmp_path: Path):
    """4-anchor v7 世界：b 帧 GT 对象漏检（→FN）+ 预测多报因子（→闸门
    挡下）；d parse 失败。两侧 id 空间刻意错开（GT 0/1，pred 3/7）。"""
    records = {
        "tok_a": make_record(
            "tok_a",
            "scene_1",
            {
                "critical_objects": [v7_object(0, object_id=0, bev_x=10.0, bev_y=0.0)],
                "risk_factors": [{"factor": "stationary_obstacle", "objects": [0]}],
                "yield_required": False,
                "speed_action": "KEEP_SPEED",
            },
        ),
        "tok_b": make_record(
            "tok_b",
            "scene_1",
            {
                "critical_objects": [
                    v7_object(0, object_id=0, bev_x=5.0, bev_y=0.0),
                    v7_object(2, object_id=1, bev_x=25.0, bev_y=1.0),  # 预测漏检
                ],
                "risk_factors": [
                    {"factor": "vru_crossing", "objects": [0, 1]},
                    {"factor": "object_near", "objects": [0]},
                ],
                "yield_required": True,
                "speed_action": "KEEP_SPEED",
            },
        ),
        "tok_c": make_record(
            "tok_c",
            "scene_2",
            {
                "critical_objects": [],
                "risk_factors": [],
                "yield_required": False,
                "speed_action": "KEEP_SPEED",
            },
        ),
        "tok_d": make_record(
            "tok_d",
            "scene_2",
            {
                "critical_objects": [],
                "risk_factors": [],
                "yield_required": False,
                "speed_action": "KEEP_SPEED",
            },
        ),
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
                "contract_version": "v7",
            }
        ),
        encoding="utf-8",
    )

    predictions_path = tmp_path / "predictions.jsonl"
    rows = [
        {  # a：完美帧——对象匹配 + 因子/对象指认全对 + BEV 偏移 (0.2, -0.2)
            "sample_token": "tok_a", "parse_ok": True, "status": "ok", "errors": [],
            "telemetry": {"generation_seconds": 1.0, "input_tokens": 900,
                          "peak_cuda_memory_bytes": 8_000_000_000},
            "output": {
                "critical_objects": [v7_object(0, object_id=3, bev_x=10.2, bev_y=-0.2)],
                "risk_factors": [{"factor": "stationary_obstacle", "objects": [3]}],
                "yield_required": False,
                "speed_action": "KEEP_SPEED",
            },
        },
        {  # b：对象漏检（gt1 无预测）+ 多报 oncoming_traffic（GT 无，闸门挡下）
            "sample_token": "tok_b", "parse_ok": True, "status": "ok", "errors": [],
            "telemetry": {"generation_seconds": 2.0, "input_tokens": 950,
                          "peak_cuda_memory_bytes": 8_000_000_000},
            "output": {
                "critical_objects": [v7_object(0, object_id=7, bev_x=5.4, bev_y=0.2)],
                "risk_factors": [
                    {"factor": "vru_crossing", "objects": [7]},
                    {"factor": "oncoming_traffic", "objects": [7]},
                ],
                "yield_required": False,
                "speed_action": "STOP",
            },
        },
        {  # c：完美空帧
            "sample_token": "tok_c", "parse_ok": True, "status": "ok", "errors": [],
            "telemetry": {"generation_seconds": 3.0, "input_tokens": 1000,
                          "peak_cuda_memory_bytes": 8_000_000_000},
            "output": {
                "critical_objects": [],
                "risk_factors": [],
                "yield_required": False,
                "speed_action": "KEEP_SPEED",
            },
        },
        {  # d：parse 失败
            "sample_token": "tok_d", "parse_ok": False, "status": "schema_failure",
            "errors": [{"category": "schema_failure"}], "telemetry": None, "output": None,
        },
    ]
    predictions_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )

    config_path = tmp_path / "eval_v3_test.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "status": "provisional",
                "eval_version": "eval_v3",
                "evaluation_split": SPLIT,
                "matching": {"iou_thresholds": IoU, "primary_iou_threshold": PRIMARY},
                "taxonomies": {
                    "motion_states": STATES,
                    "speed_actions": ACTIONS,
                    "risk_factors": RISKS_V7,
                    "categories": ["car"],
                },
                "max_items": 8,
                "risk_max_items": 8,
                "bootstrap": {"n_resamples": 50, "seed": 7, "confidence_level": 0.95},
                "counterfactual": {"subset_size": 200, "transforms": ["blank", "shuffled"]},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return dataset_root, manifest_path, predictions_path, config_path


def v3_scores(dataset_root, manifest_path, predictions_path, config_path):
    """跑一次 score_anchor_v3 批次（供聚合层键集 pin 复用）。"""
    import yaml as _yaml

    config = _yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    predictions = {
        json.loads(line)["sample_token"]: json.loads(line)
        for line in Path(predictions_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    from drivealign.evaluation.evaluator import load_gt_outputs

    gt_outputs = load_gt_outputs(dataset_root, manifest, sorted(predictions))
    return [
        score_anchor_v3(
            gt_outputs[token],
            predictions[token],
            iou_thresholds=[float(t) for t in config["matching"]["iou_thresholds"]],
            primary_iou_threshold=float(config["matching"]["primary_iou_threshold"]),
            max_items=int(config["max_items"]),
        )
        for token in sorted(predictions)
    ]


class TestRunEvaluationV3:
    @pytest.fixture()
    def world_report(self, tmp_path):
        dataset_root, manifest_path, predictions_path, config_path = build_world(tmp_path)
        report = run_evaluation_v3(
            predictions_path, manifest_path, dataset_root, config_path, tmp_path / "out"
        )
        return report, tmp_path / "out"

    def test_report_artifacts_and_new_keys_present(self, world_report):
        """三产物写出；51 标量新键 + 嵌套块全部在报告顶层。"""
        report, out = world_report
        assert (out / "eval_report.json").is_file()
        assert (out / "eval_report.md").is_file()
        assert (out / "anchor_scores.jsonl").is_file()
        for key in NEW_SCALAR_KEYS:
            assert key in report, key
        assert NEW_NESTED_KEY in report

    def test_attribution_readings_on_synthetic_world(self, world_report):
        """归因读数手算钉死：a 帧 tp=1；b 帧 vru_crossing tp=1/fn=1
        （gt1 漏检），多报 oncoming_traffic 被闸门挡下，object_near 因
        预测未报不计入。"""
        report, _ = world_report
        per_factor = report["risk_attribution_per_factor"]
        assert per_factor == {
            "stationary_obstacle": {
                "tp": 1, "fp": 0, "fn": 0, "precision": 1.0, "recall": 1.0, "f1": 1.0,
            },
            "vru_crossing": {
                "tp": 1, "fp": 0, "fn": 1, "precision": 1.0, "recall": 0.5, "f1": 2 / 3,
            },
        }
        # micro 池化全原子：tp=2（a+b）、fp=0、fn=1（gt1）→ P=1, R=2/3, F1=0.8。
        assert report["risk_attribution_f1_micro"] == pytest.approx(0.8)
        assert report["risk_attribution_f1_macro"] == pytest.approx((1.0 + 2 / 3) / 2)

    def test_bev_readings_on_synthetic_world(self, world_report):
        """BEV 读数手算钉死：两匹配对 (10.2,-0.2)/(10,0) 与
        (5.4,0.2)/(5,0) → dx 误差 {0.2, 0.4}、dy {0.2, 0.2}；
        d_gt=10 → 10_25 桶、d_gt≈5 → 0_10 桶。"""
        report, _ = world_report
        assert report["denominators"]["bev_pairs"] == 2
        assert report["bev_dx_mean"] == pytest.approx(0.3)
        assert report["bev_dy_mean"] == pytest.approx(0.2)
        assert report["bev_dxy_mean"] == pytest.approx(
            ((0.2 ** 2 + 0.2 ** 2) ** 0.5 + (0.4 ** 2 + 0.2 ** 2) ** 0.5) / 2
        )
        # 分桶：d_gt=10.0（a 帧）→ 10_25；d_gt=5.0（b 帧）→ 0_10。
        assert report["bev_dx_mean_bucket_10_25"] == pytest.approx(0.2)
        assert report["bev_dx_mean_bucket_0_10"] == pytest.approx(0.4)
        # signed 方向分解：dx 有符号误差 +0.2/+0.4（全正向）；dy 误差
        # -0.2（a 帧，pred_y=-0.2）与 +0.2（b 帧）正负各一。
        assert report["bev_signed_dx_pos_mean"] == pytest.approx(0.3)
        assert report["bev_signed_dx_neg_mean"] == 0.0
        assert report["bev_signed_dy_pos_mean"] == pytest.approx(0.2)
        assert report["bev_signed_dy_neg_mean"] == pytest.approx(-0.2)

    def test_anchor_scores_carry_v3_fields(self, world_report):
        """anchor_scores.jsonl：parse-ok 行带 bev_pairs 与
        risk_attribution_counts（bootstrap 消费面）；parse 失败行不带。"""
        _, out = world_report
        rows = {
            json.loads(line)["sample_token"]: json.loads(line)
            for line in (out / "anchor_scores.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        assert rows["tok_a"]["bev_pairs"] == [[10.2, -0.2, 10.0, 0.0]]
        assert rows["tok_a"]["risk_attribution_counts"] == {
            "stationary_obstacle": [1, 0, 0]
        }
        assert rows["tok_b"]["risk_attribution_counts"] == {"vru_crossing": [1, 0, 1]}
        assert "bev_pairs" not in rows["tok_d"]

    def test_t6_missing_predictions_fail_fast_without_artifacts(self, tmp_path):
        """T6 完整性硬前置（v2 同款）：missing 在任何产物写出前 raise。"""
        dataset_root, manifest_path, predictions_path, config_path = build_world(tmp_path)
        rows = Path(predictions_path).read_text(encoding="utf-8").splitlines()
        Path(predictions_path).write_text(
            "".join(line + "\n" for line in rows if '"tok_b"' not in line),
            encoding="utf-8",
        )
        out_dir = tmp_path / "out"
        with pytest.raises(ValueError, match="eval_v3 completeness precondition"):
            run_evaluation_v3(predictions_path, manifest_path, dataset_root, config_path, out_dir)
        assert not out_dir.exists()

    def test_t6_extra_predictions_fail_fast(self, tmp_path):
        """T6：extra predictions 同样 fail-fast。"""
        dataset_root, manifest_path, predictions_path, config_path = build_world(tmp_path)
        extra = json.dumps(
            {"sample_token": "tok_zzz", "parse_ok": True, "status": "ok",
             "errors": [], "telemetry": None, "output": None},
            sort_keys=True,
        )
        with Path(predictions_path).open("a", encoding="utf-8") as handle:
            handle.write(extra + "\n")
        with pytest.raises(ValueError, match="extra predictions"):
            run_evaluation_v3(predictions_path, manifest_path, dataset_root, config_path, tmp_path / "out")

    def test_rerun_is_byte_identical(self, tmp_path):
        """确定性（G7-4 承载面之一）：同 config 重跑三产物逐字节相同
        （含 bootstrap 同种子复现）。"""
        dataset_root, manifest_path, predictions_path, config_path = build_world(tmp_path)
        run_evaluation_v3(predictions_path, manifest_path, dataset_root, config_path, tmp_path / "out1")
        run_evaluation_v3(predictions_path, manifest_path, dataset_root, config_path, tmp_path / "out2")
        for name in ("eval_report.json", "anchor_scores.jsonl", "eval_report.md"):
            assert (tmp_path / "out1" / name).read_bytes() == (tmp_path / "out2" / name).read_bytes(), name

    def test_bootstrap_new_keys_and_exclusions(self, world_report):
        """bootstrap 新键 pin：+5 键（bev 三 mean + 归因 micro/macro）
        在 bootstrap_ci；BEV p50/p90、分桶、signed、per_factor 嵌套块
        不入 bootstrap（草案 §2.5/§3.5）。v2 基数 20 → v3 25（草案
        「10→15」为增量记法勘误，随台账登记）。"""
        report, _ = world_report
        boot = report["bootstrap_ci"]
        for key in (
            "bev_dx_mean", "bev_dy_mean", "bev_dxy_mean",
            "risk_attribution_f1_micro", "risk_attribution_f1_macro",
        ):
            assert key in boot and "ci_low" in boot[key], key
        excluded = sorted(
            key for key in boot
            if key.startswith("bev_") and key not in (
                "bev_dx_mean", "bev_dy_mean", "bev_dxy_mean"
            )
        )
        assert excluded == []
        assert NEW_NESTED_KEY not in boot

    def test_counterfactual_block_has_no_bev_or_attribution(self, tmp_path):
        """counterfactual battery 不加 BEV/归因（§6-5 维持默认）：对比块
        结构与 v2 相同，仅 action flip / output change 两率。"""
        dataset_root, manifest_path, predictions_path, config_path = build_world(tmp_path)
        cf_path = tmp_path / "cf_blank.jsonl"
        cf_path.write_text(Path(predictions_path).read_text(encoding="utf-8"))
        report = run_evaluation_v3(
            predictions_path, manifest_path, dataset_root, config_path,
            tmp_path / "out", counterfactual_paths={"blank": cf_path},
        )
        for _transform, block in report["counterfactual"].items():
            assert not any("bev" in key or "attribution" in key for key in block)
            assert "action_flip_rate" in block and "output_change_rate" in block


class TestKeySetPin:
    def test_v3_minus_v2_keys_exactly_the_frozen_51_plus_one(self, tmp_path):
        """G7-2 键集 pin：aggregate_v3 − aggregate_v2 恰为 51 标量 + 1
        嵌套块（2026-10-14 裁定按草案 §5 清单枚举；文档「50」为勘误）。"""
        dataset_root, manifest_path, predictions_path, config_path = build_world(tmp_path)
        scores = v3_scores(dataset_root, manifest_path, predictions_path, config_path)
        taxonomies = {
            "motion_states": STATES, "speed_actions": ACTIONS, "risk_factors": RISKS_V7,
        }
        common = dict(
            n_anchors_expected=4, taxonomies=taxonomies,
            primary_iou_threshold=PRIMARY, iou_thresholds=IoU,
            max_items=MAX_ITEMS, risk_max_items=8,
        )
        v3_keys = set(aggregate_v3(scores, **common))
        v2_keys = set(aggregate_v2(scores, **common))
        assert v3_keys - v2_keys == NEW_SCALAR_KEYS | {NEW_NESTED_KEY}
        assert len(NEW_SCALAR_KEYS) == 51
        # v2 报告键在 v3 下逐键保持（eval v3 = eval v2 + 增量）。
        assert v2_keys <= v3_keys

    def test_v1_score_anchor_has_no_v3_fields(self):
        """v2 路径行为不变红线：v1 score_anchor 返回不带 v3 字段
        （v3 字段只由 score_anchor_v3 追加）。"""
        gt = {
            "critical_objects": [v7_object(0, object_id=0, bev_x=10.0, bev_y=0.0)],
            "risk_factors": [{"factor": "object_near", "objects": [0]}],
            "yield_required": False,
            "speed_action": "KEEP_SPEED",
        }
        row = {
            "sample_token": "t", "parse_ok": True, "status": "ok", "errors": [],
            "output": {
                "critical_objects": [v7_object(0, object_id=3, bev_x=10.0, bev_y=0.0)],
                "risk_factors": [{"factor": "object_near", "objects": [3]}],
                "yield_required": False,
                "speed_action": "KEEP_SPEED",
            },
        }
        score = score_anchor(gt, row, iou_thresholds=IoU, primary_iou_threshold=PRIMARY, max_items=MAX_ITEMS)
        assert "bev_pairs" not in score and "risk_attribution_counts" not in score
        score_v3 = score_anchor_v3(gt, row, iou_thresholds=IoU, primary_iou_threshold=PRIMARY, max_items=MAX_ITEMS)
        # 内存形状为 tuple（json 落盘后即 list，见 anchor_scores 测试）。
        assert score_v3["bev_pairs"] == [(10.0, 0.0, 10.0, 0.0)]
        assert score_v3["risk_attribution_counts"] == {"object_near": (1, 0, 0)}

    def test_no_match_slice_bev_all_zeros_attribution_still_counts(self):
        """空集口径（G7-3，BEV 侧）：零匹配切片 BEV 49 键全 0.0（不置
        null）。归因侧不随匹配空转空——因子名两侧都有时，幻觉引用 FP /
        未覆盖 GT FN 仍是原子（边界表口径），f1 因 tp=0 计 0.0。"""
        gt = {
            "critical_objects": [v7_object(0, object_id=0, bev_x=10.0, bev_y=0.0)],
            "risk_factors": [{"factor": "object_near", "objects": [0]}],
            "yield_required": False,
            "speed_action": "KEEP_SPEED",
        }
        row = {  # 对象bbox无交集 → 零匹配
            "sample_token": "t", "parse_ok": True, "status": "ok", "errors": [],
            "output": {
                "critical_objects": [v7_object(9, object_id=3, bev_x=90.0, bev_y=0.0)],
                "risk_factors": [{"factor": "object_near", "objects": [3]}],
                "yield_required": False,
                "speed_action": "KEEP_SPEED",
            },
        }
        score = score_anchor_v3(gt, row, iou_thresholds=IoU, primary_iou_threshold=PRIMARY, max_items=MAX_ITEMS)
        assert score["bev_pairs"] == []
        report = aggregate_v3(
            [score], n_anchors_expected=1,
            taxonomies={"motion_states": STATES, "speed_actions": ACTIONS, "risk_factors": RISKS_V7},
            primary_iou_threshold=PRIMARY, iou_thresholds=IoU,
            max_items=MAX_ITEMS, risk_max_items=8,
        )
        bev_keys = [key for key in report if key.startswith("bev_")]
        assert len(bev_keys) == 49
        assert all(report[key] == 0.0 for key in bev_keys)
        # 归因：引用无落点 FP + GT 未覆盖 FN → 有原子但 F1 = 0.0。
        assert report["risk_attribution_per_factor"]["object_near"]["fp"] == 1
        assert report["risk_attribution_per_factor"]["object_near"]["fn"] == 1
        assert report["risk_attribution_f1_micro"] == 0.0

    def test_no_atoms_slice_attribution_zero_and_empty_block(self):
        """空集口径（G7-3，归因侧）：两侧无对象无因子 → 无原子：两标量
        0.0、per_factor 空块（跳过不置 null）。"""
        empty = {
            "critical_objects": [],
            "risk_factors": [],
            "yield_required": False,
            "speed_action": "KEEP_SPEED",
        }
        score = score_anchor_v3(
            empty,
            {"sample_token": "t", "parse_ok": True, "status": "ok", "errors": [], "output": empty},
            iou_thresholds=IoU, primary_iou_threshold=PRIMARY, max_items=MAX_ITEMS,
        )
        report = aggregate_v3(
            [score], n_anchors_expected=1,
            taxonomies={"motion_states": STATES, "speed_actions": ACTIONS, "risk_factors": RISKS_V7},
            primary_iou_threshold=PRIMARY, iou_thresholds=IoU,
            max_items=MAX_ITEMS, risk_max_items=8,
        )
        assert report["risk_attribution_f1_micro"] == 0.0
        assert report["risk_attribution_f1_macro"] == 0.0
        assert report["risk_attribution_per_factor"] == {}


class TestConfigPin:
    """仓库真实 config pin：eval_v3 与 eval_v2 共享块逐值相同、risk
    词表与 contract v7 契约资产同源（import 契约资产校验，防漂移）。"""

    @staticmethod
    def _configs():
        config_dir = Path(__file__).resolve().parents[2] / "configs" / "evaluation"
        v2 = yaml.safe_load((config_dir / "eval_v2.yaml").read_text(encoding="utf-8"))
        v3 = yaml.safe_load((config_dir / "eval_v3.yaml").read_text(encoding="utf-8"))
        return v2, v3

    @staticmethod
    def _v7_taxonomy_order():
        taxonomy_path = (
            Path(__file__).resolve().parents[2] / "configs" / "contracts" / "v7" / "risk_taxonomy.json"
        )
        terms = json.loads(taxonomy_path.read_text(encoding="utf-8"))["terms"]
        return [term["canonical"] for term in terms]

    def test_v3_shared_blocks_identical_to_v2(self):
        """草案 §1 不变量：matching / bootstrap / counterfactual /
        max_items / risk_max_items / split / 其余枚举全部沿用 v2 冻结值。"""
        v2, v3 = self._configs()
        for key in ("evaluation_split", "matching", "max_items", "risk_max_items",
                    "bootstrap", "counterfactual"):
            assert v3[key] == v2[key], key
        for sub in ("motion_states", "speed_actions", "categories"):
            assert v3["taxonomies"][sub] == v2["taxonomies"][sub], sub

    def test_v3_risk_vocabulary_same_source_as_contract_v7(self):
        """词表同源 pin（代际红线）：eval_v3.yaml 六枚举 = contract v7
        risk_taxonomy canonical 顺序 = evaluator 冻结枚举，逐值逐序。"""
        v2, v3 = self._configs()
        contract_order = self._v7_taxonomy_order()
        assert v3["taxonomies"]["risk_factors"] == contract_order
        assert tuple(v3["taxonomies"]["risk_factors"]) == _RISK_FACTORS_V7
        # v6 七枚举是历史事实：v2 不动、v3 必换。
        assert len(v2["taxonomies"]["risk_factors"]) == 7
        assert len(v3["taxonomies"]["risk_factors"]) == 6

    def test_v3_version_fields(self):
        v2, v3 = self._configs()
        assert v3["eval_version"] == "eval_v3"
        assert v3["status"] == "frozen"       # 2026-10-15 冻结（base v7 eval v3 定稿报告口径）
        assert v2["status"] == "frozen"       # v2 byte-frozen 不动
        assert "gate_thresholds" not in v3

    def test_bootstrap_preregistered_values(self):
        _, v3 = self._configs()
        assert v3["bootstrap"] == {
            "n_resamples": 2000, "seed": 20261003, "confidence_level": 0.95,
        }
