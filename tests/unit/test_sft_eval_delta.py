"""S12 Step 1 单测：`cli/sft_eval_delta.py`（paired-delta + 预注册守卫）。

覆盖 §3 Step 1 的 `test_sft_eval_delta`：
    ① 合成 `anchor_scores` 上的 paired-delta 数值正确（固定种子、手算可验）；
    ② 确定性：同输入两次运行逐字节一致（含写出的报告文件）；
    ③ 守恒与口径：CI 夹住 point delta；缺输入的行显式 `evaluated=false`。

启动命令：
    ``conda activate autovla_codeclean && PYTHONPATH=DriveAlign/src python -m \
    pytest DriveAlign/tests/unit/test_sft_eval_delta.py -q``
"""

from __future__ import annotations

import json

import pytest

from drivealign.cli.sft_eval_delta import (
    MAJORITY_SHARE_CEILING,
    compute_paired_delta,
    constant_predictor_reading,
    evaluate_g4,
    evaluate_guards,
    load_anchor_scores,
    main,
)

# 6 个锚点 / 3 个 scene（与 test_eval_bootstrap 同构，便于人脑核对簇结构）。
TOKENS = ["a1", "a2", "b1", "c1", "c2", "c3"]
CLUSTERS = {"a1": "s1", "a2": "s1", "b1": "s2", "c1": "s3", "c2": "s3", "c3": "s3"}
RESAMPLES = dict(n_resamples=64, seed=17)


def _score(
    token: str,
    *,
    parse_ok: bool = True,
    gt_yield: bool = True,
    pred_yield: bool = True,
    speed_gt: str = "KEEP_SPEED",
    speed_pred: str = "KEEP_SPEED",
    tp: int = 0,
    fp: int = 0,
    fn: int = 0,
) -> dict:
    """一条 `score_anchor` 形状的合成评分行（字段集与 evaluator 产物一致）。"""
    if not parse_ok:
        return {
            "sample_token": token,
            "parse_ok": False,
            "status": "generation_failure",
            "parse_error_categories": ["json_decode"],
        }
    return {
        "sample_token": token,
        "parse_ok": True,
        "status": "ok",
        "per_tau": {"0.5": {"tp": tp, "fp": fp, "fn": fn}},
        "matched_ious": [],
        "matched_categories": [],
        "unmatched_gt_categories": ["car"] * fn,
        "unmatched_pred_categories": ["car"] * fp,
        "motion_pairs": [],
        "gt_yield_required": gt_yield,
        "pred_yield_required": pred_yield,
        "gt_speed_action": speed_gt,
        "pred_speed_action": speed_pred,
        "gt_risk_factors": [],
        "pred_risk_factors": [],
        "n_pred_objects": 0,
        "n_gt_objects": 0,
        "max_items_hit": False,
    }


def _scores(**kwargs) -> dict[str, dict]:
    return {token: _score(token, **kwargs) for token in TOKENS}


def _prediction(token: str, action: str) -> dict:
    return {
        "sample_token": token,
        "parse_ok": True,
        "status": "ok",
        "output": {"speed_action": action, "yield_required": False},
    }


class TestComputePairedDelta:
    def test_hand_computable_yield_delta(self):
        """SFT 让行判定全对（F1=1），Base 一律不让行（F1=0）→ delta=1，CI 退化到点。

        GT 全为 True 保证任何非空 replicate 子集上两侧的 F1 都分别是 1 / 0，
        故 delta 的分布是常量 1（CI 上下界都等于 1）——这是可手算核对的强断言。
        """
        sft = {token: _score(token, gt_yield=True, pred_yield=True) for token in TOKENS}
        base = {token: _score(token, gt_yield=True, pred_yield=False) for token in TOKENS}
        delta = compute_paired_delta(sft, base, CLUSTERS, **RESAMPLES)

        cell = delta["yield_required_f1"]
        assert cell["point_a"] == pytest.approx(1.0)
        assert cell["point_b"] == pytest.approx(0.0)
        assert cell["delta"] == pytest.approx(1.0)
        assert cell["delta_ci_low"] == pytest.approx(1.0)
        assert cell["delta_ci_high"] == pytest.approx(1.0)

    def test_identical_systems_have_zero_delta(self):
        scores = _scores(tp=1)
        delta = compute_paired_delta(scores, scores, CLUSTERS, **RESAMPLES)
        for cell in delta.values():
            assert cell["delta"] == pytest.approx(0.0)
            # CI 必须夹住点估计（口径自洽）。
            assert cell["delta_ci_low"] <= cell["delta"] <= cell["delta_ci_high"]

    def test_deterministic_given_seed(self):
        sft = _scores(tp=1)
        base = _scores(tp=0, fp=1)
        first = compute_paired_delta(sft, base, CLUSTERS, **RESAMPLES)
        second = compute_paired_delta(sft, base, CLUSTERS, **RESAMPLES)
        assert first == second

    def test_uses_only_shared_tokens(self):
        """只在一侧出现的 token 不参与配对（否则配对 bootstrap 会 KeyError）。"""
        sft = _scores(tp=1)
        base = dict(_scores(tp=0, fp=1))
        base["zz"] = _score("zz", tp=0, fp=1)
        delta = compute_paired_delta(sft, base, CLUSTERS, **RESAMPLES)
        assert delta["parse_rate"]["n_values"] == len(TOKENS)


class TestConstantPredictorReading:
    def test_majority_share_and_class_counts(self):
        predictions = {
            "t1": _prediction("t1", "DECELERATE"),
            "t2": _prediction("t2", "DECELERATE"),
            "t3": _prediction("t3", "DECELERATE"),
            "t4": _prediction("t4", "STOP"),
            # parse 失败的行不计入分母（与内容指标同口径）。
            "t5": {"sample_token": "t5", "parse_ok": False, "status": "generation_failure"},
        }
        reading = constant_predictor_reading(predictions)
        assert reading["n_parsed"] == 4
        assert reading["majority_class"] == "DECELERATE"
        assert reading["majority_share"] == pytest.approx(0.75)
        assert reading["class_counts"] == {"DECELERATE": 3, "STOP": 1}


class TestEvaluateG4:
    def test_pass_when_parse_ok_and_primary_improves(self):
        sft = _scores(tp=1)
        base = _scores(tp=0, fp=1)
        g4 = evaluate_g4(compute_paired_delta(sft, base, CLUSTERS, **RESAMPLES))
        # parse_rate 两侧都是 1.0 → 阈值 = max(0.99, 1.0) = 1.0 → 达标。
        assert g4["parse_rate"]["threshold"] == pytest.approx(1.0)
        assert g4["parse_rate"]["passed"] is True
        assert g4["primary"]["improves"] is True
        assert g4["passed"] is True

    def test_fail_when_no_metric_improves(self):
        scores = _scores(tp=1)
        g4 = evaluate_g4(compute_paired_delta(scores, scores, CLUSTERS, **RESAMPLES))
        assert g4["primary"]["improves"] is False
        assert g4["secondary"]["improves"] is False
        assert g4["passed"] is False


def _delta_cell(point_a: float, point_b: float, low: float, high: float) -> dict:
    return {
        "point_a": point_a,
        "point_b": point_b,
        "delta": point_a - point_b,
        "delta_ci_low": low,
        "delta_ci_high": high,
    }


def _guards_delta(*, yield_low: float = -0.01, yield_high: float = 0.01) -> dict:
    """只喂 evaluate_guards 实际索引的两行（视觉依赖走 change_deltas 参数）。"""
    return {
        "overconservative_yield_rate": _delta_cell(0.83, 0.83, yield_low, yield_high),
        "overconservative_stop_rate": _delta_cell(0.30, 0.30, -0.01, 0.01),
    }


def _flip_cell(sft: float, base: float, low: float, *, high: float = 0.05) -> dict:
    return {
        "point_a": sft,
        "point_b": base,
        "delta": sft - base,
        "delta_ci_low": low,
        "delta_ci_high": high,
        "n_values": 200,
    }


class TestEvaluateGuards:
    """守卫方向口径（见 `evaluate_guards` docstring 的冻结文档差异说明）：

    - 越大越坏（overconservative）：CI **下界 > 0**（显著变大）→ 触发；
      显著变小（改善）**不**触发。
    - 越小越坏（视觉依赖）：CI **上界 < 0**（显著变小）→ 触发。
    """

    def test_pass_when_nothing_triggers(self):
        change_deltas = {
            "action_flip_blank": _flip_cell(0.15, 0.147, -0.01),
            "action_flip_shuffled": _flip_cell(0.18, 0.174, -0.01),
            "output_change_blank": _flip_cell(0.20, 0.19, -0.01),
            "output_change_shuffled": _flip_cell(0.22, 0.21, -0.01),
        }
        guards = evaluate_guards(
            _guards_delta(),
            constant_predictor={"majority_class": "DECELERATE", "majority_share": 0.75},
            change_deltas=change_deltas,
        )
        assert guards["all_evaluated"] is True
        assert guards["triggered"] == []
        assert guards["passed"] is True
        assert guards["rows"]["constant_predictor"]["share_passed"] is True

    def test_overconservative_significant_increase_triggers(self):
        # CI 完全在 0 之上 → 过度保守显著变多 → 用怂换分 → 触发。
        guards = evaluate_guards(_guards_delta(yield_low=0.002, yield_high=0.05))
        row = guards["rows"]["overconservative_yield_rate"]
        assert row["criterion"] == "delta_ci_low > 0"
        assert row["triggered"] is True
        # 未给反事实/predictions 输入 → 其余行未评估，整体不判"通过/失败"。
        assert guards["passed"] is None
        assert guards["any_triggered"] is True

    def test_overconservative_significant_decrease_does_not_trigger(self):
        # CI 完全在 0 之下（过度保守显著减少 = 不再用怂换分）→ 不得误判为退化。
        guards = evaluate_guards(_guards_delta(yield_low=-0.05, yield_high=-0.002))
        assert guards["rows"]["overconservative_yield_rate"]["triggered"] is False

    def test_visual_dependence_lower_ci_triggers(self):
        # action_flip(blank) 的 CI 完全在 0 之下 → 显著变小（不看图了）→ 触发。
        change_deltas = {
            "action_flip_blank": _flip_cell(0.10, 0.147, -0.05, high=-0.002),
            "action_flip_shuffled": _flip_cell(0.18, 0.174, -0.01, high=0.05),
            "output_change_blank": _flip_cell(0.20, 0.19, -0.01, high=0.05),
            "output_change_shuffled": _flip_cell(0.22, 0.21, -0.01, high=0.05),
        }
        guards = evaluate_guards(
            _guards_delta(),
            constant_predictor={"majority_class": "DECELERATE", "majority_share": 0.75},
            change_deltas=change_deltas,
        )
        row = guards["rows"]["visual_dependence_action_flip_blank"]
        assert row["criterion"] == "delta_ci_high < 0"
        assert row["triggered"] is True
        assert guards["passed"] is False
        assert "visual_dependence_action_flip_blank" in guards["triggered"]

    def test_constant_predictor_triggered_by_high_majority_share(self):
        change_deltas = {
            "action_flip_blank": _flip_cell(0.15, 0.147, 0.0),
            "action_flip_shuffled": _flip_cell(0.18, 0.174, 0.0),
        }
        guards = evaluate_guards(
            _guards_delta(),
            constant_predictor={"majority_class": "DECELERATE", "majority_share": 0.95},
            change_deltas=change_deltas,
        )
        row = guards["rows"]["constant_predictor"]
        assert row["share_passed"] is False
        assert row["majority_share_ceiling"] == pytest.approx(MAJORITY_SHARE_CEILING)
        assert row["triggered"] is True

    def test_constant_predictor_triggered_by_flip_below_base(self):
        change_deltas = {
            "action_flip_blank": _flip_cell(0.10, 0.147, -0.01),
            "action_flip_shuffled": _flip_cell(0.18, 0.174, -0.01),
        }
        guards = evaluate_guards(
            _guards_delta(),
            constant_predictor={"majority_class": "DECELERATE", "majority_share": 0.75},
            change_deltas=change_deltas,
        )
        assert guards["rows"]["constant_predictor"]["flip_passed"] is False
        assert guards["rows"]["constant_predictor"]["triggered"] is True

    def test_missing_inputs_are_not_silently_passed(self):
        """缺反事实输入 → 对应行 evaluated=false，整体 passed=None（未判定）。"""
        guards = evaluate_guards(_guards_delta())
        assert guards["all_evaluated"] is False
        assert guards["passed"] is None
        assert guards["rows"]["constant_predictor"]["evaluated"] is False
        assert guards["rows"]["visual_dependence_action_flip_blank"]["evaluated"] is False


def _write_jsonl(path, rows) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )


class TestMainDeterminism:
    def test_report_byte_identical_across_runs(self, tmp_path):
        sft_rows = [_score(token, tp=1) for token in TOKENS]
        base_rows = [_score(token, tp=0, fp=1) for token in TOKENS]
        manifest = {
            "anchors": {token: {"scene_token": CLUSTERS[token]} for token in TOKENS}
        }
        sft_path = tmp_path / "sft_scores.jsonl"
        base_path = tmp_path / "base_scores.jsonl"
        manifest_path = tmp_path / "manifest.json"
        sft_pred_path = tmp_path / "sft_predictions.jsonl"
        base_pred_path = tmp_path / "base_predictions.jsonl"
        _write_jsonl(sft_path, sft_rows)
        _write_jsonl(base_path, base_rows)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        _write_jsonl(
            sft_pred_path, [_prediction(token, "DECELERATE") for token in TOKENS]
        )
        _write_jsonl(
            base_pred_path, [_prediction(token, "KEEP_SPEED") for token in TOKENS]
        )

        argv = [
            "--sft-scores", str(sft_path),
            "--base-scores", str(base_path),
            "--anchor-manifest", str(manifest_path),
            "--sft-predictions", str(sft_pred_path),
            "--base-predictions", str(base_pred_path),
            "--n-resamples", "32",
            "--seed", "7",
        ]
        out_one = tmp_path / "delta_one.json"
        out_two = tmp_path / "delta_two.json"
        assert main(argv + ["--out", str(out_one)]) == 0
        assert main(argv + ["--out", str(out_two)]) == 0
        assert out_one.read_bytes() == out_two.read_bytes()

        report = json.loads(out_one.read_text(encoding="utf-8"))
        assert report["meta"]["n_shared_anchors"] == len(TOKENS)
        assert report["constant_predictor"]["majority_class"] == "DECELERATE"
        assert report["constant_predictor"]["majority_share"] == pytest.approx(1.0)
        # 只给了 predictions、没给反事实 → 守卫未完全评估（不假装通过）。
        assert report["guards"]["passed"] is None
        assert report["change_deltas"] == {}

    def test_duplicate_scores_rejected(self, tmp_path):
        path = tmp_path / "dup.jsonl"
        _write_jsonl(path, [_score("t"), _score("t")])
        with pytest.raises(ValueError, match="duplicate"):
            load_anchor_scores(path)
