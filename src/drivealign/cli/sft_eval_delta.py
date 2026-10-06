"""S12 Step 7 对比 CLI（`cli/sft_eval_delta.py`，CPU，agent sandbox 执行）。

职责（`S12_implementation_plan.md` §2.7 / §2.8 / §3 Step 7）
----------------------------------------------------------
读 **SFT** 与 **Base** 两份 `anchor_scores.jsonl`（同一 test 锚集、各自
`cli/evaluate.py` 产物），在**同一 cluster map、同一 seed、同一 B** 下调用
`evaluation.bootstrap.bootstrap_paired_delta`，输出逐指标的
point_a / point_b / delta / delta CI。评测口径本身（`eval_v1.yaml`）与
`cli/evaluate.py` **零改动**——本 CLI 只做读数与配对，不改任何指标定义。

三块输出
--------
1. `paired_delta`：全指标配对差（`subset_statistics` 与 S09 报告同一实现，
   故 point 值与 `eval_report.json` 的读数一致）。
2. `g4`：§4 G4 判据的机械判定 = `parse_rate` ≥ max(0.99, Base) **且**
   P=`object_f1_micro@0.5` / Q=`speed_action_f1_macro` **至少一个**的
   paired-delta 95% CI 下界 > 0。
3. `guards`：§2.8 预注册退化守卫（**必须在看到 test 之前冻结**）。其中
   overconservative ×2 用 **paired-delta CI** 口径（越大越坏 → CI 下界 > 0 即触发）；
   视觉依赖 ×4（action_flip / output_change × blank / shuffled）同样用
   paired-delta CI（越小越坏 → CI 上界 < 0 即触发），其配对对象是**反事实预测**
   （需 `--sft-counterfactual-*` / `--base-counterfactual-*`）；
   **常量预测器守卫是唯一保留绝对值的行**（多数类占比 ≤ 0.867 且
   action_flip ≥ Base）——它要判的是"是否只是把 Base 的常量换成了另一个常量"，
   paired-delta 表达不了这个语义（CI 只能说"两系统有差异"）。
   缺输入时该行标 `evaluated: false`，整体 `guards.passed` 置 null（未判定），
   避免"没测就算过"。

   文档 §2.8 表已于 2026-10-07 原地更正（见 §6 决策台账「F 更正」）：取
   `delta = SFT − Base`，越**大**越坏 → CI **下界 > 0** 触发；越**小**越坏 →
   CI **上界 < 0** 触发。本实现与该表逐字一致，并在每行 `criterion` 字段回显所用判据。

必报项（S11 D3 登记）
---------------------
①多数类常量基线（`constant_predictor`，对比 Base 86.7%）；
②反事实 `action_flip_rate`（blank / shuffled，`guards` 内同源给出）。

判定性质
--------
本 CLI 的 g4 / guards 结论是**机械读数**；阶段判定在 Step 7 人工复核后写入
`S12_decision_log.md`（阈值/口径不得就地改写，要改走决策台账）。

启动命令（Step 7）：
    ``conda activate autovla_codeclean && PYTHONPATH=DriveAlign/src python -m \
    drivealign.cli.sft_eval_delta \
    --sft-scores runs/S12_sft_1f/anchor_scores.jsonl \
    --base-scores runs/S09_base_benchmark/eval_v6/anchor_scores.jsonl \
    --anchor-manifest data/face_manifests/v6/anchor_policy_manifest.json \
    --sft-predictions runs/S12_sft_1f/predictions.jsonl \
    --base-predictions runs/S09_base_benchmark/predictions.jsonl \
    --sft-counterfactual-blank runs/S12_sft_1f/predictions_cf_blank.jsonl \
    --sft-counterfactual-shuffled runs/S12_sft_1f/predictions_cf_shuffled.jsonl \
    --base-counterfactual-blank runs/S09_base_benchmark/predictions_cf_blank.jsonl \
    --base-counterfactual-shuffled runs/S09_base_benchmark/predictions_cf_shuffled.jsonl \
    --out runs/S12_sft_1f/paired_delta.json``
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from drivealign.evaluation.bootstrap import bootstrap_paired_delta
from drivealign.evaluation.evaluator import subset_statistics

#: 冻结 bootstrap 口径（逐值等于 `configs/evaluation/eval_v1.yaml` 的 bootstrap 段）。
#: 做成常量是为了让 Step 7 的"同一 cluster map、同一 seed、同一 B"成为默认行为。
DEFAULT_N_RESAMPLES = 2000
DEFAULT_SEED = 20261003
DEFAULT_CONFIDENCE_LEVEL = 0.95

#: G4 用到的两个指标（P / Q）与 parse_rate 的绝对下限（§4 G4）。
PRIMARY_METRIC = "object_f1_micro@0.5"      # P
SECONDARY_METRIC = "speed_action_f1_macro"  # Q
PARSE_RATE_FLOOR = 0.99

#: §2.8 常量预测器守卫的绝对值上界（Base 的多数类占比为 0.867）。
MAJORITY_SHARE_CEILING = 0.867

#: 视觉依赖守卫覆盖的反事实变换与两种变化口径。
VISUAL_TRANSFORMS = ("blank", "shuffled")
CHANGE_MODES = ("action_flip", "output_change")

#: 触发方向的两种口径（§2.8 表）。
_HIGHER_IS_WORSE = ("overconservative_yield_rate", "overconservative_stop_rate")


# ---------------------------------------------------------------------------
# 输入
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="S12 Step 7: paired-delta (SFT vs Base) + pre-registered guards"
    )
    parser.add_argument("--sft-scores", required=True, help="SFT anchor_scores.jsonl")
    parser.add_argument("--base-scores", required=True, help="Base anchor_scores.jsonl")
    parser.add_argument(
        "--anchor-manifest",
        required=True,
        help="anchor policy manifest (provides scene_token for scene-cluster pairing)",
    )
    parser.add_argument("--out", required=True, help="output paired-delta JSON path")
    parser.add_argument("--n-resamples", type=int, default=DEFAULT_N_RESAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--confidence-level", type=float, default=DEFAULT_CONFIDENCE_LEVEL)
    # 以下为 §2.8 守卫 / 必报项的输入；缺失时对应行标 evaluated=false。
    parser.add_argument("--sft-predictions", default=None, help="SFT predictions.jsonl")
    parser.add_argument("--base-predictions", default=None, help="Base predictions.jsonl")
    parser.add_argument("--sft-counterfactual-blank", default=None)
    parser.add_argument("--sft-counterfactual-shuffled", default=None)
    parser.add_argument("--base-counterfactual-blank", default=None)
    parser.add_argument("--base-counterfactual-shuffled", default=None)
    return parser.parse_args(argv)


def load_anchor_scores(path: str | Path) -> dict[str, dict[str, Any]]:
    """读 `anchor_scores.jsonl` → token -> 逐锚点评分行（重复 token 直接报错）。"""
    scores: dict[str, dict[str, Any]] = {}
    for line_number, raw in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines()
    ):
        if not raw.strip():
            continue
        row = json.loads(raw)
        token = str(row["sample_token"])
        if token in scores:
            raise ValueError(f"duplicate anchor score for {token} (line {line_number})")
        scores[token] = row
    if not scores:
        raise ValueError(f"anchor scores file is empty: {path}")
    return scores


def load_predictions(path: str | Path) -> dict[str, dict[str, Any]]:
    """读 predictions JSONL → token -> 行（与 `evaluator.load_predictions` 同语义）。"""
    rows: dict[str, dict[str, Any]] = {}
    for line_number, raw in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines()
    ):
        if not raw.strip():
            continue
        row = json.loads(raw)
        token = str(row.get("sample_token", row.get("sample_id", "")))
        if not token:
            raise ValueError(f"predictions row {line_number} has no sample token")
        if token in rows:
            raise ValueError(f"duplicate prediction for anchor {token}")
        rows[token] = row
    return rows


def scene_clusters(
    manifest: Mapping[str, Any], tokens: Sequence[str]
) -> dict[str, str]:
    """token -> scene_token（scene-cluster bootstrap 的配对单位）。"""
    anchors = manifest["anchors"]
    missing = [token for token in tokens if token not in anchors]
    if missing:
        raise ValueError(
            f"{len(missing)} tokens missing from the anchor manifest, e.g. {missing[:3]}"
        )
    return {token: str(anchors[token]["scene_token"]) for token in tokens}


# ---------------------------------------------------------------------------
# 配对差：全指标
# ---------------------------------------------------------------------------


def _score_statistic(scores: Mapping[str, Mapping[str, Any]]):
    """把 anchor_scores 表包成 bootstrap 的 statistic（与 S09 报告同一定义）。"""

    def statistic(tokens: Sequence[str]) -> dict[str, float]:
        return subset_statistics([scores[token] for token in tokens])

    return statistic


def compute_paired_delta(
    sft_scores: Mapping[str, Mapping[str, Any]],
    base_scores: Mapping[str, Mapping[str, Any]],
    clusters: Mapping[str, str],
    *,
    n_resamples: int,
    seed: int,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> dict[str, dict[str, Any]]:
    """两套 anchor_scores 的逐指标 paired-delta（同一 cluster map / seed / B）。

    只在**两侧都有评分行**的锚点上配对（交集）：paired bootstrap 要求每个 replicate
    两侧看到同一批 token，缺行会让一侧 statistic 抛 KeyError。交集大小写进报告。
    """
    shared = sorted(set(sft_scores) & set(base_scores))
    if not shared:
        raise ValueError("SFT and Base anchor scores share no sample token")
    shared_clusters = {token: str(clusters[token]) for token in shared}
    return bootstrap_paired_delta(
        shared_clusters,
        _score_statistic(sft_scores),
        _score_statistic(base_scores),
        n_resamples=n_resamples,
        seed=seed,
        confidence_level=confidence_level,
    )


# ---------------------------------------------------------------------------
# 配对差：反事实变化率（视觉依赖守卫）
# ---------------------------------------------------------------------------


def _change_indicator(
    predictions: Mapping[str, Mapping[str, Any]],
    counterfactual: Mapping[str, Mapping[str, Any]],
    *,
    mode: str,
) -> dict[str, float]:
    """逐锚点 0/1：主跑与反事实跑的差异指示（与 evaluator 的定义逐条一致）。

    - `action_flip`：两侧 parse_ok 且 `output.speed_action` 不同。
    - `output_change`：两侧 parse_ok 且整个 `output` 不同。

    分母口径与 `evaluator.counterfactual_comparison` 相同：只统计**两侧都 parse_ok**
    的锚点（未 parse 的锚点不构成有效比较对，不计入分母）。
    """
    indicator: dict[str, float] = {}
    for token, main in predictions.items():
        other = counterfactual.get(token)
        if other is None:
            continue
        if not (main.get("parse_ok") and other.get("parse_ok")):
            continue
        if mode == "action_flip":
            differs = main["output"]["speed_action"] != other["output"]["speed_action"]
        elif mode == "output_change":
            differs = main["output"] != other["output"]
        else:
            raise ValueError(f"unknown change mode: {mode}")
        indicator[token] = 1.0 if differs else 0.0
    return indicator


def compute_change_delta(
    sft_predictions: Mapping[str, Mapping[str, Any]],
    base_predictions: Mapping[str, Mapping[str, Any]],
    sft_counterfactual: Mapping[str, Mapping[str, Any]],
    base_counterfactual: Mapping[str, Mapping[str, Any]],
    clusters: Mapping[str, str],
    *,
    mode: str,
    n_resamples: int,
    seed: int,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> dict[str, Any]:
    """SFT vs Base 的**变化率** paired-delta（同一批锚点、同一 cluster map）。"""
    sft_indicator = _change_indicator(sft_predictions, sft_counterfactual, mode=mode)
    base_indicator = _change_indicator(base_predictions, base_counterfactual, mode=mode)
    shared = sorted(set(sft_indicator) & set(base_indicator))
    if not shared:
        raise ValueError(f"no shared anchors for mode={mode}; cannot compute CI")
    shared_clusters = {token: str(clusters[token]) for token in shared}

    def make(table: Mapping[str, float]):
        def statistic(tokens: Sequence[str]) -> dict[str, float]:
            return {"rate": sum(table[token] for token in tokens) / len(tokens)}

        return statistic

    return bootstrap_paired_delta(
        shared_clusters,
        make(sft_indicator),
        make(base_indicator),
        n_resamples=n_resamples,
        seed=seed,
        confidence_level=confidence_level,
    )["rate"]


# ---------------------------------------------------------------------------
# 必报项①：常量预测器读数
# ---------------------------------------------------------------------------


def constant_predictor_reading(
    predictions: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """SFT 的 `speed_action` 分布与多数类占比（"是否只是换了个常量预测器"）。

    只统计 parse_ok 行（与内容指标同分母口径）。平票用类名字典序取大者，保证
    读数确定性（seed 无关）。
    """
    counts: dict[str, int] = {}
    for row in predictions.values():
        if not row.get("parse_ok") or row.get("output") is None:
            continue
        action = str(row["output"]["speed_action"])
        counts[action] = counts.get(action, 0) + 1
    total = sum(counts.values())
    if not total:
        raise ValueError("no parse-ok predictions; cannot read the constant-predictor baseline")
    majority_class, majority_count = max(counts.items(), key=lambda kv: (kv[1], kv[0]))
    return {
        "n_parsed": total,
        "class_counts": dict(sorted(counts.items())),
        "majority_class": majority_class,
        "majority_share": majority_count / total,
    }


# ---------------------------------------------------------------------------
# G4 判定
# ---------------------------------------------------------------------------


def evaluate_g4(
    delta: Mapping[str, Mapping[str, Any]],
    *,
    parse_rate_floor: float = PARSE_RATE_FLOOR,
) -> dict[str, Any]:
    """§4 G4 = parse_rate ≥ max(0.99, Base) 且 P/Q 至少一个 paired CI 下界 > 0。"""
    sft_parse = float(delta["parse_rate"]["point_a"])
    base_parse = float(delta["parse_rate"]["point_b"])
    threshold = max(parse_rate_floor, base_parse)
    parse_ok = sft_parse >= threshold
    p_cell = delta[PRIMARY_METRIC]
    q_cell = delta[SECONDARY_METRIC]
    p_improves = float(p_cell["delta_ci_low"]) > 0.0
    q_improves = float(q_cell["delta_ci_low"]) > 0.0
    return {
        "passed": bool(parse_ok and (p_improves or q_improves)),
        "parse_rate": {
            "sft": sft_parse,
            "base": base_parse,
            "threshold": threshold,
            "passed": bool(parse_ok),
        },
        "primary": {
            "metric": PRIMARY_METRIC,
            "delta": p_cell["delta"],
            "delta_ci_low": p_cell["delta_ci_low"],
            "delta_ci_high": p_cell["delta_ci_high"],
            "improves": bool(p_improves),
        },
        "secondary": {
            "metric": SECONDARY_METRIC,
            "delta": q_cell["delta"],
            "delta_ci_low": q_cell["delta_ci_low"],
            "delta_ci_high": q_cell["delta_ci_high"],
            "improves": bool(q_improves),
        },
    }


# ---------------------------------------------------------------------------
# §2.8 退化守卫
# ---------------------------------------------------------------------------


def evaluate_guards(
    delta: Mapping[str, Mapping[str, Any]],
    *,
    constant_predictor: Mapping[str, Any] | None = None,
    change_deltas: Mapping[str, Mapping[str, Any]] | None = None,
    majority_share_ceiling: float = MAJORITY_SHARE_CEILING,
) -> dict[str, Any]:
    """§2.8 预注册退化守卫的机械判定（触发即退化）。

    - `overconservative_*`（越大越坏）：paired-delta 95% CI **下界 > 0** → 触发。
    - `visual_dependence_{action_flip,output_change}_{blank,shuffled}`（越小越坏）：
      paired-delta 95% CI **上界 < 0** → 触发。
    - **常量预测器**（唯一绝对值口径）：`majority_share ≤ 0.867` **且** 每个
      transform 的 SFT action_flip ≥ Base → 通过；否则触发。
      ⚠️ §2.8 表该行括号里写的是**通过**条件（它判的是"是否只换了个常量"，
      CI 无法表达该语义），故此处以通过条件取反得触发。

    §2.8 表口径（2026-10-07 原地更正，见 §6 决策台账「F 更正」）：取
    `delta = SFT − Base`，越**大**越坏 → CI **下界 > 0** 触发；越**小**越坏 →
    CI **上界 < 0** 触发。每行 `criterion` 字段回显所用判据，便于 Step 7 复核。

    任何一行缺输入 → `evaluated: false`，整体 `passed` 置 None（未判定），
    防止"没测就算过"。
    """
    change_deltas = change_deltas or {}
    rows: dict[str, dict[str, Any]] = {}

    # (1) 过度保守代理（越大越坏）：delta = SFT - Base，CI 下界 > 0 → 显著变大 → 触发。
    for metric in _HIGHER_IS_WORSE:
        cell = delta[metric]
        rows[metric] = {
            "evaluated": True,
            "direction": "higher_is_worse",
            "criterion": "delta_ci_low > 0",
            "triggered": bool(float(cell["delta_ci_low"]) > 0.0),
            "sft": cell["point_a"],
            "base": cell["point_b"],
            "delta": cell["delta"],
            "delta_ci_low": cell["delta_ci_low"],
            "delta_ci_high": cell["delta_ci_high"],
        }

    # (2) 视觉依赖（越小越坏）：CI 上界 < 0 → 显著变小（不看图了）→ 触发。
    for mode in CHANGE_MODES:
        for transform in VISUAL_TRANSFORMS:
            key = f"{mode}_{transform}"
            name = f"visual_dependence_{key}"
            cell = change_deltas.get(key)
            if cell is None:
                rows[name] = {
                    "evaluated": False,
                    "reason": "missing counterfactual predictions for this transform",
                    "direction": "lower_is_worse",
                }
                continue
            rows[name] = {
                "evaluated": True,
                "direction": "lower_is_worse",
                "criterion": "delta_ci_high < 0",
                "triggered": bool(float(cell["delta_ci_high"]) < 0.0),
                "sft": cell["point_a"],
                "base": cell["point_b"],
                "delta": cell["delta"],
                "delta_ci_low": cell["delta_ci_low"],
                "delta_ci_high": cell["delta_ci_high"],
                "n_values": cell["n_values"],
            }

    # (3) 常量预测器（唯一绝对值行）。
    flip_cells = {
        transform: change_deltas.get(f"action_flip_{transform}")
        for transform in VISUAL_TRANSFORMS
    }
    if constant_predictor is None or any(cell is None for cell in flip_cells.values()):
        rows["constant_predictor"] = {
            "evaluated": False,
            "reason": "missing --sft-predictions or counterfactual action_flip inputs",
            "direction": "absolute_value_guard",
        }
    else:
        share = float(constant_predictor["majority_share"])
        share_pass = share <= majority_share_ceiling
        per_transform = {
            transform: {
                "sft_action_flip": cell["point_a"],
                "base_action_flip": cell["point_b"],
                "passed": bool(float(cell["point_a"]) >= float(cell["point_b"])),
            }
            for transform, cell in flip_cells.items()
        }
        flip_pass = all(item["passed"] for item in per_transform.values())
        rows["constant_predictor"] = {
            "evaluated": True,
            "direction": "absolute_value_guard",
            "criterion": "majority_share <= 0.867 AND sft action_flip >= base (per transform)",
            "majority_class": constant_predictor["majority_class"],
            "majority_share": share,
            "majority_share_ceiling": majority_share_ceiling,
            "share_passed": bool(share_pass),
            "flip_passed": bool(flip_pass),
            "transforms": per_transform,
            "triggered": bool(not (share_pass and flip_pass)),
        }

    evaluated_rows = [row for row in rows.values() if row["evaluated"]]
    triggered = sorted(name for name, row in rows.items() if row.get("triggered"))
    all_evaluated = all(row["evaluated"] for row in rows.values())
    return {
        "passed": (not triggered) if all_evaluated else None,
        "all_evaluated": bool(all_evaluated),
        "any_triggered": bool(triggered),
        "triggered": triggered,
        "n_evaluated": len(evaluated_rows),
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# 编排
# ---------------------------------------------------------------------------


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    """组装 Step 7 的完整对比报告（纯函数，便于单测直接调用）。"""
    manifest = json.loads(Path(args.anchor_manifest).read_text(encoding="utf-8"))
    sft_scores = load_anchor_scores(args.sft_scores)
    base_scores = load_anchor_scores(args.base_scores)

    shared = sorted(set(sft_scores) & set(base_scores))
    clusters = scene_clusters(manifest, shared)
    delta = compute_paired_delta(
        sft_scores,
        base_scores,
        clusters,
        n_resamples=args.n_resamples,
        seed=args.seed,
        confidence_level=args.confidence_level,
    )

    # 反事实变化率配对（视觉依赖守卫 + 常量预测器守卫的 flip 分量）。
    change_deltas: dict[str, dict[str, Any]] = {}
    cf_pairs = (
        ("blank", args.sft_counterfactual_blank, args.base_counterfactual_blank),
        ("shuffled", args.sft_counterfactual_shuffled, args.base_counterfactual_shuffled),
    )
    need_predictions = args.sft_predictions is not None and args.base_predictions is not None
    if need_predictions:
        sft_predictions = load_predictions(args.sft_predictions)
        base_predictions = load_predictions(args.base_predictions)
        for transform, sft_cf_path, base_cf_path in cf_pairs:
            if not (sft_cf_path and base_cf_path):
                continue
            sft_cf = load_predictions(sft_cf_path)
            base_cf = load_predictions(base_cf_path)
            # 该反事实对的锚点 = 主跑与反事实跑四份文件都有的 token（scene 簇由
            # manifest 提供），保证两边 flip 指示量在同一批锚点上配对。
            pair_tokens = sorted(
                (set(sft_predictions) & set(base_predictions))
                & (set(sft_cf) & set(base_cf))
            )
            pair_clusters = scene_clusters(manifest, pair_tokens)
            for mode in CHANGE_MODES:
                change_deltas[f"{mode}_{transform}"] = compute_change_delta(
                    sft_predictions,
                    base_predictions,
                    sft_cf,
                    base_cf,
                    pair_clusters,
                    mode=mode,
                    n_resamples=args.n_resamples,
                    seed=args.seed,
                    confidence_level=args.confidence_level,
                )

    constant = constant_predictor_reading(sft_predictions) if need_predictions else None

    return {
        "meta": {
            "sft_scores": str(args.sft_scores),
            "base_scores": str(args.base_scores),
            "anchor_manifest": str(args.anchor_manifest),
            "n_shared_anchors": len(shared),
            "n_resamples": int(args.n_resamples),
            "seed": int(args.seed),
            "confidence_level": float(args.confidence_level),
            "change_deltas_computed": sorted(change_deltas),
        },
        "paired_delta": delta,
        "g4": evaluate_g4(delta),
        "guards": evaluate_guards(
            delta, constant_predictor=constant, change_deltas=change_deltas
        ),
        "constant_predictor": constant,
        "change_deltas": change_deltas,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(args)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "out": str(out_path),
                "g4_passed": report["g4"]["passed"],
                "guards_passed": report["guards"]["passed"],
                "guards_triggered": report["guards"]["triggered"],
            },
            sort_keys=True,
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
