"""CLI wrapper for the Eval evaluators (S09 plan section 2.1; S13 v2 dispatch).

Example launch command:
    ``conda activate autovla_codeclean && cd /root/autodl-tmp/drivealign_workspace && \
    PYTHONPATH=DriveAlign/src python -m drivealign.cli.evaluate \
    --predictions runs/S09_base_benchmark/predictions.jsonl \
    --anchor-manifest data/face_manifests/v6/anchor_policy_manifest.json \
    --dataset-root data/dataset_v4 \
    --config DriveAlign/configs/evaluation/eval_v1.yaml \
    --out runs/S09_base_benchmark \
    --counterfactual-blank runs/S09_base_benchmark/predictions_blank.jsonl \
    --counterfactual-shuffled runs/S09_base_benchmark/predictions_shuffled.jsonl``

按 config 的 ``eval_version`` 分发：``eval_v1`` -> v1 evaluator（历史基线，
    stdout 打 ``anchors_scored``）；``eval_v2`` -> v2 evaluator（S13 口径，
    stdout 打 ``anchors_expected``）；``eval_v3`` -> v3 evaluator（S13 BEV +
    归因 + 词表 v7 化）。其余值 fail-fast。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from drivealign.evaluation.evaluator import run_evaluation
from drivealign.evaluation.evaluator_v2 import run_evaluation_v2
from drivealign.evaluation.evaluator_v3 import run_evaluation_v3


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluation: predictions vs frozen dataset GT (eval_v1 / eval_v2)"
    )
    parser.add_argument("--predictions", required=True, help="predictions JSONL")
    parser.add_argument(
        "--anchor-manifest",
        required=True,
        help="anchor policy manifest (token -> shard/line/hashes)",
    )
    parser.add_argument(
        "--dataset-root", required=True, help="dataset root holding the shards"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="evaluation config YAML (eval_v1.yaml / eval_v2.yaml)",
    )
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument(
        "--counterfactual-blank",
        default=None,
        help="optional blank-image predictions JSONL for visual dependence",
    )
    parser.add_argument(
        "--counterfactual-shuffled",
        default=None,
        help="optional shuffled-image predictions JSONL for visual dependence",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    counterfactual = {
        transform: path
        for transform, path in (
            ("blank", args.counterfactual_blank),
            ("shuffled", args.counterfactual_shuffled),
        )
        if path is not None
    }
    # 按 config 的 eval_version 分发（只读 dispatch，evaluator 内部会再读一次）
    eval_version = str(
        (yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}).get(
            "eval_version", ""
        )
    )
    run_kwargs = dict(
        predictions_path=args.predictions,
        anchor_manifest_path=args.anchor_manifest,
        dataset_root=args.dataset_root,
        config_path=args.config,
        out_dir=args.out,
        counterfactual_paths=counterfactual or None,
    )
    if eval_version == "eval_v2":
        report = run_evaluation_v2(**run_kwargs)
        anchor_key = "anchors_expected"
    elif eval_version == "eval_v3":
        report = run_evaluation_v3(**run_kwargs)
        anchor_key = "anchors_expected"
    elif eval_version == "eval_v1":
        report = run_evaluation(**run_kwargs)
        anchor_key = "anchors_scored"
    else:
        raise ValueError(
            f"unsupported eval_version {eval_version!r} in {args.config}; "
            "expected 'eval_v1' or 'eval_v2'"
        )
    print(
        json.dumps(
            {
                "eval_report": str(Path(args.out) / "eval_report.json"),
                "parse_rate": report["parse_rate"],
                anchor_key: report["coverage"][anchor_key],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
