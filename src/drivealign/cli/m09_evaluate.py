"""CLI wrapper for the M09 v1 evaluator (S09 plan section 2.1).

Example launch command:
    ``conda activate autovla_codeclean && cd /root/autodl-tmp/drivealign_workspace && \
    PYTHONPATH=DriveAlign/src python -m drivealign.cli.m09_evaluate \
    --predictions runs/S09_base_benchmark/predictions.jsonl \
    --anchor-manifest data/dataset_v4/anchor_policy_manifest.json \
    --dataset-root data/dataset_v4 \
    --config DriveAlign/configs/evaluation/m09_v1.yaml \
    --out runs/S09_base_benchmark \
    --counterfactual-blank runs/S09_base_benchmark/predictions_blank.jsonl \
    --counterfactual-shuffled runs/S09_base_benchmark/predictions_shuffled.jsonl``
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from drivealign.evaluation.evaluator import run_evaluation


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="M09 v1 evaluation: predictions vs frozen dataset GT"
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
        "--config", required=True, help="M09 evaluation config YAML (m09_v1.yaml)"
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
    report = run_evaluation(
        predictions_path=args.predictions,
        anchor_manifest_path=args.anchor_manifest,
        dataset_root=args.dataset_root,
        config_path=args.config,
        out_dir=args.out,
        counterfactual_paths=counterfactual or None,
    )
    print(
        json.dumps(
            {
                "m09_report": str(Path(args.out) / "m09_report.json"),
                "parse_rate": report["parse_rate"],
                "anchors_scored": report["coverage"]["anchors_scored"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
