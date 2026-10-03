"""Counterfactual subset sampler for the S09 visual-dependence diagnostics.

Purpose:
    Pre-register the blank/shuffled evaluation subset BEFORE any
    counterfactual inference (S09 plan section 4, Step 3). From the
    v5-face anchor manifest it draws ``--n`` anchors stratified by scene:
    scenes are shuffled with a fixed seed, then anchors are picked
    round-robin (one per scene per round, within-scene order shuffled with
    the same seed) until ``--n`` anchors are collected — maximizing scene
    diversity, which is what the scene-clustered metrics care about. The
    output token list is sorted (order-independent runs) and the sampler
    is deterministic: same manifest, same seed, same list.

Example launch command:
    ``conda activate autovla_codeclean && cd /root/autodl-tmp/drivealign_workspace && \
    PYTHONPATH=DriveAlign/src python -m drivealign.cli.sample_counterfactual_subset \
    --anchor-manifest data/dataset_v4/anchor_policy_manifest.json \
    --split test --n 200 --seed 20261003 \
    --out runs/S09_base_benchmark/counterfactual_subset.txt``
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-register the counterfactual (blank/shuffled) subset"
    )
    parser.add_argument("--anchor-manifest", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument(
        "--out", required=True, help="output token list (one per line, sorted)"
    )
    return parser.parse_args(argv)


def sample_subset(
    manifest: dict, *, split: str, n: int, seed: int
) -> list[str]:
    """Scene-stratified deterministic sample of ``n`` anchor tokens."""
    by_scene: dict[str, list[str]] = {}
    for token, entry in manifest["anchors"].items():
        if entry["split"] == split:
            by_scene.setdefault(str(entry["scene_token"]), []).append(token)
    rng = random.Random(seed)
    scenes = sorted(by_scene)
    for members in by_scene.values():
        members.sort()
        rng.shuffle(members)
    order: list[str] = []
    round_index = 0
    while len(order) < n:
        progressed = False
        for scene in scenes:
            members = by_scene[scene]
            if round_index < len(members):
                order.append(members[round_index])
                progressed = True
                if len(order) == n:
                    break
        if not progressed:
            break  # split exhausted before reaching n
        round_index += 1
    return sorted(order)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = json.loads(Path(args.anchor_manifest).read_text(encoding="utf-8"))
    tokens = sample_subset(
        manifest, split=args.split, n=args.n, seed=args.seed
    )
    if len(tokens) < args.n:
        print(
            f"warning: split {args.split!r} has fewer anchors than "
            f"n={args.n}; emitted {len(tokens)}"
        )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(f"{token}\n" for token in tokens), encoding="utf-8")
    scenes = {
        str(manifest["anchors"][token]["scene_token"]) for token in tokens
    }
    print(
        json.dumps(
            {
                "out": str(out),
                "n_tokens": len(tokens),
                "n_scenes": len(scenes),
                "split": args.split,
                "seed": args.seed,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
