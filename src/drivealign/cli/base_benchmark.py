"""Stage 09 base benchmark runner: frozen config -> predictions JSONL.

Purpose:
    Run the UNFINETUNED checkpoint over the frozen test anchors under one
    pre-registered configuration. Per anchor: read the shard record through
    the v5-face anchor manifest (verifying the record hash), serialize the
    ``model_inputs`` under the configured input policy, and assert the
    recomputed request hash equals the manifest value (the contract face is
    therefore self-enforcing at inference time). One anchor failing to
    generate never aborts the run (frozen S03 taxonomy: generation_failure
    records). Rows append to the output JSONL as they complete, so
    ``--resume`` continues an interrupted run idempotently by sample token.
    Counterfactual mode (``--image-transform blank|shuffled``) routes the
    transformed images to a scratch folder and MUST write to a separate
    ``--out`` file; the main predictions are never mixed with them.

Example launch command (pilot, GPU/tmux, see S09 plan section 4):
    ``tmux new-session -d -s s09_bench \\
    'source /root/miniconda3/etc/profile.d/conda.sh && conda activate autovla_codeclean && \\
    cd /root/autodl-tmp/drivealign_workspace && set -o pipefail && \\
    PYTHONPATH=DriveAlign/src python -m drivealign.cli.base_benchmark \\
    --config DriveAlign/configs/benchmark/base_1f.yaml \\
    --anchor-manifest data/dataset_v4/anchor_policy_manifest.json \\
    --dataset-root data/dataset_v4 --nuscenes-dataroot data/nuscenes/trainval \\
    --anchors-file runs/S09_base_benchmark/pilot_tokens.txt \\
    --out runs/S09_base_benchmark/predictions_pilot.jsonl \\
    2>&1 | tee runs/S09_base_benchmark/pilot.log'``
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping

import yaml

from drivealign.contracts.prompt import build_structured_prompt
from drivealign.inference.model_loader import load_model_and_processor
from drivealign.inference.structured_runner import (
    make_blank_image,
    make_shuffled_image,
    record_to_dict,
    run_structured_inference,
)
from drivealign.records.record import DriveAlignRecord
from drivealign.records.serializer import (
    SPEED_LINE_PRECISION,
    InputPolicy,
    serialize,
)

POLICY_MAP = {"1F": InputPolicy.ONE_FRAME, "4F": InputPolicy.FOUR_FRAME}
HASH_FIELD = {InputPolicy.ONE_FRAME: "request_hash_1f", InputPolicy.FOUR_FRAME: "request_hash_4f"}
IMAGE_TRANSFORMS = ("blank", "shuffled")
PROGRESS_EVERY = 25


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the base benchmark over frozen anchors (v5 face)"
    )
    parser.add_argument("--config", required=True, help="benchmark config YAML")
    parser.add_argument(
        "--anchor-manifest",
        required=True,
        help="anchor policy manifest (v5 face) with shard/line/hash entries",
    )
    parser.add_argument(
        "--dataset-root", required=True, help="dataset root holding the shards"
    )
    parser.add_argument(
        "--nuscenes-dataroot",
        required=True,
        help="nuScenes dataroot for anchor images (dataroot-relative paths)",
    )
    parser.add_argument("--out", required=True, help="predictions JSONL (appended)")
    parser.add_argument(
        "--anchors-file",
        default=None,
        help="optional file with one sample token per line (pilot/counterfactual)",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="cap the number of anchors (pilot)"
    )
    parser.add_argument(
        "--resume", action="store_true", help="skip tokens already present in --out"
    )
    parser.add_argument(
        "--image-transform",
        default=None,
        choices=IMAGE_TRANSFORMS,
        help="counterfactual image mode; requires a separate --out file",
    )
    return parser.parse_args(argv)


def load_config(path: str | Path) -> dict[str, Any]:
    config_bytes = Path(path).read_bytes()
    config = yaml.safe_load(config_bytes)
    if config.get("status") != "frozen":
        raise ValueError(
            f"benchmark config {path} is not frozen "
            f"(status={config.get('status')!r}); freeze it before running"
        )
    config["_sha256"] = hashlib.sha256(config_bytes).hexdigest()
    return config


def select_tokens(
    manifest: Mapping[str, Any],
    split: str,
    anchors_file: str | None,
    limit: int | None,
) -> list[str]:
    """Deterministic anchor selection: file order, else token-sorted."""
    anchors = manifest["anchors"]
    if anchors_file:
        tokens = [
            line.strip()
            for line in Path(anchors_file).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        unknown = [token for token in tokens if token not in anchors]
        if unknown:
            raise ValueError(f"{len(unknown)} anchors-file tokens missing from manifest, e.g. {unknown[:3]}")
    else:
        tokens = sorted(
            token
            for token, entry in anchors.items()
            if entry["split"] == split
        )
    return tokens[:limit] if limit is not None else tokens


def load_record(
    dataset_root: str | Path, manifest: Mapping[str, Any], token: str
) -> DriveAlignRecord:
    """Read one shard record and verify it against the frozen record hash."""
    entry = manifest["anchors"][token]
    shard_path = Path(dataset_root) / str(entry["shard"])
    with shard_path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle):
            if line_number == int(entry["line"]):
                record = DriveAlignRecord.from_dict(json.loads(raw))
                if record.canonical_hash() != str(entry["record_hash"]):
                    raise ValueError(
                        f"record hash mismatch for {token} in {entry['shard']}"
                        f" line {line_number}"
                    )
                return record
    raise ValueError(f"shard {shard_path} has no line {entry['line']} for {token}")


def ego_speed_line(ego_speed_mps: float, precision: int = SPEED_LINE_PRECISION) -> str:
    """The ego-speed prompt line, byte-identical to the serializer's format."""
    return (
        f"{ego_speed_mps:.{precision}f} m/s "
        "(backward difference from the previous keyframe)"
    )


def build_benchmark_request(
    record: DriveAlignRecord,
    entry: Mapping[str, Any],
    *,
    dataroot: str | Path,
    policy: InputPolicy,
    contract_version: str,
    transform: str | None = None,
    scratch_dir: str | Path | None = None,
) -> tuple[Any, Any]:
    """Build (SampleRequest, ModelRequest) with a hash-face consistency guard.

    The serializer derives the canonical request from the record; the runner
    rebuilds the same prompt from the contract text. If the two ever diverge
    (e.g. the DEFAULT contract moved), the guard raises instead of silently
    benchmarking against the wrong face.
    """
    from drivealign.inference.structured_runner import SampleRequest

    model_request = serialize(record.model_inputs, policy)
    speed_line = ego_speed_line(record.model_inputs.ego_speed_mps)
    runner_prompt = build_structured_prompt(speed_line, version=contract_version)
    if runner_prompt != model_request.prompt:
        raise ValueError(
            "runner prompt diverges from the serialized request prompt "
            "(contract face mismatch); refusing to run"
        )
    expected_hash = str(entry[HASH_FIELD[policy]])
    actual_hash = model_request.canonical_hash()
    if actual_hash != expected_hash:
        raise ValueError(
            f"request hash mismatch for {record.sample_token}: recomputed "
            f"{actual_hash} != manifest {expected_hash} (wrong contract face?)"
        )

    image_relpath = str(entry["anchor_image_relpath"])
    if transform is None:
        image = Path(dataroot) / image_relpath
    else:
        if scratch_dir is None:
            raise ValueError("counterfactual transforms require --out scratch dir")
        source = Path(dataroot) / image_relpath
        target = (
            Path(scratch_dir) / "counterfactual_images" / transform
            / f"{record.sample_token}.jpg"
        )
        if not target.is_file():
            if transform == "blank":
                make_blank_image(source, target)
            else:
                make_shuffled_image(source, target)
        image = target
    request = SampleRequest(
        sample_id=record.sample_token,
        image=image,
        available_speed=speed_line,
        contract_version=contract_version,
    )
    return request, model_request


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.image_transform is not None:
        if Path(args.out).name in ("predictions.jsonl", "predictions_pilot.jsonl"):
            raise ValueError(
                "counterfactual runs must write to a dedicated output file"
            )
    config = load_config(args.config)
    manifest = json.loads(Path(args.anchor_manifest).read_text(encoding="utf-8"))
    policy = POLICY_MAP[str(config["input_policy"])]
    contract_version = str(config["contract_version"])
    tokens = select_tokens(manifest, str(config["split"]), args.anchors_file, args.limit)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if args.resume and out_path.is_file():
        with out_path.open("r", encoding="utf-8") as handle:
            done = {
                str(json.loads(line)["sample_id"])
                for line in handle
                if line.strip()
            }

    loaded = load_model_and_processor(config)
    generation_config = dict(config["generation"])
    scratch_dir = out_path.parent
    wall_start = time.monotonic()
    n_new = 0
    with out_path.open("a", encoding="utf-8") as sink:
        for index, token in enumerate(tokens, start=1):
            if token in done:
                continue
            entry = manifest["anchors"][token]
            record = load_record(args.dataset_root, manifest, token)
            request, model_request = build_benchmark_request(
                record,
                entry,
                dataroot=args.nuscenes_dataroot,
                policy=policy,
                contract_version=contract_version,
                transform=args.image_transform,
                scratch_dir=scratch_dir,
            )
            sample_record = run_structured_inference(
                loaded, request, generation_config=generation_config
            )
            row = record_to_dict(sample_record)
            row.update(
                {
                    "sample_token": token,
                    "split": str(entry["split"]),
                    "scene_token": str(entry["scene_token"]),
                    "record_hash": str(entry["record_hash"]),
                    "input_policy": policy.value,
                    "frame_tokens": list(model_request.frame_tokens),
                    "time_offsets_s": list(model_request.time_offsets_s),
                    "ego_speed_mps": model_request.ego_speed_mps,
                    "request_hash": model_request.canonical_hash(),
                    "manifest_request_hash": str(entry[HASH_FIELD[policy]]),
                    "request_hash_match": True,
                    "image_transform": args.image_transform,
                    "run_name": str(config["run_name"]),
                }
            )
            sink.write(json.dumps(row, sort_keys=True) + "\n")
            sink.flush()
            n_new += 1
            if index % PROGRESS_EVERY == 0:
                elapsed = time.monotonic() - wall_start
                print(
                    f"[{index}/{len(tokens)}] new={n_new} "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )

    summary = {
        "run_name": str(config["run_name"]),
        "config_sha256": config["_sha256"],
        "contract_version": contract_version,
        "input_policy": policy.value,
        "image_transform": args.image_transform,
        "split": str(config["split"]),
        "anchors_selected": len(tokens),
        "anchors_done_before": len(done & set(tokens)),
        "anchors_new": n_new,
        "wall_seconds": time.monotonic() - wall_start,
        "out_path": str(out_path),
    }
    summary_path = out_path.parent / f"run_summary_{out_path.stem}.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
