"""S12 Step 6b：批量（batch>1）benchmark runner（batched_v1 path）。

Purpose:
    ``cli.base_benchmark``（S09/S12 冻结面）的**批量镜像**：请求构造、hash
    守卫、resume、反事实路由逐行复用冻结面公开函数（不复制逻辑），仅把
    "逐样本 ``run_structured_inference``" 换成 "顺序切批 +
    ``run_structured_inference_batched``"。输出仍是同 schema 的 predictions
    JSONL + run_summary，可直接喂 Step 7 评测器。

纪律边界:
    - 冻结面 ``base_benchmark.py`` 一行不改；本 CLI 是独立新 path；
    - 输出必须写独立目录（约定 ``runs/S12_sft_1f/eval_v6_batched/``），
      严禁与冻结面 predictions 混写；
    - parity gate（与冻结面输出 CPU diff）延后执行，见
      ``S09_inference_acceleration.md`` 备案。

Example launch command (smoke, GPU/tmux):
    注意 --anchor-manifest 必须用 **v6 面** manifest
    （``data/face_manifests/v6/anchor_policy_manifest.json``）；dataset_v4 下的
    ``anchor_policy_manifest.json`` 是 v5 面，hash 守卫会正确拒绝。

    ``tmux new-session -d -s s12_batch_smoke \\
    'source /root/miniconda3/etc/profile.d/conda.sh && conda activate autovla_codeclean && \\
    cd /root/autodl-tmp/drivealign_workspace && set -o pipefail && \\
    PYTHONPATH=DriveAlign/src python -m drivealign.cli.batch_benchmark \\
    --config DriveAlign/configs/benchmark/sft_1f_v6_batched.yaml \\
    --anchor-manifest data/face_manifests/v6/anchor_policy_manifest.json \\
    --dataset-root data/dataset_v4 --nuscenes-dataroot data/nuscenes/trainval \\
    --limit 8 --out runs/S12_sft_1f/eval_v6_batched/smoke_predictions.jsonl \\
    2>&1 | tee runs/S12_sft_1f/eval_v6_batched/smoke.log'``
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from drivealign.cli.base_benchmark import (
    HASH_FIELD,
    POLICY_MAP,
    PROGRESS_EVERY,
    build_benchmark_request,
    load_config,
    load_record,
    select_tokens,
)
from drivealign.inference.batched_runner import run_structured_inference_batched
from drivealign.inference.structured_runner import record_to_dict

DEFAULT_BATCH_SIZE = 8


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batched (batch>1) benchmark runner over frozen anchors "
        "(batched_v1 path; frozen bs=1 face untouched)"
    )
    parser.add_argument("--config", required=True, help="batched benchmark config YAML")
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
    parser.add_argument(
        "--out", required=True, help="predictions JSONL (appended; MUST be a "
        "dedicated batched-path file, never a frozen-face predictions file)"
    )
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
        choices=("blank", "shuffled"),
        help="counterfactual image mode; requires a separate --out file",
    )
    parser.add_argument(
        "--adapter",
        default=None,
        help=(
            "optional LoRA adapter dir; unset falls back to config['adapter_path']"
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="batch size override; default reads config['batch_size']",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.image_transform is not None:
        if Path(args.out).name in ("predictions.jsonl", "predictions_pilot.jsonl"):
            raise ValueError(
                "counterfactual runs must write to a dedicated output file"
            )
    config = load_config(args.config)
    batch_size = int(
        args.batch_size if args.batch_size is not None
        else config.get("batch_size", DEFAULT_BATCH_SIZE)
    )
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    manifest = json.loads(Path(args.anchor_manifest).read_text(encoding="utf-8"))
    policy = POLICY_MAP[str(config["input_policy"])]
    contract_version = str(config["contract_version"])
    tokens = select_tokens(manifest, str(config["split"]), args.anchors_file, args.limit)

    out_path = Path(args.out)
    if out_path.name in ("predictions.jsonl", "predictions_pilot.jsonl"):
        # 与冻结面产物的名字硬隔离，防止误写覆盖冻结面读数。
        raise ValueError(
            f"batched runs must write to a dedicated output file, got {out_path}"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if args.resume and out_path.is_file():
        with out_path.open("r", encoding="utf-8") as handle:
            done = {
                str(json.loads(line)["sample_id"])
                for line in handle
                if line.strip()
            }
    pending = [token for token in tokens if token not in done]

    # 模型装载与冻结面同源：给了 adapter（或 config 登记了 adapter_path）时走
    # SFT 评测路径（load_adapter_model），否则走 S09 base 路径——与冻结面唯一
    # 的差别只在生成时的批大小。
    adapter_dir = args.adapter or config.get("adapter_path")
    if adapter_dir:
        from drivealign.sft.train import load_adapter_model

        loaded = load_adapter_model(config, adapter_dir)
    else:
        from drivealign.inference.model_loader import load_model_and_processor

        loaded = load_model_and_processor(config)
    generation_config = dict(config["generation"])
    scratch_dir = out_path.parent
    wall_start = time.monotonic()
    n_new = 0
    with out_path.open("a", encoding="utf-8") as sink:
        for batch_tokens in [
            pending[i : i + batch_size] for i in range(0, len(pending), batch_size)
        ]:
            # 请求构造与 hash 守卫逐样本执行，与冻结面同一函数、同一报错行为。
            batch_requests = []
            batch_model_requests = []
            batch_entries = []
            for token in batch_tokens:
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
                batch_requests.append(request)
                batch_model_requests.append(model_request)
                batch_entries.append(entry)

            sample_records = run_structured_inference_batched(
                loaded,
                batch_requests,
                generation_config=generation_config,
                batch_size=batch_size,
            )
            for request, model_request, entry, sample_record in zip(
                batch_requests, batch_model_requests, batch_entries, sample_records
            ):
                token = request.sample_id
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
                if n_new % PROGRESS_EVERY == 0:
                    elapsed = time.monotonic() - wall_start
                    print(
                        f"[{n_new}/{len(pending)}] new={n_new} "
                        f"elapsed={elapsed:.1f}s batch_size={batch_size}",
                        flush=True,
                    )

    summary = {
        "run_name": str(config["run_name"]),
        "config_sha256": config["_sha256"],
        "contract_version": contract_version,
        "input_policy": policy.value,
        "adapter_path": str(adapter_dir) if adapter_dir else None,
        "image_transform": args.image_transform,
        "split": str(config["split"]),
        "runner": str(config.get("runner", "batched_v1")),
        "batch_size": batch_size,
        "anchors_selected": len(tokens),
        "anchors_done_before": len(tokens) - len(pending),
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
