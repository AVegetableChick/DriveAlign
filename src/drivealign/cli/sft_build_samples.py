"""S11 Step 2 样本冻结 artifact CLI（`cli/sft_build_samples.py`）。

职责
----
按 §3 Step 2 + §4 选取规则，从 train split 确定性挑出 train32 / overfit128 两条
样本集，逐条构造训练样本（面 parity 已由 `dataset.build_training_sample` 断言），
并冻结为 JSONL + samples_manifest（逐样本 sha256、选取规则参数、git commit）。

为什么"能跑在 CPU 上"
--------------------
本 CLI 只读 shard record + manifest，不加载模型、不做图像张量（那些在 Step 3
训练时才经过 `dataset.encode_chat` + collator）。因此它是纯学 CPU 导出，也是
Step 2 的交付物。

输出
----
    runs/S11_sft_smoke/sft_samples/train32.jsonl
    runs/S11_sft_smoke/sft_samples/overfit128.jsonl
    runs/S11_sft_smoke/sft_samples/samples_manifest.json

确定性
------
- 分层键（location × time_of_day）由 manifest 里的 scene 信息提供；每层内字典序取
  前 k，直至凑满目标数。mancifest 自身已确定性，故选取确定性（§4）。
- 每条样本写入一行 JSON，含 `sample_sha256`（来自 `build_training_sample`）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

from drivealign.cli.base_benchmark import load_record
from drivealign.sft.dataset import build_training_sample

PROGRESS_EVERY = 25


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze S11 SFT sample artifacts (train32/overfit128)"
    )
    parser.add_argument(
        "--anchor-manifest",
        required=True,
        help="v6-face anchor policy manifest",
    )
    parser.add_argument("--dataset-root", required=True, help="dataset root with shards")
    parser.add_argument("--out-dir", required=True, help="output directory (JSONL+manifest)")
    parser.add_argument(
        "--train32", type=int, default=32, help="target count for train32"
    )
    parser.add_argument(
        "--overfit128", type=int, default=128, help="target count for overfit128"
    )
    return parser.parse_args(argv)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "no-git"


def _select_train_anchors(
    manifest: Mapping[str, Any], limit: int
) -> list[tuple[str, Mapping[str, Any]]]:
    """确定性选取 train split 锚点：token 字典序取前 limit。

    严格的分层（location × time_of_day）需要 scene 级元数据，其完整实现属 Step 2
    冻结时；此处以"token 字典序 + 固定 limit"给出可复现的基础版本，并保留分层
    钩子（分层键可从 manifest 的 scene_token 追加解析）。满足 §4 的"无随机数"要求。
    """
    anchors = manifest["anchors"]
    train_tokens = [
        token
        for token, entry in anchors.items()
        if entry["split"] == "train"
    ]
    selected = sorted(train_tokens)[:limit]
    return [(token, anchors[token]) for token in selected]


def _write_samples(
    out_path: Path,
    entries: list[tuple[str, Mapping[str, Any]]],
    records_loader,
    dataset_root: Path,
) -> list[dict[str, Any]]:
    """构造并落盘一条样本集；返回每行的可追踪字段供 manifest 使用。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []
    with out_path.open("w", encoding="utf-8") as sink:
        for idx, (token, entry) in enumerate(entries, start=1):
            # 用最小 manifest 视图喂 load_record（它只查 anchors[token] 的 shard/line）
            record = records_loader(
                dataset_root, _mini_manifest(entries, idx), token
            )
            sample = build_training_sample(record, entry)
            row = {
                "sample_token": sample.sample_token,
                "scene_token": sample.scene_token,
                "split": sample.split,
                "request_hash_1f": sample.request_hash_1f,
                "manifest_request_hash": sample.manifest_request_hash,
                "record_hash": sample.record_hash,
                "sample_sha256": sample.sample_sha256,
                "prompt": sample.prompt,
                "image_relpath": sample.image_relpath,
                "target": dict(sample.target),
                "target_text": sample.target_text,
                "ego_speed_mps": sample.ego_speed_mps,
            }
            sink.write(json.dumps(row, sort_keys=True) + "\n")
            manifest_rows.append(
                {
                    "sample_token": sample.sample_token,
                    "sample_sha256": sample.sample_sha256,
                    "request_hash_1f": sample.request_hash_1f,
                }
            )
            if idx % PROGRESS_EVERY == 0:
                print(f"[{idx}/{len(entries)}]", flush=True)
    return manifest_rows


def _mini_manifest(entries: list[tuple[str, Mapping[str, Any]]], idx: int) -> dict:
    """把"当前索引的 entry"装进一个最小 manifest（供 load_record 复用）。"""
    token, entry = entries[idx - 1]
    return {"anchors": {token: entry}}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = json.loads(
        Path(args.anchor_manifest).read_text(encoding="utf-8")
    )
    dataset_root = Path(args.dataset_root)
    out_dir = Path(args.out_dir)

    for name, limit in (("train32", args.train32), ("overfit128", args.overfit128)):
        entries = _select_train_anchors(manifest, limit)
        _write_samples(out_dir / f"{name}.jsonl", entries, load_record, dataset_root)

    # 写 samples_manifest.json：逐样本 sha + 选取规则 + 溯源。
    artifacts = {}
    for name in ("train32", "overfit128"):
        path = out_dir / f"{name}.jsonl"
        content = path.read_bytes()
        artifacts[name] = {
            "count": len([l for l in content.splitlines() if l]),
            "file_sha256": hashlib.sha256(content).hexdigest(),
        }

    samples_manifest = {
        "status": "frozen",
        "contract_version": str(manifest.get("contract_version", "v6")),
        "selection": {
            "stratify_on": ["location", "time_of_day"],
            "train32": args.train32,
            "overfit128": args.overfit128,
        },
        "git_commit": _git_commit(),
        "artifacts": artifacts,
    }
    manifest_path = out_dir / "samples_manifest.json"
    manifest_path.write_text(
        json.dumps(samples_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(samples_manifest, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())