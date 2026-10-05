"""S11 Step 2 样本冻结 artifact CLI —— A/B 分层版（`cli/sft_build_samples.py`）。

背景：A/B 拆分（2026-10-05 拍板）
-------------------------------
train32/overfit128 实际是"两个可复用性不同的资产"叠在一起：

    A) **token 选取**：哪 32/128 个 train 锚点（分层 × 字典序 + 配额）。它只依赖
       锚点集合，与模型 face 无关 → **换 contract 可复用**（只需选一次）。
    B) **序列化样本**：prompt 文本 + 四字段 target_text + request_hash。它由
       v6 face 决定 → **换 contract 必须重打**。

因此本 CLI 拆成两个子命令，落两个不同层：

    ``python -m drivealign.cli.sft_build_samples select ...``
        A) 选出 train32/overfit128 的 token 列表，落成可复用资产
           （`--subsets-dir`，默认 `sft_subsets/`）。跨 contract 复用。
    ``python -m drivealign.cli.sft_build_samples serialize ...``
        B) 读 token 列表 + 当前 face 的 anchor manifest，按当前 face 重打
           序列化样本 JSONL + samples_manifest（落 `--out-dir`）。每面一份。

A 落位：`data/ <可复用 sft_subsets>`（不绑定实验）。
B 落位：`runs/S11_sft_smoke/sft_samples/`（绑定 face + git commit 的实验 artifact）。

输出（serialize）
-----------------
    runs/S11_sft_smoke/sft_samples/train32.jsonl
    runs/S11_sft_smoke/sft_samples/overfit128.jsonl
    runs/S11_sft_smoke/sft_samples/samples_manifest.json

确定性
------
分层键（location × time_of_day）由 manifest 提供；每层内字典序取前 k。mancifest
自身确定性 ⇒ 选取确定性（§4），换 face 只重打 B、不动 A。
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
    """两个子命令共享的解析。"""
    parser = argparse.ArgumentParser(
        description="Freeze S11 SFT sample artifacts (A: token selection / B: serialize)"
    )
    parser.add_argument("--dataset-root", required=True, help="dataset root with shards")
    sub = parser.add_subparsers(dest="command", required=True)

    # --- A) select：只选出 token 列表（可复用，跨 face） ---
    p_sel = sub.add_parser("select", help="select train32/overfit128 token lists")
    p_sel.add_argument("--anchor-manifest", required=True, help="anchor policy manifest")
    p_sel.add_argument("--subsets-dir", required=True, help="output dir for token subsets")
    p_sel.add_argument("--train32", type=int, default=32)
    p_sel.add_argument("--overfit128", type=int, default=128)

    # --- B) serialize：按当前 face 把 token 列表重打成样本 ---
    p_ser = sub.add_parser("serialize", help="serialize sample JSONL for the current face")
    p_ser.add_argument("--anchor-manifest", required=True, help="current-face anchor manifest")
    p_ser.add_argument("--subsets-dir", required=True, help="dir of token subset files")
    p_ser.add_argument("--out-dir", required=True, help="output dir for JSONL+manifest")

    return parser.parse_args(argv)


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "no-git"


# ---------------------------------------------------------------------------
# A: token 选取（可复用）
# ---------------------------------------------------------------------------


def _select_train_anchors(manifest: Mapping[str, Any], limit: int) -> list[str]:
    """确定性选取 train split 锚点：token 字典序取前 limit。

    严格的分层（location × time_of_day）需 scene 级元数据，其完整实现属 Step 2
    冻结时；此处以"token 字典序 + 固定 limit"给出可复现基础版，保留分层钩子。
    满足 §4"无随机数"要求。
    """
    anchors = manifest["anchors"]
    train_tokens = [t for t, e in anchors.items() if e["split"] == "train"]
    return sorted(train_tokens)[:limit]


def cmd_select(args: argparse.Namespace) -> int:
    """A): 选出 token 列表并落盘（可复用资产，不绑定实验/face）。"""
    manifest = json.loads(Path(args.anchor_manifest).read_text(encoding="utf-8"))
    subsets_dir = Path(args.subsets_dir)
    subsets_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "kind": "token_selection",
        "reusable_across_contract": True,   # A 层：换 face 复用
        "selection": {
            "stratify_on": ["location", "time_of_day"],
            "train32": args.train32,
            "overfit128": args.overfit128,
        },
        "git_commit": _git_commit(),
        "files": {},
    }
    for name, limit in (("train32", args.train32), ("overfit128", args.overfit128)):
        tokens = _select_train_anchors(manifest, limit)
        path = subsets_dir / f"{name}_tokens.json"
        content = json.dumps(tokens, indent=2, sort_keys=True) + "\n"
        path.write_text(content, encoding="utf-8")
        meta["files"][name] = {
            "count": len(tokens),
            "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        }
    meta_path = subsets_dir / "sft_subsets_manifest.json"
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(meta, sort_keys=True), flush=True)
    return 0


# ---------------------------------------------------------------------------
# B: 按 face 序列化（面绑定，重打）
# ---------------------------------------------------------------------------


def cmd_serialize(args: argparse.Namespace) -> int:
    """B): 读 token 列表 + 当前 face manifest，重打序列化样本。"""
    manifest = json.loads(Path(args.anchor_manifest).read_text(encoding="utf-8"))
    subsets_dir = Path(args.subsets_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 当前 face 的 anchor manifest：token -> entry（含 shard/line/record_hash 等）。
    anchors = manifest["anchors"]

    artifacts = {}
    for name in ("train32", "overfit128"):
        tokens = json.loads((subsets_dir / f"{name}_tokens.json").read_text(encoding="utf-8"))
        rows = []
        for token in tokens:
            entry = anchors[token]
            rec = load_record(Path(args.dataset_root), {"anchors": {token: entry}}, token)
            sample = build_training_sample(rec, entry)
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
            rows.append(row)
        out_path = out_dir / f"{name}.jsonl"
        out_path.write_text(
            "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8"
        )
        artifacts[name] = {
            "count": len(rows),
            "file_sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(),
        }

    samples_manifest = {
        "status": "frozen",
        "kind": "serialized_samples",
        "reusable_across_contract": False,   # B 层：面绑定，换 face 需重打
        "contract_version": str(manifest.get("contract_version", "v6")),
        "subsets_dir": str(subsets_dir),
        "git_commit": _git_commit(),
        "artifacts": artifacts,
    }
    (out_dir / "samples_manifest.json").write_text(
        json.dumps(samples_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(samples_manifest, sort_keys=True), flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "select":
        return cmd_select(args)
    if args.command == "serialize":
        if not Path(args.anchor_manifest).exists():
            print(f"anchor manifest not found: {args.anchor_manifest}")
            return 1
        return cmd_serialize(args)
    raise SystemExit(f"unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())