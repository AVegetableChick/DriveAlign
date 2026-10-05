"""S11 Step 3/4 训练 CLI（`cli/sft_train.py`，GPU 执行）。

包装 `sft.train`：读取冻结样本 artifact（train32 / overfit128），把每条 JSONL
样本（含 prompt/image_relpath/target_text 与面 parity 字段）经 `dataset.encode_chat`
转成训练张量、拼成 HF Dataset，然后跑 LoRA 训练并存 adapter。

用法（GPU，tmux 内执行；`| tee` 前加 `set -o pipefail`）：
    ``conda activate autovla_codeclean && PYTHONPATH=DriveAlign/src python -m \
    drivealign.cli.sft_train \
    --config DriveAlign/configs/sft/sft_smoke_1f.yaml \
    --samples runs/S11_sft_smoke/sft_samples/train32.jsonl \
    --out runs/S11_sft_smoke/checkpoints/smoke32 \
    --dataroot data/nuscenes/trainval --overfit``

本 CLI 依赖本地模型（`configs.model.path`），无法在 CPU 无模型环境跑通；它由
Step 3 在 GPU 上调用，不参与 CPU 单测。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run S11 LoRA SFT smoke")
    parser.add_argument("--config", required=True, help="sft config YAML")
    parser.add_argument("--samples", required=True, help="frozen samples JSONL (train32/overfit128)")
    parser.add_argument("--out", required=True, help="adapter output dir")
    parser.add_argument("--dataroot", required=True, help="nuScenes dataroot for images")
    parser.add_argument("--overfit", action="store_true", help="use overfit128 params")
    return parser.parse_args(argv)


def _load_dataset(samples_path: Path, dataroot: Path, processor):
    """读冻结样本 JSONL → 逐条 encode_chat → HF Dataset。

    每条冻结行已是带面 parity 的样本；这里用真实图像路径（dataroot/relpath）
    填给 `dataset.encode_chat`，产出 input_ids/labels/pixel_values/grid。
    """
    from datasets import Dataset as HFDataset

    from drivealign.sft.dataset import encode_chat

    rows = []
    for line in samples_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        encoded = encode_chat(
            processor,
            prompt=rec["prompt"],
            image_path=Path(dataroot) / rec["image_relpath"],
            target_text=rec["target_text"],
        )
        row = {
            "input_ids": encoded["input_ids"],
            "labels": encoded["labels"],
            "pixel_values": encoded["pixel_values"],
            "image_grid_thw": encoded["image_grid_thw"],
            "sample_token": rec["sample_token"],
            "sample_sha256": rec["sample_sha256"],
        }
        rows.append(row)
    return HFDataset.from_list(rows)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if config.get("status") != "frozen":
        raise ValueError(
            f"sft config {args.config} is not frozen (status={config.get('status')!r}); "
            "freeze it before training"
        )

    from drivealign.inference.model_loader import load_model_and_processor
    from drivealign.sft.train import reload_and_infer, train

    loaded = load_model_and_processor(config)
    dataset = _load_dataset(Path(args.samples), Path(args.dataroot), loaded.processor)

    trainer = train(config, dataset, output_dir=args.out, overfit=args.overfit)

    # G1 状态保真复验（§7-I）：新进程 reload 固定 batch 冒烟。
    # 取第一样本的 encode 结果构造 fixed_batch（含原图像张量）。
    fixed = dataset[0]
    batch = {
        "input_ids": fixed["input_ids"],
        "pixel_values": fixed["pixel_values"],
        "image_grid_thw": fixed["image_grid_thw"],
    }
    model, logits = reload_and_infer(config, args.out, batch)
    # model 由 reload_and_infer 返回，供 Step 3 补充逐张量状态保真比对（§7-I 硬层）。
    _ = model
    print(
        {
            "global_step": getattr(trainer.state, "global_step", None),
            "reload_logits_shape": list(logits.shape),
            "has_nan": bool(logits.isnan().any()),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())