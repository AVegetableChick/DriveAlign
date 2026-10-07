"""S11 Step 3/4 训练 CLI（`cli/sft_train.py`，GPU 执行）。

包装 `sft.train`：读取冻结样本 artifact（train32 / overfit128），把每条 JSONL
样本（含 prompt/image_relpath/target_text 与面 parity 字段）经 `dataset.encode_frozen_sample`
转成训练张量、拼成 HF Dataset，然后跑 LoRA 训练、存 adapter，并把"状态保真参照"
落盘到 adapter 目录，供 **新进程** G1 复验消费。参照含两部分：

- 硬层：adapter 权重 state_dict + global step；
- 软层：训练端在固定样本（train32 第一条）上的贪心生成输出（`run_reference_generation`），
  复验端以"reload 输出 vs 本参照"判四字段语义相等（用户 2026-10-06 拍板）。

本 CLI **只负责训练**：不做同进程 reload（同进程 reload 会让 trainer 模型与第二份
base+adapter 双驻显存，且无法证明"磁盘上的 adapter 能独立加载"）。复验请在新进程跑
``python -m drivealign.cli.sft_reload_verify``。

用法（GPU，tmux 内执行；`| tee` 前加 `set -o pipefail`）：
    ``conda activate autovla_codeclean && PYTHONPATH=DriveAlign/src python -m \
    drivealign.cli.sft_train \
    --config DriveAlign/configs/sft/sft_smoke_1f.yaml \
    --samples runs/S11_sft_smoke/sft_samples/train32.jsonl \
    --out runs/S11_sft_smoke/checkpoints/smoke32 \
    --dataroot data/nuscenes/trainval``

本 CLI 依赖本地模型（`configs.model.path`），无法在 CPU 无模型环境跑通；它由
Step 3 在 GPU 上调用，不参与 CPU 单测。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml


def _cuda_peak_memory() -> dict[str, float] | None:
    """训练峰值显存画像（§3 Step 3 资源画像必填项）。

    读的是 CUDA 分配器的高水位线：`max_memory_allocated` = 实际被张量占用的峰值，
    `max_memory_reserved` = 分配器向驱动预留的峰值（含碎片）。两者一起看才能判断
    "还能不能再加 batch"。无 GPU 环境（CPU 单测）返回 None。
    """
    if not torch.cuda.is_available():
        return None
    gib = 1024 ** 3
    return {
        "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / gib, 2),
        "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / gib, 2),
        "device_total_gib": round(
            torch.cuda.get_device_properties(0).total_memory / gib, 2
        ),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LoRA SFT (S11 smoke / S12 full)")
    parser.add_argument("--config", required=True, help="sft config YAML")
    parser.add_argument("--samples", required=True, help="frozen samples JSONL (train32/overfit128/train_full)")
    parser.add_argument("--out", required=True, help="adapter output dir (checkpoint 落盘根)")
    parser.add_argument("--dataroot", required=True, help="nuScenes dataroot for images")
    parser.add_argument("--overfit", action="store_true", help="use overfit128 params")
    # S12 §3 Step 1 新增：训练规模覆盖（不改冻结配置即可覆盖单次运行）。
    parser.add_argument("--epochs", type=int, default=None, help="override num_train_epochs")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help=(
            "override max_steps（>0 时 HF 忽略 num_train_epochs）。"
            "只服务于诊断 smoke / Step 4 的限步验证，正式全量跑不要传。"
        ),
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=None,
        help="override checkpoint 节奏（S12 正式 = 500）",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="resume training from a checkpoint dir (S12 §2.9 / Step 4)",
    )
    return parser.parse_args(argv)


def _build_params_override(args: argparse.Namespace) -> dict | None:
    """把 CLI 覆盖开关转成 `train(params_override=...)` 的字典。

    口径：只有**显式给了**的开关才进 override，未给的字段完全走冻结配置的值
    （全为 None 时返回 None，`make_trainer` 因此拿到与原行为逐值一致的参数）。
    抽成独立函数是为了让"缺省即不变"这条口径可在 CPU 上单测（见
    `tests/unit/test_sft_train_cli.py`），不必起 GPU 或加载模型。
    """
    override = {
        key: value
        for key, value in (
            ("num_train_epochs", args.epochs),
            ("max_steps", args.max_steps),
            ("save_steps", args.save_steps),
        )
        if value is not None
    }
    return override or None


def _load_dataset(samples_path: Path, dataroot: Path, processor):
    """读冻结样本 JSONL → **惰性** map-style Dataset（S12 §2.2 / §3 Step 1）。

    装载策略从 S11 的「全量预编码 + `HFDataset.from_list`」改为惰性：`__getitem__`
    被调用时才读图 + `encode_frozen_sample`。编码语义与 S11/G1 复验完全同源
    （`sft.dataset.encode_frozen_sample` 唯一实现），只是装载时机变了——train_full
    22,941 条全量预编码约 24 GB，不可驻内存。
    """
    from drivealign.sft.dataset import make_lazy_sft_dataset

    return make_lazy_sft_dataset(samples_path, processor, dataroot)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if config.get("status") != "frozen":
        raise ValueError(
            f"sft config {args.config} is not frozen (status={config.get('status')!r}); "
            "freeze it before training"
        )

    from drivealign.inference.model_loader import load_model_and_processor
    from drivealign.sft.train import (
        run_reference_generation,
        save_train_state_reference,
        train,
    )

    # 单副本加载：processor 用于编码样本，模型直接传给 train 复用，避免同进程双份 3B 基座。
    loaded = load_model_and_processor(config)
    dataset = _load_dataset(Path(args.samples), Path(args.dataroot), loaded.processor)

    # CLI 显式覆盖（只在给了参数时才进 override，否则完全走冻结配置的值）。
    params_override = _build_params_override(args)

    # 在训练前重置高水位线：之后的峰值 = 模型常驻权重 + 训练激活/梯度（正是要落档的口径）。
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    trainer = train(
        config,
        dataset,
        output_dir=args.out,
        overfit=args.overfit,
        loaded=loaded,
        params_override=params_override or None,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )

    # 训练峰值显存必须在生成参照**之前**取走：`base_runner.generate_one` 内部会
    # `reset_peak_memory_stats()`，之后读到的只是生成阶段（KV cache）的峰值。
    cuda_peak_memory = _cuda_peak_memory()

    # G1 软层参照（§7-I）：训练结束用训练态模型（`trainer.model`，正是被 save_model
    # 落盘的那份权重）在固定样本上贪心生成一次。复验端比对"reload 输出 vs 本输出"。
    generation_reference = run_reference_generation(
        loaded, trainer.model, config, args.samples, args.dataroot
    )

    # G1 参照落盘：adapter 权重 + global step + 软层生成参照，一并写进 adapter 目录。
    ref_path = save_train_state_reference(
        trainer.model,
        int(trainer.state.global_step),
        args.out,
        generation=generation_reference,
    )

    final_metrics = next(
        (entry for entry in reversed(trainer.state.log_history) if "train_loss" in entry),
        {},
    )
    print(
        {
            "adapter_dir": str(args.out),
            "state_reference": str(ref_path),
            "global_step": int(trainer.state.global_step),
            "train_loss": final_metrics.get("train_loss"),
            "train_runtime_s": final_metrics.get("train_runtime"),
            "cuda_peak_memory": cuda_peak_memory,
            "reference_generation": generation_reference,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())