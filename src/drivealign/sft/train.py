"""S11 LoRA 训练入口（§2.4 / §3 Step 3）：`sft/train.py`（供 CLI 调用）。

职责
----
把 Step 2 冻结的样本 artifact（train32 / overfit128 JSONL）喂给 HF Trainer，
以 LoRA 微调 3B 基础模型，落地：

- 训练面 == 评测面（面 parity 由 `dataset.build_training_sample` 在样本构造时
  已断言，训练阶段不再重复断言 user 文本）。
- Step 3（32 条）/ Step 4（128 条）共用同一套 Trainer 组装，仅样本文件与
  overfit 覆盖（dropout=0、epochs=3）不同（§7-F/G）。

本模块不直接执行 `python -m`：由 `cli/sft_train.py` 解析配置与参数后调用
`train()`。支持 ``reload`` 复验（G1）：训练结束把 adapter 保存到盘，可在**新进程**
里加载并跑固定输入输出做冒烟（§3 Step 3 的资源画像 + 状态保真）。

依赖约定（§2.4）
----------------
- DeFAST Lora 通过 peft <https://github.com/huggingface/peft>；未安装时 import 抛错，
  属 GPU 环境前置检查，CPU 单测不触碰本模块（本模块在 Step 3 才被执行）。
"""

from __future__ import annotations

import torch
from transformers import TrainingArguments, Trainer

from drivealign.inference.model_loader import load_model_and_processor


def build_lora_model(model, config: dict, *, lora_override: dict | None = None):
    """包一只 Peft Lora 包装（§7-C 的 r/alpha/dropout/target）。

    ``lora_override`` 让 overfit 阶段以 ``overfit.lora`` 覆盖起步参数（§7-G：
    dropout=0），其余阶段用顶层 ``config["lora"]``。
    """
    from peft import LoraConfig, get_peft_model

    lora = dict(config["lora"])
    if lora_override:
        lora.update(lora_override)  # 覆盖（如 overfit dropout=0）
    lora_config = LoraConfig(
        r=lora["r"],
        lora_alpha=lora["alpha"],
        lora_dropout=lora["dropout"],
        target_modules=lora["target_modules"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    return model


def make_trainer(
    model,
    processor,
    train_dataset,
    config: dict,
    *,
    overfit: bool = False,
) -> Trainer:
    """组装 HF Trainer（§2.4 + §7-C 参数）。

    - collator：`make_data_collator`（批融合）。
    - train 数据集：本模块不实现 IDataset；由调用方（CLI）把 JSONL 转成 HF Dataset，
      元素是 `dataset.encode_chat` 返回的 dict（processor 已在样本层面生成张量）。
    - ``remove_unused_columns=False``：collator 收到的就是 encode_chat 的 dict，不删列。
    """
    from drivealign.sft.collator import make_data_collator

    seg = "overfit" if overfit else "train32"
    params = config[seg].get("training", config["training"])

    train_args = TrainingArguments(
        output_dir=config["run_name"],
        per_device_train_batch_size=params["per_device_train_batch_size"],
        gradient_accumulation_steps=params["gradient_accumulation_steps"],
        learning_rate=params["learning_rate"],
        num_train_epochs=params["num_train_epochs"],
        fp16=False,
        bf16=True,
        optim=params["optimizer"],
        lr_scheduler_type=params["lr_scheduler_type"],
        warmup_ratio=params["warmup_ratio"],
        weight_decay=params["weight_decay"],
        max_grad_norm=params["max_grad_norm"],
        gradient_checkpointing=params["gradient_checkpointing"],
        seed=config["seed"],
        remove_unused_columns=False,
        save_strategy="steps",
        save_steps=64,
        save_total_limit=2,
        logging_steps=10,
        report_to=None,
    )
    return Trainer(
        model=model,
        args=train_args,
        data_collator=make_data_collator(processor.pad_token_id),
        train_dataset=train_dataset,
    )


def train(
    config: dict,
    train_dataset,
    *,
    output_dir: str,
    overfit: bool = False,
):
    """加载模型/处理器 → 组装 Trainer → 训练 → 保存 adapter（G1 状态保真）。"""
    from drivealign.inference.model_loader import load_model_and_processor

    loaded = load_model_and_processor(config)
    seg = "overfit" if overfit else "train32"
    lora_override = config.get(seg, {}).get("lora")
    model = build_lora_model(loaded.model, config, lora_override=lora_override)
    trainer = make_trainer(model, loaded.processor, train_dataset, config, overfit=overfit)
    trainer.train()
    trainer.save_model(output_dir)
    return trainer


def reload_and_infer(
    config: dict,
    adapter_dir: str,
    fixed_batch: dict,
):
    """G1 状态保真复验：新进程加载 adapter 后跑固定输入输出（§7-I）。

    目的
    ----
    验证"save adapter -> 可靠load"这一链路没坏：新进程里从配置基座 + adapter 目录
    能重建出与训练端一致的参数化模型，并能对固定输入完成一次 forward（无 NaN/Inf）。

    分层判定（§7-I）
    ----------------
    - 状态保真（硬）：本函数返回重建模型，由 Step 3 脚本比对其逐张量参数与训练端
      shape/dtype 同构（此处只负责"能加载"）。
    - 功能冒烟（软）：返回一次 forward 的 logits；Step 3 脚本对生成输出做四字段语义
      相等即可（容忍 token 层非确定性）。

    参数
    ----
    config : 训练关系（含 model）配置。
    adapter_dir : 已保存的 adapter 权重目录。
    fixed_batch : dict(input_ids/pixel_values/image_grid_thw)，训练时那批原样本。

    返回
    ----
    (model, logits) : 供 Step 3 继续做逐张量比对与输出语义断言。
    """
    from peft import PeftModel

    loaded = load_model_and_processor(config)
    model = PeftModel.from_pretrained(loaded.model, adapter_dir)
    model.eval()
    with torch.no_grad():
        logits = model(
            input_ids=fixed_batch["input_ids"],
            pixel_values=fixed_batch["pixel_values"],
            image_grid_thw=fixed_batch["image_grid_thw"],
        ).logits
    return model, logits