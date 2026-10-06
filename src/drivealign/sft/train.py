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
`train()`，训练结束把 adapter 保存到盘。G1 复验（§3 Step 3 状态保真 + 功能冒烟）
拆到**新进程**入口 `cli/sft_reload_verify.py`；本模块提供两端共用的模型/样本管道：

- `load_adapter_model`：新进程从磁盘重建 PeftModel。
- `forward_fixed_batch`：固定 batch 手动 forward（软层确定性部分）。
- `fixed_sample_batch`：固定样本（train32 第一条）的编码 + 拼 batch，两端共用。
- `run_reference_generation`：训练端在固定样本上贪心生成一次，作为软层"训练端输出"
  参照（软层基准 = reload 输出 vs 训练端输出，而非 vs GT；用户 2026-10-06 拍板）。

判定规则本身是纯逻辑，放在 `sft/gate.py`。

依赖约定（§2.4）
----------------
- DeFAST Lora 通过 peft <https://github.com/huggingface/peft>；未安装时 import 抛错，
  属 GPU 环境前置检查，CPU 单测不触碰本模块（本模块在 Step 3 才被执行）。
- 本模块顶层不 import peft/transformers 的模型类，只在函数内延迟 import，保证
  `import drivealign.sft.train` 本身不需要重依赖。
"""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import TrainingArguments, Trainer

from drivealign.inference.model_loader import LoadedModel, load_model_and_processor

#: 训练端落盘的"状态保真参照"文件名（G1 硬层 + 软层比对对象）。
#: 内容 = adapter 权重 state_dict + global step + 固定样本生成参照；由
#: `save_train_state_reference` 写入 adapter 目录，供新进程
#: `cli/sft_reload_verify.py` 逐张量与逐输出比对。
TRAIN_STATE_REF_NAME = "train_state_ref.pt"


def assert_text_only_lora_targets(model) -> list[str]:
    """守卫（见 S11_bug_log B7）：LoRA 必须**只**落在语言解码层，视觉塔零命中。

    为什么需要它
    ------------
    `target_modules` 若写成裸后缀 list，peft 的匹配是 `key in list` 或
    `key.endswith(f".{name}")`（peft/tuners/tuners_utils.py: check_target_module_exists），
    于是视觉塔里同名的 `visual.blocks.*.mlp.{gate,up,down}_proj` 会被一并包上 LoRA。
    这是**静默缺陷**：不报错、loss 正常下降，只在事后数 adapter 张量数（696 而非 504）
    时才暴露。在训练前做一次结构断言，把"悄悄训错模块"变成"立刻报错"。

    判据（任一不满足即抛错）
    ------------------------
    1. 至少命中一个模块——否则说明 pattern 与模型结构对不上（等于只训一个冻结模型）；
    2. 命中模块全部位于 `.layers.` 之下、且无 `.visual.` 前缀命中。Qwen2.5-VL 里
       `model.layers` 只有语言解码层，视觉塔是 `visual.blocks`，两者不相交。

    返回命中模块名列表，便于日志/调试。
    """
    wrapped = [
        name
        for name, module in model.named_modules()
        # LoRA 包装层（peft 的 LoraLayer）的结构特征：持有非空 lora_A（ModuleDict）。
        # 用结构特征而非名字后缀去发现目标，才能暴露"后缀匹配连带命中"这类问题。
        if getattr(module, "lora_A", None)
    ]
    if not wrapped:
        raise RuntimeError(
            "LoRA 未命中任何模块：config 的 lora.target_modules 与模型结构对不上，"
            "继续训练等于只跑一个冻结模型。请核对该 pattern。"
        )
    leaked = [name for name in wrapped if ".visual." in name or name.startswith("visual.")]
    if leaked:
        raise RuntimeError(
            f"LoRA 命中了视觉塔的 {len(leaked)} 个模块（前 3 个：{leaked[:3]}）。"
            "当前口径是「只训语言侧」，target_modules 必须把视觉塔排除在外"
            "（见 S11_bug_log B7）；典型原因是用了裸后缀 list，被 peft 的后缀匹配连带命中。"
        )
    outside_decoder = [name for name in wrapped if ".layers." not in name]
    if outside_decoder:
        raise RuntimeError(
            f"LoRA 命中了语言解码层之外的 {len(outside_decoder)} 个模块"
            f"（前 3 个：{outside_decoder[:3]}）。当前口径是「只训语言侧」，"
            "请检查 lora.target_modules 是否被钉死在 `model.layers.<n>.` 之下。"
        )
    return wrapped


def build_lora_model(model, config: dict, *, lora_override: dict | None = None):
    """包一只 Peft Lora 包装（§7-C 的 r/alpha/dropout/target）。

    ``lora_override`` 让 overfit 阶段以 ``overfit.lora`` 覆盖起步参数（§7-G：
    dropout=0），其余阶段用顶层 ``config["lora"]``。

    包装后立即调 `assert_text_only_lora_targets` 守卫"只训语言侧"这一口径
    （config 的 `target_modules` 必须是路径限定正则，理由见该函数与 S11_bug_log B7）。
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
    # 口径守卫（B7）：确认 LoRA 只挂在语言解码层，视觉塔零命中。放在这里是因为
    # 这是"包装已经发生、但训练尚未开始"的唯一时间点——错误在此刻就是可拦截的。
    assert_text_only_lora_targets(model)
    # PEFT + gradient checkpointing 前置（§7-C）：LoRA 冻结基座后，文本 embedding 的
    # 输出不带 grad。transformers 的 gradient_checkpointing_enable() 只在
    # `_hf_peft_config_loaded=True`（由 PeftModel.from_pretrained 置位）时才自动调用
    # enable_input_require_grads()；我们用 get_peft_model，该标志为 False，必须手动开。
    # 否则 reentrant checkpoint 会因“无输入需要 grad”把逐层输出 detach，最终
    # loss.requires_grad=False，backward 报
    # "element 0 of tensors does not require grad and does not have a grad_fn"。
    # 该调用经 PeftModel.__getattr__ -> LoraModel.__getattr__ 转发到底层 Qwen2.5-VL，
    # 在其 embed_tokens 上注册前向钩子，强制输出 requires_grad=True。
    model.enable_input_require_grads()
    return model


def make_trainer(
    model,
    processor,
    train_dataset,
    config: dict,
    *,
    overfit: bool = False,
    output_dir: str | Path,
) -> Trainer:
    """组装 HF Trainer（§2.4 + §7-C 参数）。

    - collator：`make_data_collator`（批融合）。
    - train 数据集：本模块不实现 IDataset；由调用方（CLI）把 JSONL 转成 HF Dataset，
      元素是 `dataset.encode_chat` 返回的 dict（processor 已在样本层面生成张量）。
    - ``remove_unused_columns=False``：collator 收到的就是 encode_chat 的 dict，不删列。
    - ``output_dir``：HF Trainer 的中间 checkpoint 落盘目录，由调用方传 CLI 的 `--out`。
      **不能**再用 `config["run_name"]`：那是相对路径，`save_steps=64` 一旦触发就会在
      当前工作目录下凭空建出 `./sft_smoke_1f/checkpoint-64`（train32 只有 16 step 所以
      从未暴露；overfit128 有 96 step 必然触发），既污染工作目录又让 checkpoint 与
      adapter 分家。
    """
    from drivealign.sft.collator import make_data_collator

    seg = "overfit" if overfit else "train32"
    # 基座训练参数在顶层 config["training"]（batch/lr/optimizer 等全量）；
    # 各阶段（train32/overfit）只对其中部分字段做覆盖（如 epochs）。合并而非替换，
    # 否则阶段段只含覆盖字段会丢基础参数（见报错 KeyError per_device_train_batch_size）。
    params = dict(config["training"])
    params.update(config.get(seg, {}).get("training", {}))

    train_args = TrainingArguments(
        output_dir=output_dir,
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
        data_collator=make_data_collator(processor.tokenizer.pad_token_id),
        train_dataset=train_dataset,
    )


def train(
    config: dict,
    train_dataset,
    *,
    output_dir: str,
    overfit: bool = False,
    loaded: LoadedModel | None = None,
):
    """加载模型/处理器 → 组装 Trainer → 训练 → 保存 adapter（G1 状态保真）。

    ``loaded`` 允许调用方传入已加载的 `LoadedModel`（复用同一份基座 + processor）。
    这一步是显存刚需：CLI 需要 processor 把 JSONL 编码成训练张量，若在此再
    `load_model_and_processor`，同进程就会驻留**两份 3B 基座**（≈6GB 额外显存，
    Step 3 实测 31.5/32.7GB 并触发 meta-device/offload 告警）。传入后全程单副本。
    """
    if loaded is None:
        loaded = load_model_and_processor(config)
    seg = "overfit" if overfit else "train32"
    lora_override = config.get(seg, {}).get("lora")
    model = build_lora_model(loaded.model, config, lora_override=lora_override)
    trainer = make_trainer(
        model,
        loaded.processor,
        train_dataset,
        config,
        overfit=overfit,
        output_dir=output_dir,
    )
    trainer.train()
    trainer.save_model(output_dir)
    return trainer


def save_train_state_reference(
    model,
    global_step: int,
    adapter_dir: str | Path,
    *,
    generation: dict,
) -> Path:
    """把训练端"状态保真参照"落盘（G1 硬层 + 软层比对对象）。

    内容（供新进程 `cli/sft_reload_verify.py` 消费）：
        - ``adapter_state``：训练结束时 **adapter（LoRA）权重**的 CPU state_dict
          （`sft/gate.adapter_state_dict`，选取口径见该函数 docstring）。
        - ``global_step``：训练结束时 `Trainer.state.global_step`。
        - ``generation``：训练端在固定样本上的贪心生成参照
          （`run_reference_generation` 的返回：sample_token/text/tokens）。软层判定
          比对"复验生成 vs 本参照"，故必须与权重同批落盘，避免两者对不上号。

    写进 adapter 目录（与 `adapter_model.safetensors` 同处），随 adapter 一起归档。
    """
    from drivealign.sft.gate import adapter_state_dict

    path = Path(adapter_dir) / TRAIN_STATE_REF_NAME
    torch.save(
        {
            "global_step": int(global_step),
            "adapter_state": adapter_state_dict(model),
            "generation": dict(generation),
        },
        path,
    )
    return path


def load_adapter_model(config: dict, adapter_dir: str | Path) -> LoadedModel:
    """在**新进程**里从配置基座 + adapter 目录重建 PeftModel（G1 复验入口）。

    为什么必须新进程
    ----------------
    §3 Step 3 要求"训练结束 → 新进程 reload"。同进程内 reload 会同时驻留 Trainer
    的模型与第二份 base+adapter（≈6GB 额外显存）；且同进程 load 无法暴露"只存在于
    磁盘上的 artifact 有问题"（例如 adapter 未真正落盘）。新进程只从磁盘重建，
    才真正验证"save adapter → 可靠 load"这条链路。

    返回
    ----
    LoadedModel，其中 `.model` 是已 `eval()` 的 PeftModel（含 LoRA 权重）、
    `.processor` 与训练同源。可直接喂 `forward_fixed_batch` 与
    `inference.base_runner.generate_one`。
    """
    from peft import PeftModel

    loaded = load_model_and_processor(config)
    loaded.model = PeftModel.from_pretrained(loaded.model, str(adapter_dir))
    loaded.model.eval()
    return loaded


def fixed_sample_batch(
    processor, samples_path: str | Path, dataroot: str | Path
) -> tuple[dict, dict]:
    """取冻结样本**第一条** → 训练同款编码 + collator 拼 batch（G1 软层"固定输入"）。

    训练端（`run_reference_generation` 生成参照）与复验端（生成复验）都走这一个
    函数，保证两端"固定输入"逐位一致；若各写一份编码逻辑，任何漂移都会让软层比对
    失去意义。

    复用 `dataset.encode_frozen_sample` + `collator.make_data_collator`：collator 一步
    完成张量复位、加批维与 attention_mask 生成（避免 HF Dataset 行索引把张量退化成
    list，见 S11_bug_log B5）。

    返回 ``(record, batch)``：`record` 是原始 JSONL 记录（取 prompt/image_relpath/
    target_text/sample_token），`batch` 是可直接喂 `forward_fixed_batch` 的 tensor dict。
    """
    from drivealign.sft.collator import make_data_collator
    from drivealign.sft.dataset import encode_frozen_sample, load_frozen_records

    record = load_frozen_records(samples_path)[0]
    row = encode_frozen_sample(processor, record, dataroot)
    collate = make_data_collator(processor.tokenizer.pad_token_id)
    return record, collate([row])


def forward_fixed_batch(model, fixed_batch: dict) -> torch.Tensor:
    """对给定模型跑一次固定 batch 的手动 forward，返回 logits。

    用途：G1 软层的确定性部分——验证 reload 后模型能完成 1F forward 且 logits 无
    NaN/Inf（配合 `sft/gate.nonfinite_parameter_names` 覆盖权重侧）。

    注意（两处历史坑，见 S11_bug_log B5/B6）
    ----------------------------------------
    - `fixed_batch` 必须是**张量**：直接传 collator 输出，勿传 HF Dataset 行索引的
      裸 dict（后者把张量列退化成 list，`embed_tokens` 会报
      "argument 'indices' must be Tensor, not list"）。
    - 手动 forward 不经 HF Trainer 的 `_prepare_inputs`，需自行把 batch 张量搬到
      模型输入设备；否则 `device_map="auto"` 下报 "Expected all tensors to be on the
      same device ... cuda:0 and cpu"。
    """
    forward_kwargs = {
        "input_ids": fixed_batch["input_ids"],
        "pixel_values": fixed_batch["pixel_values"],
        "image_grid_thw": fixed_batch["image_grid_thw"],
    }
    # collator 输出带 attention_mask（单样本时全 1）；有则显式传入，与训练 forward 一致。
    if fixed_batch.get("attention_mask") is not None:
        forward_kwargs["attention_mask"] = fixed_batch["attention_mask"]
    device = model.device
    forward_kwargs = {
        k: (v.to(device) if torch.is_tensor(v) else v) for k, v in forward_kwargs.items()
    }
    with torch.no_grad():
        logits = model(**forward_kwargs).logits
    return logits


def prepare_model_for_generation(model) -> None:
    """把训练态模型切到可生成态（软层生成参照/复验共用同一份状态处理）。

    三件事，缺一即可能让贪心生成不确定或直接报错：

    1. ``eval()``：关 dropout。训练态残存的 LoRA dropout（顶层 r/dropout=0.05）会在
       前向里注入随机性，让"贪心"生成也不再可复现。
    2. ``gradient_checkpointing_disable()``：训练期开的重计算会拦 generate，且会把
       每层输出 detach（生成不需要、也不该带）。
    3. ``config.use_cache = True``：训练期检查点会强制 ``use_cache=False``（训练日志里
       的 "`use_cache=True` is incompatible with gradient checkpointing" 即由此而来），
       生成必须打开 KV cache，否则每一步都重算整段前缀（慢且可能被 generate 警告改写）。

    该函数只服务 G1 链路：训练已结束，模型不复用于后续训练，故不提供"恢复训练态"。
    """
    model.eval()
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    model.config.use_cache = True


def run_reference_generation(
    loaded: LoadedModel,
    model,
    config: dict,
    samples_path: str | Path,
    dataroot: str | Path,
) -> dict:
    """训练端在固定样本上贪心生成一次，落盘为软层"训练端输出"参照。

    软层基准（用户 2026-10-06 拍板）
    --------------------------------
    软层判定 = "复验进程的输出 vs **训练端同一固定样本的输出**"四字段语义相等，即验证
    `save adapter → 新进程 reload` 后**模型功能不变**（真正的状态保真）。与冻结 GT 的
    一致率降为复验报告里的信息项——"训练是否真的学会"属评测阶段的判定目标，不是 G1。

    为什么必须复用 `inference.base_runner.generate_one`：它与复验端、评测端共用同一条
    preprocess（`add_generation_prompt=True`）+ 贪心解码路径，两端逐位对齐；若训练端
    自写一段 generate，编码差异会直接污染比对。

    ``model`` 传的是训练结束的 `trainer.model`（正是 `save_model` 落盘的那份权重）。
    返回 dict：``sample_token`` / ``text`` / ``input_tokens`` / ``output_tokens``。
    """
    from drivealign.inference.base_runner import generate_one
    from drivealign.sft.dataset import load_frozen_records

    prepare_model_for_generation(model)

    record = load_frozen_records(samples_path)[0]
    # 借用 LoadedModel 只为一处：`generate_one` 的入参类型（model + processor 同源）。
    ref_loaded = LoadedModel(
        model=model,
        processor=loaded.processor,
        model_path=loaded.model_path,
        dtype=loaded.dtype,
    )
    result = generate_one(
        ref_loaded,
        Path(dataroot) / record["image_relpath"],
        record["prompt"],
        generation_config=config.get("generation"),
    )
    return {
        "sample_token": record["sample_token"],
        "text": result.text,
        "input_tokens": int(result.input_tokens),
        "output_tokens": int(result.output_tokens),
    }