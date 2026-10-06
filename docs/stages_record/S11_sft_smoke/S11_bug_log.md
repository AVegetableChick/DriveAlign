# S11 SFT Smoke —— 训练链路 Bug 台账（时间序）

> 目的：记录 Step 3（1F 32 条 train/save/reload）训练链路首次打通时暴露的运行时 bug
> 及其解决方案，供 S12 全量 SFT 复用，避免重复踩坑。
> 记录范围：2026-10-05 ~ 2026-10-06 冒烟首跑排障（tmux `hkz1`）。
> 命令模板：
>
> ```
> PYTHONPATH=DriveAlign/src python -m drivealign.cli.sft_train \
>   --config DriveAlign/configs/sft/sft_smoke_1f.yaml \
>   --samples runs/S11_sft_smoke/sft_samples/train32.jsonl \
>   --out runs/S11_sft_smoke/checkpoints/smoke32 \
>   --dataroot data/nuscenes/trainval 2>&1 | tee runs/S11_sft_smoke/checkpoints/smoke32/train32.log
> ```

## 0. 摘要

六个 bug 沿训练链路依次暴露（组装 → collate → forward → backward → reload×2），每修一个就
前进一步，属于典型的"首跑 bring-up"序列，非方案性缺陷。B1–B4 阻断训练，修完后训练已
16/16 step 跑通（train_loss≈0.73）；B5–B6 出现在**训练跑通之后**的 G1 reload 复验段，
修复后用已保存 adapter 独立复验通过（logits `[1, 2384, 151936]`，无 NaN/Inf）。

| # | 暴露阶段 | 报错（首行） | 根因一句话 | 解决 | 状态 |
|---|---|---|---|---|---|
| B1 | Trainer 组装 | `AttributeError: 'Qwen2_5_VLProcessor' object has no attribute 'pad_token_id'` | `pad_token_id` 挂在 tokenizer 上，processor 不代理该属性 | 改用 `processor.tokenizer.pad_token_id` | 已解决 |
| B2 | 首个 batch collate | `TypeError: expected Tensor as element 0 in argument 0, but got list` | HF DataLoader 交付后 `pixel_values` 由 tensor 退化为 list，`torch.cat` 拒绝 | 拼接前用 `torch.as_tensor` 复位 | 已解决 |
| B3 | 首个 forward | `TypeError: forward() got an unexpected keyword argument 'seq_len_max'` | collator 多返回了一个 logging 辅助 key，被整份透传给 `model(**inputs)` | 从 collator 输出中移除该 key | 已解决 |
| B4 | backward | `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn` | LoRA + gradient checkpointing：`get_peft_model` 未置 `_hf_peft_config_loaded`，transformers 跳过 `enable_input_require_grads()` | 构建 LoRA 后手动调用 `model.enable_input_require_grads()` | 已解决 |
| B5 | G1 reload/infer | `TypeError: embedding(): argument 'indices' (position 2) must be Tensor, not list` | HF Dataset 行索引把张量列退化成 list，且缺批维 | 复用 collator 拼固定 batch | 已解决 |
| B6 | G1 reload/infer forward | `RuntimeError: Expected all tensors to be on the same device ... cuda:0 and cpu` | 手动 forward 不经 Trainer，batch 张量留在 CPU | forward 前 `.to(model.device)` | 已解决 |

---

## B1：processor 无 `pad_token_id`

- **现象**
  ```
  File ".../src/drivealign/sft/train.py", line 102, in make_trainer
      data_collator=make_data_collator(processor.pad_token_id),
  AttributeError: 'Qwen2_5_VLProcessor' object has no attribute 'pad_token_id'
  ```
- **根因**：`Qwen2_5_VLProcessor` 是图像+文本处理器的组合壳，`pad_token_id`
  实际定义在其持有的 `tokenizer` 上，processor 本身没有该属性（也不做透明代理）。
- **解决**：在 `sft/train.py` 的 `make_trainer` 中改为
  `make_data_collator(processor.tokenizer.pad_token_id)`。
- **涉及文件**：`src/drivealign/sft/train.py`（`make_trainer`）。

## B2：`torch.cat` 首元素是 list

- **现象**
  ```
  File ".../src/drivealign/sft/collator.py", line 74, in collate_fn
      pixel_values = torch.cat([p for p in pixel_list if p is not None], dim=0)
  TypeError: expected Tensor as element 0 in argument 0, but got list
  ```
- **根因**：`dataset.encode_chat` 产出的 `pixel_values` 本是张量，但经 HF `DataLoader`
  取 batch 时经历了隐式的元素级转换（tensor → list / numpy），到 collator 时已不是
  `Tensor`，`torch.cat` 因此报首元素类型错误。
- **解决**：在 `sft/collator.py` 的 `collate_fn` 中，拼接前统一复位类型：
  ```python
  pixel_list = [torch.as_tensor(b["pixel_values"]) for b in batch if b["pixel_values"] is not None]
  grid_list  = [torch.as_tensor(b["image_grid_thw"]) for b in batch if b["image_grid_thw"] is not None]
  ```
- **涉及文件**：`src/drivealign/sft/collator.py`（`collate_fn`）。

## B3：collator 多返回 key 被透传进模型

- **现象**
  ```
  File ".../peft/tuners/tuners_utils.py", line 222, in forward
      return self.model.forward(*args, **kwargs)
  TypeError: forward() got an unexpected keyword argument 'seq_len_max'
  ```
- **根因**：`collate_fn` 除模型入参外还返回了一个日志辅助字段 `seq_len_max`。而
  `train.py` 设了 `remove_unused_columns=False`，HF Trainer 会把 collator 输出的
  **整份 dict** 原样透传给 `model(**inputs)`；Qwen2.5-VL 的 `forward` 没有
  `seq_len_max` 形参，于是第一个 step 就崩。`seq_len_max` 全仓库无消费方，属纯冗余。
- **解决**：从 `sft/collator.py` 的返回值中移除 `seq_len_max`，并在 docstring 中固化
  约束——"返回 dict 的 key 必须全部是模型 forward 能接收的入参"。
- **回归防护**：`tests/unit/test_sft_collator.py` 新增
  `test_collate_keys_are_model_inputs_only`，断言输出 key 集合恰好为
  `{input_ids, labels, attention_mask, pixel_values, image_grid_thw}`，
  防止未来再混入辅助字段。
- **涉及文件**：`src/drivealign/sft/collator.py`、`tests/unit/test_sft_collator.py`。

## B4：backward 报 loss 无 grad_fn

- **现象**
  ```
  UserWarning: None of the inputs have requires_grad=True. Gradients will be None
  ...
  File ".../accelerate/accelerator.py", line 2359, in backward
      loss.backward(**kwargs)
  RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
  ```
- **根因**：LoRA + gradient checkpointing 的经典冲突，链条如下：
  1. LoRA 冻结基座参数，文本 `embed_tokens` 的输出 `requires_grad=False`。
  2. transformers 的 `gradient_checkpointing_enable()` 只在
     `_hf_peft_config_loaded=True`（由 `PeftModel.from_pretrained` 置位）时，才自动调用
     `enable_input_require_grads()`；本链路用的是 `get_peft_model`，该标志为 `False`，
     于是自动兜底被跳过。
  3. 落入 torch 的 **reentrant** checkpoint（transformers 默认 `use_reentrant=True`），
     `CheckpointFunction.forward` 检测到"无输入需要 grad"，把每层输出 detach；该
     detach 逐层级联（第 0 层输出无 grad → 第 1 层输入无 grad → …），最终
     `loss.requires_grad=False`，backward 抛错。
- **解决**：在 `sft/train.py` 的 `build_lora_model` 中，`get_peft_model` 之后补：
  ```python
  model.enable_input_require_grads()
  ```
  该调用经 `PeftModel.__getattr__` → `LoraModel.__getattr__` 转发到底层 Qwen2.5-VL，
  在其 `embed_tokens` 上注册前向钩子，强制输出 `requires_grad=True`，使梯度能穿过
  冻结层回流到 LoRA 层。
- **验证**：以 tiny 本地 Qwen2 + `get_peft_model` 实测——调用不报错、`_require_grads_hook`
  注册成功、forward 输出 `requires_grad=True`。
- **备选（未启用）**：若后续仍遇同类问题，可在 `TrainingArguments` 加
  `gradient_checkpointing_kwargs={"use_reentrant": False}` 作为兜底（非 reentrant 路径
  无该 requires_grad 检查）。
- **涉及文件**：`src/drivealign/sft/train.py`（`build_lora_model`）。

## B5：reload 时 input_ids 是 list 而非张量

- **现象**（训练已 16/16 跑完、adapter 已保存后，进入 G1 reload 复验时触发）
  ```
  File ".../src/drivealign/cli/sft_train.py", line 96, in main
      model, logits = reload_and_infer(config, args.out, batch)
  File ".../src/drivealign/sft/train.py", line 172, in reload_and_infer
      logits = model(
  File ".../transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py", line 1792, in forward
      inputs_embeds = self.model.embed_tokens(input_ids)
  TypeError: embedding(): argument 'indices' (position 2) must be Tensor, not list
  ```
- **根因**：`cli/sft_train.py` 用 `fixed = dataset[0]` 取固定样本，但 HF `Dataset`
  的行索引会把 Arrow 存储的数值列**退化成 Python list**（张量 → list，且丢失批维）。
  这个裸 dict 被直接喂给 `model(input_ids=...)`，`embed_tokens` 收到 list 即报错。
  与 B2 同源（都是"张量经 HF 数据容器往返后退化为 list"）。
- **解决**：在 `cli/sft_train.py` 中复用**训练同一个 collator** 把单样本拼成 batch，
  一步完成张量复位、加批维与 `attention_mask` 生成，保证与训练 forward 逐位对齐：
  ```python
  from drivealign.sft.collator import make_data_collator
  collate = make_data_collator(loaded.processor.tokenizer.pad_token_id)
  batch = collate([dataset[0]])
  ```
  同时 `sft/train.py` 的 `reload_and_infer` 改为把 collator 产出的 `attention_mask`
  （单样本时全 1）一并显式传入 forward。
- **涉及文件**：`src/drivealign/cli/sft_train.py`、`src/drivealign/sft/train.py`
  （`reload_and_infer`）。

## B6：reload forward 张量与模型不同设备

- **现象**（B5 修复后、同一个 reload 段继续暴露；独立复验时可见 batch 已是正确张量）
  ```
  batch shapes: {'input_ids': (1, 2384), 'pixel_values': (7296, 1176), 'image_grid_thw': (1, 3), ...}
  File ".../modeling_qwen2_5_vl.py", line 1792, in forward
      inputs_embeds = self.model.embed_tokens(input_ids)
  RuntimeError: Expected all tensors to be on the same device, but found at least two
  devices, cuda:0 and cpu! (when checking argument for argument index ...)
  ```
- **根因**：`reload_and_infer` 是**手动 forward**，不经过 HF Trainer 的 `_prepare_inputs`
  （后者会自动把 batch 张量搬到模型设备）。collator 产出的张量在 CPU，而
  `device_map="auto"` 下模型权重在 `cuda:0`，`embed_tokens` 的 index_select 因此撞设备不一致。
- **解决**：在 `sft/train.py` 的 `reload_and_infer` 中，forward 前把 batch 张量搬到
  模型输入设备（`model.device`，即 `device_map` 下首参数/embedding 所在设备）：
  ```python
  device = model.device
  forward_kwargs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in forward_kwargs.items()}
  ```
- **复验结果**（用已保存 adapter 独立跑，不重训）：
  `reload_logits_shape=[1, 2384, 151936]`，`has_nan=False`，`has_inf=False` → PASS。
- **备注**：training 侧不受此影响（Trainer 已代管设备搬运）；本项只影响手动推理/reload 路径。
- **涉及文件**：`src/drivealign/sft/train.py`（`reload_and_infer`）。

---

## 5. 非阻断性告警与操作项（不属 bug，登记备查）

- `tee: runs/S11_sft_smoke/checkpoints/smoke32/train32.log: No such file or directory`
  —— 输出目录不存在导致日志未落盘（训练不受影响）。**操作项**：跑前先
  `mkdir -p runs/S11_sft_smoke/checkpoints/smoke32`。
- `huggingface/tokenizers: The current process just got forked ...`
  —— DataLoader 多进程 fork 后的 tokenizers 并行告警，可按提示设
  `TOKENIZERS_PARALLELISM=false` 消除，不影响正确性。
- `` `use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False` ``
  —— transformers 自动纠正，无需处理。
- `No label_names provided for model class 'PeftModelForCausalLM' ...`
  —— `label_names` 为空不影响损失计算（loss 由 `labels` 入参直接产生），可忽略。