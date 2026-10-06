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

七个 bug 沿训练链路依次暴露（组装 → collate → forward → backward → reload×2 → 配方复核），
每修一个就前进一步，属于典型的"首跑 bring-up"序列，非方案性缺陷。B1–B4 阻断训练，修完后训练已
16/16 step 跑通（train_loss≈0.73）；B5–B6 出现在**训练跑通之后**的 G1 reload 复验段，
修复后用已保存 adapter 独立复验通过（logits `[1, 2384, 151936]`，无 NaN/Inf；2384 是
原生分辨率时期的序列长度，A1 降分辨率后为 `[1, 1008, 151936]`）。
>
> **后续演进（2026-10-06，Step 3 冻结时同步）**：G1 复验已从训练 CLI 内的**同进程**
> reload 拆成**新进程**入口 `cli/sft_reload_verify.py`（同进程 reload 会双驻 3B 基座，
> 且无法验证"只存在于磁盘的 artifact"）。`reload_and_infer` 已删除：B5 的修复落在
> `sft/train.py: fixed_sample_batch`，B6 的修复落在 `sft/train.py: forward_fixed_batch`。
>
> **B7 是另一类**（2026-10-06，Step 3 冻结后复盘发现）：它**不阻断**训练、**不报错**、
> loss 正常下降，是被"数 adapter 张量数除不尽"（696 vs 504）倒查出来的**静默缺陷**——
> 代价不是崩溃，而是训练配方被悄悄改掉（ViT 被意外训练，与"只训语言侧"的口径不符）。
> 它与 B1–B6 的排障性质不同，故单列并额外补了**运行时守卫**与**回归单测**，防止复发。

| # | 暴露阶段 | 报错（首行） | 根因一句话 | 解决 | 状态 |
|---|---|---|---|---|---|
| B1 | Trainer 组装 | `AttributeError: 'Qwen2_5_VLProcessor' object has no attribute 'pad_token_id'` | `pad_token_id` 挂在 tokenizer 上，processor 不代理该属性 | 改用 `processor.tokenizer.pad_token_id` | 已解决 |
| B2 | 首个 batch collate | `TypeError: expected Tensor as element 0 in argument 0, but got list` | HF DataLoader 交付后 `pixel_values` 由 tensor 退化为 list，`torch.cat` 拒绝 | 拼接前用 `torch.as_tensor` 复位 | 已解决 |
| B3 | 首个 forward | `TypeError: forward() got an unexpected keyword argument 'seq_len_max'` | collator 多返回了一个 logging 辅助 key，被整份透传给 `model(**inputs)` | 从 collator 输出中移除该 key | 已解决 |
| B4 | backward | `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn` | LoRA + gradient checkpointing：`get_peft_model` 未置 `_hf_peft_config_loaded`，transformers 跳过 `enable_input_require_grads()` | 构建 LoRA 后手动调用 `model.enable_input_require_grads()` | 已解决 |
| B5 | G1 reload/infer | `TypeError: embedding(): argument 'indices' (position 2) must be Tensor, not list` | HF Dataset 行索引把张量列退化成 list，且缺批维 | 复用 collator 拼固定 batch（现 `fixed_sample_batch`） | 已解决 |
| B6 | G1 reload/infer forward | `RuntimeError: Expected all tensors to be on the same device ... cuda:0 and cpu` | 手动 forward 不经 Trainer，batch 张量留在 CPU | forward 前 `.to(model.device)`（现 `forward_fixed_batch`） | 已解决 |
| B7 | Step 3 冻结后复盘 | *（无报错，静默）* | `lora.target_modules` 用裸后缀 list，被 peft 的后缀匹配连带命中视觉塔 MLP，**ViT 被意外训练** | 改路径限定正则 + 运行时守卫 `assert_text_only_lora_targets` + 回归单测 | 已解决 |

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
- **解决**：复用**训练同一个 collator** 把单样本拼成 batch，一步完成张量复位、加批维与
  `attention_mask` 生成，保证与训练 forward 逐位对齐：
  ```python
  from drivealign.sft.collator import make_data_collator
  collate = make_data_collator(processor.tokenizer.pad_token_id)
  batch = collate([row])          # row = dataset.encode_frozen_sample(...)
  ```
  同时把 collator 产出的 `attention_mask`（单样本时全 1）一并显式传入 forward。
  Step 3 冻结时这段逻辑已收敛到 `sft/train.py: fixed_sample_batch`，由**两端共用**：
  训练端 `run_reference_generation` 用它生成软层参照，复验端 `cli/sft_reload_verify.py`
  用它生成待比对输出。共用一个函数是**判定前提**——若两端各写一份编码，任何漂移都会
  让"软层语义相等"这个结论失去意义。
- **涉及文件**：`src/drivealign/sft/train.py`（`fixed_sample_batch`，供
  `cli/sft_reload_verify.py` 与 `cli/sft_train.py` 调用）。

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
- **解决**：在 `sft/train.py` 的 `forward_fixed_batch` 中，forward 前把 batch 张量搬到
  模型输入设备（`model.device`，即 `device_map` 下首参数/embedding 所在设备）：
  ```python
  device = model.device
  forward_kwargs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in forward_kwargs.items()}
  ```
- **复验结果**（用已保存 adapter 独立跑，不重训）：
  `reload_logits_shape=[1, 2384, 151936]`（原生分辨率时期；A1 后为 `[1, 1008, 151936]`），
  `has_nan=False`，`has_inf=False` → PASS。
- **备注**：training 侧不受此影响（Trainer 已代管设备搬运）；本项只影响手动推理/reload 路径。
- **涉及文件**：`src/drivealign/sft/train.py`（`forward_fixed_batch`）。

## B7：ViT 的 MLP 被 LoRA 意外训练（静默缺陷）

- **现象**：*（无报错）*。Step 3 冻结后核对 adapter 规模时发现张量数对不上：
  `n_adapter_tensors = 696`，而按 `7 类模块 × 36 层 × (lora_A + lora_B) = 504` 应为 **504**，
  多出 192。
- **定位**：直接读 `adapter_model.safetensors` 的 key 分组，得
  ```
  252 个模块  model.layers.<0..35>.{self_attn.{q,k,v,o}_proj, mlp.{gate,up,down}_proj}   ← 期望
   96 个模块  visual.blocks.<0..31>.mlp.{gate,up,down}_proj                              ← 泄漏
  ```
  换算张量数：252×2 = 504（语言侧）+ 96×2 = 192（视觉侧）= 696。
  192/2 = 96 = 3 类 × 32 层视觉 block，与"只有 MLP 三件套漏进来"吻合。
- **根因**：`configs/sft/sft_smoke_1f.yaml` 把 `lora.target_modules` 写成**裸后缀列表**
  `[q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]`。peft 对 **list** 的
  匹配语义是"精确相等 **或** 裸后缀"（peft 0.17.1，`peft/tuners/tuners_utils.py:
  check_target_module_exists`）：
  ```python
  elif key in config.target_modules:
      target_module_found = True
  else:
      target_module_found = any(key.endswith(f".{target_key}") for target_key in config.target_modules)
  ```
  视觉塔的 MLP 与语言侧**同名**（`visual.blocks.N.mlp.gate_proj` 以 `.gate_proj` 结尾），
  于是被一并命中。视觉塔的注意力用的是 `qkv`/`proj`、merger 用的是 `mlp.0`/`mlp.2`，
  都不匹配这 7 个后缀——这正是泄漏恰好只有 MLP 三件套的原因。
- **影响**：训练"看起来"完全正常——不报错、loss 从 ~2 降到 0.80。但配方已偏离
  「只训语言侧」的口径：ViT 被训练了。这与 A4（缓存 ViT embedding）、
  `S11_vit_embedding_cache.md` 中"ViT 必须真的冻结"这一前置条件**直接冲突**，
  故修正后 Step 3 的产物需作废重跑、重新登记 sha。
- **解决**：把 `target_modules` 换成**路径限定正则字符串**：
  ```yaml
  target_modules: 'model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))'
  ```
  peft 对 **str** 走 `re.fullmatch(pattern, key)`（`peft/utils/other.py:
  match_target_against_key`），对模块**全路径**做全匹配，范围被钉死在 `model.layers.<n>.`
  之下，`visual.*` 永不入选。（trl 亦有同类用法可参照：
  `examples/sft_diffusion_gemma/sft_diffusion_gemma.py` 的
  `target_modules=r"model\.(encoder\.language_model|decoder)\.layers\.\d+"`。）
- **防复发（两道闸门）**
  1. **运行时守卫** `sft/train.py: assert_text_only_lora_targets`——`get_peft_model` 后立即
     按结构特征（持有非空 `lora_A` 的包装层）收集命中模块，要求"非空 + 无 `.visual.` 命中 +
     全部在 `.layers.` 之下"，任一不满足即抛 `RuntimeError`，把"悄悄训错模块"变成
     "开训前立刻报错"。
  2. **回归单测** `tests/unit/test_sft_lora_targets.py`——直接读**冻结配置里的真实
     pattern**，用 peft 自己的 `check_target_module_exists` 校验：命中 7 类语言模块、
     不命中 `visual.*` 与 `embed_tokens`/`lm_head`/`norm`；并固化"裸后缀 list 会命中
     `visual.blocks.*.mlp.gate_proj`"这一根因，防止有人把配置改回去。
  实测验证（meta device 上跑真实 Qwen2.5-VL 结构，不占显存）：修正后守卫放行 **252** 个
  模块；换回裸后缀 list 时守卫拦下 **96** 个视觉模块。
- **教训**：LoRA 的 `target_modules` 用"裸名字"在**多模态**模型上不安全——视觉塔与语言塔
  层名大量重名，`endswith` 语义会把两边同名层一起收走。凡"只想训某一侧"，就必须用路径限定
  （正则）表达；`n_adapter_tensors` 这类**结构不变量**应被当作断言，而不是只记进日志。
- **涉及文件**：`configs/sft/sft_smoke_1f.yaml`（`lora.target_modules`）、
  `src/drivealign/sft/train.py`（`assert_text_only_lora_targets`）、
  `tests/unit/test_sft_lora_targets.py`（新增）。

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