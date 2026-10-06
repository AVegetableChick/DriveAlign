# S11 A4：ViT 图像 embedding 缓存（方法与理由）

> **状态**：候选方案，未落地。属 `S11_training_acceleration.md` 的 A4 项，为 4F/全量训练
> 做的前置调研。
> **关联**：`experiment_record/qwen-1f-sft-accel/accel_record.md` §3（4F 会把 A1 的加速吃回去，
> 故 4F 需单独立项，A4 由"后备"升级为正经候选）。

## 1. 一句话

ViT 的输出（图像 embedding）在训练期是**常量**，可以算一次存起来，之后每 epoch 直接查表，
跳过重复的 ViT 前向。

## 2. 可缓存性从哪来（机制）

注入点（transformers 4.49.0 `modeling_qwen2_5_vl.py` L1791–1809，已核对源码）：

```python
inputs_embeds = self.model.embed_tokens(input_ids)
image_embeds  = self.visual(pixel_values, grid_thw=image_grid_thw)
# 校验 (input_ids == image_token_id).sum() == image_embeds.shape[0]
inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
```

- `image_embeds` **只**由 `(ViT 权重, pixel_values, image_grid_thw)` 决定，与 `input_ids`
  的文本段、`labels`、batch 组合、optimizer step 全无关。
- 只要 LoRA 不挂在 ViT 上，ViT 权重不更新 ⇒ 同一张图在第 1 epoch 与第 3 epoch 过 ViT 的
  结果**逐位相同**。
- 可复用次数 = epoch 数（当前 2–3）× 视角数（4F 时每个视角各一张图）。
- 预处理侧已有确定性保证：`encode_frozen_sample` 走冻结 processor 并逐样本落 sha256
  ⇒ 同图的 `pixel_values` 字节稳定，可以直接当缓存 key。

## 3. ⚠️ 前提：ViT 必须真的冻结（B7 已修复并重跑确认）

**这个前提一度被破坏**（2026-10-06 复盘发现，记为 [S11_bug_log B7](S11_bug_log.md)）：
实测 `runs/S11_sft_smoke/checkpoints/smoke32/adapter_model.safetensors` 的 696 个张量里，

| 来源 | 数量 | 说明 |
|---|---|---|
| 语言侧 `layers.{0..35}` × 7 模块 × 2 | 504 | 符合 §7-C 设计意图 |
| **ViT `visual.blocks.{0..31}.mlp.{gate,up,down}_proj` × 2** | **192** | **非预期：ViT 的 MLP 也在被训练** |

**根因**：`config["lora"].target_modules` 当时是裸后缀列表（`q/k/v/o_proj` + `gate/up/down_proj`），
peft 对 **list** 按"精确相等 **或** 名字尾"匹配（`key.endswith(f".{name}")`）；而 ViT 的 MLP 恰好
也叫 `gate_proj` / `up_proj` / `down_proj`（ViT 注意力用的是 `qkv` / `proj`，不匹配——所以只有 MLP 中招）。

**后果（当时）**：ViT 权重每步都在变 ⇒ `image_embeds` 每步不同 ⇒ **缓存不成立**。

**修复（已落地并重跑确认）**：`target_modules` 改为**路径限定正则字符串**，把范围钉死在
`model.layers.<n>.` 之下（peft 对 **str** 走 `re.fullmatch`）；并新增运行时守卫
`sft/train.py: assert_text_only_lora_targets` 与回归单测 `tests/unit/test_sft_lora_targets.py` 防复发。
重跑后实测：`n_adapter_tensors` **696 → 504**、`visual` 命中 **192 → 0**（train32 与 overfit128 皆是），
G1 双层判定均 PASS（见 [S11_decision_log.md](S11_decision_log.md) D4/D6）。**故"ViT 冻结"这一前提现在成立。**

> **副作用提醒**：这是配方变更，原 smoke32（其 adapter 含 ViT MLP 训练）**不可比**，
> 已按 D4/D6 的口径重跑并重新登记 sha（bug 期产物改名留证为 `*_vit_leak_bug/`）。

## 4. 两种实现

### 设计 A：进程内拦截（推荐先做）

monkeypatch `model.visual.forward`，命中缓存则直接返回、**不跑** ViT。

- **key**：`pixel_values` 的**内容摘要**。为什么不能用样本 id 或 grid：拦截点只拿得到张量、
  拿不到样本身份；且多张不同的图可能共享同一 `grid_thw`。
- **容器**：smoke 用进程内 dict（128 图 × 1.8 MB ≈ 0.2 GB）；全量必须落盘/分片。
- **优点**：不改 batch 契约、不改 `model.forward`，且**不碰 `embed_tokens`**（见 §6 坑 1）。

### 设计 B：离线预计算落盘

训练前把每张图的 `image_embeds` 写成 shard（`.safetensors`），训练时按 key 读。

- **优点**：训练进程零 ViT 成本，且可跨实验复用；**缺点**：磁盘占用见 §5，需额外维护
  key→文件索引与一致性校验。

## 5. 收益量化

**缓存体积**（= post-merger 视觉 token × 2048 × 2 B）：

| 口径 | 视觉 token | 单图 | 全量 22,941 帧 |
|---|---|---|---|
| A1 1F | 448 | ≈ 1.8 MB | ≈ 42 GB |
| 原生 1F | 1824 | ≈ 7.5 MB | ≈ 172 GB |
| A1 4F | 448 / 视角 | ≈ 1.8 MB | ≈ 169 GB（91,764 图） |

> A1 的附带好处：缓存体积同比缩到 **1/4**。

**算力**（**FLOP 量级估算，非实测**）：ViT 前向 ≈ 2.8 TFLOP / 图（depth 32、hidden 1280、
1792 patch）；LM 侧 1F@A1 开梯度检查点后 ≈ 14 TFLOP / 样本 ⇒ ViT 约占 **~16%**；
4F@A1 时 ViT（4 视角）≈ 11.2 对 LM ≈ 23 ⇒ 约占 **~33%**。

⇒ 端到端预期收益约 **1.2×（1F）/ 1.4×（4F）**，**不是 4×**。所谓"4×"仅指省下的
**ViT 那部分算力**（4 视角省掉 3/4 的重复），不可当作整体加速比。**真实占比须 profiling
确认后再决定是否值得做。**

**显存**：ViT 激活（1792 patch × 32 block）不再驻留，对 4F 的显存预算尤其有价值。

## 6. 坑

1. **不要用 `inputs_embeds` 绕过 `embed_tokens`**。B4 的 `enable_input_require_grads()`
   把钩子挂在 `embed_tokens` 上；绕过它，reentrant gradient checkpointing 会再次把每层
   输出 detach，backward 报 "does not require grad"（**B4 复发**）。正确做法是保持
   `embed_tokens` 路径不变，只在 `self.visual` 处拦截。
2. **key 要对 dtype/device 稳定**：`pixel_values` 在 forward 内被 `.type(self.visual.dtype)`
   转成 bf16，取摘要时应统一按转换后的张量（或固定按 bf16）计算。
3. **缓存与确定性强耦合**：ViT 一旦被 LoRA 训练（§3）、或 processor 分辨率变化，旧缓存
   全部失效。key 里最好并入 `max_pixels`/分辨率与 ViT 权重版本。
4. **全量落盘 42–172 GB**，需先定存储位置与清理策略——与 A3 属同一类"全量开工前置"问题。
5. **与 A3 的交互**：A3 是"图像不预编码"，A4 是"ViT 输出预计算"。两者都在解全量内存/算力，
   应合并设计：一旦缓存了 embedding，原始 `pixel_values` 反而**不必长期驻留**。

## 7. 落地时的验证步骤

1. 先排除 ViT 的 LoRA（§3），重训 smoke，确认 `n_adapter_tensors = 504` 且 G1 PASS。
2. 加拦截 + 缓存后复验 G1 软层：reload 输出与训练端参照仍应**四字段语义相等**（缓存不得改数值）。
3. 断言"命中路径 vs 直接计算路径"的 `max|Δ| == 0`，并记录命中率。
4. profiling 取 ViT 真实占比，再决定是否为全量做落盘版。
