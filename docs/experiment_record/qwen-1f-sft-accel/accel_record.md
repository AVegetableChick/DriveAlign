# SFT Smoke 训练加速记录（1F）

> 记录 SFT smoke 阶段"发现问题 → 做实验 → 敲定参数"的完整链路与实测数字，供后续
> 4F/全量训练复用。时间 2026-10-06；决策台账见
> `stages_record/S11_sft_smoke/S11_decision_log.md`（D4 冻结 / D5 batch A/B）。
>
> **最终参数**：`max_pixels=360000` + `batch=1` + `grad-accum=4` + `lr=1e-4`
> （配置 `configs/sft/sft_smoke_1f.yaml`，现行 sha256 `05231b34…`）。
> 注：本文的 A/B 与显存读数取自 **B7 修复前**的配置（当时 sha `3d91290f…`，LoRA 误训了 ViT MLP）；
> 因 B7 只改 LoRA 目标、不改 batch/分辨率，**上述结论与数字不受影响**，但复现锚点已按现行 sha 更新（见 §5）。

---

## 1. 发现问题

1. **太慢**。train32（32 条 / 2 epoch / 16 optimizer step）在原生分辨率下
   `train_runtime = 290.86 s`，即 18.18 s/step（4.54 s/micro-batch）。
   全量 train split = **22,941 帧**，按"每样本一次 micro-batch"一阶外推：

   | 口径 | 计算 | 结果 |
   |---|---|---|
   | 每 epoch | 22,941 × 4.54 s | ≈ **29 h** |
   | 3 epoch | ×3 | ≈ **87 h（≈3.6 天）** |

   且该口径**不含**数据加载/编码开销，实际墙钟只会更长。

2. **显存读数一度被 bug 污染**。早期观测到"batch=1 已占满 31.5/32.7 GB"，据此曾把
   "加大 batch 换吞吐"列为首选杠杆。该读数实为**同进程双驻 3B 基座**的产物（CLI 加载
   一次 + `train()` 内部再加载一次），修掉后 batch=1 的真实峰值见 §2.3。
   ——教训：**先确认资源读数未被 bug 污染，再据此做容量决策。**

## 2. 实验

### 2.1 A1 输入分辨率减半（`max_pixels=360000`）—— 唯一有效的杠杆

- **机制澄清**（曾误述为"改 patch_size"）：`patch_size=14` **不变**；是 processor 用
  `max_pixels` 触发 `smart_resize`，在**原图**上重采样（1600×900 → 784×448），
  ViT 的 patch 数按面积线性下降。
- 链路：原生 7296 patch / 视觉 token 1824 / 序列 2384 → A1 1792 patch / **448** / **1008**。
- **实测**（同 train32、同 batch/accum）：

  | 口径 | runtime | s/step | s/micro-batch | 加速 |
  |---|---|---|---|---|
  | 原生 | 290.86 s | 18.18 | 4.54 | — |
  | A1 | **90.35 s** | 5.65 | 1.41 | **≈3.2×** |

### 2.2 A2 加大 batch（`batch=2 + accum=2`，有效 batch 仍 4、step 仍 16）—— 实测否决

**实测 A/B**（同 train32）：

| 设置 | 有效 batch | step | runtime | samples/s | peak alloc | peak reserved |
|---|---|---|---|---|---|---|
| bs=1 + accum=4 | 4 | 16 | **90.35 s** | **0.708** | 10.13 GiB | 15.08 GiB |
| bs=2 + accum=2 | 4 | 16 | 108.08 s | 0.592 | 12.73 GiB | 19.01 GiB |

两次样本通行总量相同（各 64 次单样本前向），故是纯效率差异：bs=2 使 runtime
**+19.6%**、每样本吞吐 **−16%**、显存多占 **3.9 GiB**。

**判据不在"慢了一点"，而在方向**：若 bs=1 处于 batch 饥饿区间，加大 batch 应提升吞吐，
实际反而下降 ⇒ **bs=1 已不饥饿，富余显存换不来速度**。

变慢的机制假设（**未做 profiling，不采信为结论**）：① 短样本被右 pad 到批内最大长度仍
走完整计算；② 出现真实 padding 后 `attention_mask` 不再全 1，FlashAttention2 失去
快捷路径；③ 显存跨阈值后分配器/选核换区制。

> 附带收获：经 collator 的 **batch>1 集成路径**由此获得一次真实执行，G1 复验亦 PASS
> （硬层 696 张量、`worst_abs_diff=0.0`——该读数取自 bug 期，696 含 192 个 `visual.*`；
> B7 修复后正确值应为 504）。

### 2.3 显存画像（A1 + batch=1）

HF `Trainer` **默认不记显存**，为此在 `cli/sft_train.py` 加了
`torch.cuda.reset_peak_memory_stats()` + `_cuda_peak_memory()`。注意**取值顺序**：
必须在生成 G1 软层参照**之前**读走——`base_runner.generate_one` 内部会再 reset 一次，
否则读到的只是生成阶段（KV cache）的峰值。

| 指标 | 值 |
|---|---|
| peak allocated | **10.13 GiB** |
| peak reserved | **15.08 GiB** |
| 卡总显存 | 31.48 GiB |
| train_loss / runtime | 0.7988 / 90.35 s |

## 3. 敲定参数：短期 A/B + 长期预算

**短期（A/B 实测）**：加 batch 在方向上是负收益，见 §2.2。

**长期（决定性理由：显存要为 4F/多视角预留）**：

| 口径 | 视觉 token | 序列长度 |
|---|---|---|
| 1F @ A1（当前，实测） | 448 | 1008 |
| **4F @ A1（推算）** | **4×448 = 1792** | **≈2380** |
| 原生 1F（A1 之前，实测） | 1824 | 2384 |

> 4F 序列为**推算**：1F@A1 的文本/目标段 ≈ 1008−448 = 560，4F 的视觉段 4×448 = 1792，
> 外加帧标签等少量 token，合计 ≈2380——与原生 1F 的 2384 **几乎相同**。

即 **4F 的序列长度与视觉 patch 数基本回到原生 1F 的水平**——A1 省下的开销（视觉 patch
4×、序列 2.4×）会被 4F 原样吃回去。bs=2 现在多占的 3.9 GiB 到那时几乎必然要退回，
所以加 batch 是**一笔注定要撤销的改动**。

**推论（对后续排期有直接影响）**：A1 的加速是**一次性的**。4F@A1 单 micro-batch 耗时
回到 ≈4.54 s，全量 4F 训练又回到 ~29 h/epoch 量级 ⇒ **4F 落地时不能沿用 1F 的加速结论，
需单独立项**。其中 `A4 缓存冻结 ViT 图像 embedding` 从"后备"升级为正经候选：4 个视角
各过一次 ViT、每 epoch 都在重算同一批确定性输出，缓存可省掉其中 **3/4 的 ViT 算力**
（**端到端约 1.4×，不是 4×**），算力与显存双省。

> **A4 有前置条件（B7，已修复并重跑确认）**：实测 smoke32 的 adapter 里有 **192 个张量落在
> `visual.blocks.*.mlp.*`** 上——LoRA 的裸后缀 `target_modules` 把 ViT 的 MLP 一起命中了，
> **ViT 并未真正冻结**，故当时缓存并不成立。2026-10-06 已按 bug 修正：`target_modules`
> 改为路径限定正则（`model\.layers\.\d+\....`），并加运行时守卫
> `sft/train.py: assert_text_only_lora_targets` 与回归单测
> `tests/unit/test_sft_lora_targets.py`。重跑后 adapter 张量 **696 → 504**、`visual` 命中
> **192 → 0**，G1 双层判定 PASS ⇒ **前提现已成立**。方法与理由见
> `stages_record/S11_sft_smoke/S11_vit_embedding_cache.md`，缺陷记录见
> `stages_record/S11_sft_smoke/S11_bug_log.md` B7。

**由此确定的形状**：`batch=1` 长期不动；提高有效 batch 的诉求一律交给 **grad-accum**
（accum 不占额外显存，batch 才占）——"小 batch + 大 accum"是本项目当前正确的形状。

## 4. 后续候选项（尚未落地）

- **A3 惰性数据集（S12 前置，非本阶段范围）**：`cli/sft_train.py::_load_dataset` 现为
  "预编码全部样本"（列表推导 + `HFDataset.from_list`）。按每样本 `pixel_values`
  ≈ 4.2 MB（A1；原生 ≈17.2 MB）估算：

  | 规模 | 预编码所需内存 |
  |---|---|
  | overfit128 | ≈ 0.54 GB（无碍） |
  | 全量 22,941 帧 | ≈ **96 GB**（原生 ≈ **394 GB**） |

  即 A1 把它从 394 GB 压到 96 GB，但**全量仍需改为 `__getitem__` 即时 decode+resize**
  （配 `dataloader_num_workers>0`）。
- **A4 冻结 ViT embedding 缓存**：见 §3 推论。

## 5. 复现锚点

| 对象 | 位置 / 值 |
|---|---|
| 配置 | `configs/sft/sft_smoke_1f.yaml`（sha `05231b34…`，B7 修复后的现行版） |
| train32（bs=1，冻结） | `runs/S11_sft_smoke/checkpoints/smoke32/`，runtime 90.87 s |
| train32（bs=2，A/B 证据，bug 期） | `runs/S11_sft_smoke/checkpoints/smoke32_bs2_vit_leak_bug/`，runtime 108.08 s |
| G1 报告 | `runs/S11_sft_smoke/reports/g1_smoke32.json` / `g1_smoke128.json`；bug 期 A/B 报告在 `reports/vit_leak_bug/` |
| 逐条细节 | `stages_record/S11_sft_smoke/S11_training_acceleration.md`（A1–A5） |
| B7 缺陷（ViT 被意外训练） | `stages_record/S11_sft_smoke/S11_bug_log.md`（B7） |
