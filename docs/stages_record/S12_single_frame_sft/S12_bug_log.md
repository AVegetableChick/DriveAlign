# S12 单帧 SFT —— 批量推理链路 Bug 台账（时间序）

> 记录 S12 batched_v1 path 开发与运行中的关键 bug、发现过程、根因与修复。
> ⚠️ 本文件曾因编辑器误撤销丢失，2026-10-07 依会话记录重建，内容与原版事实一致。

## B1（主 bug）：批量推理静默退化为单样本（2026-10-07 发现）

### 现象与发现过程（四步定位链）

1. **观察异常**：smoke 运行时 GPU 利用率仅 ~35%，与 bs=8 预期不符；
2. **telemetry 反查**：smoke 产物 JSONL 的 telemetry 字段里**没有 `batch_size` 键**
   （正常批量产物必有）——说明产线根本没走 `batched_runner.generate_batch`；
3. **小样本复现**：`--limit 2 --batch-size 4` 复跑，直接拿到完整 traceback：
   `Image features and image tokens do not match: tokens: 0, features 896`；
4. **CPU 层隔离**：不走 GPU，仅用 processor 复现编码过程，确认
   `input_ids.shape == (2, 1)`（预期 (2, ~2000)），定位到 prompt 塌缩。

### 根因（双层故障链）

- **触发层**：`generate_batch` 调 `apply_chat_template` 时把单个 message dict
  直接传入（应为 `[message]` 列表）。jinja 模板按 dict 的 **key 迭代**渲染，
  产出仅 1 个 token 的空 prompt → 图像 pad token 全部丢失 → 与 896 个视觉
  feature 数对不上而抛错；
- **回退层（放大器 1）**：外层 `except Exception` **静默吞掉异常**，逐条降级
  到冻结面单样本路径（`base_runner.generate_one`），8 条样本全部以 bs=1 跑完
  （131s），无任何日志提示——批量路径"看起来跑通了"；
- **放大器 2**：回退产物与正常产物**逐字节不可区分**（JSONL 行无 runner 标记），
  仅能靠 telemetry 缺 `batch_size` 键反推。

### 修复

1. [batched_runner.py] `apply_chat_template(messages=[...])` 包列表；
2. [batched_runner.py] 回退分支打印完整 traceback 留痕，不再静默。

### 验证链

- 11 个单测全绿（含批切分确定性、EOS 截断计数、never-abort 契约、telemetry 摊平）；
- smoke 复跑：**131.3s → 37.2s**（同一批 8 条），telemetry 确认 `batch_size: 8`；
- parity 抽查（8 条 vs eval_v6 冻结面）：8/8 结构完全一致（对象/类别/motion_state
  全同），差异全部为 bbox 个位扰动（如 `910.4` vs `906.0`）且 output_tokens
  逐条相等——bf16 批归约的预期签名，非行为退化；
- GPU 利用率认知修正：decode 阶段受显存带宽限制，**正常批量的利用率本就
  ~35-50%**；正确的 KPI 是吞吐而非 nvidia-smi 百分比（详见
  S12_batched_inference_speed.md §3 效应④）。

### 修复后读数基线（bs 扫描前，32 条 smoke 外推）

| 配置 | 每步成本 | 全量 5468 条外推 | 加速 |
|---|---|---|---|
| bs=1（eval_v6 冻结面实测） | 50.9 ms/token | ~13.6 h | 1× |
| bs=8（smoke 实测） | 75.5 ms/step（1.48×） | ~5.6 h | 2.4× |
| bs=16（外推） | — | ~3.2 h | 4.2× |
| bs=32（外推） | — | ~1.8 h | 7.5× |

（正式扫参读数以 S12_batched_inference_speed.md 为准：bs=16 定档 3.48×。）

## B2：manifest 面误用触发 hash 守卫（2026-10-07，守卫按设计工作）

- **现象**：smoke 首跑报 `request_hash mismatch`，token `0017c2…` 的
  `request_hash_1f` 与 manifest 不符；
- **根因**：smoke 命令抄了 [base_benchmark.py] docstring 里 **S09 v5 时代**
  的示例，`--anchor-manifest` 传了 `data/dataset_v4/anchor_policy_manifest.json`
  （v5 面），而批量配置是 v6 面（v5 `5c322035…` vs v6 `838dd0c6…`）；
- **修复**：[batch_benchmark.py] docstring 示例改为 v6 manifest 并加显眼警告；
- **登记**：守卫正确拦截"跑错面"，**不应因易踩而放松**；教训是示例命令
  必须与最新面同步。

## 运维注记（固化约定，供 batched_v2 / vLLM 面沿用）

1. **eval_v6（bs=1 参照面）**：在 164/5468 处中断（为批量实验让 GPU），
   恢复命令 = 原版 `1f_base_run.sh` 电池（`base_benchmark` + `--resume`），
   跑完后与 `eval_v6_batched/predictions_bs16.jsonl` 做 CPU diff 补 parity gate；
2. **回退留痕**：任何"批内失败降级单样本"路径的产物，telemetry 必须显式标记
   实际 runner path（如 `runner: fallback_single`），禁止静默回退；
3. **批大小承载**：`batch_size` 由配置文件承载（`sft_1f_v6_batched.yaml`），
   CLI 不再覆盖，保证 run_summary 的 config sha256 与实际行为一一对应。
