# AutoVLA Loader / Collator Batch Sample 格式（Reference）

- 上游：pinned checkout `third_party/AutoVLA`（`dataset_utils/sft_dataset.py` 的 `SFTDataset` 与 `DataCollator`，上游代码零修改）
- 配置：`config/training/qwen2.5-vl-3B-mix-sft.yaml`（仅覆盖 `json_dataset_path`、`sensor_data_path=None` 与两个相对路径绝对化）
- 验证：`artifacts/s05_autovla_prep_smoke/loader_smoke_report.json`，23/23 检查 PASS；检查入口 `DriveAlign/src/drivealign/cli/autovla_loader_smoke.py`
- 数据：Stage 05 mini preprocessing 产物 164 样本（无 DriveLM，`cot_output=[]`）

## 1. 单条样本 `dataset[i]` 返回值

| 键 | 值 | 说明 |
|---|---|---|
| `text` | str | `apply_chat_template(messages, add_generation_prompt=True)` 全文，含 system/user/assistant 三角色 |
| `image_inputs` | None | nuScenes 路径只用 video，无独立 image |
| `video_inputs` | list[3][4] | 3 相机 × 4 帧，PIL Image，已由 processor `smart_resize` 缩放至 **224×420** |
| `gt_trajectory` | tensor [10,2] | `gt_pos_raw`，未来 10 步 (x_forward, y_left)，0.5s 间隔 |
| `gt_action` | tensor [1,10] | 未来轨迹经 codebook（`codebook_cache/agent_vocab.pkl` TokenProcessor）匹配出的 action token 索引序列 |
| `has_cot` | bool | raw 路径恒 False；DriveLM 5 段 `cot_output` 齐全时为 True |
| `data_path` | str | 来源 JSON 绝对路径 |

## 2. 消息结构（chat messages）

- `system`：驱动 agent 角色设定
- `user`：
  - 状态注入文本：`velocity`（m/s）与 `acceleration`（m/s²）标量、`instruction`（`go straight` / `turn left` / `turn right`，小写）
  - 3 个 `{"type": "video", "video": [4 个 file:// 绝对路径]}`，相机顺序 **front → front_left → front_right**，帧顺序 **[t-1.5, t-1.0, t-0.5, t]**（最旧→最新，与 preprocessing JSON 逐一对齐）
  - 每路 video 消息级 `min_pixels=max_pixels=28*28*128=100352`
- `assistant`：
  - `has_cot=True`：`<think>…5 段推理…</think><answer>The final output action is: …</answer>`
  - `has_cot=False`：`The final output action is: …`（验证到的 no-CoT 分支文本含 "straightforward scenario" 引导）

## 3. `DataCollator` 输出 batch

| 键 | shape | 说明 |
|---|---|---|
| `input_ids` / `attention_mask` | (B, L) | mini 实测 (2, 1058) |
| `labels` | (B, L) | `<|im_start|>assistant`（token `[151644, 77091]`）**之前全部 -100**，仅监督 assistant 段 |
| `pixel_values_videos` | (N_patch, 1176) | 见 3.1 的组织方式解析 |
| `gt_action` | (B, 1, 10) | action token 序列 stack |
| `gt_trajectory` | (B, 10, 2) | 位置序列 stack |
| `has_cot` | list[bool] | 逐样本 CoT 标志 |

### 3.1 `pixel_values_videos` 的组织方式

三个维度层层叠加，最后一维决定"一行是什么"：

**① 时间维（temporal_patch_size=2）**：每路视频的 4 帧先两两配对成 2 个"帧对"：[(f0,f1), (f2,f3)]。因此**一行携带的是相邻 2 帧的信息，不是 1 帧**。

**② 空间维（patch_size=14）**：每帧 224×420 按 14×14 切块 → 16 行 × 30 列 = 480 块/帧。

**③ 行的解剖（特征维 1176）**：张量的 1 行 = 帧对中**同一空间位置**的 14×14 块 × 2 帧 × 3 通道，按 (C, T, P, P) 展平：

```
1 行 = 3 通道 × 2 帧 × 14 × 14 = 1176 个数
```

消费端 `Qwen2_5_VisionPatchEmbed`（transformers `modeling_qwen2_5_vl.py`）把每行 `view(-1, 3, 2, 14, 14)` 还原后过 `Conv3d(kernel=2×14×14, stride=同)`，**1 行 → 1 个视觉 embedding**。

**行数账目（batch 维平铺拼接）**：

```
每视频 patch 数 = 2 帧对 × 16 × 30 = 960
每样本 patch 数 = 3 路摄像机 × 960     = 2880
总 batch 的 patch 数 = 2 样本 × 2880    = 5760   → shape (5760, 1176)
```

行内排序 t→row→col（与模型 position_ids 计算的 `(t, h, w)` 展开一致）。不同样本/视频的行在 batch 维直接顺序拼接，边界由 `video_grid_thw`（每视频 [2, 16, 30]）记录。

**下游 token 数**：embedding 再做 2×2 空间合并（spatial_merge_size=2），960 → **240 个视觉 token/视频**，即文本模板中 `<|video_pad|>` 占位符的展开数；每样本 3 视频 = 720 个视觉 token 嵌入 `input_ids`（mini 实测 L=1058）。

## 4. 验证要点与遗留

- 逐项验证：video 路径序列与 preprocessing JSON 字段**顺序敏感对齐**（spy 实测 `process_vision_info` 入参）；帧数/分辨率/像素预算（H·W=94080 ≤ 100352，且为 28 倍数）；文本中 3 个 `<|video_pad|>`；状态标量注入；`gt_action` [1,10]；labels 掩码；张量 stack 与 patch 数自洽。
- **CoT 分支未在真实数据上触发**（无 DriveLM 标注），`<think>` 目标的监督格式留待 Stage 07 引入 DriveLM 后补验。
- Stage 06 复用边界：`video_inputs`/消息组织/processor 数值契约可复用于 CAM_FRONT 单相机；`gt_action`/`gt_trajectory` 属 target 侧；`instruction` 为 future-derived（Reference only）。
