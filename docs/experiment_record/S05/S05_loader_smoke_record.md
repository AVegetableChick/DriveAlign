# S05：AutoVLA Loader / Collator Smoke 检查记录

## 1. 目标与边界

原样驱动 pinned checkout 的 `SFTDataset` / `DataCollator`（上游代码零修改），确认 loader → processor → collator 的真实契约。本记录只存执行流程与结论；**batch sample 格式的权威版本**见 `references/autovla_loader_batch_format.md`（含 3.1 节 `pixel_values_videos` (5760, 1176) 的组织方式解析）。

边界：三相机、`acceleration`、`instruction`（future-derived）仅为 Reference only，不进入 DriveAlign 输入。

## 2. 输入与产物

- 输入：
  - Stage 05 preprocessing 产物 164 样本（`artifacts/s05_autovla_prep_smoke/train/`，无 DriveLM，`cot_output=[]`）
  - 上游 `config/training/qwen2.5-vl-3B-mix-sft.yaml`（仅覆盖 `json_dataset_path` → 164 样本目录、`sensor_data_path` → None、两个相对路径绝对化）
  - `codebook_cache/agent_vocab.pkl`、本地 `Qwen2.5-VL-3B-Instruct` processor
- 检查入口：`DriveAlign/src/drivealign/cli/autovla_loader_smoke.py`（docstring 含示例启动命令）
- 产物：`artifacts/s05_autovla_prep_smoke/loader_smoke_report.json`

## 3. 执行流程

1. 前置自检：`navsim`/`nuplan`/`qwen_vl_utils`/`peft` import 链通过，`agent_vocab.pkl` 存在。
2. 在 harness 内对 `process_vision_info` 做被动 spy，捕获 loader 实际传给 processor 的 video 路径序列（上游代码不改）。
3. 实例化 `SFTDataset`（`using_cot=True`），对首条、末条、随机样本（seed=42）逐项断言。
4. `DataCollator` 对 2 样本小批量检查标签掩码与张量 stack。
5. 首次运行 22/23：唯一 FAIL 为 harness 自身预期写错（把 patch 展平张量当 per-sample 张量核对行数），修正检查逻辑后复跑全过——上游行为始终正常。

## 4. 验证结果（23/23 PASS）

| 检查项 | 结果 |
|---|---|
| `len(dataset)` | 164 == 目录 json 数 |
| video 路径（spy 实测） | 每样本 3 相机 × 4 帧，`file://` 序列与 JSON `front/front_left/front_right_camera_paths` 逐一对齐（顺序敏感） |
| 帧分辨率 | 每帧 224×420（H·W=94080 ≤ 像素预算 100352，28 的倍数） |
| 文本契约 | 3 个 `<|video_pad|>`；velocity/acceleration/instruction 注入；no-CoT 分支 `The final output action is:` |
| 目标张量 | `gt_action` [1,10]、`gt_trajectory` [10,2] |
| Collator | `input_ids` (2,1058)；assistant（token [151644,77091]）前 labels 全 -100；`gt_action` (2,1,10)、`gt_trajectory` (2,10,2)；`pixel_values_videos` (5760,1176) |

## 5. 遗留与交接

- **CoT 分支未在真实数据上触发**（无 DriveLM 标注），`<think>` 监督格式留待 Stage 07 引入 DriveLM 后补验。
- Stage 06 复用边界：`video_inputs`/消息组织/processor 数值契约可复用于 CAM_FRONT 单相机；`gt_action`/`gt_trajectory` 属 target 侧；`instruction`、三相机、`acceleration` 均 Reference only。
- 字段级 policy 判定见 `references/autovla_preprocessing.md`。
