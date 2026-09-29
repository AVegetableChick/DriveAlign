# AutoVLA nuScenes Preprocessing 字段映射（Reference）

- 上游：pinned checkout `third_party/AutoVLA`，入口 `tools/preprocessing/nusc_sample_generation.py`（命令与跳过逻辑见 `docs/experiment_record/S05/S05_preprocessing_smoke_record.md`）
- 验证数据：mini v1.0 train split 164 样本（2026-09-29 smoke，确定性重跑逐字节一致）
- policy 取值：**复用**（进入 DriveAlign）/ **target only**（仅训练监督侧）/ **Reference only**（记录契约，禁止进入 DriveAlign 输入）

## 1. 字段映射表

| 原始来源（raw field） | 上游字段（upstream field） | 上游用途（upstream use） | DriveAlign policy |
|---|---|---|---|
| nuScenes `sample` / `sample_data` 表，`sample.prev` 链（2Hz keyframe，`his_ts=3`+当前帧） | `{side}_camera_paths` ×6，每路 4 个绝对路径，顺序 [t-1.5, t-1.0, t-0.5, t]（`insert(0)`，最旧→最新） | loader 取 3 前向相机（front/front_left/front_right）各 4 帧组装 `type:"video"` 消息 | **CAM_FRONT 4 帧：复用**（与 S04 验证窗口一致）；其余 5 路：**Reference only** |
| nuScenes `ego_pose` 未来全局位姿（`fut_ts=16`，即 10 keyframe） | `gt_trajectory` 10×3 (x, y, heading)，0.5s 间隔，LIDAR_TOP lcf → x-forward/y-left（`x'=y, y'=-x`） | TokenProcessor 匹配 codebook → `gt_action` [1,10]，训练监督 | **target only**（坐标系约定可复用于 target 构建） |
| nuScenes `ego_pose` 历史差分（t 与 t-0.5s） | `velocity`（标量 m/s） | user 消息状态注入文本 | **复用**（纯历史信息，因果安全） |
| 同上 | `acceleration`（标量 m/s²） | user 消息状态注入文本 | **Reference only**（DriveAlign 因果白名单禁入，S03 决策） |
| 未来轨迹终点横向偏移（阈值 ±2m） | `instruction`（Go Straight / Turn Left / Turn Right） | user 消息文本（小写注入） | **Reference only**（**future-derived**，违反因果白名单；DriveAlign 指令语义待 M09/S07 另行设计） |
| DriveLM `v1_1_train_nus.json`（未接入） | `cot_output`（5 段：fov / 关键对象 / 运动意图 / 意图推理 / 规划动作） | `len==5` → `has_cot=True` → assistant `<think>…</think><answer>` 监督目标 | **target only**（Stage 07 引入 DriveLM 后生效；只进训练目标，不进输入） |
| DriveLM 样本覆盖关系 | `cot_output=[]` ↔ 5 段（`has_cot` 分支） | mix-sft 中 CoT 与 no-CoT 样本共存 | **target only**（分支开关本身可复用） |
| nuScenes val split 未来帧有效性 | `future_mask`（仅 val 输出） | 目标侧掩码 | **target only** |
| nuScenes `sample.token` | `token` + 文件名 `{token}.json`、`dataset_name="nuscenes"` | 样本索引、loader `glob(*.json)+sorted`、`data_path` 关联 | **复用**（作为样本 ID / 溯源键） |

## 2. 消息级数值契约（loader 侧实测）

| 契约项 | 值 | policy |
|---|---|---|
| video 消息 `min_pixels=max_pixels` | `28*28*128 = 100352`（config 写 109760，消息级 override 生效） | 复用 |
| 帧分辨率（smart_resize 结果） | 224×420，H·W=94080 ≤ 预算，28 倍数 | 复用 |
| 相机顺序 | front → front_left → front_right | **仅 CAM_FRONT 复用** |
| 帧顺序 | [t-1.5, t-1.0, t-0.5, t] 最旧→最新 | 复用 |
| assistant 起始 token | `[151644, 77091]`（`<|im_start|>assistant`），之前 labels=-100 | 复用 |

## 3. 使用说明

- Stage 06 构建 DriveAlign 训练集时，只搬运标记**复用**的路径与契约，重新实施因果白名单。
- 本表与 `references/autovla_loader_batch_format.md`（batch 格式）、`docs/experiment_record/S05/`（执行记录）互为索引。
