# S06：adapter（记录装配器）作用与流程

> 代码：`DriveAlign/src/drivealign/records/adapter.py`（`build_record` + `BuildResult`）。
> 定位：全项目**唯一**有权生产 `DriveAlignRecord` 的入口（Stage 06 产出点）。

## 1. 作用

adapter 回答的问题是："这条样本的**完整事实**是什么？"——把原始 nuScenes 数据库的一条锚点样本，装配成训练与评测共用的标准档案 `DriveAlignRecord v1`，并对坏数据做确定性质检分类。

| 项 | 内容 |
|---|---|
| 输入 | `nusc`（NuScenes 索引）、`anchor_token`（锚点 keyframe sample token）、`builder_commit` |
| 输出 | `BuildResult(record, reason_code, detail)`：**要么**完整 Record（reason_code=None），**要么**隔离分类（record=None），二者互斥 |
| 保证 | 数据级失败永不抛异常；只有装配产物自身未通过 `validate_record`（推导 bug）才 raise ValueError |

## 2. 装配流程（四步）

1. **读 4 帧历史链**：复用 S04 `read_keyframe_chain`（严格沿 `sample.prev`，禁用高频 sweep 链），取 anchor + 3 个 prev 的 CAM_FRONT 关键帧（旧→新）。链路失败直接透传 6 个 reason code。
2. **解析 ego pose**：对每个历史帧经 `sample_data → ego_pose / calibrated_sensor` 解析全局位姿（yaw 由四元数给出）；不可解析 → `MISSING_CALIBRATION`。
3. **派生当前速度（只用过去）**：anchor 与前一帧的全局 planar (x,y) 位移差 ÷ 实际时间差（≈0.5s）→ `ego_speed_mps`（backward difference，因果合法）；非有限/负 → `NUMERIC_ANOMALY`。
4. **采集未来证据**：沿 `sample.next` 走 `NUM_FUTURE_POSES=6` 帧（3.0s @ 2Hz）取全局位姿，进 `oracle_only`。scene 末尾自然截断（<6 帧），**不算隔离**；未来时间戳非严格递增复用 `NON_INCREASING_TIME`，坐标非有限 → `NUMERIC_ANOMALY`。

最后组装 7 键 Record（provenance 用 `DEFAULT_CONTRACT_VERSION="v3"`、`TEMPORAL_POLICY_VERSION="v1"`、`builder_commit`、`nuscenes_version`），先过 `validate_record` 再返回。

## 3. 隔离分类（quarantine taxonomy，8 类）

| 来源 | reason code |
|---|---|
| 透传 nuscenes_io | `INSUFFICIENT_HISTORY` / `CROSS_SCENE` / `NON_INCREASING_TIME` / `MISSING_IMAGE` / `BAD_GAP` / `NO_CAM_FRONT` |
| adapter 新增 | `MISSING_CALIBRATION`（ego_pose / calibrated_sensor 不可解析，含退化四元数）、`NUMERIC_ANOMALY`（速度或未来坐标非有限） |

## 4. 验证结果（Step 2 内联验证，6 组断言全过）

- happy path（scene-0061 锚点）：speed=8.414 m/s、future=6、`validate_record` 为空
- 同输入两次构建 + `to_dict → from_dict` roundtrip：canonical hash 一致（确定性）
- 场景首样本 → `INSUFFICIENT_HISTORY` 隔离正确
- 10 个 scene 末尾锚点：future=0 仍 valid（截断不隔离决策生效）
- 空 `builder_commit` 被 `validate_record` 拒绝（ValueError 路径）
- v1.0-mini 全量 sweep：**374 valid + 30 INSUFFICIENT_HISTORY**，全部确定性归类

## 5. 边界

- adapter 不决定模型看到什么（那是 serializer 的职责）；未来证据只进 `oracle_only`，`training_targets` 在 S06 恒为 None（S07/S08 填充）。
- `frame_token` = keyframe sample_token（已确认决策）；`sample_data_token` 在 v1 中裁剪。
