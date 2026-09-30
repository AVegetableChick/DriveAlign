# S06：DriveAlignRecord v1 格式与作用记录

> 状态：**Stage 06 完成**——record / adapter / serializer / 单测 / CLI 冒烟全部落地，五项 gate 通过（见 `S06_record_build_smoke.md`）。
> 代码：`DriveAlign/src/drivealign/records/record.py`（frozen dataclass + 校验），`DriveAlign/src/drivealign/records/__init__.py`。

## 1. Record 的定位与作用

Record 是训练与评测共同消费的**标准样本单元**（统一中间表示），但不是模型实际"看到"的格式。三层严格区分：

| 层 | 是什么 | 产出方 |
|---|---|---|
| Record | 样本完整档案（含全部 namespace） | `build_record`（S06 adapter，唯一产出点） |
| model request | 模型实际看到的 prompt + 图像 | serializer 从 `model_inputs` 派生 |
| batch 张量 | processor/collator 输出 | 训练侧再加工 |

Namespace 即因果守卫：

- 训练取 `model_inputs`（拼输入）+ `training_targets`（拼监督答案）；评测取 `model_inputs`（拼同样输入）+ `oracle_only`（取 GT 打分）。
- serializer 类型上只接收 `model_inputs`，是 Record 与 model request 之间的唯一衔接路径（Step 3 实现）。
- 未来证据只进 `oracle_only`；泄漏扫描用 `oracle_only` 里的真实未来 sample_token / 时间戳做指纹（无合成 canary 字段）。

## 2. v1 顶层结构（恰好 7 键，不可增减）

```
DriveAlignRecord
├── record_version = "v1"
├── sample_token          # anchor（当前帧）keyframe sample token
├── scene_token
├── model_inputs          # 模型可见输入（serializer 唯一输入）
│   ├── frames[4]（旧→新）
│   │   ├── image_relpath   # dataroot 相对路径（机器无关规范化）
│   │   ├── frame_token     # keyframe sample token（已确认语义；sample_data_token 在 v1 裁剪）
│   │   ├── timestamp_us    # int，微秒
│   │   └── time_offset_s   # 相对 anchor，anchor=0.0，精度 6 位
│   └── ego_speed_mps      # backward difference：当前帧 vs frames[-2] ego pose
│                          # 的 planar (x,y) 位移差 / 0.5s，只用过去，精度 3 位，非负
├── training_targets      # 最终监督标签（S07/S08 填充，此前为 None）
│   ├── expected_output     # dict|None（S08 填充）
│   └── language_reference  # dict|None（S07 DriveLM join 填充）
├── oracle_only           # 未来证据，永不进入模型输入
│   └── future_ego_poses    # 沿 sample.next 走 6 帧（3.0s @ 2Hz，已确认）；
│                           # scene 末尾自然截断，不视为 quarantine
│                           # 每项 (sample_token, timestamp_us, x, y, yaw)，精度 3 位
└── provenance
    ├── contract_version        # "v3"，绑定 configs/contracts/v3/ 的 prompt+schema
    ├── temporal_policy_version # "v1"（1F/4F 策略分类版本）
    ├── builder_commit          # 构建时仓库 git commit
    └── nuscenes_version        # 如 "v1.0-mini"
```

已确认的设计决策：`frame_token` = keyframe sample_token；未来视距 6 帧 / 3.0s；ego_speed 用 planar (x,y) 口径。

## 3. 确定性与 canonical hash

- 所有浮点在**构造期**定精度 round（offset 6 位 / speed 3 位 / pose 3 位），含噪声输入（如 `5.199999999`）规范化后与干净输入同 hash。
- `canonical_hash()` = 全 Record sorted-keys 紧凑 JSON 的 sha256，无排除特例；同输入同配置必须同 hash。

## 4. 两层校验

| 层 | 入口 | 检查内容 |
|---|---|---|
| 结构 | `DriveAlignRecord.from_dict`（classmethod， raises ValueError） | 各 namespace 键白名单（缺失/多余均拒绝）、`record_version=="v1"`、恰好 4 帧、类型严格（int 不接受 bool） |
| 语义 | `validate_record()`（返回问题列表，空列表即合法） | 时间戳严格递增且唯一；`time_offset_s` 与 `(ts−anchor)/1e6` 一致（6 位）；`frames[-1].frame_token == sample_token`；speed 有限非负；未来 pose 时间严格递增且晚于 anchor；未来 token 唯一且不与历史帧重叠（泄漏前置校验）；`contract_version ∈ {v1,v2,v3}`；`temporal_policy_version=="v1"`；`builder_commit`/`nuscenes_version` 非空 |

## 5. 验证结果（Step 1 内联验证，9 组断言全过）

- roundtrip `to_dict → from_dict` 相等且 hash 一致；确定性成立。
- 浮点规范化、不同 speed → 不同 hash 均符合预期。
- 7 种结构错误（多余键 / 错版本 / 3 帧 / 缺键 / 非数值 / 非整时间戳 / 缺 provenance 键）全部被 `from_dict` 拒绝。
- anchor 错位、offset 不一致、负 speed、未知 contract_version、未来 token 与历史重叠均被 `validate_record` 捕获。
- docstring 示例启动命令原文可执行。

## 6. 遗留与交接

- `training_targets` 两字段在 S06 恒为 None，S07（DriveLM join）/ S08 填充。
- adapter（Step 2）已落地：`BuildResult(record | reason_code)`，quarantine 分类 = `nuscenes_io` 六个 reason code + `MISSING_CALIBRATION` + `NUMERIC_ANOMALY`，详见 `S06_adapter.md`。
- serializer（Step 3）已落地：从 `model_inputs` 派生 1F/4F 请求，prompt 恒取 v3 冻结文本，详见 `S06_serializer.md`。
- 单测（Step 4，91 passed）与 CLI 冒烟（Step 5/6，31 disjoint 窗口五项 gate 全绿 + rerun diff 一致）已完成，执行流程与 gate 结果见 `S06_record_build_smoke.md`。
