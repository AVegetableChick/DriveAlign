# S06：serializer（序列化器）作用与流程

> 代码：`DriveAlign/src/drivealign/records/serializer.py`（`InputPolicy` / `ModelRequest` / `serialize` / `to_processor_inputs` / `collate` / 指纹扫描）。
> 定位：Record 与 model request 之间的**唯一衔接路径**，因果守卫的强制点。

## 1. 作用

serializer 回答的问题是："模型**允许看到**什么？"——把 Record 档案切片成模型实际消费的请求（prompt + 图像），并在类型层面保证 `oracle_only` 的未来信息物理上进不了模型输入。

| 项 | 内容 |
|---|---|
| 输入 | **只收** `ModelInputs`（`model_inputs` namespace，运行期 isinstance 强制 TypeError） |
| 输出 | `ModelRequest`（1F/4F）→ `ProcessorInputs` → `Batch`（最小 processor/collator 契约层，张量级处理留待 S07） |
| 保证 | 同一 `model_inputs` 派生的 1F/4F 请求 prompt 完全相同；请求 hash 确定性（sorted-keys 紧凑 JSON sha256） |

## 2. 派生流程

1. **选帧（InputPolicy）**：`ONE_FRAME("1F")` 只取 `frames[-1]`（当前帧）；`FOUR_FRAME("4F")` 取全部 4 帧（旧→新）。图像、frame_token、time_offset 三个对齐数组同步切换。
2. **拼 prompt（恒定冻结）**：恒取 S03 冻结的 v3 `prompt.txt`，仅允许前置一行因果速度信息（`Currently available ego speed: 8.414 m/s (backward difference from the previous keyframe)`，精度 3 位与 Record 一致）。**绝不改写任务文本**，因此 1F/4F prompt 逐字节相同——这是 gate"当前帧、prompt、监督答案完全相同"的前提。
3. **携带元数据（不进 prompt）**：帧序号（frame_token）与时间偏移只作为 `ModelRequest` 的机器可读字段随行，满足"temporal wrapper 可增加帧序号与时间偏移，但不得改写任务要求"。
4. **下游投影**：`to_processor_inputs` 无损投影为 processor 级契约（prompt + 图像路径），`collate` 保序堆叠为 `Batch`；三层均带 `canonical_hash()`，供冒烟做确定性 gate 与指纹扫描。

## 3. 泄漏扫描（无合成 canary）

- `future_fingerprints(record)`：从 `oracle_only` 提取**真实**未来证据字符串——每个未来 sample_token + 每个未来 timestamp 的十进制串。
- `scan_future_fingerprints(fingerprints, payloads)`：报告哪些文本载荷（prompt_1f / prompt_4f / processor JSON / batch JSON）命中任一指纹，返回排序后的命中名单；**空列表 = 该层无泄漏**。
- 设计依据：真实未来数据本身就是最严格的探针，且不引入额外字段，保证 Record 字节确定（已确认决策，替代早期 canary 方案）。

## 4. 验证结果（Step 3 内联验证，7 组断言全过）

- docstring 示例命令原文可执行；向 `serialize` 传入非 `ModelInputs` 对象被 TypeError 拒绝
- prompt = 冻结 v3 文本 + 恰好一行速度前导；1F/4F prompt 一致
- 1F 只含末帧（offset=0.0），4F 含全 4 帧旧→新
- 同输入请求 hash 一致、改速度则 hash 改变（确定性 + 敏感性）
- 真实 mini Record 端到端：4 层载荷（两个 prompt + processor + batch）指纹命中 0；阳性对照（注入未来 token 的载荷）被正确捕获
- 1F/4F 共享 anchor frame_token、prompt、ego_speed

## 5. 边界

- serializer 不做任何语义改写/增强：不碰 `training_targets`（S07/S08 的监督答案另行拼接），不读 `oracle_only`（扫描函数是唯一以全 Record 为入参的只读探针，不参与请求生成）。
- 像素级 vision processor（tensor 化）不在本阶段，S07 训练集成时接入；本阶段的 processor/batch 层只负责确立可扫描、可哈希的契约形态。
