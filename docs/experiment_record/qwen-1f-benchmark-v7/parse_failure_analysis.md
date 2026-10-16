# Base v7 parse_failed 模式分析（457 条全量归因）

- **分析日期**：2026-10-15
- **数据源**：`runs/S13_base_1f_v7_batched/predictions_1f_rescaled.jsonl`（5,468 条，与定稿报告 `eval_report.md` 同输入，parse_ok 5,011 / parse_failed 457）
- **方法**：逐条读取 predictions 自带 `errors`（category/field/message）+ `raw_text` + `telemetry.output_tokens`；schema 失败另用 `raw_text` 独立重解析 + `drivealign.contracts.output.parse_structured_output` 真实校验做反事实量化
- **裁决（2026-10-15 用户拍板）**：**本部分格式问题不走后处理挽救**——parse 侧零改动（不开 fence 宽松、不加 schema 修复），格式纪律交给 SFT 监督文本直接覆盖

## 总账

| error category | 条数 | 一句话定性 |
|---|---|---|
| json_parse_failure | 298 | **截断主导**（284 条尾部截断 + 9 条括号失衡 + 5 条 fence 提取缺口） |
| schema_failure | 155 | 三类附加属性/引用串位（bev_z、objects、bbox 坐标当 id） |
| semantic_range_failure | 4 | oncoming_traffic 引用未声明 id |

失败率 457/5,468 = 8.36%（parse_rate 0.9164）；失败横切：打满 700 预算占失败的 **65.2%**，成功侧 84.5% 输出 <400 token。

## 模式一：截断主导（65.2%）

- **308/5,468（5.6%）打满 `max_new_tokens=700`，其中 298 全部解析失败（96.8%）**；成功侧仅 10 条打满（0.2%）。
- 284 条尾部截断（`does not end with '}'`，断在 risk_factors / speed_action 中途），output_tokens **全部 = 700，无例外**；另 9 条 fence 包裹完整但内部括号失衡（截断后强行闭合）。
- 失败有场景聚集性：91/150 个 scene 有失败，top scene 失败率 40–60%（繁忙场景 → 对象多 → 输出长 → 截断）。

## 模式二：fence 包裹是 base 惯性，本身不致命

- **69.3% 的成功样本（3,472/5,011）raw 同样以 ```` ```json ```` 开头**，pipeline 提取后正常解析（`extracted_from_fences=True`）。
- 285/298 json 失败带开 fence——fence 与截断强相关但不是死因，截断才是。

## 模式三：schema 失败三类主因（155 条，共 550 个错误）

| 模式 | 锚点数 | 内容 | schema 依据 |
|---|---|---|---|
| 自创 `bev_z` | 45（178 个元素） | 模型补了 z 维 | `critical_object` 仅定义 bev_x/bev_y，additionalProperties=false |
| `critical_objects` 误带 `objects` 字段 | 54（169 个元素） | 与 `risk_factors` 的 `{factor, objects}` 形状串位 | 同上 |
| `risk_factors.objects` 引用悬空/越界 id | 32 | **大 id 本质 = bbox 坐标串位** | objects 元素 integer 0–7 |

**bbox 串位实证**：`0dc3d1fb` 归因引用 `[295, 234, 511, 408]` = 其 0 号对象的 `bbox_2d: [295, 234, 511, 408]`；`0a0371e1` 引用 168/210 = motorcycle bbox 前两维。模型抄了坐标没抄 id。

**可挽救量化（仅诊断，不入流水线）**：去掉 `bev_z` / `objects` 两类附加属性后 **97/155 能通过 validator**；剩余 58 条 = id>7（63 错，即 bbox 串位）/ 重复 id 9 / 顶层形状错位 12 / 空 objects 数组 6 / 缺 bev_x 或 bev_y 10 / 类别枚举违例 1（`stop_sign`）。

## 模式四：parse 侧唯一技术性缺口（5 条）

- [output.py 的 `_FENCED_ENVELOPE`](../../src/drivealign/contracts/output.py)（`^\s*```(?:json)?\s*\n(?P<body>.*)\n\s*```\s*$`）要求**闭合** ```` ``` ````；这 5 条开 fence 无闭合（输出在 700 内恰好写完 JSON 但未写闭合标记），提取未触发 → 计为 json_parse_failure。
- strip 后 **5/5 全部通过 validator**。**维持原判**（S03 契约"从不静默修复"）：不放宽提取正则，paired-delta 解读时知晓这 5 条属"提取边界"而非模型能力问题即可。

## 模式五：semantic 4 条

全部为 `oncoming_traffic` 引用未声明 id（id 在 0–7 合法范围但不在该响应自己的 `critical_objects` 里）——schema 层过、语义层拒，防线分工正确。

## 分类示例（真实样本摘录）

### json_parse_failure 四亚型

**① fence + 截断（271 条）** — `0017c2623c914571a1ff2a37f034ffd7`，out_tok=700：

````json
```json
{
    "critical_objects": [
        {
            "id": 0,
    …（中略，断在 risk_factors 之后）…
    "yield_required": true,
    "speed
```
````
error: `Expecting value: line 1 column 1 (char 0) (raw text does not end with '}': output looks truncated, check the generation token budget)`

**② 无 fence + 截断（13 条）** — `15f61a56df7a4ecab63dda6cff25e7a0`，out_tok=700（注意缩进是 2 空格非 4——base 发射形态不稳定）：

```json
{
  "critical_objects": [
    {
      "id": 0,
      "category": "barrier",
    …（中略）…
  "yield_required": true,
  "
```
error: `Unterminated string starting at: line 79 column 3 (char 1595) (…truncated…)`

**③ fence + 完整但括号失衡（9 条）** — `1646a03caf1e497bb7687d0caafa7a85`，out_tok=700（尾部以 `}` 收口但 risk_factors 数组未闭合，截断后强行闭合）：

````json
```json
{
    "critical_objects": [
        { … },
    …（中略）…
            "factor": "oncoming_traffic",
            "objects": [1, 2, 3, 4, 5, 6, 7]
        }
```
````
error: `Expecting value: line 1 column 1 (char 0)`（fence 未闭合 + 括号失衡双重失败）

**④ fence 提取缺口（5 条）** — `666e8405840b434bb91c9dc66ad2efb5`，out_tok=700（JSON 本身完整合法，仅缺闭合 ```` ``` ````，`_FENCED_ENVELOPE` 未触发；strip 后 validator 通过）：

````json
```json
{
    "critical_objects": [
        { … },
    …（中略）…
    "yield_required": true,
    "speed_action": "DECELERATE"
}
```
````

**对照（模式二佐证）**：`0018da40529f48678419d44d09da5369`，out_tok=688，带闭合 ```` ``` ```` → 提取成功、parse_ok——同样 fence 开头，成败只差闭合标记与预算。

### schema_failure 八类

| # | 类别 | 锚点 | 违例片段（真实样本） |
|---|---|---|---|
| 1 | 自创 `bev_z` | `01411f03…` | 见下 ① |
| 2 | `critical_objects` 误带 `objects` | `021dac45…` | 见下 ② |
| 3 | bbox 坐标当 id（悬空引用 + id>7 同族） | `0a0371e1…` / `0dc3d1fb…` | 见下 ③ |
| 4 | `objects` 重复元素 | `1f0a495a…` | `{"factor": "vru_crossing", "objects": [1, 1]}` |
| 5 | `objects` 空数组（minItems 1） | `3eb6673b…` | `{"factor": "oncoming_traffic", "objects": []}` |
| 6 | 缺 `bev_x`/`bev_y`（required） | `248b69fa…` | 见下 ④ |
| 7 | 类别枚举违例（`stop_sign` 不在 10 类） | `a49f9ebf…` | 见下 ⑤ |
| 8 | 顶层形状错位（决策字段嵌进对象元素） | `146a1aef…` | 见下 ⑥ |

**①** `01411f03…`（前 6 键全合法，第 7 键自创）：

```json
{ "id": 0, "category": "car", "bbox_2d": [349, 236, 478, 340],
  "motion_state": "same_direction", "bev_x": -1.2, "bev_y": 0.0,
  "bev_z": 0.0 }
```

**②** `021dac45…`（把归因形状错搬进对象元素）：

```json
{ "id": 0, "category": "car", "bbox_2d": [349, 239, 492, 353],
  "motion_state": "same_direction", "bev_x": -1.2, "bev_y": 0.0,
  "objects": [0] }
```

**③** bbox 串位实锤——`0a0371e1…`：

```json
"critical_objects": [
  { "id": 0, "bbox_2d": [0, 280, 159, 448], … },
  { "id": 1, "bbox_2d": [168, 210, 218, 424], … } ]
"risk_factors": [ { "factor": "vru_crossing", "objects": [168, 210] } ]
```
引用的 168/210 = 1 号对象 bbox 前两维；`0dc3d1fb…` 同型（引用 295/234/511/408 = 0 号对象完整 bbox）。

**④** `248b69fa…`（BEV 字段整体缺失）：

```json
{ "id": 1, "category": "car", "bbox_2d": [346, 238, 365, 250],
  "motion_state": "stationary" }
```

**⑤** `a49f9ebf…`（v6 之外的自创类别）：

```json
{ "id": 0, "category": "stop_sign", "bbox_2d": [623, 82, 693, 156],
  "motion_state": "stationary", "bev_x": -0.0, "bev_y": 0.0 }
```

**⑥** `146a1aef…`（每个 critical_object 元素里又嵌了一份 risk_factors/yield_required/speed_action，顶层同时还有一份）：

```json
{ "critical_objects": [
    { "id": 0, "category": "car", "bbox_2d": [349, 235, 492, 355],
      "motion_state": "same_direction", "bev_x": -1.0, "bev_y": 0.0,
      "risk_factors": ["object_near"], "yield_required": false,
      "speed_action": "KEEP_SPEED" }, … ], "risk_factors": […], … }
```
error: `Additional properties are not allowed ('risk_factors', 'speed_action', 'yield_required' were unexpected)`

### semantic_range_failure（4 条）

`8e86bdba7c1f434780fdc7924d9d2286`，out_tok=371（schema 全过、语义层拒）：

```json
"critical_objects": ids [0, 1, 2, 3]
"risk_factors": [ { "factor": "oncoming_traffic", "objects": [3, 4] } ]
```
error: `risk_factors entry 'oncoming_traffic' references unknown object id: 4`——id 4 在 0–7 合法范围但未在该响应的 critical_objects 里声明。schema 检查范围、语义检查存在性，防线分工正确。

## 裁决与后续

1. **parse 侧零改动**（2026-10-15 用户拍板：格式问题不再后处理挽救）：fence 提取正则不放宽、附加属性不清洗、悬空 id 不重映射。S13 冻结的 parse/eval 面保持 byte-frozen。
2. **格式纪律由 SFT 监督文本覆盖**：三大失败模式全部落在监督文本的直接教学范围——紧凑 separators（V7-A2）、精确 schema 形状（无 bev_z/无 objects 串位）、归因引用 id（V7-B1）。预期 SFT parse_rate 从 0.9164 逼近 1.0，parse 修复本身构成 paired-delta 收益的一部分。
3. **700 预算维持 V7-A2 冻结值**：base 的 5.6% 打满率属未训模型冗长惯性；SFT target 最坏 620 token（预算余 80）。SFT 评测后复核打满率应≈0，若仍显著非零则升级台账。

## 复现口径

- 逐条分类统计：读 `predictions_1f_rescaled.jsonl` 的 `errors[].category`（primary）+ `raw_text` 首尾特征（```` ``` ```` 开头 / `}` 结尾）+ `telemetry.output_tokens`。
- 截断判定：`raw_text.rstrip()` 不以 `}` 结尾 且 output_tokens=700。
- schema 可挽救量化：`raw_text` strip fence 后 `json.loads`，删 `critical_objects[*]` 的 `bev_z`/`objects` 键，再过 `parse_structured_output`（jsonschema 2020-12，v7 [output_schema.json](../../configs/contracts/v7/output_schema.json)）。
