# qwen-1f-sft-v6 —— parse 失败形态分析（json / schema 两类）

> 记录 SFT-1F（1 epoch）在 v6 面 test 全量推理中 195 条 parse 失败的形态学分析
> 与典型实例（2026-10-07）。全量细节与决策链见
> `stages_record/S12_single_frame_sft/S12_sft_vs_base_eval.md` §4（本文件为其 parse 部分移植，
> 原 `sft_vs_base_eval.md` §4 已移出至此）。
>
> 数据源：
> - `runs/S12_sft_1f/eval_v6_batched/predictions_bs16.jsonl` 逐条 `raw_text` 复算
> - 探针工具：`DriveAlign/src/drivealign/contracts/schema_failure_probe.py`
>   （contracts 层诊断工具，复用 eval 同款解析器 + jsonschema 重放）
> - 探针报告：`runs/S12_sft_1f/eval_v6_batched/schema_probe_report.json`
>
> 对照基线：Base v6 parse_rate 99.93%（失败 4 条：schema 3 + semantic 1）。

## 1. 总量分解

parse_failed 4→195（parse_rate 99.93%→96.43%，CI [95.3, 97.3]）。

| 类别 | 条数 | 说明 |
|---|---|---|
| json_parse_failure | 45 | `json.loads` 失败（含 markdown fence 容错后仍失败） |
| schema_failure | 150 | JSON 合法但违反 v6 output schema |
| semantic_range_failure | 0 | 无越界类失败 |

探针解析对账：ok 5,273 / json 45 / schema 150 / semantic 0，与 eval_report
`parse_error_counts` 逐数一致。

## 2. json_parse_failure（45 条）典型形式

| 形态 | 条数 | 占比 | 典型样例 |
|---|---|---|---|
| **`//` 注释混入** | **25** | 56% | `"bbox_2d": [1368.0, 486.0, 1600.0, 640.4], // Oncoming traffic` |
| **`#` 注释混入** | **14** | 31% | `"bbox_2d": [122.0, 525.4, 330.4, 611.9],  # Red car` |
| 512 截断（max_new_tokens 打满未闭合） | 5 | 11% | 尾部停在 `"motion_state":` 中间 |
| 其他语法破损 | 1 | 2% | — |

要点：

1. **注释混入占 39/45（87%）**，是唯一主导形态——模型在 objects 数组里给目标写
   语义标注（`// Oncoming traffic`、`# Red car`），把"解释习惯"带进了 JSON。
   位置分布：35 条在 objects 数组内（bbox 行后）、4 条在字段之间；出错点
   相对位置集中在文本中段（39/45）。
2. **截断不是主因**（5/45）：调大 max_new_tokens 最多修 5 条（+0.09pp），无谓动手。
3. **错误信息画像**：42/45 报 `Expecting property name enclosed in double quotes`
   （逗号/注释后紧跟非键 token 的特征报错），与注释形态自洽。

### 实例 A —— `//` 注释混入（`01176d20420b404e`，out=161）

bbox 行后直接写英文注释，其余结构完美：

```json
{
    "critical_objects": [
        {
            "category": "car",
            "bbox_2d": [0.0, 0.0, 653.4, 900.0],
            "motion_state": "stationary"
        },
        {
            "category": "car",
            "bbox_2d": [1368.0, 486.0, 1600.0, 640.4], // Oncoming traffic
            "motion_state": "oncoming"
        }
    ],
    "risk_factors": ["vehicle_merging", "corridor_conflict"],
    "yield_required": true,
    "speed_action": "KEEP_SPEED"
}
```

### 实例 B —— `#` 注释混入且与 category 自相矛盾（`1808fbc3531c415f`，out=220）

第三对象 category=car，注释却写 `# Pedestrian`——注释与结构化字段脱钩，
是纯粹的"解释习惯"泄漏：

```json
{
    "category": "car",
    "bbox_2d": [1228.0, 468.0, 1301.2, 580.0],  # Pedestrian
    "motion_state": "same_direction"
}
```

### 实例 C —— 512 截断（`012f334ee21b46be`，out=512）

多对象场景输出超长，尾部恰好停在键名中间：

```json
…, {"category": "car", "bbox_2d": [665.0, 432.0, 800.0, 550.0], "motion_state":
```

## 3. schema_failure（150 条）排查结果

**L1 字段定位**（181 条违规，分布于 150 anchor）：

| 字段（归一化路径） | 关键字 | 条数 |
|---|---|---|
| `$.critical_objects[*]` | required | **180** |
| `$.critical_objects` | maxItems | 1 |

**L2 具体问题**：180 条全部是同一件事——**critical_objects 的对象缺
`motion_state` 必填键**。无任何 enum / additionalProperties / type / 范围违规
（同义词归一化后词表纪律完好，唯一问题就是漏字段）。

**88–89 tok 奇异峰（84 anchor，56%）= 同一刻板坏模板**：单对象输出只写
`category` + `bbox_2d`、独缺 `motion_state`，其余字段完全正常：

```json
{"critical_objects": [{"category": "car", "bbox_2d": [465.0, 486.1, 570.0, 548.0]}],
 "risk_factors": ["lead_vehicle_braking"], "yield_required": false,
 "speed_action": "KEEP_SPEED"}
```

**实例 D —— 多对象部分缺 motion_state**（`03f6a362b0774343`，out=135）：
同一数组里第一个对象完整、第二个独缺 `motion_state`——漏字段是逐对象的
随机行为而非全样本模式：

```json
{
    "critical_objects": [
        {"category": "car", "bbox_2d": [194.0, 483.0, 411.0, 568.0],
         "motion_state": "stationary"},
        {"category": "car", "bbox_2d": [830.8, 463.2, 918.0, 519.0]}
    ],
    "risk_factors": [],
    "yield_required": false,
    "speed_action": "DECELERATE"
}
```

其余 66 anchor 为同形态散发；另有 1 条密集场景 objects 列表 9 项超 maxItems=8。

## 4. 可救回上限与口径纪律

- 若加"剥注释"后处理，39 条 json 失败可被救回 → parse_rate 上限 99.15%；
  schema 失败的"补默认 motion_state"不属于确定性归一（值不可凭空补），不计入。
- **改解析口径 = 变更 eval 合同**，必须走新 eval 版本预注册，不得悄悄改现面；
  治本在训练侧（输出格式纪律）。

## 5. 归因方向（下一步）

1. 反查 SFT 训练 target 是否存在缺 `motion_state` 的对象：
   存在 → 数据噪声，模型忠实复现；不存在 → 对比推理 prompt 的 schema 说明
   与训练 target 字段集是否逐字一致（contract drift）；
2. 注释混入：检查训练 target / prompt few-shot 示例中是否出现注释样式，
   修复以数据清洗 + few-shot 强约束为主；
3. 多训 1 epoch 观察 parse_rate 与过保守工作点的联动变化。
