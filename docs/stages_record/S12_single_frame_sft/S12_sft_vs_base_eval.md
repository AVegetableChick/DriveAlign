# S12 Step 7 —— SFT-1F（1 epoch）vs Base-1F 评测对比记录

> 目的：落档 SFT-1F 末点（1 epoch）与 1F-Base 在 v6 合同面上的全量指标对比，
> 并统计 SFT 侧 json_parse_failure 的典型失败形式（2026-10-07 新增分析）。
>
> 数据源：
> - Base：`runs/S09_base_benchmark/eval_v6/eval_report.json`（bs=1 冻结面电池）
> - SFT-1F：`runs/S12_sft_1f/eval_v6_batched/eval_report.json`（batched_v1，bs=16）
> - 失败形态统计：`eval_v6_batched/predictions_bs16.jsonl` 逐条 `raw_text` 复算
>
> 口径前提：
> - 同一 anchor manifest（sha `2abe8c4f…`，v6 面）、同一 eval_v1 配置（sha `e6690f6f…`）、
>   同一 bootstrap seed（20261003，scene 级 2000 次重采样）→ 配对可比；
> - SFT 侧两处**设计内**配置差异（`sft_1f_v6.yaml` 头注登记）：adapter 挂载 +
>   `max_pixels=360000`（A1，与训练侧同值；Base 走 processor 默认 → input p50 2313 vs 937 tok）；
> - SFT 侧读数**待 parity 复核**（bs=1 参照面 eval_v6/ 仅 164 行，见 S12_bug_log.md）；
> - ⚠️ overconservative_* 的跨模型对比是**伪象**，见 §3（Base KEEP_SPEED 塌缩所致，
>   此结论已在 S12 台账落盘，本记录沿用该口径）。

## 1. 核心指标总表

| 指标 | Base-1F | SFT-1F（1ep） | Δ | 95% CI 重叠？ |
|---|---|---|---|---|
| speed_action_f1_macro | 0.162 | **0.595** | **+43.3pp（3.7×）** | 否 ✅ |
| motion_state_accuracy | 0.313 | **0.760** | **+44.7pp（2.4×）** | 否 ✅ |
| object_f1_micro@0.5 | 0.336 | **0.583** | +24.7pp | 否 ✅ |
| object_f1_macro@0.5 | 0.216 | **0.399** | +18.3pp | 否 ✅ |
| object_precision_micro@0.5 | 0.402 | 0.684 | +28.2pp | ✅ |
| object_recall_micro@0.5 | 0.289 | 0.508 | +21.9pp | ✅ |
| yield_required_f1 | 0.045 | **0.486** | **+44.1pp（10.8×）** | 否 ✅ |
| risk_factors_f1_macro | 0.088 | 0.335 | +24.7pp（3.8×） | ✅ |
| max_items_hit_rate | 0.040 | 0.051 | +1.2pp | ✅ |
| matched_iou_mean | 0.732 | 0.792 | +6.0pp | — |
| parse_rate | **0.9993** | 0.9643 | **-3.49pp ⚠️** | 显著回退（§4） |

object 分类目 @0.5 全部上涨，弱项改善最大：barrier 0.025→0.361、pedestrian 0.072→0.421、
traffic_cone 0.051→0.234、trailer 0.177→0.335；强项续升：car 0.410→0.681、bus 0.586→0.727。

## 2. 行为质变：Base 输出塌缩，SFT 出现真决策

**speed_action（per-class F1 与混淆结构）**

| 类别 | Base | SFT-1F |
|---|---|---|
| ACCELERATE | 0.000 | 0.470 |
| DECELERATE | 0.054 | 0.281 |
| KEEP_SPEED | 0.595 | 0.765 |
| STOP | 0.000 | **0.865** |

Base 5,337 个 motion pair 里 **99.5% 预测 KEEP_SPEED**（5,312 条），全程只输出过 1 次 STOP；
GT=STOP 的 1,064 条中 1,036 条被判 KEEP_SPEED——典型多数类塌缩，其 macro F1=0.162 无行为含义。

**motion_state**：Base 把 stationary 判成 same_direction/oncoming（stationary recall **6.0%**，
125/2,096）；SFT stationary recall **89.6%**（4,789/5,346），四态混淆矩阵呈真实结构。

**检测配对**：matched_pairs 5,337→9,098（+70%）；micro@0.5 [TP,FP,FN]
[5,337, 7,954, 13,123]→[9,098, 4,212, 8,800]（FP -47%、FN -33%）。

**yield**：Base tp=24 / fn=924（几乎不让行）；SFT tp=762 / fn=155。

**visual dependence**：blank action flip 6.5%→**16.1%**（2.5×）、shuffled 5.0%→8.9%——
SFT 的输出真正依赖图像内容；Base 的 blank output_change=100% 是塌缩输出的退化伪象，不作正向读数。

## 3. overconservative_*：跨模型对比无效（伪象声明）

Base 的 overconservative_yield_rate=2.1%、stop_rate=0% **不是校准好**，而是
KEEP_SPEED 塌缩的直接后果——从不让行/急停，自然没有假让行/假急停
（yield tp 仅 24、stop 预测全程 1 次）。**这两个键只在同工作点的模型族内
（SFT 变体之间）做 paired-delta 才有意义**，Base→SFT 的 +31.5pp / +4.8pp
不构成 SFT 的退证据。

SFT 自身工作点（正确读法）：yield precision **34.3%**（762/2,220）、
recall **83.1%**（762/917）——从 Base 的 recall 2.5% 到 83.1% 是让行能力的出现；
代价是让得偏宽（precision 1/3），属工作点选择问题，留待 RL/阈值阶段调整。

## 4. parse_rate 回退分解与 json_parse_failure 典型形式

**总量**：parse_failed 4→195（parse_rate 99.93%→96.43%，CI [95.3, 97.3]）。
Base 4 条 = schema 3 + semantic 1；SFT 195 条 = **schema_failure 150 + json_parse_failure 45**。

### 4.1 json_parse_failure（45 条）典型形式（2026-10-07 逐条复算）

| 形态 | 条数 | 占比 | 典型样例 |
|---|---|---|---|
| **`//` 注释混入** | **25** | 56% | `"bbox_2d": [1368.0, 486.0, 1600.0, 640.4], // Oncoming traffic⏎ "motion_state": ...` |
| **`#` 注释混入** | **14** | 31% | `"bbox_2d": [122.0, 525.4, 330.4, 611.9],  # Red car⏎ ...` |
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
4. **可救回上限**：若加"剥注释"后处理，39 条可被救回 → parse_rate 上限 99.15%。
   但**改解析口径 = 变更 eval 合同**，必须走新 eval 版本预注册，不得悄悄改现面；
   治本在训练侧（输出格式纪律）。

### 4.2 schema_failure（150 条）排查结果（2026-10-07，探针脚本落档）

排查工具（持久化，可复用于任意 run）：
`DriveAlign/src/drivealign/contracts/schema_failure_probe.py`
（contracts 层诊断工具，随包走；复用 eval 同款 `parse_structured_output` +
jsonschema 重放，L1 定位字段 / L2 定位具体问题）；
详细报告：`runs/S12_sft_1f/eval_v6_batched/schema_probe_report.json`。

**解析对账**：ok 5,273 / json 45 / schema 150 / semantic 0 —— 与 eval_report 逐数一致。

**L1 字段定位**（181 条违规，分布于 150 anchor）：

| 字段（归一化路径） | 关键字 | 条数 |
|---|---|---|
| `$.critical_objects[*]` | required | **180** |
| `$.critical_objects` | maxItems | 1 |

**L2 具体问题**：180 条全部是同一件事——**critical_objects 的对象缺 `motion_state`
必填键**。无任何 enum / additionalProperties / type / 范围违规（同义词归一化后
词表纪律完好，唯一问题就是漏字段）。

**88–89 tok 奇异峰（84 anchor，56%）= 同一刻板坏模板**：单对象输出只写
`category` + `bbox_2d`、独缺 `motion_state`，其余字段（risk_factors /
yield_required / speed_action）完全正常：

```json
{"critical_objects": [{"category": "car", "bbox_2d": [465.0, 486.1, 570.0, 548.0]}],
 "risk_factors": ["lead_vehicle_braking"], "yield_required": false,
 "speed_action": "KEEP_SPEED"}
```

其余 66 anchor 为同形态散发（多对象里部分缺 motion_state）；另有 1 条密集场景
objects 列表 9 项超 maxItems=8。

**归因方向（下一步）**：反查 SFT 训练 target 是否存在缺 `motion_state` 的对象
（若存在 → 数据噪声，模型忠实复现；若不存在 → 对比推理 prompt 的 schema 说明
与训练 target 字段集是否逐字一致，定位 contract drift）。

### 4.3 典型实例（真实样本节选，2026-10-07 补录）

**实例 1 —— `//` 注释混入**（`01176d20420b404e`，out=161，json 失败）：
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

**实例 2 —— `#` 注释混入且与 category 自相矛盾**（`1808fbc3531c415f`，out=220）：
第三个对象 category=car，注释却写 `# Pedestrian`——注释是模型自己的
"解释习惯"泄漏，与结构化字段脱钩：

```json
{
    "category": "car",
    "bbox_2d": [1228.0, 468.0, 1301.2, 580.0],  # Pedestrian
    "motion_state": "same_direction"
}
```

**实例 3 —— 512 截断**（`012f334ee21b46be`，out=512）：多对象场景输出超长，
尾部恰好停在键名中间：

```json
…, {"category": "car", "bbox_2d": [665.0, 432.0, 800.0, 550.0], "motion_state":
```

**实例 4 —— schema 失败：多对象部分缺 motion_state**（`03f6a362b0774343`，
out=135）：同一数组里第一个对象完整、第二个独缺 `motion_state`——漏字段是
逐对象的随机行为而非全样本模式：

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

> 同步说明：本节（§4 全部）的分析已移植至
> `docs/experiment_record/qwen-1f-sft-v6/parse_failure_analysis.md`；
> 该目录下的 `sft_vs_base_eval.md` 已将 parse 分析移出并指向新文档。

## 5. 成本对比（混合效应，谨慎外推）

| 项 | Base | SFT-1F | 变化 |
|---|---|---|---|
| latency_p50 / p90 | 3.98 / 6.66 s | 2.34 / 2.72 s | p50 1.7× |
| throughput | 0.220 anchors/s | 0.457 anchors/s | **2.08×** |
| GPU hours（主跑+反事实） | 6.90 | 3.32 | 2.08× |
| input_tokens_p50 | 2,313 | 937 | A1 分辨率对齐所致 |
| peak CUDA mem | 8.5 GB | 13.2 GB | bs=16 KV/激活增量 |

注意：2.08× 是 **bs=16 批量化（纯效应 3.48×，见 S12_batched_inference_speed.md）×
输入更短（A1）** 的混合结果，不能直接当批量加速读数引用。

## 6. 结论

1. SFT-1F 在 v6 面核心能力全面大幅提升（速度决策 3.7×、运动状态 2.4×、检测 +24.7pp、
   让行 10.8×、视觉依赖 2.5×），全部通过 scene 级 bootstrap 显著性检验；
2. Base 的多数指标是塌缩读数，跨模型比较时 overconservative_* 与
   visual_dependence_output_change 两个键须按伪象处理；
3. 真实代价集中在 **parse_rate -3.5pp**：json 失败 87% 是注释混入（单一、可修）；
   schema 失败 100% 是 `critical_objects` 对象缺 `motion_state` 键（§4.2，
   其中 56% 来自同一刻板坏模板）；
4. 后续优先级建议：① 反查 SFT 训练 target 是否存在缺 `motion_state` 的对象
   （schema 失败已定位为该单一形态，见 §4.2）；② 训练侧修注释混入
   （数据清洗/few-shot 强约束）；③ 多训 1 epoch 观察 parse_rate 与过保守工作点；
   ④ parity gate 补跑后复核本记录读数。
