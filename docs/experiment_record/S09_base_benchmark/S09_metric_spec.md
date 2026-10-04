# S09 Eval v1 指标口径汇总（Metric Spec）

> 状态：**frozen**（2026-10-05 随 eval_v1.yaml 冻结同步转 frozen）；本文档是 Eval v1 的权威指标口径参考。
> 代码绑定：`src/drivealign/evaluation/{matching,metrics,bootstrap,evaluator}.py`、`cli/evaluate.py`、`configs/evaluation/eval_v1.yaml`（冻结版 sha256 `e6690f6fe950a1d1accdae3286dadbf6a1fad6cb503e3b8fe04ff22bbcf91f2d`，2026-10-05 登记）。
> 与规划的关系：`S09_implementation_plan.md` §3 是摘要表；本文档逐键展开定义、分母与手算示例，两者键名一一对应。命名裁定（2026-10-03）：原 "M09 族" 更名 eval 族，见规划 §2.2。
> 命名约定：所有报告键名遵循 `<字段>_<统计量>[_<聚合>]`，IoU 阈值以 `@τ` 内嵌（如 `object_f1_micro@0.5`）；键名与实现一一对应，冻结后不可变更，新增指标走 eval_v2（§4）。

## 1. 输入与分母体系

### 1.1 评测输入三元组

| 输入 | 来源 | 角色 |
|---|---|---|
| predictions.jsonl | `cli/base_benchmark.py` 产出 | 模型侧：每 anchor 一行（`sample_token`、`parse_ok`、`output`、`status`、`errors`、telemetry） |
| anchor manifest | `data/dataset_v4/anchor_policy_manifest.json`（v5-face） | 定位：token → shard/line/record_hash/scene_token/split |
| dataset_v4 shards | `data/dataset_v4/{split}/shard-NNNN.jsonl` | GT 侧：逐 record 校验 record_hash 后取 `training_targets.expected_output` |

评测全程只读；record_hash 校验失败即报错终止（防 GT 错位）。

### 1.2 GT 五字段（`expected_output` 结构）

| 字段 | 类型 | 枚举/结构 |
|---|---|---|
| `critical_objects` | 对象列表（≤8） | 每项 `{category, bbox_2d, motion_state}`；category ∈ 10 类 nuScenes 集合，motion_state ∈ {stationary, same_direction, oncoming, crossing} |
| `risk_factors` | 字符串集合 | 7 枚举：pedestrian_crossing / vehicle_merging / lead_vehicle_braking / corridor_conflict / stationary_obstacle / congestion / oncoming_traffic |
| `reasoning` | 英文文本 | 感知→风险→决策固定叙事序（见 §2.5：v1 不评分） |
| `yield_required` | bool | 让行判定（corridor 冲突反事实外推派生） |
| `speed_action` | 4 枚举 | ACCELERATE / KEEP_SPEED / DECELERATE / STOP（3.0s 未来轨迹窗口派生，优先级 STOP > DECELERATE > ACCELERATE > KEEP_SPEED） |

闭枚举以 `eval_v1.yaml` `taxonomies` 为准，与 contract v5 schema 由 pin 单测逐字节互锁。

### 1.3 四级分母漏斗

每个指标键都属于且只属于一个分母层级；全部原始分母显式落 `report["denominators"]`：

```
L0 全部 anchors（test = 5,468）          ← coverage / parse 块
L1 parse_ok 集（parse 失败仅进 parse 块）  ← 一切内容指标（对象/动作/风险/诊断）
L2 配对集（主阈值下匹配成功的对象对）      ← matched_iou_*、motion_state_*
L3 反事实子集（200，blank/shuffled）      ← visual dependence 诊断
```

**排除语义**：parse 失败 anchor 不进任何内容指标（含对象数、混淆矩阵），只贡献 `parse_rate` / `parse_error_counts` / `coverage`——失败样本无结构化输出可评，计入内容指标会引入方向性偏差。

**零分母约定**：F1 在 tp=0 时恒为 0.0（`f1_from_counts` 冻结约定），避免未定义值污染聚合；分母为 0 的比率型指标同样记 0.0，且真实分母可从 `denominators` 复核。

## 2. 逐 GT 字段指标

### 2.0 手算示例集（与 `test_eval_metrics.py` fixture 同口径）

全文示例共用一个 4-anchor 合成场景（taxonomies 同 eval_v1.yaml，IoU 阈值 {0.5}，max_items=8）：

| Anchor | GT | 预测 | 结果 |
|---|---|---|---|
| a | 2 对象（car stationary + pedestrian crossing，bbox 相同）、risk={pedestrian_crossing}、yield=false、KEEP_SPEED | 与 GT 完全一致 | 完美帧 |
| b | 1 对象（**car** same_direction，IoU=1.0 位置）、risk={corridor_conflict, congestion}、yield=**true**、KEEP_SPEED | 1 对象（**truck** same_direction 同位置）、risk={corridor_conflict}、yield=false、**STOP** | 类别错误 + 漏报让行 + 假急停 |
| c | 空对象、risk={}、yield=false、KEEP_SPEED | 1 对象（car，幻觉）、risk={congestion}（幻觉）、yield=**true** | 幻觉 + 假让行 |
| d | 与 a 相同 | **parse 失败**（schema_failure） | 仅进 parse 块 |

全局期望：coverage {expected=4, scored=4, parse_ok=3, failed=1}。以下各小节只展开本字段相关计算。

### 2.1 `critical_objects`（对象级）

**前置：类内贪心匹配**（`matching.py`，2026-10-03 用户裁定，替代 Hungarian）

1. normalized category 相同的对象才允许配对（跨类别不配对 → 类别错误 = 1 FP + 1 FN）；
2. 贪心顺序：IoU 降序 → bbox 字典序 tie-break（与预测输出顺序无关，确定性可复现）；
3. IoU < τ 的对不允许配对；同一 GT 至多配一次，落选预测计 FP，落选 GT 计 FN；
4. 对每个报告阈值 τ ∈ {0.3, 0.5, 0.7} 独立跑一遍；主指标 τ=0.5；
5. 规模 ≤ 8×8，贪心与全局最优的数值差异可忽略，无 scipy 依赖。

**micro 指标**（分母 L1：parse_ok 全体预测框与全部 GT 池化合并后计 TP/FP/FN）：

示例：a 贡献 tp=2；b 类别错配不成 → fp=1（truck 预测）、fn=1（car GT）；c 幻觉 → fp=1。合并 **tp=2, fp=2, fn=1**：

- `object_precision_micro@0.5` = 2/(2+2) = **0.5**
- `object_recall_micro@0.5` = 2/(2+1) = **2/3**
- `object_f1_micro@0.5` = 2·P·R/(P+R) = **4/7 ≈ 0.571**

**macro 指标**（分母 L1，但仅计入 GT 非空类别；类别数落 `macro_categories_counted`）：

逐类别（主阈值）计 (tp, fp, fn) 后各算 F1 等权平均。示例：car = {tp:1(a), fp:1(c 幻觉), fn:1(b 类别错)} → F1=0.5；pedestrian = 1.0；truck 无 GT → **排除**。`object_f1_macro@0.5` = (0.5+1.0)/2 = **0.75**，counted=2。
注意：macro 与 risk_factors macro（§2.4）的类别计入规则**不同**——对象 macro 排除 GT 空类别，风险 macro 冻结 7 类全平均。

**IoU 分布**（分母 L2 配对集）：`matched_iou_p50` / `matched_iou_p90`（线性插值分位数）/ `matched_iou_mean`，配对数落 `matched_pair_count`。示例：2 对 IoU 全 1.0 → 三键均 1.0。

**motion_state**（分母 L2 配对集）：只对匹配成功的对象对计 motion_state 相等率与 4×4 混淆矩阵（行 GT 列 pred）。类别错误的对象根本不配对，其 motion_state 不进本指标——类别错误的惩罚已完整落在对象 F1 上，不重复计账。示例：配对对 (stationary,stationary) 与 (crossing,crossing) → `motion_state_accuracy` = 1.0。

**`max_items_hit_rate`**（分母 L1）：预测对象数 == maxItems(8) 的帧占比（S03-Q5 监控项，防"凑满 8 个刷 recall"）。示例：0/3 = 0.0。

| 键名 | 分母 | 计算 |
|---|---|---|
| `object_precision_micro@τ` / `object_recall_micro@τ` / `object_f1_micro@τ` | L1 池化框 | 类内贪心匹配后 TP/FP/FN，跨类别合并 |
| `object_f1_per_category@0.5`，`object_f1_macro@0.5`，`macro_categories_counted` | L1，GT 非空类别 | per-class F1 等权平均 |
| `matched_iou_p50/p90/mean`，`matched_pair_count` | L2 配对集 | 配对 IoU 分布 |
| `motion_state_accuracy`，`motion_state_confusion` | L2 配对集 | 相等占比 + 4×4 混淆 |
| `max_items_hit_rate` | L1 | 预测数==8 帧占比 |

### 2.2 `speed_action`（帧级 4 类）

**`speed_action_confusion`**：4×4 混淆矩阵，行 GT 列 pred；冻结枚举全形（无样本类补 0 行列）。分母 L1。
示例：a/c 均 KEEP→KEEP，b KEEP→STOP → 行 KEEP_SPEED = {KEEP:2, STOP:1, 其余 0}。

**`speed_action_per_class_f1` + `speed_action_f1_macro`**：每类 one-vs-rest F1（该类为正，其余全为负），4 类**等权**平均——无样本类计 0 参与平均，这是 macro 对稀疏类敏感的固有性质，不是 bug。
示例：KEEP_SPEED: tp=2, fn=1 → F1=0.8；其余三类 0 → macro = 0.8/4 = **0.2**。

**`overconservative_stop_rate`**（过度保守代理，分母 L1）：GT ∈ {KEEP_SPEED, ACCELERATE} 而预测 STOP 的帧占比。分母 = "本不该停的帧数"（落 `denominators.overconservative.stop_gt_keep`）。
示例：不该停的帧 {a,b,c}=3，假急停 {b}=1 → **1/3**。S12 gate 以 S09 读数为退化参照（动机：SFT 最易学到的捷径是"拿不准就 STOP"）。

### 2.3 `yield_required`（帧级 2 类）

**`yield_required_f1`**（分母 L1）：二值 F1，正类 = true；四格 tp/fp/fn/tn 全部落 `denominators.yield_counts`。
示例：a(tn)、b(GT true 预测 false → fn)、c(GT false 预测 true → fp) → tp=0, fp=1, fn=1 → F1 = **0.0**。

**`overconservative_yield_rate`**（假让行率，分母 L1）：GT `yield_required=false` 而预测 true 的帧占比。
示例：GT false 的帧 {a,c}=2，假让行 {c}=1 → **0.5**。

设计说明：yield 与 speed_action 是**独立行为轴**（S08 冒烟：STOP 帧 85% yield=false），因此 over-conservative 用两个正交比率分别捕捉"假让行"与"假急停"，不合并。

### 2.4 `risk_factors`（帧级 7 类集合）

**集合语义**（与 S08 GT 派生的双向精确口径对齐）：每帧 GT 与预测各为一个集合；对每个风险类 one-vs-rest 计帧数——GT 集与预测集**都含** = TP，仅 GT 含 = FN，仅预测含 = FP。

**`risk_factors_f1_per_class`**（7 键）+ **`risk_factors_f1_macro`**（分母 L1）：7 类 F1 各自报告，macro 为 7 类等权平均——**冻结 7 类全平均，不排除无样本类别**（与 §2.1 对象 macro 的"GT 非空才计入"规则不同，此处枚举冻结、类类等权）。
示例：pedestrian_crossing tp=1 → 1.0；corridor_conflict tp=1 → 1.0；congestion tp=0（fp 在 c，fn 在 b）→ 0.0；其余 4 类无样本 → 0.0。macro = 2/7 ≈ **0.286**。

### 2.5 `reasoning`（文本）——v1 不评分

**Eval v1 对 reasoning 不设任何自动评分指标**，仅随 record 落档。理由：(1) 无可靠自动文本指标（n-gram/语义分对"固定模板序的短文本"区分度差，且易被模板回声欺骗）；(2) reasoning GT 的规则冲突已在 S08 构建侧过滤（叙事序、枚举锚定、无距离数字），质量由构建 gate 背书；(3) 内容质量在 S12 以人工抽检承担。后续版本若引入自动文本指标，走 eval_v2 并在决策台账留痕。

### 2.6 管线健康度（非 GT 字段，评测有效性前提）

| 键名 | 分母 | 计算 |
|---|---|---|
| `parse_rate` | L0 | parse_ok 帧占比 |
| `parse_error_counts` | L0 | 按 S03 taxonomy（generation_failure / json_parse_failure / schema_failure / semantic_range_failure）分列；`small_following_gap` 触发的违规仍单列观察（v5 prompt 已不教唆，若出现即为 Base 先验行为） |
| `coverage`（六键） | L0 | anchors_expected / anchors_scored / parse_ok / parse_failed / missing_predictions / extra_predictions（missing 附前 5 个样例 token） |

示例：parse_rate = 3/4 = **0.75**，parse_error_counts = {schema_failure: 1}。

### 2.7 视觉依赖诊断（反事实子集，L3；诊断不 gate）

反事实法：预注册 200-anchor 子集（场景轮转分层抽样，`counterfactual_subset.txt`）上分别以 blank（全灰，摧毁场景内容）与 shuffled（patch 打乱，保留纹理摧毁布局）图像重推理，与正常输出逐 anchor 比较：

| 报告键（规划 §3 名） | 报告结构（`counterfactual.<transform>.`） | 计算方式 |
|---|---|---|
| `visual_dependence_action_flip_blank/_shuffled` | `action_flip_rate` | 换图后 `speed_action` 翻转的帧占比 |
| `visual_dependence_output_change_blank/_shuffled` | `output_change_rate` | 换图后结构化输出（整个 `output` dict）任一字段变化的帧占比 |

两个比率只统计**双侧都 parse_ok** 的 anchor 对；比较对数落 `n_compared`。解读：变化率 ≈ 0 → 模型未使用图像（纯文本先验"背答案"）；变化率高 → 输出由图像内容驱动。Base 阶段仅建立读数；S12 gate 防止 SFT 把模型训得更不看图。

### 2.8 成本摘要（全量，L0）

| 键名 | 计算 |
|---|---|
| `latency_p50` / `latency_p90` | 逐 anchor generation_seconds 分位数（线性插值） |
| `throughput_anchors_per_s` | 有 telemetry 的 anchor 数 / 总生成秒数 |
| `input_tokens_p50` | 输入 token p50——**承载规划原 `visual_tokens_p50` 键**：S02 telemetry 记录的是总 prompt token（文本+图像），无图像 token 单独计数，拆分需动冻结栈，不为之破坏（Step 6 台账复核项） |
| `peak_cuda_mem_bytes` | 峰值显存（逐 anchor 最大值） |
| `gpu_hours` | 总生成秒数 / 3600 |

## 3. CI 与 scene-cluster bootstrap

### 3.1 为什么点估计不够

S09 的分数不是终点，是 **S12 的对照基线**——对照就需要知道"差多少才算真差"。两个具体压力：

1. **小分母类别**：test 集 `construction_vehicle` 仅 162 个 GT 实例（`gt_distribution_report.json`），该类 F1 翻转 1 个预测动 ~0.6pp，翻 5 个动 ~3pp；macro-F1 中占 1/10 权重。Base 与 SFT 在这 162 个实例上的偶然差异就能单独摆动 macro-F1 ±0.3pp。
2. **gate 判定**：S12 的"over-conservative 未越过退化阈值""内容指标相对 SFT 改善"都需要判定观测差值是真实变化还是波动——CI 是唯一可复算的裁判标尺。

CI 度量的问题：**若换一批同等规模、同等构建规则的场景集，指标会在哪个范围内波动**。bootstrap 从"只有一份测试集"的现实出发，用重采样模拟这个波动。

### 3.2 为什么按场景整簇抽，而不是按帧抽

同一场景的相邻 anchor 高度相关（同一辆车、同一路况、帧距 0.5s）。若把 5,468 个 anchor 当独立样本重抽，等于虚报独立信息量 → CI 假性偏窄 → 高估置信度。scene-cluster 重抽以 `scene_token` 为簇（取自 anchor manifest）整场景有放回抽取：**场景内相关结构被原样保留**，场景间独立可抽。这是分母显式落 `n_scenes` 的原因——CI 宽度由场景数（≈数百）而非帧数（5,468）驱动。

### 3.3 机制（`bootstrap.py`，逐位确定）

```
输入  clusters: anchor_token -> scene_token（来自 anchor manifest）
      statistic: 一组计数函数（与 aggregate 用同一套 helper，"一个定义、两个消费者"）
参数  n_resamples=2000, seed=20261003, confidence_level=0.95（eval_v1.yaml 预注册）
流程  1) 全量 anchor 算一遍 → point_estimate
      2) RandomState(seed) 生成 2000×n_scenes 重采样矩阵（一次生成，所有指标共享）
      3) 每行: 按场景索引有放回抽场景 → 拼出 anchor 子集 → 重算全部统计量
      4) 每键取 2000 个 replicate 值的 P2.5 / P97.5（percentile 法）→ ci_low / ci_high
```

关键设计：**共享重采样矩阵**——所有指标的 CI 来自同一组重采样子集，相互一致（不会出现"A 的 CI 高于 B 但点估计更低"的抽样错位）；**逐位确定性**——全部随机性经 `numpy.random.RandomState(seed)`，同输入同字节。

每键报告结构：`{point_estimate, ci_low, ci_high, n_scenes, n_values, n_resamples, seed, confidence_level}`。

### 3.4 覆盖指标清单（`bootstrap_ci` 实际收录的 10 键）

与规划 §3"主指标均报告 point estimate + 95% CI"一致，`subset_statistics` 收录：

| # | 键名 | 备注 |
|---|---|---|
| 1 | `parse_rate` | 重采样子集内分母=子集帧数 |
| 2 | `object_f1_micro@0.5` | 主指标 |
| 3 | `object_f1_macro@0.5` | 主指标 |
| 4 | `yield_required_f1` | |
| 5 | `motion_state_accuracy` | |
| 6 | `speed_action_f1_macro` | |
| 7 | `risk_factors_f1_macro` | |
| 8 | `overconservative_yield_rate` | S12 gate 参照 |
| 9 | `overconservative_stop_rate` | S12 gate 参照 |
| 10 | `max_items_hit_rate` | 监控项 |

不进 bootstrap：per-τ@0.3/0.7 全家、matched_iou 分布、混淆矩阵、per-class 明细（结构非标量或非主指标，点估计已落报告）、成本摘要（非评测质量量）、visual dependence（子集仅 200，重采样无意义）。若 S12 需要为某键补 CI，走 eval_v2。

### 3.5 paired-delta：S09 vs S12 的对比判定（`bootstrap_paired_delta`）

两系统（如 Base 与 SFT 后）在**同一 anchor 集**上评测，传入同一 cluster map 与同一 seed → 重采样矩阵逐行相同 → 每个 replicate 是同一批场景下 A、B 各算一遍后取差：

```
输出每键: {point_a, point_b, delta, delta_ci_low, delta_ci_high, n_scenes, ...}
其中 delta = point_a - point_b，CI 为 2000 个逐 replicate 差值的 P2.5/P97.5
```

配对的意义：同一重采样子集下两系统面对完全相同的场景组合，场景难易的波动在差值中抵消，**差值 CI 只反映系统差异的噪声**——比两个独立 CI 相减窄得多也更诚实。判定规则：`delta` 的 95% CI 不含 0 → 差异可判定为真；含 0 → 不得声称提升/退化（gate PASS/FAIL 依此）。

### 3.6 解读边界（三条，防误读）

1. **不是频率陈述**。"95% CI" 指构造方法在重复抽样下的覆盖率，不是"真值有 95% 概率落在这个区间里"。
2. **test 集是确定性构建，不是随机样本**。CI 度量的是"换同等规模、同规则场景集"的假设波动，用于系统间对比与 gate 判定；它不（也不能）修正 test 集本身的覆盖偏差——那由 S07 构建侧的分布表背书。
3. **CI 宽 ≠ 指标算错**。CI 宽是分母小（场景少、类实例少）的诚实反映；把宽 CI 的指标从报告里删掉只会隐藏不确定性，不会消除它。

## 4. 冻结流程与变更规则

1. `eval_v1.yaml` `status: provisional → frozen`（Step 5，冻结发生在正式报告生成前），重新登记 sha256 于规划 §4；
2. 冻结后**报告键名、分母定义、枚举、bootstrap 参数全部不可变**；`m09_version`→`eval_version: eval_v1` 与 config sha256 落报告 `meta` 块；
3. 正式报告（`eval_report.{json,md}` + `anchor_scores.jsonl`）仅在 frozen 配置下生成，方可引用为冻结 Base benchmark；
4. 新增/修改指标（含补 CI）→ eval_v2：复制 yaml、改 `eval_version`、代码分支按版本分发，决策台账留痕；
5. 确定性验收：同一 predictions 两次评测，`eval_report.json` 逐字节一致（sorted keys、无时间戳、无 git 状态）。

## 5. 验收核对清单

- [ ] 本文 §2 键名表与规划 §3 表逐键对应（含 `input_tokens_p50` 承载裁定）；
- [ ] 每个比率型指标的分母可从 `report["denominators"]` 复核；
- [ ] §2.0 示例数值与 `test_eval_metrics.py` 断言逐值一致；
- [ ] §3.4 bootstrap 覆盖键与 `evaluator.subset_statistics` 返回键逐键一致；
- [ ] 二次评测逐字节一致（§4.5）；
- [x] 冻结后本文档 status → frozen，与 eval_v1.yaml 同步（2026-10-05 完成）。
