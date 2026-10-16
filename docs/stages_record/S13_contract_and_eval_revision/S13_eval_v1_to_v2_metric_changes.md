# S13 Eval v1 → v2 指标变更台账

## 0. 元信息与分类口径

- **文档定位**：S13（契约与评测修订）Step 1 的「口径变更台账」人工版。对应计划 Step 1 Gate 的“每条口径变更均有台账登记”；机器版为 `configs/evaluation/eval_v2.yaml`。
- **修订基准**：`eval_v1`，配置 `configs/evaluation/eval_v1.yaml`（`status: frozen`，config sha256 `e6690f6fe950a1d1accdae3286dadbf6a1fad6cb503e3b8fe04ff22bbcf91f2d`，冻结于 2026-10-05）；参考实现 `src/drivealign/evaluation/metrics.py` 与 `docs/stages_record/S09_base_benchmark/S09_metric_spec.md`。
- **修订目标**：`eval_v2`。产出 `eval_v2.yaml` 与 `eval_v2` evaluator（gate 机制取消，见第 9 节）；`eval_v1` 报告保留为历史基线。
- **条目命名**：`<field>_<statistic>[_<aggregate>]`，IoU 阈值以 `@` 后缀（如 `object_f1_micro@0.5`）；与实现键名一一对应。
- **分类口径**（严格三类）：
  - **保留**：键名与计算方式都不变。
  - **新增**：`eval_v1` 报告中不存在的键。
  - **替换**：v2 引入替代指标、v1 对应旧指标退役（未标注「同名沿用」者，即该 v1 键在 v2 报告中移除）；含“键名沿用但定义/覆盖属性发生变化”的情形，此类在指标名列标注 `（同名沿用，口径变更）`，明细统一见第 7 节。
- **填写规则**：每（子）表固定 4 列 `指标名 | 指标含义及计算方式 | 设置该指标的原因 | 类别`。「原因」列必须给可追溯依据（冻结规范行号 / M09 规则 / 审计文档条目 / S12 记录结论），不得写主观判断。标「替换」的行引用第 7 节的 `T编号`。
- **状态说明**：本台账随 Step 1 推进增量填实。凡 v2 口径尚未定稿者，显式写「待 Step 1 定稿」，不留静默缺口。

---

## 1. 全局与解析（Global & Parse）

| 指标名 | 指标含义及计算方式 | 设置该指标的原因 | 类别 |
|---|---|---|---|
| `coverage`（`anchors_expected` / `parse_ok` / `parse_failed`） | 三级分母漏斗的原始计数：manifest 期望 anchor 数、解析成功数、解析失败数；`anchors_scored` 退役，由 T6 硬前置保证恒等于 `anchors_expected` | 使每个内容指标的分母可复核（S09_metric_spec §1.3），排除语义显式化 | 替换（见 T6） |
| `parse_rate` | `parse_ok / anchors_expected`；分母为唯一锚点分母（T6 不变性下与 v1 分母 `anchors_scored` 数值等价） | 解析成功率是所有内容指标的前置门槛，退化会污染全部内容读数 | 替换（同名沿用，见 T6） |
| `parse_error_counts` | 按 S03 冻结错误类别对解析失败计数（键排序） | 失败可归因（格式 / 枚举 / bbox 等），避免只报单一比率 | 保留 |
| `denominators`（顶层 `anchors_expected` / `parse_ok`） | 顶层分母计数（`anchors_scored` 依 T6 退役） | 使 `parse_rate` 与内容指标分母可由报告独立复算 | 替换（同名沿用，见 T6） |

---

## 2. critical_objects

### 2.1 检测与分类质量（micro 全网格 / macro 等权）

> 注：micro 与 macro 共享同一份类内贪心匹配计数（一份 counting helper、两个消费者，见 T4），差别仅在聚合层——micro 跨类合并（频率加权）、macro 逐类等权平均；两者背离（v1 实测 @0.5：micro 0.336 vs macro 0.216）即“主导类掩盖长尾”的判据，故并表登记、不互替。

| 指标名 | 指标含义及计算方式 | 设置该指标的原因 | 类别 |
|---|---|---|---|
| `object_precision_micro@τ` / `object_recall_micro@τ` / `object_f1_micro@τ`（τ = 0.3、0.5、0.7，3×3 全网格） | 每个 τ 独立：在 parse-ok anchors 上按类内贪心匹配汇总 (tp, fp, fn)，precision = tp/(tp+fp)、recall = tp/(tp+fn)、F1 由 (tp, fp, fn) 直算；主指标 τ=0.5 | 对象检测整体质量主指标族（频率加权），与 macro 分工；全网格暴露结论对匹配严格度的依赖。依据 S09_metric_spec §2.1 步骤 4（τ ∈ {0.3, 0.5, 0.7} 独立跑、主指标 τ=0.5）与键表 `@τ` 行；实现与 v1 报告均 9 键齐全（2026-10-09 勘误：此前误写为仅 τ=0.3、0.7，漏列 @0.5 的 precision/recall）；precision/recall 的 CI 覆盖 2026-10-09 定稿补入 bootstrap（6 键，见表 6 bootstrap 行） | 保留 |
| `object_f1_per_category@0.5` | 同一匹配计数按类拆分：每类 (tp, fp, fn) → F1；仅统计 GT 非空类（`counts[0] + counts[2] > 0`） | 暴露类间不均衡与长尾失败 | 保留（口径澄清见 T5） |
| `object_f1_macro@0.5` | 上述 per-category F1 的等权平均 | 要求每类都可用，防止主导类掩盖长尾类 | 保留 |
| `macro_categories_counted` | 参与 macro 的类别数 | 使 macro 的“类数”分母可复核 | 保留 |
| per-category bootstrap CI（已定稿） | bootstrap 块新增 `object_f1_per_category@0.5` 条目：对每个入选类别（GT 非空类，T5 过滤，共 `macro_categories_counted` 个）逐类给出与既有条目同构的八字段 CI（ci_low / ci_high / point_estimate / confidence_level / n_resamples / n_scenes / n_values / seed）；方法 = scene-cluster 重采样（n=2000、seed=20261003、0.95），与聚合共用 counting helper（T4）；点估计键 `object_f1_per_category@0.5` 保持 {类别: F1} 结构不变，CI 只落 bootstrap 块；块内表示 2026-10-09 定稿为嵌套 `{<类别>: 八字段}`，与点估计键结构对称（见表 6 bootstrap 行） | 长尾类点估计波动大，无 CI 无法判断差异是否真实（S12 教训：per-class 判定曾无 CI 来源）；沿用 v1 bootstrap 块表示惯例（点估计与 CI 分离），主键结构不变以免破坏下游消费者 | 新增 |
| `matched_pair_count` | 主阈值下匹配对总数 | 内容指标的关键分母来源 | 保留 |
| `denominators.micro_counts` | 每个 τ 的 `[tp, fp, fn]` | 使 micro 指标可独立复算 | 保留 |

### 2.2 匹配质量（bbox_2d / IoU）

| 指标名 | 指标含义及计算方式 | 设置该指标的原因 | 类别 |
|---|---|---|---|
| `matched_iou_mean` | 主阈值下匹配对 IoU 的均值 | 区分“勉强匹配”与“高质量匹配” | 保留 |
| `matched_iou_p50` | 匹配对 IoU 的中位数 | 抗离群，反映典型匹配质量 | 保留 |
| `matched_iou_p90` | 匹配对 IoU 的 90 分位 | 反映高质量匹配的上界 | 保留 |

### 2.3 运动属性（motion_state）

| 指标名 | 指标含义及计算方式 | 设置该指标的原因 | 类别 |
|---|---|---|---|
| `motion_state_confusion` | t0 配对对象（IoU≥0.5 匹配对）的 GT × pred 4×4 混淆矩阵 | 可诊断少数态的错误方向（谁被误判为谁） | 保留 |
| `motion_state_accuracy`（v1 键，v2 退役） | 配对对象集上 `correct / total`；分母 = 配对对象数 | 旧口径；在类别不平衡下有盲区 | 替换（见 T1） |
| `motion_state_f1_macro`（+ `motion_state_f1_per_class`） | 同配对集按 4 类 one-vs-rest F1，再类别等权平均；附逐类 F1 | 与 speed_action 口径统一，对少数态敏感 | 替换（见 T1） |
| `denominators.motion_pairs` | 配对对象数 | 使运动属性指标分母可复核 | 保留 |

### 2.4 计数与工作点

| 指标名 | 指标含义及计算方式 | 设置该指标的原因 | 类别 |
|---|---|---|---|
| `max_items_hit_rate` | parse-ok anchors 中预测对象数恰等于 `max_items`(8) 的比例 | 反映预测是否撞满上限、可能存在截断 | 保留 |
| `n_pred_objects` / `n_gt_objects` 分布 | 逐 anchor 计数后落整数直方图 `{k: anchor_count}`（k = 0..8 全桶覆盖、含 0 计数桶；8 = max_items 封顶）；0 桶即空集方向四格的 GT∅ / Pred∅ 边界，块内可复算 | v1 仅逐条留存、不聚合成分布，无法看截断与幻觉方向 | 新增 |
| 空集方向性：`object_empty_direction_counts` + `object_empty_hallucination_rate` / `object_empty_miss_rate` | parse-ok anchors 上按对象集基数分 4 格计数 {gt_empty_pred_empty, gt_empty_pred_nonempty, gt_nonempty_pred_empty, gt_nonempty_pred_nonempty}；两方向率取条件分母：hallucination = gt_empty_pred_nonempty ÷ GT∅ 数（该空不空），miss = gt_nonempty_pred_empty ÷ GT 非空数（该有不有）；分母为 0 取 0.0（与既有实现守卫一致）；两率入 scene-cluster bootstrap（T4 同法：n=2000、seed=20261003、0.95） | 对称 F1 不含方向信息，幻觉与漏报代价不对称须分开计数；与 T2/T3 保守代理同构（同为条件方向率）；4 格计数即 `n_pred_objects` / `n_gt_objects` 分布的零值边界，报告内可复算 | 新增 |

---

## 3. risk_factors

### 3.1 集合级（基数 / 空集）

| 指标名 | 指标含义及计算方式 | 设置该指标的原因 | 类别 |
|---|---|---|---|
| risk 基数分布 | 逐 anchor 取 `len(risk_factors)` 落整数直方图 `{k: anchor_count}`（k = 0..8 全桶覆盖、含 0 计数桶；8 = v6 契约 risk_factors maxItems 封顶）；0 桶即 risk 空集方向四格边界 | 观察风险数量是否系统性多于/少于 GT | 新增 |
| risk 空集方向性：`risk_empty_direction_counts` + `risk_empty_hallucination_rate` / `risk_empty_miss_rate` | 同上四格与两式，集合 = risk_factors 数组（仅基数、类无关）；分母为 0 取 0.0；两率入 scene-cluster bootstrap（T4 同法） | 与对象空集同理，方向性信息不被逐类 F1 覆盖；risk GT∅ 子集小，条件率必须配 CI 才可判定 | 新增 |

### 3.2 逐类（per-class）

| 指标名 | 指标含义及计算方式 | 设置该指标的原因 | 类别 |
|---|---|---|---|
| `risk_factors_f1_per_class` | 冻结 7 类逐类 one-vs-rest (tp, fp, fn) → F1 | 稀疏类（如 congestion）的失败可被看见 | 保留（纳入 CI 见 T4） |

### 3.3 macro

| 指标名 | 指标含义及计算方式 | 设置该指标的原因 | 类别 |
|---|---|---|---|
| `risk_factors_f1_macro` | 冻结 7 类 F1 的等权平均（固定全 7 类，不变长） | 与冻结枚举一致，防常见类掩盖稀疏类 | 保留 |

---

## 4. yield_required

| 指标名 | 指标含义及计算方式 | 设置该指标的原因 | 类别 |
|---|---|---|---|
| `yield_required_f1` | 正类 = `true` 的 (tp, fp, fn) → F1 | 二元安全字段主指标，权衡“该让不让”与“不该让却让” | 保留 |
| `yield_miss_rate` | `yield_fn / yield_gt_true`（GT=true 中预测 false 的比例，“该让不让”；数值上 = 1 − recall）；恒定 false（从不让行）塌缩下取 1.0（塌缩不变性定稿见 T2）；入 scene-cluster bootstrap（T4 同法） | F1 对称合并两方向、方向概率不可见；与 `yield_overconservative_rate` 配成完整方向对；可由 `denominators.yield_counts` 独立复算 | 新增 |
| `yield_overconservative_rate`（自表 6 移入并改名，原名 `overconservative_yield_rate`，见 T2） | `yield_fp / yield_gt_false`（GT=false 中预测 true 的比例，“不该让却让”）；塌缩不变性与 CI 定义见 T2；可由 `denominators.yield_counts` 独立复算 | 安全代理：不该让却让；归属 yield 字段，与 `yield_miss_rate` 同表对照；键名 field-first 对齐 §0 条目命名 | 替换（见 T2） |
| `denominators.yield_counts` | `tp/fp/fn/tn`；方向对两率的条件分母由其派生（yield_gt_true = tp+fn、yield_gt_false = fp+tn），原 `denominators.overconservative` 的 yield 侧由此覆盖（见 T3） | 使 yield 指标（含方向对两率）可独立复算 | 保留 |

---

## 5. speed_action

| 指标名 | 指标含义及计算方式 | 设置该指标的原因 | 类别 |
|---|---|---|---|
| `speed_action_confusion` | 4×4 GT × pred 混淆（行 GT、列 pred，固定形状） | 动作误判方向可诊断 | 保留 |
| `speed_action_per_class_f1` | 4 类逐类 one-vs-rest F1 | 少数动作（如 `STOP`）的失败可见 | 保留（纳入 CI 见 T4） |
| `speed_action_f1_macro` | 4 类 F1 的等权平均 | 对少数动作敏感，避免被 `KEEP_SPEED` 主导 | 保留 |
| `stop_overconservative_rate`（自表 6 移入并改名，原名 `overconservative_stop_rate`，见 T3） | `stop_fp / stop_gt_keep`（GT∈{KEEP_SPEED, ACCELERATE} 中预测 `STOP` 的比例，“不该停却停”）；塌缩不变性与 CI 定义见 T3；可由 `denominators.stop_counts` 独立复算 | 安全代理：不该停却停；归属 speed_action 字段，与 `stop_miss_rate` 同表对照；键名 field-first 对齐 §0 条目命名 | 替换（见 T3） |
| `stop_miss_rate` | `stop_fn / stop_gt_stop`（GT∈{DECELERATE, STOP} 中预测 ∉ `STOP` 的比例，“该停不停”）；恒定不输出 `STOP` 的塌缩下取 1.0（塌缩不变性定稿见 T3）；入 scene-cluster bootstrap（T4 同法）；可由 `denominators.stop_counts` 独立复算 | 混淆矩阵含方向信息但非标量判决；与 `stop_overconservative_rate` 配成完整方向对，与表 4 的 yield 方向对同构 | 新增 |
| `denominators.stop_counts` | `stop_fp / stop_gt_keep / stop_fn / stop_gt_stop`（前两键沿用 v1 `denominators.overconservative` 的 stop 侧，后两键新增；原块撤销并按字段拆分，登记移入本表，见 T3） | 使 stop 方向对两率（`stop_miss_rate` / `stop_overconservative_rate`）可独立复算；归位 speed_action 字段所在表 | 替换（见 T3） |

---

## 6. 系统级代理与诊断

| 指标名 | 指标含义及计算方式 | 设置该指标的原因 | 类别 |
|---|---|---|---|
| `visual_dependence_action_flip_*` / `visual_dependence_output_change_*` | 主预测与反事实预测（blank / shuffled）在共同 anchor 上的翻转率与输出变化率 | 证明模型确实使用了视觉输入 | 保留 |
| bootstrap CI（v1 10 键 → v2 22 标量键 + 逐类条目） | 方法不变：以 `scene_token` 为簇的 scene-cluster 重采样（n=2000，seed=20261003，0.95），条目八字段同 v1。v2 块内容：① 沿用 v1 的 7 键（`max_items_hit_rate`、`parse_rate`、`object_f1_micro@0.5`、`object_f1_macro@0.5`、`risk_factors_f1_macro`、`speed_action_f1_macro`、`yield_required_f1`）；② v1 另 3 键退役（`motion_state_accuracy` 见 T1、`overconservative_stop_rate` 见 T3、`overconservative_yield_rate` 见 T2）；③ 新增标量键 15 个（`motion_state_f1_macro`；`yield_miss_rate` / `yield_overconservative_rate`；`stop_miss_rate` / `stop_overconservative_rate`；`object_empty_hallucination_rate` / `object_empty_miss_rate`；`risk_empty_hallucination_rate` / `risk_empty_miss_rate`；`object_precision_micro@τ` / `object_recall_micro@τ`，τ=0.3、0.5、0.7 共 6 键——2026-10-09 定稿补入，原 §8 待定项撤销）；④ 逐类条目（2026-10-09 定稿块内表示）：`object_f1_per_category@0.5` 逐 GT 非空类（见 2.1 已定稿行）、`risk_factors_f1_per_class` 7 类、`speed_action_per_class_f1` 4 类、`motion_state_f1_per_class` 4 类（T1⑤、T4），条目数依数据而定（v1 test：10+7+4+4=25）；块内表示 = 嵌套 `{<类别>: 八字段}`，与各点估计键的 `{类别: 值}` 结构对称，四个逐类键同规则 | 判定配对差异是否显著 | 保留（覆盖扩展见 T1、T4 及表 4、表 5、2.4、3.1） |
| cost 6 键 | `latency_p50` / `latency_p90` / `throughput_anchors_per_s` / `input_tokens_p50` / `peak_cuda_mem_bytes` / `gpu_hours` | 成本与资源可追溯 | 保留 |

---

## 7. 替换变更明细（v1 → v2）

本节承载「替换」类的全部解释责任。其余表里标「替换」的行只给键名与 `T编号`，细节一律在此展开。

**变更类型词表**（可多选）：`换统计量` / `改分母` / `改聚合层级` / `扩覆盖` / `加不变性` / `键名沿用` / `改键名`。

**列定义**（7 列）：

| 编号 | 变更类型 | v1 指标：口径 | v2 指标：口径 | 逐项变化点 | 变更理由与依据 | 关联表 |
|---|---|---|---|---|---|---|

- **v1 指标 / v2 指标：口径**：统一按四要素写全——定义、分子/分母、聚合层级、是否入 bootstrap。
- **逐项变化点**：用 ①②③ 逐条列，每条写“从 X 改成 Y”，不给笼统结论。
- **变更理由与依据**：可追溯来源（审计文档条目 / 冻结规范行号 / S12 记录结论）。
- **关联表**：该 v2 指标所在的表号。

### 明细

| 编号 | 变更类型 | v1 指标：口径 | v2 指标：口径 | 逐项变化点 | 变更理由与依据 | 关联表 |
|---|---|---|---|---|---|---|
| T1 | 换统计量、扩覆盖 | `motion_state_accuracy`：t0 配对对象集（IoU≥0.5 匹配对）上 `correct / total`；分母 = 配对对象数；只报点估计，不入 bootstrap | `motion_state_f1_macro` + `motion_state_f1_per_class`：同配对集按 4 类 one-vs-rest F1，再类别等权平均；分母逐类为各自 tp/fp/fn；入 bootstrap | ① 统计量 accuracy → macro-F1；② 分母仍为配对对象集，但聚合改为逐类 tp/fp/fn；③ 少数态塌缩不再被主导类掩盖（v1 下少数态全放弃仍可读较高 accuracy）；④ 与 speed_action（已是 macro-F1）口径统一，消除同任务双标；⑤ 新增 per-class 与 95% CI | 同构任务双标；accuracy 在类别不平衡下有盲区；S12 实测 macro-F1 与 accuracy 方向一致但机制不同 | 2.3、6 |
| T2 | 改键名、加不变性、扩覆盖 | `overconservative_yield_rate`：`yield_fp / yield_gt_false`（GT=false 中预测 true 比例）；点估计 | 改名沿用：键名改为 `yield_overconservative_rate`（field-first 对齐 §0 条目命名），登记移入表 4；显式定义“模型退化为常量保守输出（全 `yield_required=true`）”时的取值（= 1.0，方向对另一侧同回合取 0.0）；入 bootstrap | ① 键名改为 `yield_overconservative_rate`，v1 键退役；② 登记位置移至表 4，与 `yield_miss_rate` 配成方向对；③ 塌缩不变性定稿（2026-10-09）：恒定输出 false ⇒ `yield_miss_rate` = 1.0、本率 = 0.0；恒定输出 true ⇒ 本率 = 1.0、`yield_miss_rate` = 0.0；例外仅当恒定输出恰为 GT 常量（两率同 0.0，属真实全对）；分母为 0 沿用 0.0 守卫；纯规范条款、评估过程零新增计算；④ 纳入 95% CI | v1 缺塌缩不变性，无法区分“真保守”与“恒定策略”；S12 已判定 `overconservative_*` 跨模型对比存在伪象；field-first 使键名满足 §0 命名约定 | 4、6 |
| T3 | 改键名、加不变性、扩覆盖 | `overconservative_stop_rate`：`stop_fp / stop_gt_keep`（GT∈{KEEP_SPEED, ACCELERATE} 中预测 `STOP` 比例）；点估计 | 改名沿用：键名改为 `stop_overconservative_rate`（field-first），登记自表 6 移入表 5，与新增 `stop_miss_rate` 配成方向对；显式定义“退化为恒定 `STOP`”时的取值（= 1.0，方向对另一侧同回合取 0.0）；入 bootstrap | ① 键名改为 `stop_overconservative_rate`，v1 键退役；② 登记位置移至表 5；③ 塌缩不变性定稿（2026-10-09）：恒定不输出 `STOP` ⇒ `stop_miss_rate` = 1.0、本率 = 0.0；恒定输出 `STOP` ⇒ 本率 = 1.0、`stop_miss_rate` = 0.0；例外仅当恒定输出恰为 GT 常量（两率同 0.0，属真实全对）；分母为 0 沿用 0.0 守卫；纯规范条款、评估过程零新增计算；④ 纳入 95% CI；⑤ 原 `denominators.overconservative` 块撤销并按字段拆分：yield 侧（`yield_fp` / `yield_gt_false`）由既有 `denominators.yield_counts`（tp/fp/fn/tn，两条件分母为其派生和，v1 实测 96=fp、4516=fp+tn）覆盖，stop 侧改立 `denominators.stop_counts`（沿用 `stop_fp` / `stop_gt_keep` + 新增 `stop_fn` / `stop_gt_stop`），登记移入表 5，两方向率均可独立复算 | 同 T2；恒定 `STOP` 策略会结构性抬高/压低该值而无判别力；方向率成对才可区分“真保守/真漏停”与恒定策略 | 5、6 |
| T4 | 键名沿用、扩覆盖 | per-class F1：`object_f1_per_category@0.5`、`speed_action_per_class_f1`、`risk_factors_f1_per_class`——v1 只报点估计，被排除在 bootstrap 之外 | 同名沿用并纳入 scene-cluster bootstrap，给出 95% CI | ① 键名不变；② 从“只报点”到“点 + 95% CI”；③ 重采样使用与聚合相同的 counting helper（一份定义、两个消费者）；④ `object_f1_macro` 的 per-category 过滤（`counts[0] + counts[2] > 0`）显式化 | per-class 是关键诊断量却被排除在 CI 外；S12 表中 `risk_factors_f1_macro` 的 CI 判定无来源 | 2.1、3.2、5、6 |
| T5 | 改分母（口径澄清） | `object_f1_macro@0.5`：per-category 仅统计 GT 非空类（`counts[0] + counts[2] > 0`），该过滤在实现里隐含、未在规范中显式登记 | 同名沿用：过滤规则写进 v2 规范，分母 = 参与 macro 的类数（即 `macro_categories_counted`） | ① 键名与数值不变；② 过滤规则由“实现隐含”变为“规范显式”；③ 分母语义与 `macro_categories_counted` 绑定 | macro 分母不透明会导致跨报告类数不一致而无法核对 | 2.1 |
| T6 | 键名沿用、改分母、加不变性 | `coverage` 四级漏斗：`anchors_expected`（manifest 期望数）/ `anchors_scored`（实际载入并评分的预测记录数 = expected − missing + extra）/ `parse_ok` / `parse_failed`；missing/extra 仅作事后诊断计数（missing 附前 5 个样例 token），不阻断评测；`parse_rate = parse_ok / anchors_scored` | `coverage` 三级漏斗：`anchors_expected` / `parse_ok` / `parse_failed`；`anchors_scored` 退役，唯一锚点分母 = `anchors_expected`；`missing_predictions ≠ 0` 或 `extra_predictions ≠ 0` 为硬前置，违反即评测中止、不产出报告（错误信息附 missing 前 5 个 token 便于定位）；`parse_rate = parse_ok / anchors_expected` | ① `anchors_scored` 字段退役，锚点分母唯一化为 `anchors_expected`；② “expected == scored” 由操作保证（推理 manifest 驱动；S09_v6_face_rerun 与 S11_step0 实测 missing 0 / extra 0）升级为注册不变性（硬前置）；③ `parse_rate` 分母由 `anchors_scored` 改写为 `anchors_expected`（不变性下数值等价）；④ 消除 v1 缺口：run 中断导致的缺预测在 v1 会静默缩小内容指标分母、漏报混入 parse 侧，v2 直接失败而非静默 | 两个计数字段在既定管线下恒等，双计数冗余；S09_metric_spec 的六键 coverage 中 missing/extra 本就是为捕捉该差异而设，v2 将其从“事后诊断”升级为“事前阻断” | 1 |

---

## 8. 未纳入指标与显式留白理由

| v1 键 / 候选 | 处理 | 理由 |
|---|---|---|
| `motion_state_confusion` / `speed_action_confusion` | 保留但不进 bootstrap | 混淆矩阵是诊断载体而非判决标量，CI 无对应口径 |
| 完整 4×4 / 7×N 混淆的逐格显著性 | 不纳入 | 逐格 CI 的多重比较问题未预先注册，纳入会造成选择性报告 |
| `reasoning_*`（原 §6 reasoning 首版最小口径，2026-10-09 整表移除） | 不纳入 | contract v6 已将 reasoning 移出生成契约（S11 v6 model-face 裁定）：`configs/contracts/v6/output_schema.json` 的 required 仅含 critical_objects / risk_factors / yield_required / speed_action 四字段，prompt 不再教授 reasoning；无字段即无评分对象，v1 亦从未评分 |

---

## 9. 回归证据与冻结信息

- **回归证据**（Step 2 产物）：以同一批 predictions 零重推理重算，未变更键须与 `eval_v1` 报告逐值一致；变更键给出对照与差异归因。证据文件路径待补。
- **冻结信息**：`eval_v2.yaml` 在正式报告前转 `status: frozen` 并登记 sha256（待补）。
- **gate 机制取消**（2026-10-09 裁定）：v2 不设 `gate_thresholds` 块、报告不含 PASS/FAIL 判定；指标优劣由人工观察 + 跨模型同批对照（配对 + CI）承担——比较的是模型间报告，而非指标对预设阈值。曾拟以 4 对方向率（8 键）为 gate 清单、并已定稿“报告回显判定、不回显阈值”的机制设计，随本裁定一并作废。T6 硬前置不受影响（完整性不变性，非阈值 gate）。
- **关联**：`eval_v1` 基准 sha256 见第 0 节；缺陷议事清单另见 S13 审计文档（待出）。

---

## 10. eval v2 → v3 变更登记（2026-10-14，S13 Step 5）

> 本节为 eval v3（BEV 误差族 + risk 因子归因族 + risk 词表 v7 化）的同款变更登记。权威口径 = [S13_eval_v3_metric_additions_draft.md](./S13_eval_v3_metric_additions_draft.md)（已拍板定稿 2026-10-14）；实现 = `evaluation/metrics_bev.py` + `evaluation/metrics_attribution.py`（纯函数）+ `evaluation/evaluator_v3.py`（编排）+ [configs/evaluation/eval_v3.yaml](../../../configs/evaluation/eval_v3.yaml)。

### 10.0 元信息

| 项 | 值 |
|---|---|
| 修订基准 | eval_v2（[eval_v2.yaml](../../../configs/evaluation/eval_v2.yaml)，frozen 2026-10-09，sha256 `661f28f1c70f26893fe0259a91db2f6114bcd976fdb1fc03753af259cb816fc4`，byte-frozen 不动） |
| 新 config | [eval_v3.yaml](../../../configs/evaluation/eval_v3.yaml)：v2 副本仅两处差异——`eval_version: eval_v3` + `status: provisional`；`taxonomies.risk_factors` 换 v7 六枚举（与 `configs/contracts/v7/risk_taxonomy.json` canonical 顺序逐值同源，import 校验 pin 测试钉死）；matching / bootstrap 三参数 / counterfactual / max_items / risk_max_items / evaluation_split 逐值同 v2 |
| provisional sha256 | `e897e8c893c4e08133a5c8067a532f86a68b87a0d4e5b07f8111ef880328a496`（正式报告前转 frozen 并重登记，A.3 惯例） |
| 代际红线 | v7 记录四字段 GT + v6 词表不可混用；v3 字段（`bev_pairs` / `risk_attribution_counts`）只由 `score_anchor_v3` 追加，v1/v2 打分与报告路径行为零改变 |

### 10.1 变更总览（类别 = 新增）

| # | 族 | 内容 | 键数 |
|---|---|---|---|
| N1 | BEV 顶层 | `bev_{dx,dy,dxy}_{mean,p50,p90}`，主阈值匹配对上计算，全非负误差语义 | 9 |
| N2 | BEV 分桶 | `..._bucket_{0_10,10_25,25_40,40p}`，桶归属只看 GT 距离 `d_gt = hypot(x_g, y_g)`（模型间可比），四桶 [0,10) / [10,25) / [25,40) / 40+ 开放桶（2026-10-14 拍板，末桶开放吸收量程外尾巴） | 36 |
| N3 | BEV 偏置诊断 | `bev_signed_{dx,dy}_{pos,neg}_mean`：方向分解（用户裁定：严禁代数均值——正负偏差相互抵消，+2m/−2m 对半分布代数均值为 0 偏置全丢），正/负子集分别求均值，零误差两侧都不进，空子集 0.0 | 4 |
| N4 | 归因标量 | `risk_attribution_f1_micro`（主读数）/ `risk_attribution_f1_macro`（六因子均值，无原子因子跳过）：conditional 双向口径（仅两侧因子名集合之交产生归因原子，2026-10-14 拍板），对象 id 经主阈值 IoU 几何桥接（两侧独立编号空间，严禁编号直比） | 2 |
| N5 | 归因诊断 | `risk_attribution_per_factor` 嵌套块：六因子各自 {tp, fp, fn, precision, recall, f1}，无原子因子跳过不置 null，不入 bootstrap | 1（嵌套块） |
| N6 | 词表 v7 化 | `taxonomies.risk_factors`：v6 七枚举 → v7 六枚举（vru_crossing / vehicle_crossing / lead_vehicle_braking / stationary_obstacle / object_near / oncoming_traffic） | — |

**冻结键集：51 标量键 + 1 嵌套块**（逐键清单见草案 §5）；bootstrap 标量键 22 → **27**（增量 +5：bev 三 mean + 归因 micro/macro；p50/p90 / 分桶 / signed / per_factor 均不入 bootstrap）。bootstrap 总键 31 = 27 标量 + 4 嵌套（`subset_statistics_v3` 空集键数实测）。

**勘误登记（2026-10-14）**：
- 草案 §5 早前"50 标量键"系算术笔误（9 + 36 + 2 + 4 = 51），用户裁定按逐键清单枚举为准；
- 草案 §5 早前"bootstrap 标量键 10 → 15"基数有误——v2 实际 bootstrap 标量基数为 22（`subset_statistics_v2` 空集键数实测：22 标量 + 4 嵌套），v3 追加 5 键后为 27。

### 10.2 关键口径条款（与 v2 先例同款钉死）

- **计算对象**：BEV 与归因都只在主阈值（IoU 0.5）匹配对上计算；未匹配对象不产生 BEV 样本（缺样本 ≠ 零误差），检测层面的漏检/幻觉由 object_f1 族承担，不向 BEV/归因泄漏；
- **全非负误差语义**（2026-10-13 拍板）：dx = |Δx|、dy = |Δy|、dxy = hypot；带符号偏置信息走 N3 方向分解键，主族不保留符号；
- **分桶条件量**：桶归属只看 GT 最近点距 `d_gt`，与预测误差无关——保证分桶条件量在模型间可比；边界左闭右开（`lo <= d_gt < hi`），40+ 为开放桶；
- **分位数口径**：p50/p90 与 `matched_iou_p90` 同一 `_percentile` 线性插值实现；
- **空集口径**：无匹配对切片全部 0.0（沿用 aggregate_v2 空集口径，不置 null）；per_factor 无原子因子跳过；macro 跳过无原子因子；
- **归因 id 桥接**：预测引用 id → 该 pred 对象经 primary IoU 匹配落到的 GT 对象 id；预测引用对象是幻觉 / IoU<0.5 → FP；GT 对象漏检 → FN；conditional 双向——多报因子不计、GT 因子未报不计，同一错误不在因子指标与归因指标双计；
- **T6 完整性硬前置不变**：missing/extra 在任何产物写出之前 fail-fast（消息 `eval_v3 completeness precondition violated`）；
- **counterfactual battery 不加** BEV/归因读数（2026-10-14 默认关闭，草案 §6-5）；cut/near 分层 BEV 读数走诊断脚本不入正式报告（§6-4）。

### 10.3 回归证据（Step 5 产物）

- 全量单测 **492 passed**（原 442 + 新增 50：`test_eval_metrics_bev.py` / `test_eval_metrics_attribution.py` / `test_eval_evaluator_v3.py`），v1/v2 路径零影响。键集 pin：v2 键集 ⊆ v3、v3−v2 恰为 51 标量 + 1 嵌套块、v1 `score_anchor` 无 v3 字段；
- 冒烟（`runs/S13_eval_v2_and_contract_v7/step5_smoke/`，30 真实 test anchor + GT 确定性变换合成预测：id 反转 7−id、bev 偏移 (+0.3, −0.2)、因子引用同步重映射）：**SMOKE PASS**；byte-identical 重跑 PASS；49 BEV 键全在位（bev_pairs=240）；signed 分离读数 dx_pos=0.3000 / dy_neg=−0.2000（代数均值口径下相互抵消归零）；dxy_mean=0.3606=hypot(0.3, 0.2) 验算通过；四桶全有读数；归因 micro/macro=1.0（id 桥接经刻意错位编号验证）；bootstrap v3 新键恰为预期 5 键；
- 复现：workspace 根目录 `PYTHONPATH=DriveAlign/src /root/miniconda3/envs/autovla_codeclean/bin/python runs/S13_eval_v2_and_contract_v7/step5_smoke/run_smoke.py`。

### 10.4 冻结信息

- `eval_v3.yaml` **已于 2026-10-15 冻结**（base v7 eval v3 定稿报告跑批前回写）：`status: provisional → frozen`，冻结 sha256 `fae363adc8dd085d90e87709fcb9e2c19f31afa86c64b4c00a39474b65e7c7cf`；
- 定稿报告 `runs/S13_base_1f_v7_batched/eval_v3_cf/`（主输入 `predictions_1f_rescaled.jsonl` 5,468 anchors + counterfactual blank/shuffled 各 200，dataset root = `data/dataset_v7`）；主指标与首跑报告 `eval_v3/` 逐值一致（同输入确定性复证）；
- gate 机制沿用 v2 裁定：不设 `gate_thresholds` 块、报告不含 PASS/FAIL 判定。

---

## 附录：S13 台账整改与冻结登记（2026-10-09）

### A.1 anchor manifest 唯一入口裁定
- 裁定：anchor manifest 一律以 `data/face_manifests/<v>/anchor_policy_manifest.json` 为唯一入口；`data/dataset_v4/` 只承载 record shards，不再存放 manifest 副本。
- 依据：v2 首跑误用 `data/dataset_v4/anchor_policy_manifest.json`（v5-face，sha256 `fd0d4dc40bf2bd728dd314cbeebca5b922203467ecc7bbf43c37ea9d07a2b09e`），报告 meta `contract_version: v5` 与 v1 基线（v6）不同源。
- 验证：`data/face_manifests/v5/anchor_policy_manifest.json` 与被删副本逐字节相同（sha256 同为 `fd0d4dc4...`），v5-face 证据链无损；v6-face 权威件 sha256 `2abe8c4feb43067e294f6af743efcc01e72729aacb94f92764d0f349c239acd1` 与 v1 报告 `anchor_manifest_sha256` 一致。
- 配套：v5 哈希复现测试改指向归档（FACE_MANIFESTS_V5），14 项 contract 对齐测试全绿；5 处 docstring 示例路径改 v6 入口。
- 待办：数据盘物理删除由用户手动执行 `rm /root/autodl-tmp/datasets/drivealign_dataset/v4/anchor_policy_manifest.json`。

### A.2 v2 首跑 meta 误用与修正
- 首跑（meta 引用 v5-face）：指标与 v1 全部逐位等值（GT 内容未变），仅 meta 两行不同源；首跑产物已被正确重跑覆盖，无遗留引用。
- 重跑（修正后）：meta `contract_version: v6`、`anchor_manifest_sha256: 2abe8c4f...`，与 v1 基线完全同源；eval v2 报告定稿。

### A.3 config 冻结登记
- `DriveAlign/configs/evaluation/eval_v2.yaml`：`status: provisional -> frozen`，冻结于 2026-10-09。
- 冻结前 provisional sha256：`f5e64dbf8c3efbd74360fdd487d05db2e83821eec1474a6a9b2a9dbc3cea7ecf`。
- 冻结后 sha256：`661f28f1c70f26893fe0259a91db2f6114bcd976fdb1fc03753af259cb816fc4`（脚本自动登记）。
- 后续：已定稿报告 meta 的 config_sha256 为 provisional 期值，与冻结件相差一行注释；如需逐字节对齐，重跑一次 eval v2（指标不变，仅 meta 更新）。
