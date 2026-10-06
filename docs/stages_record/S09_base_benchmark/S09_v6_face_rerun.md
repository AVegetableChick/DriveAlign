# S09 附录：1F Base v6 面重跑（S11 Step 0.3/0.4）

> 状态：**已完成（2026-10-06）**，G0.3/G0.4 PASS。
> 定位：本文是 [S11 Step 0 实施计划](../S11_sft_smoke/S11_step0_v6_face_transition.md) Step 0.3（v6 面全量重跑）与 Step 0.4（冻结落档）的交付，也是 [S09 Benchmark 记录](S09_benchmark_full_run.md)（v5 面存档）的 **v6 面重跑附录**。
> 时间：重跑跨 2026-10-05 夜至 2026-10-06；主跑 `wall_seconds = 25,267.6`（≈7.0 h），反事实两组各约 14 分钟。
> 面标签：本文除对照表左列外，所有读数均为 **v6 面**（contract v6，模型面删除 reasoning 字段）。v5 与 v6 读数**不得直接互比**，见 §双面标注规则。
> 背景：v6 与 v5 的唯一面差异是模型面 prompt 删除 reasoning 槽位指令、schema 删除 reasoning 的 required+properties（§面 diff 审计）。本次重跑的目的即在面变更后重建 Base 对照基准，并定位该变更对读数的实际影响面。

## 执行口径

- **主评测** = v6 面全锚集：test split 全部 **5,468 锚**（无 `--limit`、无 `--anchors-file`）；输入政策 ONE_FRAME，checkpoint `models/Qwen2.5-VL-3B-Instruct`（未微调 Base），greedy 解码。
- **反事实** = blank / shuffled **各 200 锚子集**，取自 `runs/S09_base_benchmark/counterfactual_subset.txt`（与 S09 v5 电池同构、同一子集文件）。
- **实际执行脚本** = `DriveAlign/scripts/s11_step03_v6_base_rerun.sh`：严格串行 fail-fast（`set -euo pipefail`），四步依次为主评测 → blank → shuffled → 指标汇总；三步推理均带 `--resume`（幂等，中断后重跑即续跑）。全程 **GPU 独占**执行。
- **注意事项**：反事实两步必须带 `--anchors-file runs/S09_base_benchmark/counterfactual_subset.txt`；若漏掉该 flag，会退化为对全量 5,468 锚各跑 blank/shuffled（即 5468×2 帧），耗时量级与产物语义全错。脚本已在预检段对子集文件做存在性守卫。
- **耗时**：主跑约 25,268 s（≈7.0 h）；blank 831.2 s、shuffled 844.7 s（各约 14 分钟）。较 v5 面（gpu_hours 8.38）更快，与"v6 输出少 reasoning 段、总 token 更短"的预期一致。

## 面 diff 审计（异常归因的前提）

先证明"yield 轴量级差异"来自面变更而非 bug——这是 G0.4 的硬要求。

1. **八件套比对**：`configs/contracts/v5/` 与 `configs/contracts/v6/` 逐文件 diff，**仅 `prompt.txt` 与 `output_schema.json` 不同**；其余 6 件（`category_vocab.json` / `motion_vocab.json` / `risk_taxonomy.json` / `enum_phrase_map.json` / `gt_rule_config.json` / `reasoning_templates.json`）**sha256 全同**（逐字节一致）。
2. **prompt diff**：唯一差异 = v5 第 7 行 `- "reasoning": a concise explanation grounded in the current image and the available ego speed, if provided.` 被删除。`yield_required` 与 `speed_action` 两条字段指令句**逐字节未动**，包络规则、枚举说明、风险条款其余措辞也未动。
3. **schema diff**：字段级约束**仅 reasoning 一项被删**（required 列表去掉 `"reasoning"`，properties 去掉 `reasoning` 块）；`$id`/`title` 由 v3 升至 v6、`description` 增第 7 条（记录"S11 v6 模型面裁定：输出恰为四字段，prompt 不再教 reasoning"），属面元数据版本更新，不改变其余字段约束。
4. **v1–v5 字节钉子未动**：`configs/contracts/v1..v5/` 目录零改动（git diff 空，字节钉子单测 pin 住）。
5. **结论**：本次读数变化的**唯一变量 = 推理期 prompt 里 CoT 槽位被删除**（连带 schema 不再要求 reasoning），与 GT、锚集、评测指标定义、其余枚举资产均无关。

## v5→v6 主指标对照表

v5 = 含 reasoning（S09 存档 `eval_v1`）；v6 = 删 reasoning（本次 `eval_v6`）。双侧均为 test split 5,468 锚、同一 eval_v1 口径。

| 指标 | v5 | v6 | Δ |
|---|---|---|---|
| parse_rate | 0.9965 | 0.9993 | +0.0027 |
| parse_failed（锚数） | 19 | 4 | −15 |
| yield_required_f1 | 0.2812 | 0.0449 | −0.2362 |
| overconservative_yield_rate | 0.8339 | 0.0213 | −0.8126 |
| overconservative_stop_rate | 0.0006 | 0.0000 | −0.0006 |
| speed_action_f1_macro | 0.1146 | 0.1622 | +0.0476 |
| motion_state_accuracy | 0.3257 | 0.3133 | −0.0125 |
| object_f1_macro@0.5 | 0.2173 | 0.2160 | −0.0013 |
| object_f1_micro@0.5 | 0.3384 | 0.3362 | −0.0022 |
| risk_factors_f1_macro | 0.0890 | 0.0882 | −0.0007 |
| max_items_hit_rate | 0.0250 | 0.0395 | +0.0146 |
| visual_dependence_action_flip_blank | 0.1472 | 0.0650 | −0.0822 |
| visual_dependence_action_flip_shuffled | 0.1744 | 0.0500 | −0.1244 |
| gpu_hours | 8.3823 | 6.9042 | −17.6%（相对） |
| latency_p50 (s) | 4.9771 | 3.9785 | −20% |
| throughput (anchors/s) | 0.1812 | 0.2200 | +21% |

补充事实（同源自两份 `eval_report.json`，不在此表重复计账）：

- **parse 错误分解**：v5 `parse_error_counts` = {json_parse_failure 14, schema_failure 4, semantic_range_failure 1}；v6 = {schema_failure 3, semantic_range_failure 1}。删掉 reasoning 后 `json_parse_failure` 归零（该字段的截断型 JSON 破坏源消失），parse_rate 因此改善。
- **coverage**：两侧均 anchors_expected 5468 / scored 5468 / missing 0 / extra 0。
- **反事实比较对数** `n_compared`：v5 = blank 197、shuffled 195（n_shared 200）；v6 = blank 200、shuffled 200。
- **输入 token**：v5 `input_tokens_p50` = 2335、v6 = 2313。
- **speed 逐类 F1**：v5 = {ACCELERATE 0.0, DECELERATE 0.2846, KEEP_SPEED 0.1683, STOP 0.0056}；v6 = {ACCELERATE 0.0, DECELERATE 0.0544, KEEP_SPEED 0.5945, STOP 0.0}。

## 行为层发现

### 1. 感知轴对 CoT 不敏感，决策轴极度敏感

object / risk / motion 三类感知指标两侧全部在 CI 内（Δ 最大约 0.013，均属噪声量级）；而 yield / speed 两个**离散决策头**完全翻面。这说明删除 CoT 槽位改变的不是"看图能力"，而是"从感知到动作"的映射取向。

### 2. 两个面都是"常量预测器"，只是常量不同

这是判断"v6 变好"是否成立的关键。两面各自塌缩为一对相反常量：

- **v5 ≡ "一律 DECELERATE + 一律 yield=true"**：预测 yield=true 占 83.0%（4,524/5,449 parse_ok）、预测 DECELERATE 占 86.7%（4,724/5,449）。
- **v6 ≡ "一律 KEEP_SPEED + 一律 yield=false"**：预测 KEEP_SPEED 占 97.2%（5,312/5,464 parse_ok）、预测 yield=true 占 2.2%（120/5,464）。

用 GT 边际算出的**常量预测器参考水位**（平凡常量基线）：

- yield 轴"一律 true" 的 F1 ≈ **0.2957**（GT yield 正类率 ≈ 17.35%）；
- speed 轴"一律 KEEP_SPEED" 的 macro F1 ≈ **0.1491**（GT KEEP_SPEED ≈ 42.5%，其余三类预测阳性为 0 计 0 参与 4 类等权）。

据此：

- **v5 的 yield F1 0.2812 实质就是常量全 true 的水平（还略低于它）**——v5 并非"学会了让行判断"，而是恰好押在多数类 true 上；
- **v6 的 speed macro 0.1622 只比常量全 KEEP 的 0.1491 高约 0.013（噪声级）**；
- **v5 的 speed macro 0.1146 反而低于该常量水位**（v5 押 DECELERATE 押错了多数类）。

**结论**：v6 的 `speed_action_f1_macro +0.0476` 不是能力提升，而是常量恰好落到多数类（GT KEEP_SPEED ≈42.5%）——一个**多数类假象**。任何把它写成"v6 速度决策变好"的表述都是错的。

**附带（与面无关的能力地板）**：两面**都从不输出 ACCELERATE**（v5 = 0/5468，v6 = 0/5468），而 GT 中 ACCELERATE 占比 ≈20.6%。这是 3B Base 的能力下限，不随面变化。

### 3. 视觉依赖 flip 率腰斩只能有限解读

`action_flip_rate` 从 0.1472→0.0650（blank）、0.1744→0.0500（shuffled）看似"翻倍下降"，但**常量预测器的 flip 率按定义趋近 0**——输出恒为一个常量时，换任何图都不会翻转。所以降幅中有一部分是"塌缩"的算术后果，**不能直接读成"v6 更不看图"**。

真正的问题在两个面都**不看图**。直接统计反事实子集（各 200 锚，按 `output` 字段计）的预测分布：

| 反事实子集 | 预测 speed_action 主类 | 预测 yield=true |
|---|---|---|
| v5 blank | DECELERATE 97.0% | **94.0%** |
| v5 shuffled | DECELERATE 90.4% | 88.9% |
| v6 blank | KEEP_SPEED 97.0% | **0.0%** |
| v6 shuffled | KEEP_SPEED 97.5% | 1.0% |

即两面的常量行为在**纯空白图**上不但没有解除，反而**比全锚集更极端**（v5 全锚 DECELERATE 86.7% → blank 97.0%、yield=true 83.0% → blank 94.0%；v6 全锚 KEEP_SPEED 97.2% → blank 97.0%）：动作头主要由**先验**驱动，图像输入对离散决策几乎没有贡献。此外 `visual_dependence_output_change_*` 两侧均 = 1.0（文本总有变化，因 frozen Base 对输入噪声极敏感），说明该指标不具判别力，判读须以 `action_flip_rate` 为准。

## 双面标注规则（硬性）

- **S09 存档读数**（`runs/S09_base_benchmark/eval_v1/`）= **v5 面**，继续有效。
- **S12 的 Base→SFT 对比基准 = `runs/S09_base_benchmark/eval_v6/`（v6 面）**，**不是** S09 存档的 v5 读数。
- **v5 与 v6 读数不得直接对比**；此后任何文档引用数字必须带面标签（v5 / v6）。
- 双面并存期最大账务风险 = **数字混淆**，故该规则为硬性要求。

## 冻结物 sha256 登记表

| 冻结物 | 路径 | sha256 |
|---|---|---|
| v6 prompt | `configs/contracts/v6/prompt.txt` | `242b632b279e87d9cf355fe42803f7381fa983adebe82d07b5597cbbe692ec10` |
| v6 output schema | `configs/contracts/v6/output_schema.json` | `22efe8f8738aadb14e654a7357f02892049085ba42ab9e17fbf86cffe205419a` |
| v6 benchmark 配置 | `configs/benchmark/base_1f_v6.yaml` | `4c729e66784d30fc0422cb24996eb7da8a74b5fd5342c477f0430357035fb012` |
| v5 benchmark 配置（原件，未动） | `configs/benchmark/base_1f.yaml` | `1aab7db306b17f3688587e5d6006a98bb6a3fcc16326c26b8c8a2066d1332cb4` |
| eval 配置（v5/v6 共用，未动） | `configs/evaluation/eval_v1.yaml` | `e6690f6fe950a1d1accdae3286dadbf6a1fad6cb503e3b8fe04ff22bbcf91f2d` |
| v6 anchor manifest | `data/face_manifests/v6/anchor_policy_manifest.json` | `2abe8c4feb43067e294f6af743efcc01e72729aacb94f92764d0f349c239acd1` |
| v5 anchor manifest | `data/face_manifests/v5/anchor_policy_manifest.json`（与 `data/dataset_v4/anchor_policy_manifest.json` 逐字节相同） | `fd0d4dc40bf2bd728dd314cbeebca5b922203467ecc7bbf43c37ea9d07a2b09e` |
| v6 eval 报告 | `runs/S09_base_benchmark/eval_v6/eval_report.json` | `be2cb6b902e062202dc2f841357d9711dbb1c744091968da0aaf048348255013` |

**提醒**：两侧 `eval_report.json` 的 `meta.config_sha256` 字段都等于 `e6690f6f…`（那是 **eval** 配置，v5/v6 共用），**不是** benchmark 配置的 sha。辨识一份报告属于哪一面，请用 `meta.contract_version` 与 `meta.anchor_manifest_sha256`，不要看 `config_sha256`。

## 回退与残余风险

- **回滚**：`DEFAULT_CONTRACT_VERSION` 改回 `"v5"` 一行即可回退；v6-face 的 manifest / runs 产物可直接删除。
- **残余风险**：双面数字混淆（v5/v6 读数被误并或误比）——靠上面"双面标注规则"兜底。

## 对 S12 的提示

- v6 base 是"什么都不做"的常量基线，SFT 相对它的提升会**系统性虚高**（摆脱塌缩是廉价的）。故 S12 除 base-relative delta 外**必须同时报绝对值**，并**加一条"多数类常量预测器"地板基线**（进一步可加 GT 边际常量基线），否则"学会了看图"与"不再一味 KEEP_SPEED"无法区分。
- `overconservative_stop_rate` 的参照点已退化（v5 0.0006 → v6 0.0000，阈值 0 无法判别），S12 该指标的 gate 语义需重新审视。
- counterfactual `action_flip_rate` 应列为 SFT 的**重点监控项**：v6 base 仅 0.065（blank）/ 0.050（shuffled）；SFT 若不回升，本身就是"reasoning 曾承担了什么"的结论。
