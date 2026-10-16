# S13 ctxbug：问题发现与 A/B 对比实验定案

## 0. 元信息与定位

- **日期**：2026-10-15。
- **状态**：定案——「SFT 训练-推理上下文错位（ctxbug）导致格式回退」的归因获对比实验直接证据支撑，判读标准在实验执行前预注册（见 §3，于脚本交付对话中先行给出、用户执行）。
- **关联文档**：
  - 决策台账 §9「SFT v7 训练-推理上下文错位（ctxbug）修复 + 重训裁定」条目：[S13_contract_v7_decision_log.md](S13_contract_v7_decision_log.md)；
  - base v7 parse_failed 归因（截断主导模式，本文 §4 观察三与之闭环）：[parse_failure_analysis.md](../../experiment_record/qwen-1f-benchmark-v7/parse_failure_analysis.md)；
  - 实施规划 §6.2/§6.3（重训与评测交接）：[S13_v7_implementation_plan.md](S13_v7_implementation_plan.md)。
- **证据归档**：`runs/S13_sft_1f_v7/ctxbug_ab_evidence/`（`ctx_ab_test.log` 全量运行日志 + `ctx_ab_results.jsonl` 24 条逐臂原始输出；smoke8 predictions 留 run 根目录作对照基准）。
- **实验动机（用户质疑原话）**："但是你确定空格存在与否和这个有关吗，你能不能做个对比试验，看一下如果评测和训练适配的情况下空格是否存在？"

## 1. 问题发现与定性

### 1.1 发现（SFT smoke8）

SFT 1-epoch 训练收口后（adapter `7facb03d…`，frozen 评测 config `c462db1d…`），smoke8 八锚点 7/8 parse_ok，唯一失败 0018da40 打满 700 预算截断。**全部 8 条发射格式均为 `json.dumps(indent=4)` 美化 JSON，而非训练监督的紧凑单行 JSON**（v7 契约 = `separators=(",", ":")`，V7-A2）；内容侧（键序、字段集、id 引用、无 fence）与训练目标完全一致。

### 1.2 定性：ctxbug（训练-推理上下文错位）

1. [dataset.py encode_chat](../../../src/drivealign/sft/dataset.py) 旧实现以 `apply_chat_template(add_generation_prompt=False)` 渲染上下文后直接拼 target——训练态边界 = `...<|im_end|>\n{target}`，**22,941 条训练样本的上下文从未出现 `<|im_start|>assistant\n` 开仓标记**；
2. 推理端 [batched_runner.py](../../../src/drivealign/inference/batched_runner.py)（及冻结面 `generate_one`）均 `add_generation_prompt=True`，推理时模型站在**训练中从未见过的状态**上续写；
3. 在该未见状态下，格式回退 base 先验（美化 JSON），内容仍受 adapter 牵引。

影响量化：美化发射使 8 对象场景 ≈720–760 tok > 700 预算 → 截断；GT 7–8 对象锚点占 test 12.8%，主跑 parse_rate 预期 ~0.87–0.93，paired-delta 被污染。

## 2. 待验证命题与竞争假设

| 假设 | 内容 | 若成立 |
|---|---|---|
| **H1（ctxbug 归因）** | 空格/缩进差异由「推理上下文 = 训练未见状态」引起；在训练见过的状态上模型会发射紧凑格式 | 按已裁定方案重训（修复已合入） |
| **H2（竞争假设）** | 美化输出与上下文态无关，如 LoRA 对格式 token 欠拟合、格式监督强度不足——则换任何状态都美化 | 重训前须重审，修复可能无效 |

## 3. 对比实验设计

### 3.1 控制变量

| 变量 | 取值 | 来源 |
|---|---|---|
| adapter | 同一只 buggy adapter（`archive/sft_1f_v7_ctxbug/best`，sha `7facb03d…`；实验执行时位于 `runs/S13_sft_1f_v7/checkpoints/sft_1f_v7_ctxbug/best`，2026-10-15 转移，见 §6） | 归档产物 |
| 锚点 | smoke8 全部 8 锚（prompt 与图像原样复用 predictions 字段） | `runs/S13_sft_1f_v7/predictions_smoke8.jsonl` |
| 解码 | greedy、`max_new_tokens=700`，与冻结评测面一致 | [benchmark/sft_1f_v7.yaml](../../../configs/benchmark/sft_1f_v7.yaml) |
| 模型装载 | 基座 + PeftModel 重建，复用 `load_adapter_model` | 训练/评测同源路径 |
| **唯一自变量** | 上下文态（渲染方式） | — |

### 3.2 三臂定义

| 臂 | 上下文构造 | 语义 |
|---|---|---|
| **A_true** | `add_generation_prompt=True`，上下文止于 `<\|im_start\|>assistant\n` | 推理态 = 训练未见状态；等价复现 smoke8 生成条件 |
| **B_train_free** | `add_generation_prompt=False`，上下文止于 `<\|im_end\|>\n`，不强制任何 token | 训练见过的状态，观察模型自发续写 |
| **B_train_forced** | 同 B_free + 上下文末尾追加 1 个 token（id=4913，`'{"'`） | 精确复现训练监督的第一个条件转移 `P(target首token \| ctx)`；该首 token 取自训练 JSONL 首条 `target_text`，全部训练样本同构故恒同 |

### 3.3 预注册判读标准（执行前给出）

- **A 输出美化 且 B（尤其 B_forced）输出紧凑** → H1 成立，按已裁定方案重训；
- **B 也输出美化** → H1 不成立，转入 H2 方向重审，重训前暂停。

## 4. 结果（8 锚 × 3 臂，全部完成 24/24）

`shape` 判类：`indented` = 缩进美化（检测 4/8 空格缩进）、`compact` = 紧凑单行、`other` = 其余（含下文两处口径注记）。括号内为生成 token 数。

| 锚点 | smoke8（bs=8） | A_true（推理态） | B_free（训练态） | B_forced（训练态+首token） |
|---|---|---|---|---|
| 0017c262 | indented (312) | indented (312) | **compact (277)** | compact* (276) |
| 00185acf | indented (381) | indented (381) | **compact (328)** | compact* (327) |
| 0018da40 | indented (700 截断) | indented (**700 截断**) | **compact (515 完整收尾)** | compact* (514) |
| 00460925 | indented (143) | indented (143) | **compact (94)** | compact* (93) |
| 005cfc5e | indented (653) | indented (653) | **compact (440)** | compact* (438) |
| 006cbdb2 | indented (574) | indented (574) | **compact (513)** | compact* (512) |
| 006d520c | indented (400) | indented (400) | **compact (276)** | compact* (275) |
| 0071536e | indented (225) | other (225)† | **compact (399)** | compact* (404) |

\* B_forced 被脚本判为 `other` 纯属展示口径：强制 token `{"` 计入输入段、不在生成段，生成段文本以 `critical_objects":` 开头；完整发射 `{"critical_objects":[...]` 为紧凑单行（各锚 `n_new` 恰比 B_free 少 1，自洽）。

† 锚点 8 的 A_true 在 bs=1 重跑下发射 **2 空格缩进**美化 JSON（smoke8 bs=8 下为 4 空格）——同属美化先验家族，缩进宽度漂移属 S09 备案预注册的「bs 差异下 bf16 归约顺序 → 偶发 token 翻转」已知差异类，不影响本实验判定。

### 关键观察

1. **A_true 逐锚复现 smoke8**：8/8 token 数完全一致（312/381/700/143/653/574/400/225），对照基线忠实。
2. **唯一自变量翻转格式**：同一 adapter、同一图像、同一 prompt、同解码参数下，A 臂 8/8 美化发射，B 臂 8/8 紧凑单行（无缩进、无 `", "`/`": "` 空格）——**评测与训练适配（上下文对齐）的情况下，空格不存在**。
3. **token 预算与 parse_failure_analysis 截断主导闭环**：0018da40 在 A 臂 700 打满、截在第 8 对象；B_free 同为 8 对象内容 515 tok 完整收尾至 `}<|im_end|>`——缩进开销即截断根因，同内容下格式成本差 ≈185 tok。
4. **内容跨臂一致、仅格式翻转**：bbox/bev/category/id 引用/risk_factors 结构逐臂近乎逐值（数值存在贪心路径漂移，如 0017c262 首框 A `[425.9, 484.1, 517.0, 630.1]` vs B `[425.9, 480.0, 518.5, 628.1]`）——adapter 在两种状态下都牵引内容，**唯独格式随上下文态切换**，纯格式先验回退，无能力差异。B 臂 token 数由内容（对象数）与格式共同决定（锚点 8 B_free 对象更多故更长），token 数佐证仅在内容同规模的 0018da40 上成立。

## 5. 结论

1. **H1 成立（判读标准第一支命中）**：空格/缩进差异确由训练-推理上下文错位引起；上下文对齐（训练态）时 buggy adapter 即发射紧凑格式。美化输出不是 LoRA 欠拟合（H2 排除）。
2. **ctxbug 修复（encode_chat 改 `add_generation_prompt=True` + 回归钉 + 全量 504 passed）方向正确**，按台账 §9 已裁定方案重训，无需重审。
3. 修复后的重训预期由本实验间接支撑：训练态与推理态将逐位一致（回归钉断言），模型在推理时 standing 在「见过的状态」上，紧凑发射 + 700 预算内完整收尾（参考 B 臂读数）。

## 6. 产物与证据溯源

| 项 | 位置 | 说明 |
|---|---|---|
| 运行日志 | `runs/S13_sft_1f_v7/ctxbug_ab_evidence/ctx_ab_test.log` | 全量 24 生成 + 汇总表 |
| 逐臂原始输出 | `runs/S13_sft_1f_v7/ctxbug_ab_evidence/ctx_ab_results.jsonl` | 24 行（8 锚 × 3 臂），含 `text`/`n_new_tokens`/`shape`/`smoke8_shape` |
| 实验脚本 | `/tmp/ctx_ab_test.py`（临时诊断，用后已清理） | 设计全文收录于本文 §3；模型装载复用 `drivealign.sft.train.load_adapter_model`，渲染/编码镜像 [batched_runner.py](../../../src/drivealign/inference/batched_runner.py) `generate_batch` |
| buggy adapter | `archive/sft_1f_v7_ctxbug/best`（workspace 根） | sha `7facb03d…` 移后复验无损；**2026-10-15 自 `runs/S13_sft_1f_v7/checkpoints/sft_1f_v7_ctxbug/` 精简转移**（用户拍板：best/ + 根 adapter + 溯源件保留，11 个中间 checkpoint-\* 删除释放 ~4.1G；台账 §9 作废登记不变） |
| 对照基准 | `runs/S13_sft_1f_v7/predictions_smoke8_ctxbug.jsonl` | smoke8 八锚 predictions（A 臂复现对象；2026-10-16 自 `predictions_smoke8.jsonl` 改名保留，供重训 smoke8 fresh 重写原路径） |

## 7. 遗留事项

- 无未决项。重训后流程照旧：日志/adapter 核验 → best 重冻（`best_freeze.json` v2，台账 §9 补溯源）→ smoke8 重验（预期 8/8 parse_ok、紧凑发射、无打满）→ 实施规划 §6.3 主评测（主 5,468 + CF blank/shuffled 各 200）→ eval v3 → paired-delta。
