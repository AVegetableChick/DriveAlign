# S09 决策台账（单帧未微调 Base Benchmark）

> 状态：**已关闭（2026-10-05）**。本台账按时间序记录 S09 全部关键决策、指标缺口备案与 gates 判定；
> 参数冻结 hash 见 §2，阶段关闭核对见 §3。姊妹文档：实施规划（`S09_implementation_plan.md`）、
> 指标规格（`S09_metric_spec.md`）、run 记录三份、专题备案两份。

## 1. 决策记录（时间序）

### #1 执行链路裁定（2026-10-03，用户拍板）

- **决策**：S09 → S11（仅 1F 半边）→ S12；S10 与 S11-4F 后置。
- **后果**：S09 履行 base_4f.yaml 预注册义务（决策 #6）；S11 实验记录须标注 "1F closed / 4F deferred"。

### #2 contract v5 承载 prompt 修正（2026-10-03，用户拍板；commit `41810e5`）

- **背景**：v4 `prompt.txt` 逐字节继承 v3，risk_factors 措辞仍列 8 项（含 small_following_gap），与 v4 schema 的 7 项不一致——Base 被 prompt 教唆输出被 parser 拒绝的枚举值。
- **决策**：不可原地改 v4 prompt（request hash 覆盖 prompt 文本，原地改会使冻结资产 hash 永久不可复现）；以 v5 = v4 八文件 + 修正版 prompt（7 项）承载，`MODEL_FACE_LINEAGE["v5"]="v5"`；v4 冻结不动、dataset_v4 不重建、S08 历史 gate 记录不受影响。
- **验证**：prompt↔schema 一致性 pin 单测随 Step 1 交付（255 全绿含此）。

### #3 v5-face anchor manifest 替换（2026-10-03，用户拍板；commit `d93485a`）

- **决策**：**替换** `data/dataset_v4/anchor_policy_manifest.json` 为 v5-face 版（数据集目录承载最新权威入口表，拒绝"最新 manifest 在 runs/"的布局）；原 v3-face 版归档 `runs/S08_gt_backfill/anchor_policy_manifest_v3face_frozen.json`；物理盘替换由用户 tmux 执行。
- **后果**：face 时代分账（规划 §8）——v3-face 归档件与 v5-face 现行件并存，任何 parity 审计须先声明 face 时代，禁止跨代比对 request hash；dataset_v4 逐字节确定性重跑基线由归档件承载。

### #4 eval 族更名（2026-10-03，用户拍板；commit `6791785`）

- **决策**：原 "M09 族" 更名 **eval 族**——`eval_v1.yaml` / `cli/evaluate.py` / `eval_report.{json,md}` / config 字段 `eval_version`。理由：评测器为 S09→S12 对比复用而建，名字不绑阶段号。
- **边界**：blueprint 的 M01–M09 模块编号与 S03–S08 历史记录保留 "M09" 称谓不改写；v5 contract 内 "M09 oracle" 文案属 request-hash face 冻结不动。
- **后果**：eval_v1.yaml hash 沿革两版 provisional（`e9312af9…` → `8bc0f419…47eb`），冻结版见 §2。

### #5 类内贪心匹配替代 Hungarian（2026-10-03，用户拍板）

- **决策**：≤8×8 规模下用类内贪心（IoU 降序 → bbox 字典序 tie-break）一对一匹配，替代原拟 Hungarian。理由：与全局最优数值差异可忽略、类内匹配天然禁跨类别（类别错误 = FP+FN 语义更干净）、消除 δ_cat 参数、无 scipy 依赖。实现于 `evaluation/matching.py`。

### #6 解码与配置预注册（2026-10-03，Step 2；commit `3eaf93d`）

- **决策**：greedy 解码冻结（`do_sample: false`、`temperature: 0.0`、`max_new_tokens: 512`——沿 S03 截断教训）；base_1f.yaml（frozen）与 base_4f.yaml（**仅预注册不运行**）在测试推理前登记 hash，履行 stage 09 gate"Stage 10 配置 hash 已在本阶段运行前登记"。hash 见 §2。
- **验证**：runner 逐 anchor 断言重算 request hash == v5-face manifest 存储值，且 runner 侧 prompt 与 serializer 侧 prompt 逐字节一致——合约 face 错位即刻报错而非静默产出。

### #7 反事实子集预注册（2026-10-03，用户拍板；commit `8ee51c0`）

- **决策**：N=200、blank+shuffled 各一遍、按 scene 轮转分层抽样（seed=20261003）、test split；子集清单 `runs/S09_base_benchmark/counterfactual_subset.txt` 在 pilot 前产出。诊断不 gate。

### #8 bootstrap 参数预注册（2026-10-03，Step 2；随 Step 5 冻结）

- **决策**：scene-cluster paired bootstrap，B=2000、seed=20261003、confidence_level=0.95、cluster = scene_token；全部主指标共用同一重采样矩阵；S09→S12 对比用 paired-delta bootstrap（同 cluster map 同 seed）。参数预注册于 eval_v1.yaml，Step 5 冻结后不可变。

### #9 `input_tokens_p50` 承载 `visual_tokens_p50` 复核（Step 6 要求，2026-10-05 复核）

- **背景**：规划 §3 成本摘要行原列 `visual_tokens_p50`；S02 冻结 telemetry 只记总 prompt input tokens（文本 + 图像），无图像 token 单独计数，拆分需动冻结的 S02 加载栈。
- **裁定**：**维持 `input_tokens_p50` 承载，不做拆分**。理由：① 拆分属诊断增强而非 gate 项，不为它破坏冻结栈；② S09→S12 对比中该键同口径同分布，可比性不受影响；③ 报告侧已加 NOTE 注释说明承载关系（`evaluator.py` cost_summary）。
- **登记**：规划 §4 注意点与本条互指，缺口非静默。

### #10 反事实分母口径修复（2026-10-05；commit `7ed35a3`）

- **背景**：官方报告首跑发现 `counterfactual_comparison` 分子跳过非 parse_ok 对、但分母误用 `n_shared`（200），与冻结规格 §2.7（`n_compared` = 双侧 parse_ok 对数）不符。
- **裁定**：属"符合冻结规格的 bug 修复"，非指标定义变更——eval_v1.yaml 无需重新冻结（hash 不变，修复只触 `evaluator.py` 实现层）。
- **落地**：分母改 `n_compared`、新增 `n_shared` 追溯字段、新增回归单测（parse 失败对排除断言）；全量单测 255 passed；报告重生成后反事实读数与反事实记录手算值逐值一致（blank 29/197 = 14.72%、shuffled 34/195 = 17.44%）；主指标逐值不变。

### #11 溯源缺口备案：run summary 未 stamp git commit（2026-10-05）

- **事实**：`run_summary_predictions_*.json` 含 config_sha256 / contract_version / input_policy，但无 checkpoint 显式路径与 git commit 字段。
- **裁定**：接受现状，不回改 runner。绑定锚以 config sha 为准：checkpoint 路径与解码参数内嵌于冻结 base_1f.yaml（sha `1aab7db3…`），推理时逐 anchor request hash 对账（5,468/5,468）已兜底"跑的就是冻结请求"；代码侧推理窗起点 HEAD = `8ee51c0`（其后 commit 均为 docs/config，不触代码），由 commit 时间线旁证（规划 §4 台账已留痕）。

## 2. 参数冻结（Step 5/6 汇总）

| 产物 | sha256 | 状态 |
|---|---|---|
| `configs/benchmark/base_1f.yaml` | `1aab7db306b17f3688587e5d6006a98bb6a3fcc16326c26b8c8a2066d1332cb4` | frozen（2026-10-03），Step 4 以此运行 |
| `configs/benchmark/base_4f.yaml` | `793db1beee23f26b4dfc7ad92f425ae7c34e2908133b92156b51bdb5812d5e90` | 仅预注册，**S09 全程未运行**（S10 起才可执行） |
| `configs/evaluation/eval_v1.yaml` | `e6690f6fe950a1d1accdae3286dadbf6a1fad6cb503e3b8fe04ff22bbcf91f2d` | frozen（2026-10-05，commit `339afbc`），正式报告以此 hash 为准 |
| v5-face anchor manifest | `fd0d4dc40bf2bd728dd314cbeebca5b922203467ecc7bbf43c37ea9d07a2b09e` | `data/dataset_v4/anchor_policy_manifest.json`（sha 落 eval report meta） |

## 3. 阶段关闭核对（2026-10-05）

### 3.1 Stage 09 完成标准（`docs/stages_plan/09_single_frame_base_benchmark.md` §1）

> "Base-1F 的全量预测、主指标、95% CI、denominator 与资源统计可由固定配置重现。"

- 全量预测 ✓（`predictions_full.jsonl`，5,468 anchors 一次成型）；
- 主指标 + 95% CI ✓（`bootstrap_ci` 10 键 × {point, ci_low, ci_high, n_resamples, seed, confidence_level, n_scenes, n_values} 全覆盖）；
- denominator ✓（报告 `denominators` 键显式落档：matched_pairs 5,228 / parse_ok 5,449 / micro TP-FP-FN 按 τ / overconservative 分子分母）；
- 资源统计 ✓（cost_summary：gpu_hours 8.38 / latency / throughput / 显存）；
- 可由固定配置重现 ✓（Gate 2：固定脚本重跑，`eval_report.json` + `anchor_scores.jsonl` 逐字节一致）。

### 3.2 Stage 09 gates（§4）× 实施规划 §5 八项判定

| # | gate | 判定 | 证据 |
|---|---|---|---|
| 1 | 一一对应 + request_hash 对账（v5-face） | **PASS** | coverage 5,468/5,468 missing=0 extra=0；hash match 5,468/5,468 |
| 2 | 可复算（指标层逐字节） | **PASS** | 重跑 `eval_v1_verify_gate2/` 与正式报告两文件逐字节一致（runs/ 内留档） |
| 3 | 反事实控制 | **PASS** | blank/shuffled 独立文件 + 标记元数据；读数与手算值一致（反事实记录 gates 1/2 PASS） |
| 4 | 预注册义务（base_4f 未运行） | **PASS** | hash 于 Step 2 登记（§2 表）；runs/ 无 4F 产物 |
| 5 | 输入政策审计（1F + future fingerprint 零命中） | **PASS** | `test_base_benchmark_requests` 单测 9 条全绿（255 passed 含此）；run summary `input_policy=1F` |
| 6 | CI 完整性 | **PASS** | bootstrap_ci 10 键全带 CI + scenes=150 + denominator |
| 7 | 溯源 | **PASS**（缺口备案见 #11） | report meta 四件套 + §2 冻结 hash 表 + checkpoint 路径在冻结 config 内 |
| 8 | prompt↔schema 一致性 + v4 资产零改动 | **PASS** | pin 单测随 255 全绿；v4 目录未触碰（git 可证） |

唯一非 PASS 项：全量 run 记录 gate 3（resume 幂等生产抽查）**未触发**——全量一次成型无中断，不阻塞关闭，留作未来 resume 运行的顺带抽查项（`S09_benchmark_full_run.md`）。

### 3.3 指标缺口备案（非静默丢弃）

- **temporal dependence**：1F 无历史输入，归 S10（4F 阶段同口径补测）；
- **per-prediction observability / unscorable**：GT 池已在 S08 构建时可见性门控，评测侧无该输入；如需挂回须评测时访问 nuScenes（本阶段不引入）；
- **reasoning 不评分**：Base 阶段 reasoning 为模板/自由文本，无对齐基线可评；S12 起按 `S09_reasoning_supervision.md` 路线裁定后再定评测口径。

### 3.4 交接（→ S11/S12/S10）

- **S12 对比基线**：`runs/S09_base_benchmark/` 全套（predictions_full / eval_v1 报告 / anchor_scores.jsonl）；对比一律 paired-delta bootstrap（同 anchor 同 scene 配对）；
- **S12 gate 参照**：overconservative_yield_rate = 83.4%（CI [80.4, 86.2]）、action_flip ≈ 14.7%/17.4%——退化阈值在 S12 预注册时以本读数为参照；
- **S11/S12**：一律运行 contract v5 模型面（训练侧 prompt 与 S09 评测 prompt 同源）；dataset_v4 记录仍为数据源（GT 与 prompt 无关）；
- **S10**：只允许加载 base_4f.yaml（sha `793db1be…`），不得针对单帧 test 结果调参；
- **4F 显存可行性**：S11-4F 前未验证（已知后置风险，S12 前不阻塞，S11 实验记录须标注）。
