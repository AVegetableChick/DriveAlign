# qwen-1f-sft-benchmark-v7：SFT v7 首个正式评测存档（ctxbug 修复后重训 adapter）

- **日期**：2026-10-16
- **阶段**：S13（contract v7 + eval v3）；SFT（1 epoch，LoRA r16）checkpoint 首个 v7 面全量评测
- **来源**：`runs/S13_sft_1f_v7/eval_v3/`（定稿报告）+ `runs/S13_sft_1f_v7/paired_delta.json`
- **eval_report.md / eval_report.json / paired_delta.json**：定稿原样拷贝（eval_v3.yaml 冻结口径，sha `fae363ad…`）

## Provenance（全链 sha）

| 件 | 值 |
|---|---|
| benchmark config | `DriveAlign/configs/benchmark/sft_1f_v7.yaml` frozen `c462db1d…`（run_summary 实测一致） |
| adapter | `runs/S13_sft_1f_v7/checkpoints/sft_1f_v7/best` = v2 重训末点，`da1971e4…`（best_freeze.json v2） |
| eval config | `DriveAlign/configs/evaluation/eval_v3.yaml` frozen `fae363ad…`（meta.config_status=frozen 实测一致） |
| anchor manifest | `data/face_manifests/v7/anchor_policy_manifest.json` sha `c7df2a4e…`（与 base 逐位同源） |
| runner | batched_v1，bs=16，max_new_tokens=700，greedy；主跑 wall 14,940 s |
| 前置 | ctxbug 修复（encode_chat add_generation_prompt=True）+ A/B 对比实验定案（见 S13_v7_ctxbug_ab_experiment.md） |

## 评测输入

| 段 | 文件 | n | 说明 |
|---|---|---|---|
| 主 | `predictions_1f.jsonl` | 5,468 | SFT 面原生输出原图坐标，无需 rescale（base 需 CPU 反变换后处理，SFT 免除） |
| CF blank | `predictions_blank.jsonl` | 200 | `--anchors-file` base 同子集（counterfactual_subset.txt） |
| CF shuffled | `predictions_shuffled.jsonl` | 200 | 同上 |

## 核验快照

- 主跑 parse_ok 5,465/5,468（99.95%），base 91.64%；3 条失败全部为 bbox x1>x2 倒置（2f5de0ae id0 / 5e10fe79 id1 / 84c24cd1 id0），schema 校验拒绝，非截断非串位，占 0.05%
- 打满 700：0（max out_tok 563，mean 191；base 打满 308/5468、mean 325.7）
- 发射形态：5,465/5,465 紧凑单行，fence 零命中（ctxbug 修复后格式全程受监督，parse_failure_analysis 的截断/串位/fence 三大模式同步消失）
- CF blank：200/200 parse_ok，仅 3 种输出（全为空对象：KEEP_SPEED ×128 / ACCELERATE ×41 / STOP ×31），mean 21 tok
- CF shuffled：200/200 parse_ok，mean 148 tok

## 主指标（base 定稿 eval_v3_cf vs SFT，同锚同 GT；base 输入为 rescaled 文件，SFT 原生，同一原图空间）

| 指标 | base | SFT | Δ |
|---|---|---|---|
| parse_rate | 0.9164 | 0.9995 | +0.0830 |
| object_f1_micro@0.5 | 0.3764 | 0.6064 | +0.2301 |
| object_f1_macro@0.5 | 0.2434 | 0.4423 | +0.1989 |
| precision/recall micro@0.5 | 0.3898 / 0.3639 | 0.6810 / 0.5466 | +0.29 / +0.18 |
| matched_iou_mean | 0.7365 | 0.7864 | +0.0499 |
| yield_required_f1 | 0.3690 | 0.7008 | +0.3318 |
| speed_action_f1_macro | 0.1443 | 0.5725 | +0.4283 |
| motion_state_f1_macro | 0.2896 | 0.6734 | +0.3839 |
| risk_factors_f1_macro | 0.1064 | 0.3799 | +0.2735 |
| risk_attribution_f1_micro | 0.2116 | 0.6008 | +0.3892 |
| stop_miss_rate | 0.9973 | 0.5243 | -0.4730 |
| yield_overconservative_rate | 0.6628 | 0.1138 | -0.5490 |
| object_empty_hallucination_rate | 1.0000 | 0.0569 | -0.9431 |
| object_empty_miss_rate | 0.0000 | 0.0935 | +0.0935 |

## BEV 族（eval v3 新增；pairs 5,757 → 10,091，+75%）

| 指标 | base | SFT |
|---|---|---|
| bev_dxy_mean | 24.99 m | 1.72 m |
| bev_dx_mean / dy_mean | 24.31 / 4.52 | 1.48 / 0.64 |
| bev_dxy_p50 / p90 | — / — | 0.90 / 3.89 m |

分桶（dxy_mean，按 GT 距离，误差随位置条件化）：

| bucket | base | SFT |
|---|---|---|
| [0,10) | 8.62 | 0.65 |
| [10,25) | 19.42 | 1.40 |
| [25,40) | 34.95 | 2.62 |
| 40+ | 44.42 | 5.57 |

单调随距离增长（近处 0.65 m → 远处 5.57 m），是内容条件化的形态而非常量先验偏移。

## G4 / 守卫（paired-delta，同 cluster map / seed 20261003 / B=2000）

- G4 通过：parse_rate 0.9995 ≥ 0.99；P=object_f1_micro@0.5 delta +0.2301 [CI +0.2087, +0.2504] 下界>0；Q=speed_action_f1_macro delta +0.4283 [CI +0.4003, +0.4564] 下界>0——双指标均显著为正
- 6 条预注册守卫机械触发（逐条人工解读，最终阶段判定留用户复核）：
  1. constant_predictor：判据为合取（majority_share ≤ 0.867 且 action_flip ≥ base），实际失败在后者——SFT flip（blank 0.18 / shuffled 0.07）< base（0.70 / 0.55）。但 base 的 flip 被其 blank 57% parse 失败的垃圾输出抬高（base blank 可比对仅 86/200），属退化基线伪影；SFT 非常量预测器有独立证据：blank 输出 3 种分布、speed_action F1 macro 0.573（混淆矩阵铺开）、object F1 macro 0.442
  2. overconservative_stop_rate：SFT 4.52% vs base 0%（CI [+0.034, +0.058]）。base 恒 0 是因为它几乎从不 STOP（stop_miss 99.7%）——退化基线伪影；SFT 首次出现真实 STOP 行为（stop_miss 52.4%）代价是 4.5% 过度保守
  3-6. visual_dependence_* ×4：SFT 的 output_change/flip 低于 base（delta CI 全负）。base 的 100% output_change 由其 blank 下的解析垃圾贡献；SFT blank 输出全为合法空场景 JSON（78% 与 anchor 不同 = anchor 场景确有物体），object_empty_hallucination 5.7% vs base 100% 是更直接的接地证据

## BEV 反事实验证（BEV rules §6.1，判先验抄近路）

三证据全部否定"先验抄近路"：
1. blank 下物体归零：200/200 空场景输出（若为类别先验抄近路，blank 下应仍发射典型物体/BEV——base 正是如此，empty 场景 hallucination 100%）
2. shuffled 下输出响应：output_change 76.5%，物体发射随内容变化
3. 锚点误差随 GT 距离单调条件化（见分桶表）

结论：BEV 为视觉接地输出，非先验捷径。

## 事件记录（过程如实登记）

- CF 首跑误用 --limit 200（manifest 前 200）而非 §6.3 注册命令的 --anchors-file 子集 → 与 base 锚集不可配对；产物改名 predictions_blank_limit200.jsonl / predictions_shuffled_limit200.jsonl 保留，mis-caliber eval 目录删除后按注册命令重跑并重新 eval（本轮定稿）。同日 CF 命令首版另有一次参数名错误（--counterfactual → 实为 --image-transform），argparse 启动前拒绝，无副作用
- 速度口径备注：generation_seconds 为批级时间复制到批内成员，逐条对比应按批聚合；SFT 端到端 ~3.0-3.2 s/条 vs base 3.42（wall 4.15 h vs 5.2 h）——提速主要来自发射更短（out_tok -42%），端到端大头为两侧同款的批间数据管道开销（~45-50 s/批，telemetry 视野之外，值得后续插桩）

## BEV §6.2 审计（2026-10-16 补，BEV rules §6.2）

锚点分组 BEV dxy_mean（per-anchor 均值再平均；GT 来自 test shards 5,468 条记录）：

| 分组 | n 锚 | dxy | pairs/锚 | 其余锚 dxy |
|---|---|---|---|---|
| 截断（out_tok≥700） | **0** | — | — | — |
| 近距底边出画（GT y2≥899） | 864 | **1.28 m** | 3.1 | 1.90 m |
| 近距（GT min bev_x<10） | 1,879 | 1.44 m | 3.0 | 2.05 m |

判定：**通过，不污染整体读数**。SFT 无截断锚点；近距/出画锚点误差反而低于全局（1.77 m）且配对密度更高——BEV rules §6.2 "底边线索失效误差预计偏高"的预期被证伪：近距对象在图上更大更清晰，模型表现更好。对照组 base 侧见 base README 同名审计节（base 截断锚点仅 8 个有 bev 配对，截断的主要危害是配对缺失而非误差膨胀）。

## 遗留

- ~~阶段判定（Step 7 人工复核）~~ **已通过（2026-10-16 用户拍板，台账 §9 收官条目）**
- speed F1 macro 0.573 中 STOP 类召回仍弱（stop_miss 52.4%），可作为下阶段训练侧议题（非本阶段判据）
