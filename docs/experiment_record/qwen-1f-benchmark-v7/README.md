# Qwen 1F Base Benchmark v7 评测存档（S13）

- **存档日期**：2026-10-15
- **阶段**：S13（contract v7 + eval v3）；base（未训）checkpoint 首个 v7 面全量评测
- **来源**：`runs/S13_base_1f_v7_batched/eval_v3_cf/`（定稿报告；首跑 `eval_v3/` 留档）
- **eval_report.md / eval_report.json**：定稿报告原样拷贝（eval_v3.yaml 冻结口径，sha `fae363ad…`）
- **parse_failure_analysis.md**：457 条 parse_failed 全量归因（截断主导 65.2% / schema 串位三类 / fence 提取缺口 5 条）+ 裁决"格式不走后处理挽救，由 SFT 监督覆盖"（2026-10-15）

## 运行配置

| 项 | 值 |
|---|---|
| checkpoint | Qwen2.5-VL-3B base（未训，无 adapter） |
| runner | batched_v1，bs=16（S12 Step 6b 先例独立预注册面） |
| benchmark config | `configs/benchmark/base_1f_v7_batched.yaml`，frozen sha `a3480104…` |
| 视觉口径 | max_pixels=360000 → smart-resize 784×448（与 SFT 面 A1 对齐，2026-10-14 用户拍板） |
| 生成 | max_new_tokens=700，SFT 发射形态为紧凑 JSON |
| 数据根 | `data/dataset_v7`（**v7 face 专用**，勿用 dataset_v4） |
| manifest | `data/face_manifests/v7/anchor_policy_manifest.json`（5,468 test anchors） |
| 评测输入 | `predictions_1f_rescaled.jsonl`（bbox 已从 784×448 空间反变换回原图 1600×900） |
| counterfactual | blank / shuffled 各 200（`counterfactual_subset.txt`，seed 20261003），bbox 天然原图坐标 |

## 关键读数速览

- parse_rate **0.9164**（5011/5468；json_parse 298 / schema 155 / semantic 4）
- object_f1_micro@0.5 **0.3764**（P 0.3898 / R 0.3639），matched_iou_mean **0.7365**
- motion_state_f1_macro **0.2896**；yield_required_f1 **0.3690**；stop_miss_rate **0.9973**
- BEV dxy_mean **24.99m**（p90 41.31m；未训模型量级）；signed dx_pos +73.33 / dx_neg −24.23（长尾不对称）
- risk_attribution micro/macro **0.2116 / 0.2144**；risk_factors_f1_macro 0.1064
- 反事实基线：blank output_change 1.0 / action_flip 0.6977（仅 86/200 可比对）；shuffled 1.0 / 0.5515（165/200）
  —— base 未训读数，正式视觉依赖判读留 SFT 后 paired-delta

## 口径备注

1. base 模型未受训练教"原图绝对坐标"，输出处于 resize 空间；v7 起流水线在 parse 端按 config `bbox_output_space: [784, 448]` 内联反变换（决策台账 §9 bbox 条目）。主 predictions 因早于该修复跑出，走等价的手动 rescale 文件；GPU 冒烟已证两法逐框等价（16/16 raw_text 一致、66/66 框 ≤0.05）。
2. **SFT v7 评测不需要 rescale**（训练 target 教原图坐标），`benchmark/sft_1f_v7.yaml` 不设 `bbox_output_space`。
3. bs=16 与 bs=1 存在 padded attention 数值路径差异（同 8 token parity 3/8 逐值一致），跨代绝对值对比需谨慎；S13 内部 paired-delta 两侧同用 bs=16 自洽。
4. `batch_benchmark` 为追加写：覆盖同名 `--out` 重跑会叠加产生 duplicate，evaluator fail-fast 拦截。

## BEV §6.2 审计（2026-10-16 补，BEV rules §6.2；SFT 定稿后回填）

| 分组 | n 锚 | dxy | pairs/锚 | 其余锚 dxy |
|---|---|---|---|---|
| 截断（out_tok≥700） | 8（308 个截断锚中仅 8 个有 bev 配对） | 27.60 m | 2.1 | 25.62 m |
| 近距底边出画（GT y2≥899） | 658 | 19.49 m | 2.1 | 27.11 m |
| 近距（GT min bev_x<10） | 1,473 | 20.47 m | 2.0 | 29.61 m |

判定：截断锚点 dxy 27.60 vs 全局 25.62（n=8，量级相同）——截断的主要危害是**配对缺失**（308 锚仅 8 个产生 bev 配对）而非误差膨胀，不污染整体读数；近距/出画组误差反而更低，是 base 先验偏移误差结构的体现（近距 GT 绝对值小，§6.1 已判 base 为先验行为）。

## 相关文档

- **SFT v7 评测定稿存档**：`DriveAlign/docs/experiment_record/qwen-1f-sft-benchmark-v7/README.md`（paired-delta 对照侧）

- 决策台账：`DriveAlign/docs/stages_record/S13_contract_and_eval_revision/S13_contract_v7_decision_log.md` §9
- eval 口径：`S13_eval_v3_metric_additions_draft.md`（定稿）+ eval 台账 `S13_eval_v1_to_v2_metric_changes.md` §10
- 实施规划：`S13_v7_implementation_plan.md`
