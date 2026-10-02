# Stage 07 阶段收尾：gate 核对与产物清单

> 2026-10-01。S07 全部六步完成；分步记录见 `S07_input_validation.md`（决策台账）、`S07_scene_split.md`（Step 1）、`S07_dataset_build.md`（Step 2）、`S07_anchor_policy_manifest.md`（Step 3）。本文只做阶段级核对。

## 1. Stage 07 Gate 核对（5/5 全 PASS）

| Gate | 结论 | 证据 |
|---|---|---|
| train/val/test scene 集合两两不交叉 | PASS | Step 1 gate 6/6（assignment_matches_devkit 等），630+70 vs 150 scene 不相交 |
| 1F/4F manifest anchor tokens 一一对应、数量相同 | PASS | Step 3 gate `anchor_sets_identical`：30,940 = 22,941 + 2,531 + 5,468，双策略覆盖同一集合，parity 0 问题 |
| 100% candidates 归为 valid 或带 reason code 的 quarantine | PASS | 31,599 = 30,940 valid + 659 quarantine（全部 BAD_GAP，8 code 冻结断言有单测锁定） |
| validation/test 不参与任何 fitted state | PASS | 全链路纯规则（D4 字典序切分、D7 无模型、无统计拟合），无任何从 val/test 估计的参数 |
| 相同 raw+代码+配置 → 相同 manifests 与 Record hash | PASS | `determinism_report.json`：重建 38 个文件（32 shard + quarantine.jsonl + dataset_manifest.json + anchor_policy_manifest.json + 3 报告）逐字节一致，pinned builder_commit `817187e` |

## 2. 单元测试

117 passed（S06 既有 91 + S07 新增 26：split 规则/确定性/哈希 8、分片与 roundtrip/防篡改 12、anchor parity 与分布 6）。启动命令见各测试文件 docstring。

## 3. 产物清单

数据资产（`data/dataset_v1/`，物理 `/root/autodl-tmp/datasets/drivealign_dataset/v1`）：
- `{train,val,test}_scene_manifest.json`（Step 1，sha256 见 S07_scene_split.md §6）
- `{train,val,test}/shard-NNNN.jsonl` 共 32 片、30,940 valid 记录（Step 2）
- `quarantine.jsonl` 659 条（全部 BAD_GAP）
- `dataset_manifest.json`（token→shard 相对路径/行号/record_hash 索引，sha256 `fe91b719…`）
- `anchor_policy_manifest.json`（30,940 anchor 双策略指纹，sha256 `f94b4bad…`，Step 3）

运行产物（`runs/S07_dataset/`）：split/build/manifest 三日志、reports/{split_distribution,build,record_distribution,determinism}_report.{json,md}、smoke 产物与 verify_out（可复现工作区）。

## 4. 关键数字备忘

- 规模：train 630 scenes / 23,425 候选 / 22,941 valid；val 70 / 2,605 / 2,531；test 150 / 5,569 / 5,468
- 隔离率 2.1%，全部 BAD_GAP（>0.6s keyframe 间隔），对分布影响 |Δpp| ≤ 0.5（Step 3 审计）
- benchmark 口径：test 有效记录 **5,468**（不是 6,019 keyframes）

## 5. 交接 S08

- 训练/评测样本迭代入口：`data/dataset_v1/anchor_policy_manifest.json`（token→shard/line/record_hash/request_hash_1f/4f）；读取 shard 行后 `DriveAlignRecord.from_dict`。
- 1F/4F 是全局策略选择，由 val 过拟合信号决定；两视图 anchor 集合已证明一致。
- S08 回填 `training_targets`（expected_output = 结构 GT + reasoning 模板渲染，骨架感知→风险→决策）会改变 record hash 代际——须升 contract_version 并重跑 Step 2/3（教训已记录在 S07_anchor_policy_manifest.md §7）。
- DriveLM 仅作 S08 可选语言增强包（D7）；任何 DriveLM 文本进 reasoning 须过 D5/D6 过滤链。
