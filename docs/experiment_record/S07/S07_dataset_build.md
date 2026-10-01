# Stage 07 Step 2：全量数据集构建记录（anchor 枚举 → 分片落盘）

> 2026-10-01 执行。决策依据：D3（数据资产/运行产物分离）、D7（纯 nuScenes 规则构建），承接 `S07_scene_split.md` 冻结的三个 scene manifest。

## 1. 目标与边界

从三个冻结 scene manifest 出发，把每个场景的全部 4F 窗口候选（keyframe 游走序 index≥3，即 prevs≥3）逐个经 `records.adapter.build_record`（唯一构建路径）构建为 v1 Record，按确定性规则分片落盘，并产出 quarantine 清单与根索引。不涉及 1F/4F 对应表与分布对比（Step 3），不填 training_targets（S08）。

## 2. 输入 / 输出

- 输入：`data/dataset_v1/{train,val,test}_scene_manifest.json`（Step 1 冻结，载入时重算 sha256 校验）+ `data/nuscenes/trainval`（v1.0-trainval）。
- 规则：split 内按（scene manifest token 序 → 场景 keyframe 游走序）全局递增；`shard = global_index // 1000`，行内 line（0-based）`= global_index % 1000`；shard 每行 = canonical JSON（sort_keys compact，`sha256(line)` 即 `record_hash`）；场景末尾自然截断 future poses（非隔离条件）。
- 代码：`src/drivealign/dataset/build_dataset.py`（含 5 项 gate + shard roundtrip 全量复核）。
- 输出（数据资产，D3）：`data/dataset_v1/{split}/shard-NNNN.jsonl`（32 片）+ `quarantine.jsonl` + `dataset_manifest.json`（token→{split, shard 相对路径, line, record_hash}，绑定三个 scene manifest 指纹，自带 sha256，纯相对路径）。
- 输出（运行产物，D3）：`runs/S07_dataset/reports/build_report.{json,md}`；日志 `runs/S07_dataset/build_dataset.log`。

## 3. 执行命令

先 smoke（每 split 前 2 场景，写 scratch 目录，不触碰真实资产；沙箱内可跑）：

```
PYTHONPATH=DriveAlign/src python -m drivealign.dataset.build_dataset \
  --max-scenes 2 --out runs/S07_dataset/smoke_out --reports runs/S07_dataset/reports
```

全量（写 `data/dataset_v1`，外部物理盘，须用户 tmux）：

```
tmux new-session -d -s s07_build \
  'source /root/miniconda3/etc/profile.d/conda.sh && conda activate autovla_codeclean && \
   cd /root/autodl-tmp/drivealign_workspace && \
   PYTHONPATH=DriveAlign/src python -m drivealign.dataset.build_dataset 2>&1 | tee runs/S07_dataset/build_dataset.log'
```

## 4. Gate 结果（5/5 全 PASS）

scene_manifests_intact（载入时 sha256 重算一致）/ scene_sets_disjoint（三集两两不交）/ full_coverage（valid+quarantine==candidates，31,599 全归类）/ quarantine_codes_valid（code 属于 8-code 分类且不混入 index）/ shard_roundtrip（32 片逐行重读：sha256(line)==record_hash==from_dict 往返 hash、token 位置与 index 一致，0 问题）。

## 5. 构建 Summary

| split | scenes | keyframes | candidates | valid | quarantine | shards |
|---|---|---|---|---|---|---|
| train | 630 | 25,315 | 23,425 | 22,941 | 484 | 23 |
| val | 70 | 2,815 | 2,605 | 2,531 | 74 | 3 |
| test | 150 | 6,019 | 5,569 | 5,468 | 101 | 6 |
| 合计 | 850 | 34,149 | 31,599 | 30,940 | 659 | 32 |

- 候选数与 Step 1 预估精确吻合（keyframes − 3×scenes）。
- 隔离 659 条 **全部为 BAD_GAP**（keyframe 间隔 >0.6s，2Hz 丢帧），占 2.1%；散布 205 个场景，每场景 1–6 个连续窗口（一个丢帧污染其前后连续候选段，最重场景 6 条）。smoke 期的 scene-0126 BAD_GAP 与全量结果一致，证明管线敏感而非 bug。
- 其余 7 个 reason code 零触发：INSUFFICIENT_HISTORY/CROSS_SCENE 被枚举方式结构性排除；NO_CAM_FRONT/MISSING_IMAGE/MISSING_CALIBRATION/NUMERIC_ANOMALY 零触发说明 trainval 资产完好。
- 溯源链闭合：builder_commit `817187e` + contract v3 + scene manifest sha256（见 Step 1 记录 §6）+ manifest 自身 sha256（§6）。
- 资产体积 64MB；smoke 产物留存于 `runs/S07_dataset/smoke_out/`（scratch，可随时删）。

## 6. 产物指纹（sha256）

- `dataset_manifest.json`：`edea772da2dd86e9a285046eeab0b7ddd89619a2764b099d3d5831932c00efff`
- `quarantine.jsonl`：`27ab38eef1ff70013818ffc2c8972b227619db6e4e722c1c11315f55df6a3f15`
- 抽样锚点 `train/shard-0000.jsonl`：`407fd12c13438eb1ea28f52cbd1b964d39818970f346d9918e4fb0c26389d9cd`

同 raw+代码+配置 的跨进程重建比对（byte-identical）归入 Step 5/6 执行。

## 7. 已知限制

- BAD_GAP 阈值 0.6s 继承 S04 的 `DEFAULT_MAX_GAP_S`；2Hz 丢帧窗口被整体隔离而非降级为 1F，1F/4F 策略选择在 Step 3 manifest 层面仍只对 valid 记录成立。
- 隔离的 659 个 anchor 的 keyframe 不进入任何 split 的训练/评估；其场景仍在 manifest 中（split 边界在 scene 级冻结，不受影响）。
- training_targets 两字段恒为 None（S08 回填）；当前 manifest 的 record_hash 已覆盖该状态，S08 回填将产生新 hash 代际（v1 record 不变式不受影响）。

## 8. 交接（Handover）

- Step 3（manifest.py）：只对 30,940 条 valid 记录建 1F/4F anchor 一一对应表 + train/val/test 分布对比表（location × 时段 × description 高频词，仅报告不筛选）；数据源为 `dataset_manifest.json` + scene manifests，不改任何资产。
- test 5,468 条 valid 是 benchmark 口径（BAD_GAP 隔离后），后续评测表须以 manifest 的 record 集为准，不得回退到 keyframe 计数。
- 沙箱无法写 `data/dataset_v1`：凡重建/再分片一律走用户 tmux（§3 命令模式）。
