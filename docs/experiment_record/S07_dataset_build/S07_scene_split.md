# Stage 07 Step 1：分层字典序场景划分记录

> 2026-10-01 执行。决策依据：D3（数据资产/运行产物分离）、D4（分层字典序切分），见 `S07_input_validation.md` 与 `docs/project_stages/07_dataset_build_and_split.md`（2026-10-01 D3/D4 冻结版）。

## 1. 目标与边界

以 scene_token 为最小划分单位，把 nuScenes v1.0-trainval 冻结为 train / val / test 三个 scene manifest（不展开到 anchor，不构建 Record）。一切后续样本级构建（Step 2）只能基于这三个 manifest。

## 2. 输入 / 输出

- 输入：`data/nuscenes/trainval`（v1.0-trainval，850 scenes / 34,149 keyframes，已端到端验证）；devkit `nuscenes.utils.splits` 的 train（700 names）/ val（150 names）名单。
- 规则（D4）：train split 700 scenes 按 (log.location × time_of_day) 分桶；time_of_day = 首 keyframe UTC hour ∈ [06,18) → day，否则 night；桶内 scene_token 字典序，前 90% = train、后 10% = val（half-up 取整）；test = devkit val 名单全部 150 scenes，不筛选；天气不进分层键。
- 代码：`src/drivealign/dataset/split.py`（含 6 项 scene-level gate）。
- 输出（数据资产，D3）：`data/dataset_v1/{train,val,test}_scene_manifest.json`（排序 tokens + 分桶表 + 规则快照 + builder commit + manifest_sha256，仅相对路径约定自本阶段 manifest 起）。
- 输出（运行产物，D3）：`runs/S07_dataset/reports/split_distribution_report.{md,json}`。

## 3. 执行命令

```
tmux new-session -d -s s07_split \
  'source /root/miniconda3/etc/profile.d/conda.sh && conda activate autovla_codeclean && \
   cd /root/autodl-tmp/drivealign_workspace && \
   PYTHONPATH=DriveAlign/src python -m drivealign.dataset.split 2>&1 | tee runs/S07_dataset/split.log'
```

## 4. Gate 结果（6/6 全过）

train_val_disjoint / test_disjoint / train_plus_val_is_700 / test_is_150 / val_nonempty / assignment_matches_devkit 全部 True。首跑曾 FAIL：gate 误拿 scene token 与 devkit scene **name** 比对，修复为 name 集合比对后 PASS（bug 已修，教训：跨标识符比对必须显式换算）。

## 5. 划分 Summary

### 5.1 规模

| split | scenes | keyframes | 预估 4F 窗口（keyframes − 3×scenes） |
|---|---|---|---|
| train | 630 | 25,315 | ~23,425 |
| val | 70 | 2,815 | ~2,605 |
| test | 150 | 6,019 | 5,569 |
| 合计 | 850 | 34,149 | ~31,599 |

train+val = 700 / 28,130，与交接预估一致；test 5,569 与"6,019 − 3×150"一致。

### 5.2 分桶明细（train split 700）

| location × tod | scenes | train | val |
|---|---|---|---|
| boston-seaport / day | 193 | 174 | 19 |
| boston-seaport / night | 197 | 177 | 20 |
| singapore-onenorth / day | 75 | 67 | 8 |
| singapore-onenorth / night | 73 | 66 | 7 |
| singapore-queenstown / day | 71 | 64 | 7 |
| singapore-hollandvillage / day | 70 | 63 | 7 |
| singapore-queenstown / night | 21 | 19 | 2 |

每桶 val = half-up(10%)，最小桶（21）取 2，无空桶。

### 5.3 分布对比（仅报告，不筛选）

location 占比 %：boston-seaport 55.7/55.7/51.3，onenorth 21.1/21.4/23.3，queenstown 13.2/12.9/15.3，hollandvillage 10.0/10.0/10.0（train/val/test 顺序）。**train/val 逐桶对齐**（分层直接效果）；test 略偏 Boston，官方 val split 固有构成。

时段占比 %：day 58.4/58.6/**33.3**，night 41.6/41.4/**66.7** → **test 为 night-heavy（2:1）**，官方 val split 固有偏斜，train/val 不补偿。

描述关键词 /1k scenes（train/val/test）：parked 639.7/628.6/746.7，peds 490.5/600.0/486.7，intersection 384.1/328.6/386.7，bus 209.5/300.0/286.7，bicycle 158.7/100.0/153.3，vehicle 73.0/28.6/106.7。val 与 train 同带（n=70 小样本噪声正常），无系统性题材漂移。

## 6. 产物指纹（manifest_sha256）

- train：`cf4d4e2895746af234a8ea53276e4edcf7fae9b58d0ffd91d1f0d9190c1d31fb`
- val：`21bcbe71bdda318f839edf2a1a3386760e3f20222dfbc431ea4efc0bf2349f69`
- test：`d5987b2f95ee5582e2bc2088ac71a6fd9284196a21ca2b1c45e3924aa43c7f19`

canonical json（sort_keys）构建；跨进程重跑逐字节比对归入 Step 5/6。

## 7. 已知限制

- test night-heavy 是官方 val split 固有偏斜（D4 冻结：不筛选）。
- "day/night" 为 UTC 小时代理：Boston（UTC−5）≈当地日照；新加坡（UTC+8）语义有偏移（night 桶≈当地凌晨/清晨暗光）。若未来需语义准确时段，须重议 D4 并重建 manifest——当前分层键的确定性/均衡性不受影响。

## 8. 交接（Handover）

- Step 2（build_dataset.py）必须只从这三个 manifest 取 scene：枚举 prevs≥3 anchor → `records.adapter.build_record`（唯一构建路径）→ 按确定性规则分片写 `data/dataset_v1/{train,val,test}/shard-NNNN.jsonl`（1,000 条/片）+ 根目录 `quarantine.jsonl`（8 个 reason code）。
- 4F 窗口预估：train ~23,425 / val ~2,605 / test 5,569；全部 candidates 必须归为 valid 或带 reason code 的 quarantine。
- test 不参与任何 fitted state；同配置重建 hash 一致性在 Step 5/6 验证。
