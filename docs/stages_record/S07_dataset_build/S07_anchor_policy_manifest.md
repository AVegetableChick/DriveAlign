# Stage 07 Step 3：1F/4F anchor 对应表与 record 级分布审计

> 2026-10-01 执行。决策依据：D3（1F/4F 对应表为独立数据资产，不改动 dataset_manifest.json）、D4（分布仅报告不筛选）。承接 `S07_dataset_build.md`。

## 1. 目标与边界

在不动任何记录的前提下，把 Step 2 资产目录化成两份产物：(a) 对 30,940 条 valid 记录经 Stage 06 serializer 推导 1F/4F 双视图指纹，落独立资产 `anchor_policy_manifest.json`，承载 Stage 07 gate"1F/4F manifest anchor tokens 一一对应、数量相同"；(b) record 级分布对比报告，审计 BAD_GAP 隔离是否引入分布漂移。不构建记录、不修改 dataset_manifest.json、不筛选。

## 2. 输入 / 输出

- 输入：`data/dataset_v1/dataset_manifest.json` + 三个 scene manifest（全部重算 sha256，且 scene 指纹必须与 dataset manifest 绑定一致）+ `data/nuscenes/trainval`（scene 元数据与 scene 级基线）。
- 方法：逐行重读 32 个 shard，复核落位（token/shard/line/`sha256(line)==record_hash`）→ `DriveAlignRecord.from_dict` → `serialize(model_inputs, 1F|4F)` → `request.canonical_hash()` 指纹；scene 级基线复用 `split.distribution_stats`（与 Step 1 同口径）。
- 代码：`src/drivealign/dataset/manifest.py`（含 6 项 gate）。
- 输出（数据资产，D3）：`data/dataset_v1/anchor_policy_manifest.json`（30,940 anchor：request_hash_1f/4f、anchor_image_relpath、落位、scene_token；绑定 dataset_manifest_sha256 与 scene 指纹，自带 sha256）。
- 输出（运行产物，D3）：`runs/S07_dataset/reports/record_distribution_report.{json,md}`；日志 `runs/S07_dataset/manifest.log`。

## 3. 执行命令

```
tmux new-session -d -s s07_manifest \
  'source /root/miniconda3/etc/profile.d/conda.sh && conda activate autovla_codeclean && \
   cd /root/autodl-tmp/drivealign_workspace && \
   set -o pipefail && PYTHONPATH=DriveAlign/src python -m drivealign.dataset.manifest 2>&1 | tee runs/S07_dataset/manifest.log'
```

（`set -o pipefail` 必加：裸 `| tee` 会掩盖 python 退出码，smoke 期曾掩盖一次 KeyError。）

## 4. Gate 结果（6/6 全 PASS）

assets_intact / scene_binding_match / anchor_sets_identical（counts==预期 30,940，两策略覆盖同一 anchor 集）/ policy_parity（prompt 一致、anchor token 一致、1F 图==4F 末帧、speed 一致，0 问题）/ placement_reverified（32 片全部落位复核 0 问题）/ split_paths_consistent。

## 5. Summary

### 5.1 对应表规模

| split | anchors | request_hash_1f/4f |
|---|---|---|
| train | 22,941 | 全部派生成功 |
| val | 2,531 | 全部派生成功 |
| test | 5,468 | 全部派生成功 |
| 合计 | 30,940 | 与 dataset_manifest record_count 精确一致 |

### 5.2 分布审计（scene% → record%，Δpp）

- **tod**：train 0.0/0.0；val day −0.5、night +0.5；test day +0.4、night −0.4。
- **location**：全部 |Δ| ≤ 0.5pp（最大 onenorth train −0.2 / val +0.3）。
- **keywords /1k**：train parked 639.7→639.9、peds 490.5→492.1；val/test 同带（parked 625/749，peds 604/488），与 Step 1 scene 级结论一致。

**结论：BAD_GAP 隔离（659 条）对 split 代表性无可测影响（|Δ| ≤ 0.5pp），分布审计通过。**

### 5.3 溯源链

builder_commit `817187e`（与 dataset_manifest 同 commit）+ contract v3 + dataset_manifest_sha256 `fe91b719…` + scene 指纹三件套（Step 1 §6）+ 资产自哈希（§6）。

## 6. 产物指纹（sha256）

- `anchor_policy_manifest.json`：`f94b4bad9de3e36955df052c1ef77ee84122d6b97a3738564e9794b1fb03ca8b`（17.9MB，自哈希已独立复核）

## 7. 已知限制

- request_hash 绑定 contract v3 prompt 文本；S08 若动 prompt 或 serializer 需升 contract_version 并重建本表（hash 代际）。
- 分布审计关键词口径复用 split.py 的 stopword 表；新词表需两个 Step 同步改。
- anchor_policy_manifest 是 shard 的派生物，二者由 dataset_manifest_sha256 绑定；任何 shard 重建都必须重跑 Step 3。

## 8. 交接（Handover）

- Step 4（单测）：split 确定性、shard roundtrip、gate 断言、anchor parity 单测（可用 mini 构造的小 record 或 smoke_out 固定产物做 fixture）。
- Step 5（CLI + 确定性重跑）：`cli/` 入口风格参照 `record_build_smoke.py`；全量重跑 diff 验证 byte-identical（Step 2/3 的产物指纹是比对基线）。
- S08 消费入口：`anchor_policy_manifest.json` 的 anchors 表即训练/评测样本迭代索引（token→shard/line/record_hash/双策略指纹），1F/4F 为全局策略选择。
