# S09 Benchmark 记录：反事实子集推理（blank / shuffled，200 anchors）

> 状态：**运行中/待回填**。本文档预置于推理启动前，`结果记录`区在跑完后由 agent 核验回填。
> 配置：同全量（`configs/benchmark/base_1f.yaml` frozen）；唯一差异是 `--image-transform` 与 `--anchors-file`。
> 子集：`runs/S09_base_benchmark/counterfactual_subset.txt`（200 tokens / 150 scenes / seed=20261003，test split，场景轮转分层抽样，预注册）。

## 命令（tmux 内直接执行；两段串行，GPU 单卡）

```bash
source /root/miniconda3/etc/profile.d/conda.sh && conda activate autovla_codeclean && \
cd /root/autodl-tmp/drivealign_workspace && \
set -o pipefail && \
PYTHONPATH=DriveAlign/src python -m drivealign.cli.base_benchmark \
  --config DriveAlign/configs/benchmark/base_1f.yaml \
  --anchor-manifest data/dataset_v4/anchor_policy_manifest.json \
  --dataset-root data/dataset_v4 --nuscenes-dataroot data/nuscenes/trainval \
  --anchors-file runs/S09_base_benchmark/counterfactual_subset.txt \
  --image-transform blank \
  --resume \
  --out runs/S09_base_benchmark/predictions_cf_blank.jsonl 2>&1 | tee runs/S09_base_benchmark/cf_blank.log
```

```bash
source /root/miniconda3/etc/profile.d/conda.sh && conda activate autovla_codeclean && \
cd /root/autodl-tmp/drivealign_workspace && \
set -o pipefail && \
PYTHONPATH=DriveAlign/src python -m drivealign.cli.base_benchmark \
  --config DriveAlign/configs/benchmark/base_1f.yaml \
  --anchor-manifest data/dataset_v4/anchor_policy_manifest.json \
  --dataset-root data/dataset_v4 --nuscenes-dataroot data/nuscenes/trainval \
  --anchors-file runs/S09_base_benchmark/counterfactual_subset.txt \
  --image-transform shuffled \
  --resume \
  --out runs/S09_base_benchmark/predictions_cf_shuffled.jsonl 2>&1 | tee runs/S09_base_benchmark/cf_shuffled.log
```

## 目标边界

- 反事实语义：blank = 全灰图（摧毁全部场景内容）；shuffled = patch 打乱（保留纹理摧毁空间布局）；`make_blank_image`/`make_shuffled_image` 已有实现与单测；
- 产物独立成文件，**不污染主预测** `predictions_full.jsonl`；
- 消费方式：evaluator 的 counterfactual 块与主预测逐 anchor 比较（仅双侧 parse_ok 的对），产出 `visual_dependence_action_flip_{blank,shuffled}` 与 `visual_dependence_output_change_{blank,shuffled}`，定义见 `S09_metric_spec.md` §2.7；
- **诊断不 gate**（Base 内容质量低可接受；本记录为 S12 退化参照提供基线读数）。

## 执行约定

- 必须在主跑（全量）之后串行执行，不与主跑并行抢占 GPU；
- `--resume` 幂等，中断重跑同一条命令即续跑；
- blank 与 shuffled 是两次独立推理（不同文件、不同 log），不可合并。

## 结果记录（跑完后回填）

- [ ] blank：anchors_scored = 200；起止时间/耗时；
- [ ] shuffled：anchors_scored = 200；起止时间/耗时；
- [ ] `n_compared`（双侧 parse_ok 对数）与两变换 × 两比率的四个读数（Step 5 evaluator 产出后回填）；
- [ ] 与主跑同 token 的输出差异定性观察（flip 集中在哪些场景）。

## 验收 gates

1. 两个产物文件各恰 200 行，token 集合与 counterfactual_subset.txt 完全一致；
2. 反事实行带 image-transform 标记元数据（blank/shuffled 可区分）；
3. 读数解读备案：变化率 ≈ 0 → 输出未使用图像（文本先验"背答案"）；变化率高 → 输出由图像驱动——无论哪种都如实记录，不做好坏评判。

