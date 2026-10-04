# S09 Benchmark 记录：反事实子集推理（blank / shuffled，200 anchors）

> 状态：**已完成（2026-10-05）**，结果区已回填；gates 1/2 PASS，gate 3（解读备案）见结果记录。
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

## 结果记录（2026-10-05 回填）

- [x] blank：200 行、token 集合与子集完全一致（in=200 / missing=0）、**parse_ok 200/200（100%）**、耗时 1,131.8 s（≈18.9 min）；run summary：config sha `1aab7db3…1332cb4` 与登记一致、`image_transform: blank`；
- [x] shuffled：200 行、token 集合一致、**parse_ok 198/200**（2 例 schema_failure：`7f4c0c97…`、`c3d70d9f…`，配对时按双侧 parse_ok 规则排除）、耗时 1,034.7 s（≈17.2 min）；`image_transform: shuffled`；
- [x] **视觉依赖配对读数**（主跑 vs 反事实，双侧 parse_ok 对）：

| 指标键 | blank | shuffled |
|---|---|---|
| `n_compared` | 197/200 | 195/200 |
| `visual_dependence_action_flip_*` | **29/197 = 14.72%** | **34/195 = 17.44%** |
| `visual_dependence_output_change_*` | 197/197 = 100% | 195/195 = 100% |
| （逐字段变化：critical_objects） | 100% | 100% |
| （逐字段变化：risk_factors） | 77.16% | 32.82% |
| （逐字段变化：reasoning） | 100% | 99.49% |
| （逐字段变化：yield_required） | 23.35% | 21.54% |
| （逐字段变化：speed_action） | 14.72% | 17.44% |

- [x] 定性观察：
  - **output_change = 100%** 由 critical_objects 字段机械驱动——图内容被摧毁后对象输出必然与真实场景输出不同，该键的诊断力弱，**action_flip 才是主读数**；
  - **action_flip ≈ 15–17%**：图像内容摧毁后仍有约 1/6 帧的 speed_action 翻转——Base **确实使用了图像**（非纯文本先验"背答案"），但 83–85% 的动作在无图信息下不变，动作主要由文本先验（available_speed）+ 模型先验驱动。两变换读数接近（14.7% vs 17.4%，shuffled 略高）符合预期（blank 摧毁更彻底但 Base 对二者的依赖差异不大）；
  - 此读数为 **S12 退化参照基线**：SFT 后 `visual_dependence_action_flip_*` 若显著跌向 0（模型学会贴标签先验、更不看图）即触发 gate 关注。

## 验收 gates

1. 两个产物文件各恰 200 行，token 集合与 counterfactual_subset.txt 完全一致——**PASS**（各 200 行、in=200/missing=0）；
2. 反事实行带 image-transform 标记元数据（blank/shuffled 可区分）——**PASS**（run summary 与逐行 `image_transform` 字段均正确）；
3. 读数解读备案：变化率 ≈ 0 → 未用图像；变化率高 → 图像驱动——**已备案**（action_flip 15–17%，介于两端：有真实视觉依赖但动作主导仍是文本先验；不做好坏评判，作为 S12 对照基线记录）。

