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

## reasoning 监督方式选项（S11/S12 设计输入，2026-10-03 备案）

**背景观察**（来自 pilot/full 产物）：Base 未微调模型的 reasoning 输出格式自由、内容多样。而 S08 回填的 GT reasoning 是模板生成（3–5 变体、hash 确定性选句、固定叙事序、枚举锚定），信息熵低。SFT 监督方式因此成为一个真实的设计权衡，四条路线记录如下：

| # | 路线 | 优点 | 代价/风险 | 状态 |
|---|---|---|---|---|
| 1 | 模板 reasoning 监督（GT 现状，逐 token 计算 loss） | reasoning↔GT 字段双向一致可验；监督信号干净 | 模板塌缩：reasoning 沦为填空回声，信息量趋零；教模型"解释"的能力退化 | GT 资产形态（S08 冻结），默认监督源 |
| 2 | **不监督 reasoning（loss masking）**（2026-10-03 用户方向） | 模板梯度为零，从根上杜绝模板塌缩；reasoning 保持 Base 自由风格；实现近零成本（collator 对 reasoning 值域 token 置 loss 权重 0，schema 不变、模型仍输出 reasoning 字段） | reasoning 与结构化字段的一致性**无监督约束**，推理时可能出现文本与 `speed_action`/`yield_required` 矛盾的输出；reasoning 质量只能人工抽检（Eval v1 本就不评 reasoning，无 gate 冲突） | 用户意向，待 S11 规划拍板 |
| 3 | 自蒸馏 rejection sampling（STaR 式） | 兼得流畅性与可验证性：采样多条自由 reasoning，仅保留结构化字段与 GT 逐项一致的样本 | 需额外采样轮与过滤管线；v2 工作量 | 未来方向 |
| 4 | DriveLM 增强包（人工/大模型自由文本） | 语言多样性最高 | 覆盖 14.5%、~93 QA/帧需重过滤、与规则 GT 冲突需清洗；增强包当前关闭 | 备选 |

**与评测侧的衔接**：Eval v1 对 reasoning 不设任何自动指标（见 `S09_metric_spec.md` §2.5），S09 的全量 predictions 已将 Base 自由 reasoning 全量留档（`raw_text`），四种路线切换都不影响 benchmark 与 gate；若未来要评 reasoning 一致性/多样性，走 eval_v2（候选诊断：reasoning↔结构化字段一致率、distinct-n 多样性）。
