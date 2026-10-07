# qwen-1f-sft-v6 —— SFT-1F（1 epoch）vs Base-1F 评测对比

> 记录 SFT-1F 末点（1 epoch）与 1F-Base 在 v6 合同面上的全量指标对比。
> parse_rate 回退分解与失败形态分析已移至同目录
> `parse_failure_analysis.md`（2026-10-07）。
> 全量细节与决策链见 `stages_record/S12_single_frame_sft/S12_sft_vs_base_eval.md`（本文件为其关键内容移植）。
>
> 数据源：
> - Base：`runs/S09_base_benchmark/eval_v6/eval_report.json`（bs=1 冻结面电池）
> - SFT-1F：`runs/S12_sft_1f/eval_v6_batched/eval_report.json`（batched_v1，bs=16）
>
> 口径前提：
> - 同一 anchor manifest（sha `2abe8c4f…`，v6 面）、同一 eval_v1 配置（sha `e6690f6f…`）、
>   同一 bootstrap seed（20261003，scene 级 2000 次重采样）→ 配对可比；
> - SFT 侧两处**设计内**配置差异（`sft_1f_v6.yaml` 头注登记）：adapter 挂载 +
>   `max_pixels=360000`（A1，与训练侧同值；Base 走 processor 默认 → input p50 2313 vs 937 tok）；
> - SFT 侧读数**待 parity 复核**（bs=1 参照面 eval_v6/ 仅 164 行，见 S12_bug_log.md）；
> - ⚠️ overconservative_* 的跨模型对比是**伪象**，见 §3（Base KEEP_SPEED 塌缩所致）。

## 1. 核心指标总表

| 指标 | Base-1F | SFT-1F（1ep） | Δ | 95% CI 重叠？ |
|---|---|---|---|---|
| speed_action_f1_macro | 0.162 | **0.595** | **+43.3pp（3.7×）** | 否 ✅ |
| motion_state_accuracy | 0.313 | **0.760** | **+44.7pp（2.4×）** | 否 ✅ |
| object_f1_micro@0.5 | 0.336 | **0.583** | +24.7pp | 否 ✅ |
| object_f1_macro@0.5 | 0.216 | **0.399** | +18.3pp | 否 ✅ |
| object_precision_micro@0.5 | 0.402 | 0.684 | +28.2pp | ✅ |
| object_recall_micro@0.5 | 0.289 | 0.508 | +21.9pp | ✅ |
| yield_required_f1 | 0.045 | **0.486** | **+44.1pp（10.8×）** | 否 ✅ |
| risk_factors_f1_macro | 0.088 | 0.335 | +24.7pp（3.8×） | ✅ |
| max_items_hit_rate | 0.040 | 0.051 | +1.2pp | ✅ |
| matched_iou_mean | 0.732 | 0.792 | +6.0pp | — |
| parse_rate | **0.9993** | 0.9643 | **-3.49pp ⚠️** | 显著回退（§4） |

object 分类目 @0.5 全部上涨，弱项改善最大：barrier 0.025→0.361、pedestrian 0.072→0.421、
traffic_cone 0.051→0.234、trailer 0.177→0.335；强项续升：car 0.410→0.681、bus 0.586→0.727。

## 2. 行为质变：Base 输出塌缩，SFT 出现真决策

**speed_action（per-class F1 与混淆结构）**

| 类别 | Base | SFT-1F |
|---|---|---|
| ACCELERATE | 0.000 | 0.470 |
| DECELERATE | 0.054 | 0.281 |
| KEEP_SPEED | 0.595 | 0.765 |
| STOP | 0.000 | **0.865** |

Base 5,337 个 motion pair 里 **99.5% 预测 KEEP_SPEED**（5,312 条），全程只输出过 1 次 STOP；
GT=STOP 的 1,064 条中 1,036 条被判 KEEP_SPEED——典型多数类塌缩，其 macro F1=0.162 无行为含义。

**motion_state**：Base 把 stationary 判成 same_direction/oncoming（stationary recall **6.0%**，
125/2,096）；SFT stationary recall **89.6%**（4,789/5,346），四态混淆矩阵呈真实结构。

**检测配对**：matched_pairs 5,337→9,098（+70%）；micro@0.5 [TP,FP,FN]
[5,337, 7,954, 13,123]→[9,098, 4,212, 8,800]（FP -47%、FN -33%）。

**yield**：Base tp=24 / fn=924（几乎不让行）；SFT tp=762 / fn=155。

**visual dependence**：blank action flip 6.5%→**16.1%**（2.5×）、shuffled 5.0%→8.9%——
SFT 的输出真正依赖图像内容；Base 的 blank output_change=100% 是塌缩输出的退化伪象，不作正向读数。

## 3. overconservative_*：跨模型对比无效（伪象声明）

Base 的 overconservative_yield_rate=2.1%、stop_rate=0% **不是校准好**，而是
KEEP_SPEED 塌缩的直接后果——从不让行/急停，自然没有假让行/假急停
（yield tp 仅 24、stop 预测全程 1 次）。**这两个键只在同工作点的模型族内
（SFT 变体之间）做 paired-delta 才有意义**，Base→SFT 的 +31.5pp / +4.8pp
不构成 SFT 的退证据。

SFT 自身工作点（正确读法）：yield precision **34.3%**（762/2,220）、
recall **83.1%**（762/917）——从 Base 的 recall 2.5% 到 83.1% 是让行能力的出现；
代价是让得偏宽（precision 1/3），属工作点选择问题，留待 RL/阈值阶段调整。

## 4. parse_rate 回退分析（已移出）

parse_rate 99.93%→96.43%（失败 4→195，全部为 SFT 侧）的分解与形态学分析
（json 45 条：注释混入 39 / 截断 5 / 其他 1；schema 150 条：缺 `motion_state`
180 违规等）已整体移至同目录 **`parse_failure_analysis.md`**。

## 5. 成本对比（混合效应，谨慎外推）

| 项 | Base | SFT-1F | 变化 |
|---|---|---|---|
| latency_p50 / p90 | 3.98 / 6.66 s | 2.34 / 2.72 s | p50 1.7× |
| throughput | 0.220 anchors/s | 0.457 anchors/s | **2.08×** |
| GPU hours（主跑+反事实） | 6.90 | 3.32 | 2.08× |
| input_tokens_p50 | 2,313 | 937 | A1 分辨率对齐所致 |
| peak CUDA mem | 8.5 GB | 13.2 GB | bs=16 KV/激活增量 |

注意：2.08× 是 **bs=16 批量化（纯效应 3.48×，见 S12_batched_inference_speed.md）×
输入更短（A1）** 的混合结果，不能直接当批量加速读数引用。

## 6. 结论

1. SFT-1F 在 v6 面核心能力全面大幅提升（速度决策 3.7×、运动状态 2.4×、检测 +24.7pp、
   让行 10.8×、视觉依赖 2.5×），全部通过 scene 级 bootstrap 显著性检验；
2. Base 的多数指标是塌缩读数，跨模型比较时 overconservative_* 与
   visual_dependence_output_change 两个键须按伪象处理；
3. 真实代价集中在 **parse_rate -3.5pp**，失败形态已定位为两个可命名问题：
   json 失败 87% 是注释混入、schema 失败 100% 是 `critical_objects` 对象缺
   `motion_state` 键（详见同目录 `parse_failure_analysis.md`）；
4. 后续优先级建议：① 反查 SFT 训练 target 是否存在缺 `motion_state` 的对象；
   ② 训练侧修注释混入（数据清洗/few-shot 强约束）；③ 多训 1 epoch 观察
   parse_rate 与过保守工作点；④ parity gate 补跑后复核本记录读数。
