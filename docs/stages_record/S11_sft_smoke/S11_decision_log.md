# S11 SFT Smoke —— 决策日志（时间序）

> 本文件按时间序记录 S11 的关键裁定；格式沿 S09 `S09_decision_log.md`。关闭核对见文档末尾。

## 2026-10-05 裁定记录

### D1：reasoning 字段从模型面删除（v6）

- **裁定**：reasoning 不再作为模型输出字段、不进训练目标（四字段全监督）。
- **论证/被否方案**：见 [S11_reasoning_supervision_decision.md](S11_reasoning_supervision_decision.md)。
- **落地**：contract v6（四字段 schema + 删 reasoning prompt 行），详见 Step 0。

### D2：SFT 样本冻结 = A/B 分层（本轮拍板）

- **背景**：train32/overfit128 是"token 选取 + 序列化样本"两种复用性不同资产的叠合。
- **裁定**：
  - **A）token 选取**（可复用、跨 contract）→ `data/` 可复用目录 + `sft_subsets/`。
  - **B）序列化样本**（面绑定、换 face 重打）→ `runs/S11_sft_smoke/sft_samples/`。
- **不放 `data/dataset_v4/`**：那是 S08 冻结基座，保持不可变不变式。
- **落位原则再细分**：面级可复用资产（face manifest）与实验 artifact 分属不同层，
  避免"定位模糊"；本次先把 SFT 样本拆清，face manifest 的 T2 层归位（v5/v6）记录为
  后续清理项，不阻塞 Step 2。
- **落地**：`cli/sft_build_samples.py` 拆 `select`（A）/ `serialize`（B）两子命令；
  计划 §3 Step 2 已按此改写（见 `S11_implementation_plan.md`）。
- **保留项（后续）**：~~v5 面 manifest 自 `data/dataset_v4/` 迁至独立面 manifest 层；
  v6 面 manifest 自 `runs/S11_sft_smoke/face_v6/` 归位。~~ **已完成（2026-10-05）**：
  v6 整体迁至 `data/face_manifests/v6/`；v5 复制至 `data/face_manifests/v5/`
  （原位 `data/dataset_v4/anchor_policy_manifest.json` 保留，不破坏 S09 冻结陈述）。

---

## 2026-10-06 裁定记录

### D3：v6 面过渡落地 + 1F Base v6 重跑（Step 0.3/0.4 执行记录）

- **三跑覆盖**：主评测 = test split 全量 **5,468 锚**（v6 面）；反事实 blank / shuffled 各 **200 锚**（`runs/S09_base_benchmark/counterfactual_subset.txt`，与 S09 v5 电池同构）。串行 fail-fast（`set -euo pipefail`）、三步推理均带 `--resume`、**GPU 独占**；主跑 `wall_seconds = 25,267.6` s（≈7.0 h），两组反事实各约 14 分钟。反事实必须带 `--anchors-file`，否则会误跑成全量 5468×2。
- **G0.1–G0.5 判定：全 PASS**（G0.3/G0.4/G0.5 于 2026-10-06 补判；G0.1/G0.2 于 2026-10-05 已判）。
- **冻结物 sha256**（完整表见 [S09_v6_face_rerun.md §冻结物](../S09_base_benchmark/S09_v6_face_rerun.md)）：

  | 冻结物 | 路径 | sha256 |
  |---|---|---|
  | v6 prompt | `configs/contracts/v6/prompt.txt` | `242b632b279e87d9cf355fe42803f7381fa983adebe82d07b5597cbbe692ec10` |
  | v6 output schema | `configs/contracts/v6/output_schema.json` | `22efe8f8738aadb14e654a7357f02892049085ba42ab9e17fbf86cffe205419a` |
  | v6 benchmark 配置 | `configs/benchmark/base_1f_v6.yaml` | `4c729e66784d30fc0422cb24996eb7da8a74b5fd5342c477f0430357035fb012` |
  | v6 anchor manifest | `data/face_manifests/v6/anchor_policy_manifest.json` | `2abe8c4feb43067e294f6af743efcc01e72729aacb94f92764d0f349c239acd1` |
  | v6 eval 报告 | `runs/S09_base_benchmark/eval_v6/eval_report.json` | `be2cb6b902e062202dc2f841357d9711dbb1c744091968da0aaf048348255013` |

- **产物位置**：`runs/S09_base_benchmark/eval_v6/`（`predictions.jsonl` + `predictions_cf_blank.jsonl` / `predictions_cf_shuffled.jsonl` + `main.log` / `cf_blank.log` / `cf_shuffled.log` / `evaluate.log` + `run_summary_*.json` + `eval_report.json` / `eval_report.md` + `anchor_scores.jsonl`）。**执行脚本**：`DriveAlign/scripts/s11_step03_v6_base_rerun.sh`。
- **行为结论登记**：v5/v6 两面的 action 决策轴均退化为**常量预测器**（方向相反：v5 恒 DECELERATE + 让行，v6 恒 KEEP_SPEED + 不让行），v6 的 speed F1 提升是**多数类假象**而非能力提升；感知轴（object / risk / motion）两侧无显著变化。据此把"**多数类常量基线 + counterfactual action_flip_rate**"登记为 **S12 的必报项**。完整论证与双面标注规则见 [S09_v6_face_rerun.md](../S09_base_benchmark/S09_v6_face_rerun.md)。

---

### D4：Step 3 冻结（train32 链路打通 + G1 双层判定 PASS）

> **修订（2026-10-06，B7 修复后重跑）**：首次冻结的 smoke32 是在**错误的 LoRA 目标**下产出的
> ——`target_modules` 用裸后缀 list，被 peft 的后缀匹配连带命中视觉塔 MLP（adapter 696 张量，
> 其中 192 个落在 `visual.blocks.*.mlp.*`），即 **ViT 被意外训练**，与"只训语言侧"口径不符，
> 属 [S11_bug_log B7](S11_bug_log.md) 的**静默缺陷**。已修正配置并**重跑 + 重新冻结**；
> 下表为**重跑后**的 sha。bug 期产物改名留证：`smoke32_vit_leak_bug/`、
> `smoke128_vit_leak_bug/`、`reports/vit_leak_bug/`。

- **裁定**：Step 3（1F 32 条 train → save adapter → 新进程 reload）判定 PASS，**冻结**，
  转入 Step 4（overfit128）。冻结含义：本步骤的配置、样本、adapter、参照与报告 sha 如下，
  后续阶段不得就地改写这些 artifact，只能新增。

- **冻结物 sha256（B7 修复后重跑版）**：

  | 冻结物 | 路径 | sha256 |
  |---|---|---|
  | SFT 配置 | `configs/sft/sft_smoke_1f.yaml` | `05231b343307ce6e6b95020d882a714fd9af344e49e69ba06c09acfb32c97085` |
  | train32 样本 | `runs/S11_sft_smoke/sft_samples/train32.jsonl` | `72285c15e3be653a166db8e43959b2515424f2679476a0206f51bf365cb5bb0f` |
  | overfit128 样本 | `runs/S11_sft_smoke/sft_samples/overfit128.jsonl` | `aec424f786cc65a4c20b6870bb65b4c2ae7892f8050392143009133b9b4d6fcc` |
  | 样本 manifest | `runs/S11_sft_smoke/sft_samples/samples_manifest.json` | `5896118e52f597a78a459edf868e9c2238a44666ab05470a09bcfb79528e35a2` |
  | smoke32 adapter | `runs/S11_sft_smoke/checkpoints/smoke32/adapter_model.safetensors` | `3dfe30c1dd8935f13bd3132b9a63a510b4e89260ae4eddedea11549c62e46f70` |
  | smoke32 状态参照 | `runs/S11_sft_smoke/checkpoints/smoke32/train_state_ref.pt` | `63667acebbcff6e18d30a11f45d9baa96fbbbf3886959dbb8bef5759491418d1` |
  | G1 报告 | `runs/S11_sft_smoke/reports/g1_smoke32.json` | `0b870fb29be64a8f4b5c2016afbe6af530f69eb7916d87ac289e28633ab0a4ff` |

  > **溯源提示**：本表 sha 即"现行配置 + 现行 adapter"的**同一次产出**，不存在首次冻结时
  > "配置与产出仅差注释"那种偏差。样本三项的 sha 与首次冻结**逐字相同**（分辨率与 LoRA 目标
  > 都不进样本 JSONL），这是预期的——B7 只改训练面，不改样本面。

- **G1 判定结果（§7-I 双层）：PASS**
  - 硬层（状态保真）：`n_adapter_tensors=504`（**= 36 层 × 7 模块 × 2**）、`missing=[]`、
    `unexpected=[]`、`shape_mismatch={}`、`dtype_mismatch={}`、`worst_abs_diff=0.0 < atol=1e-6`、
    `nonfinite_parameters=[]`、`global_step=16 == expected=16`。
  - 软层（功能冒烟）：`logits_finite=true`（`[1,1008,151936]`）、`parse_ok=true`、
    `four_fields_equal_to_reference=true`（reload 输出与训练端参照**逐 token 相同**，81 token）、
    `sample_token_match=true`。
  - **信息项（不参与判定）**：`vs_gt.four_fields_equal=false`，差异字段 `speed_action`
    （模型 `ACCELERATE` vs GT）。32 样本 × 2 epoch 不足以拟合 GT，属预期；这正是把
    软层基准从"vs GT"改为"vs 训练端输出"的原因——否则"训练不充分"会被误报成 G1 失败。
  - 训练侧资源画像：`train_loss=0.7981`、`train_runtime=90.87 s`、
    峰值显存 allocated **10.10 GiB** / reserved **15.03 GiB**（卡 31.48 GiB）。
    与 bug 期读数（90.35 s / 10.13 / 15.08 GiB）几乎一致——排除 ViT MLP 后耗时与显存
    **无实质变化**，因为 ViT 在 1F@A1 下本就只占端到端算力的一小部分（见 A4 估算 ~16%）。

- **口径确认**：分辨率维持 **A1 的 `max_pixels: 360000`**（原图 `smart_resize` 到
  784×448，visual token 1824→448、序列 2384→1008，实测 3.17× 加速）。**原生分辨率
  因训练过慢已被否，不再作为候选口径**（2026-10-06 用户明确纠正）。

- **遗留修复（随本次冻结一并做掉）**：
  - `tests/unit/test_sft_collator.py` 中 `... or True` 的**恒真空断言**已改为对
    `image_grid_thw` 拼接数值的真实校验（原期望列数也与实现不符）。
  - `sft/train.py: make_trainer` 的 `output_dir` 由 `config["run_name"]`（相对路径）
    改为调用方传入的 `--out`：否则 `save_steps=64` 触发时会在 CWD 下凭空建出
    `./sft_smoke_1f/checkpoint-64`（train32 仅 16 step 未暴露，Step 4 的 96 step 必然触发）。
  - `S11_bug_log.md` B5/B6 的 `reload_and_infer` 引用已同步为
    `fixed_sample_batch` / `forward_fixed_batch`。
  - **B7 修正（本次重跑的触发原因）**：`configs/sft/sft_smoke_1f.yaml` 的
    `lora.target_modules` 由裸后缀 list 改为**路径限定正则字符串**
    （`model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)`）；
    并新增运行时守卫 `sft/train.py: assert_text_only_lora_targets`（开训前断言
    "非空 + 无 `visual.` 命中 + 全在 `.layers.` 之下"）与回归单测
    `tests/unit/test_sft_lora_targets.py`（读冻结配置的真实 pattern，用 peft 的
    `check_target_module_exists` 校验）。修复效果：adapter 张量 **696 → 504**、
    `visual` 命中 **192 → 0**。详见 [S11_bug_log B7](S11_bug_log.md)。

### D5：batch/accum 维持 1/4（显存余量不做 batch，A/B 实测否决）

- **背景**：Step 3 首跑实测峰值 reserved 15.08 GiB / 卡 31.48 GiB，尚有富余，遂试
  `batch=2 + accum=2`（有效 batch 仍 4、optimizer step 仍 16，配方不变）。
- **实测 A/B（同为 train32）**：

  | 设置 | 有效 batch | step | runtime | samples/s | peak alloc | peak reserved |
  |---|---|---|---|---|---|---|
  | bs=1 + accum=4 | 4 | 16 | **90.35 s** | **0.708** | 10.13 GiB | 15.08 GiB |
  | bs=2 + accum=2 | 4 | 16 | 108.08 s | 0.592 | 12.73 GiB | 19.01 GiB |

- **裁定**：**维持 `batch=1 + accum=4`**。
- **论证**：两次样本通行总量相同（各 64 次单样本前向），故这是纯效率差异——bs=2 使
  runtime **+19.6%**、每样本吞吐 **−16%**、显存多占 3.9 GiB。判据不在"慢了一点"，而在
  **方向**：若 bs=1 处于 batch 饥饿区间，加大 batch 应提升吞吐，实际反而下降 ⇒ bs=1 已
  不饥饿，富余显存换不来速度。可信度排序的机制假设（未做 profiling，不采信为结论）：
  ① 短样本被右 pad 到批内最大长度仍走完整计算；② 出现真实 padding 后 attention_mask
  不再全 1，FlashAttention2 失去快捷路径；③ 显存跨过阈值后分配器/选核换区制。
- **追加论证（用户提出，比 A/B 更具决定性）：显存要为 4F/多视角预留**。量化：
  1F@A1 视觉 token 448、序列 1008；**4F@A1 视觉 token 4×448=1792、序列 ≈2300，与原生
  1F（1824 / 2384）基本持平**——即 A1 省下的开销（视觉 patch 4×、序列 2.4×）会被 4F
  原样吃回去。bs=2 现在多占 3.9 GiB，等 4F 落地时几乎必然要退回 1。故维持 batch=1
  不只是"当前更快"，更是**避免一次注定要撤销的改动**。
  推论：提高有效 batch 的诉求应由 **grad-accum** 承担（accum 不占额外显存，batch 才占），
  这也是"小 batch + 大 accum"是本项目正确形状的原因。
- **附带结论**：bs=2 那轮 G1 复验亦 PASS（硬层 696 张量、`worst_abs_diff=0.0`），
  说明经 collator 的 batch>1 集成路径可用；A/B 证据保留在
  `runs/S11_sft_smoke/checkpoints/smoke32_bs2_vit_leak_bug/` 与
  `runs/S11_sft_smoke/reports/vit_leak_bug/g1_smoke32_bs2.json`。
  > **注（B7 修复后补）**：该轮与 bs=1 基线同处 bug 期，adapter 同为 696 张量（含 192 个
  > `visual.*`）。本决策比的是**同一配方下 bs 的差异**，两组同错故相对结论不受影响；
  > 但这两个目录属 bug 期产物，**不可**当作配额基线复用。
- **再试条件**：若未来显存/吞吐格局变化需重议，先跑 torch profiler 定位瓶颈再动
  batch，不要凭"显存还有富余"直接加。

---

### D6：Step 4 冻结（overfit128，G1 双层判定 PASS；B7 修复后重跑）

- **裁定**：Step 4（1F 128 条 overfit，`dropout=0` + `epochs=3`，即 96 step）判定 PASS，**冻结**。
- **背景说明（时间序）**：Step 4 曾在 **B7 暴露之前**跑过一次（`smoke128`，实测 531.55 s、
  96 step、loss 0.5729），但未做 G1 复验、也未登记；其 adapter 同样是 696 张量 / 192 个
  `visual.*`，**与本 bug 同源、同样作废**。该目录已改名留证为 `smoke128_vit_leak_bug/`。
  B7 修复后按同一配置重跑，下表为重跑版。
- **冻结物 sha256（B7 修复后重跑版）**：

  | 冻结物 | 路径 | sha256 |
  |---|---|---|
  | SFT 配置（同 D4） | `configs/sft/sft_smoke_1f.yaml` | `05231b343307ce6e6b95020d882a714fd9af344e49e69ba06c09acfb32c97085` |
  | overfit128 样本（同 D4） | `runs/S11_sft_smoke/sft_samples/overfit128.jsonl` | `aec424f786cc65a4c20b6870bb65b4c2ae7892f8050392143009133b9b4d6fcc` |
  | smoke128 adapter | `runs/S11_sft_smoke/checkpoints/smoke128/adapter_model.safetensors` | `db035370c66c8b9b1d16526cc42f60f362b3b1c04167c099b4434738d16a4242` |
  | smoke128 状态参照 | `runs/S11_sft_smoke/checkpoints/smoke128/train_state_ref.pt` | `372989acd0f1417a423c964c92c476d9c155fa5219356b895a46e7c43e868911` |
  | G1 报告 | `runs/S11_sft_smoke/reports/g1_smoke128.json` | `b0bbd6e6ed41b2e3e7213e5676dee7b525d509213c8b68a949849bf24a3c9842` |

- **G1 判定结果（§7-I 双层）：PASS**
  - 硬层（状态保真）：`n_adapter_tensors=504`、`missing=[]`、`unexpected=[]`、
    `worst_abs_diff=0.0 < atol=1e-6`、`nonfinite_parameters=[]`、
    `global_step=96 == expected=96`（128 样本 × 3 epoch ÷ accum 4）。
  - 软层（功能冒烟）：`logits_finite=true`（`[1,1008,151936]`）、`parse_ok=true`、
    `four_fields_equal_to_reference=true`（125 token 逐 token 相同）、`sample_token_match=true`。
  - **信息项**：`vs_gt.four_fields_equal=**true**`、`field_diffs=[]`——128 条 × 3 epoch
    已足以把固定样本**过拟合到 GT**，与 train32 的 `vs_gt` 不符形成对照。这佐证了软层
    基准选"vs 训练端输出"是对的：若拿 GT 当基准，train32 那轮会被误判失败。
  - 训练侧资源画像：`train_loss=0.5732`、`train_runtime=525.42 s`、
    峰值显存 allocated **10.14 GiB** / reserved **13.90 GiB**。
    （bug 期为 531.55 s / 10.17 / 13.94 GiB，差异在噪声范围。）
- **落盘纪律**：`save_steps=64` 在本步**必然触发**，已验证 `checkpoint-64` / `checkpoint-96`
  落在 `--out` 指定的 adapter 目录内（D4 遗留修复中的 `output_dir` 改动生效），
  CWD 下未再产生游离目录。

---

### D7：Step 5 收尾（Gate 范围收缩 + G3/G4/G6/G7 判定 + G5/G2 记录）

- **背景**：Step 4 完成后，§5 原列的 G2–G7 六项因**逐项复核**发现"真正需要跑"的远少于六项，
  用户据此提出收缩（"我感觉 G2–G7 可以收缩数量"）。本轮逐项判定每项的真实信息量与
  已有凭证，作出收缩裁定并执行。

- **裁定 1（Gate 范围收缩）：G2–G7 六项 → 2 项实跑（G6/G7）+ 4 项凭证/记录。**

  | Gate | 收缩后处置 | 理由 |
  |---|---|---|
  | G2 | **降级为读数记录** | §7-D 阈值（最终 ≤ 初始 ×0.3，或初始 >1.0 时降 ≥75%）在本 run 实测曲线上**会判 FAIL**（仅降 53.4%），而同一 run G1 双层 PASS 且 `vs_gt` 四字段全等 ⇒ 阈值判失败、证据表明学得好，阈值是噪声来源。且 loss 下降的实质信息已被 G1+G6 覆盖；`train32.log` 在 16 step 下仅 1 个日志点，本就不构成曲线。 |
  | G3 | **以"构造期已强制"为凭证，不单独实跑判定** | `sft/dataset.build_training_sample` 在构造期即以 `ValueError` 强制 `request_hash_1f == v6 manifest`，不等即拒绝——判定发生在更早阶段。另附 `disk` 子命令的**独立复核**（160 条 0 失配）作交叉验证。 |
  | G4 | **保留一次纯 CPU 重跑**（唯一有争议项） | "重建能否复现"此前**从未实证**：D4 里"样本 sha 与首次冻结逐字相同"只说明"文件没被动过"（B7 只改训练面、样本从未重打），**不等于**"重打能复现"。此项为 S12 的"序列化不得改变"提供可复现基线，故保留。实测证明该保留有价值：重建与冻结**逐字节一致**。 |
  | G5 | **折入 G6 报告的记录字段** | 本就非 PASS/FAIL（S03-Q5 监控项），与 G6 同一次推理即可顺带统计，无需独立执行。 |
  | G6 | **实跑（不可省）** | parse_rate/截断率/输出长度分布是"生成链路 + 解析器 + 契约贯通"的唯一端到端证据。 |
  | G7 | **实跑（不可省）** | blank-image 反事实是**本阶段唯一能证明"模型确实在用图"的检查**；B7 修复后 ViT 已冻结（只有语言侧 LoRA 在训），该确认反而更必要（S04 教训：图像 token 未正确传入会静默降级）。 |

- **裁定 2（执行器）**：新增 `src/drivealign/cli/sft_closeout.py`，两个子命令——
  `disk`（纯 CPU：G4 确定性重跑 + G3 独立复核 + G2 曲线读数，报告 `reports/g34_disk.json`）、
  `model`（GPU：G6 全量自推理 + G5 触顶统计 + G7 blank-image 反事实，报告 `reports/g67_model.json`）。
  G7 判定原语（离散字段变更）加在 `src/drivealign/sft/gate.py`
  （`changed_semantic_fields` / `discrete_fields_changed` / `DISCRETE_FIELD_NAMES`）。
  两子命令均以退出码 0/1 表达判定。回归单测 `tests/unit/test_sft_gate.py` **14 passed**
  （新增 4 项：bbox 与顺序不参与语义比较、三个离散字段各自可检出、`risk_factors` 漂移**不**算改变、冻结范围常量锁定）。

- **判定结果**：
  - **G4 PASS**：以 `cli/sft_build_samples.py serialize` 的同一条代码路径重打到临时目录，
    `train32` / `overfit128` 与冻结 sha **逐字节一致**（重建 sha == 冻结 sha == manifest 登记 sha）。
  - **G3 PASS**：独立复核 160 条样本的 `request_hash_1f`（并核 `record_hash`），**0 失配**。
  - **G6 PASS**：`parse_rate=1.0000`（128/128）、截断率 `0.0000`（0/128）、未以 `}` 收尾 0、
    输出 token min/中位/max = **25/125/416**（最长距 512 预算尚有余量，与截断率 0 自洽）。
  - **G7 PASS**：overfit128 前 32 条（= train32 集合）blank-image 反事实，离散字段 ≥1 项改变者
    **25/32（78.1%）**（仅 `critical_objects` 21、`speed_action`+`critical_objects` 3、
    仅 `speed_action` 1、无改变 7）。7 条未变属"过拟合到记忆 + 该样本决策本就由文本侧决定"的合理残留。
  - **G5 记录**：`critical_objects` 触顶 **6/128（4.69%）**、`risk_factors` 触顶 0/128
    （分母只含解析成功的样本）。4.69% 值得 S12 持续观察。
  - **G2 记录**：smoke128 9 点 `0.9882 → 0.4606`（降 53.4%，回升 2 次）；smoke32 仅 1 点。

- **交付文档**：新建 [S11_smoke_run.md](S11_smoke_run.md) 作为 S11 运行结果的**唯一汇总入口**
  （原规划 §3 Step 5 的欠交物），含 Gate 一览、配置与冻结物、Step 3/4 资源画像、G2 读数、
  G3/G4、G5/G6/G7、复现命令、交付与交接。§5 Gate 总表已同步收缩后的处置列。

- **双面标注（凭证）**：S09 存档 v5 读数 vs `eval_v6` 的双面标注规则见
  [S09_v6_face_rerun.md](../S09_base_benchmark/S09_v6_face_rerun.md) §双面标注规则；
  Step 0 的 G0.4（双面标注）已于 2026-10-06 判 PASS（D3）。本阶段所有读数均为 v6 面，
  已在 `S11_smoke_run.md` 顶部标注"不得与 v5 面直接互比"。

- **bug 期产物处置：保留，不清理。**
  - 保留对象：`runs/S11_sft_smoke/checkpoints/smoke32_vit_leak_bug/`（284M）、
    `smoke32_bs2_vit_leak_bug/`（284M）、`smoke128_vit_leak_bug/`（1.1G）、
    `runs/S11_sft_smoke/reports/vit_leak_bug/`（16K），合计约 **1.7G**。
  - 理由：B7 是**静默缺陷**（peft 裸后缀匹配连带训练 ViT MLP，全程不报错），
    这四个目录是"曾发生过"的**唯一原始实证**（696 张量 / 192 个 `visual.*` 的 adapter 实体），
    且 D5 的 batch A/B 结论依赖 `smoke32_bs2_vit_leak_bug/` 这一组对照。删掉只省 1.7G，
    却使缺陷台账从"可复核"退化为"仅剩文字描述"。
  - 纪律：这些目录**不可**当作配额/性能基线复用（D5 注、D6 背景说明均已声明）。
  - 若后续磁盘吃紧需清理，应先保留 `adapter_model.safetensors` 的 sha256 与
    `safetensors` 张量名清单（证明 696/192）再删权重本体。

---

## 阶段关闭核对（Step 5 已填写，2026-10-07）

- [x] 参数冻结表（SFT 关键参数与 sha）—— 见 D4/D6（配置 + 样本 + adapter + 参照 + 报告）
- [x] Step 2/3/4 交付物 sha 登记 —— Step 2/3 见 D4，Step 4 见 D6
- [x] G1–G7 判定结果 —— G1（Step 3 见 D4、Step 4 见 D6，两次均 B7 修复后重跑）、G3/G4/G6/G7 **PASS**，
      G2/G5 **记录**（降级依据见 D7；报告 `reports/g34_disk.json` / `reports/g67_model.json`）
- [x] 双面标注（S09 存档 v5 vs `eval_v6`）—— 规则见 `S09_v6_face_rerun.md §双面标注规则`，
      G0.4 已 PASS（D3）；本阶段读数均为 v6 面并在 `S11_smoke_run.md` 顶部标注
- [x] bug 期产物处置 —— **保留**（D7），四个 bug 期目录留证不清理；不得作配额基线
