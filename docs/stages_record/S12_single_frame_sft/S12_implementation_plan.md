# S12 实施规划（正式单帧 SFT：SFT-1F，2026-10-07，v1）

> 状态：**已冻结（v1，2026-10-07）**。A–J 十项拍板见 §6；冻结后**阈值与口径不得就地改写**（要改走决策台账 + 新版本）。本阶段依据 [docs/stages_plan/12_single_frame_sft.md](../../../stages_plan/12_single_frame_sft.md)。
> 前置：S11 阶段关闭（G6/G7 判定完成、关闭核对表清零、`S11_smoke_run.md` 创建）——**已于 2026-10-07 满足**。
> 链路裁定（S11 §8 已登记）：S12 从**同一 Base checkpoint** 出发训练正式 `SFT-1F`；对比基准 = `runs/S09_base_benchmark/eval_v6/`（v6 面，**非** S09 存档 v5 读数）；S10 与 S11-4F 继续后置。
> 执行环境约定：沿用 S08/S09/S11——命令先 `conda activate autovla_codeclean`；GPU 长跑用 tmux 且 `| tee` 前必加 `set -o pipefail`；物理盘（`/root/autodl-tmp/datasets`）写入由用户执行，workspace 内 `runs/` 写入可由 agent sandbox 执行。

## 0. 前置快照

本规划文档先提交一次快照再开工代码：

```
git add docs/stages_record/S12_single_frame_sft
git commit -m "[Docs] S12 implementation plan (formal single-frame SFT)"
```

## 1. 现状基线（执行前已核实）

- **数据（S08 冻结，记录层零改动）**：`data/dataset_v4`，valid = **train 22,941 / val 2,531 / test 5,468**（合计 30,940），quarantine 659 不回填。迭代入口 `data/face_manifests/v6/anchor_policy_manifest.json`（token → split/shard/line/record_hash/request_hash_1f/anchor_image_relpath）。
- **训练目标**：record 内 `training_targets.expected_output` 的**四字段子集**（critical_objects / risk_factors / yield_required / speed_action），`reasoning` 不进训练侧（S11 D1 / v6 面）。
- **模型面**：contract v6（四字段 schema + v6 prompt），`DEFAULT_CONTRACT_VERSION="v6"`；解码冻结 greedy（`do_sample=false`、`temperature=0.0`、`max_new_tokens=512`）。
- **对比基准**：`runs/S09_base_benchmark/eval_v6/`——`predictions.jsonl` + `predictions_cf_{blank,shuffled}.jsonl` + `eval_report.json` + `anchor_scores.jsonl`。Base 关键读数：`parse_rate=0.9965`、`object_f1_micro@0.5=0.338`、`speed_action_f1_macro=0.115`、`yield_required_f1=0.281`、`risk_factors_f1_macro=0.089`、`overconservative_yield_rate=0.834`、`max_items_hit_rate=0.025`、visual dependence blank flip 14.72% / shuffled 17.44%。
- **推理栈**：`cli/base_benchmark.py`（test manifest → 逐 anchor 读 record → `serialize(ONE_FRAME)` → 推理 → predictions JSONL，支持 `--resume` / `--anchors-file` / `--image-transform`）。**当前无 adapter 加载能力**（全仓仅有 `sft/train.py: load_adapter_model` 一处 peft 重建入口）。
- **评测栈**：`cli/evaluate.py` + `evaluation/{matching,metrics,evaluator}.py`，口径冻结于 `configs/evaluation/eval_v1.yaml`（status frozen，sha 已登记）；`evaluation/bootstrap.py` **已含 `bootstrap_paired_delta`**（scene-cluster 配对差 CI），尚缺 CLI 包装。
- **训练栈（S11 交付，Step 3/4 已实跑验证）**：`sft/{targets,dataset,collator,train,gate}.py` + `cli/{sft_train,sft_build_samples,sft_reload_verify,sft_closeout}.py`；LoRA r=16/alpha=32/dropout=0.05，`target_modules` 为**路径限定正则**（B7 修复，adapter 504 张量、`visual.` 零命中），batch=1 + grad-accum=4，AdamW + cosine + warmup_ratio=0.05，seed=0。实测 ~1.37–1.41 s/micro-batch（A1 分辨率）。
- **分辨率口径（A1，2026-10-06 用户纠正后冻结）**：`max_pixels: 360000`，原图 `smart_resize` → 784×448，visual token 1824→448、序列 2384→1008。**训练侧与评测侧必须同一值**，原生分辨率已否决。
- **沿用教训**：①图像 token 未正确传入会静默降级（S04，需反事实验证）；②LoRA 裸后缀 list 会静默连带训练 ViT MLP（S11 B7）；③`save_steps` 触发时的 `output_dir` 必须钉在 `--out`，否则 CWD 下产生游离目录（S11 D4 遗留修复）；④反事实必须带 `--anchors-file`，否则会误跑成全量（S11 D3）；⑤"拿不准就 STOP/让行"是 SFT 最易学到的捷径（S09 §3.1 over-conservative 的动机）。

## 2. 核心设计裁定

### 2.1 与 S11 的差异全表（复用 / 必须新增）

| 维度 | S11 smoke | S12 正式 | 代码影响 |
|---|---|---|---|
| 训练数据 | train32 / overfit128 | **train split 全量 22,941 anchors** | §2.2 |
| 数据管道 | 全样本**预编码**进 HF Dataset | **惰性化**（22,941 × ~1 MB 张量 ≈ 24 GB，不可驻内存） | §2.2 |
| 验证 | 无 | 末点 全 val split 2,531 **单点读数** | §2.3 / §2.4 |
| checkpoint 选择 | 无 | **= 末点**；⚠️ 偏离蓝图"完全由 validation 选择"，已登记且不复议 | §2.3 |
| test 推理 | 无 | 5,468 全量一次（带 adapter） | §2.6 |
| 对比 | 无 | paired-delta vs Base-1F v6 | §2.7 |
| 训练规模 | 2 / 3 epoch | **1 epoch**（2026-10-07 拍板） | §2.5 |
| 评测口径 | 无 | `eval_v1.yaml` 零变更（保可比性） | §2.7 |

**语义不变式（本阶段硬约束）**：`sft/targets.py` 的四字段显式列举与 DUMP、`sft/dataset.py` 的 `build_training_sample` parity 断言、`sft/collator.py` 的输出格式（`input_ids` / `labels` / `attention_mask` / `pixel_values` / `image_grid_thw`）**逐条不变**。S12 只改**装载策略**，不改序列化与训练语义。

### 2.2 数据管道惰性化（本阶段最大工程项）

**问题**：`cli/sft_train.py: _load_dataset` 当前把全部样本一次性 `encode_frozen_sample` 后 `HFDataset.from_list(rows)`。单样本 `pixel_values` 约 448 × 1176 × 2 B ≈ 1.0 MB，22,941 条 ≈ **24 GB** 常驻内存，不可行。

**裁定**：改为 **map-style 惰性 Dataset**——`__getitem__(i)` 内才 `load_frozen_records → encode_frozen_sample`，collator 每 micro-batch 现取现编码。

**理由**：①显存/内存 O(1)；②batch=1 下与 S11 是**同一条编码代码路径**（`encode_frozen_sample`），parity 与确定性不受影响；③不引入 arrow 落盘缓存（备选方案 `Dataset.map(num_proc=N)` 需一次性预跑 + ~24 GB 磁盘，收益不抵复杂度）。

**实现要点**：惰性 Dataset 必须保证 `__getitem__` 返回与 `encode_chat` 同型 dict，且 `remove_unused_columns=False` 下 collator 拿到的是张量（S11 B5 教训：HF Dataset 行索引会把张量退化成 list）。

### 2.3 best checkpoint 选择（2026-10-07 拍板：末点，不再复议）

**口径**：`best := 训练最后一个 checkpoint`（末点，global_step ≈ 5,736）。**不做 val 扫描，且不再复议**——本阶段确定不启用 sweep 重选（§6-J 已关闭）。

> **⚠️ 偏离登记（须原样落入 `S12_decision_log.md` 与 `S12_run.md`）**
>
> 蓝图 Stage 9 Gate 与 [12_single_frame_sft.md](../../../stages_plan/12_single_frame_sft.md) §4 均要求"**best checkpoint 完全由 validation 选择**"。本期取末点 = **不满足该条**，属**有意偏离且不复议**。
>
> - **理由**：1 epoch + 22,941 条（smoke 数据量的 900 倍）下，单遍训练内过拟合概率极低，val 指标大概率单调改善、末点即最优。
> - **配套 val 义务（仍然执行）**：对末点跑**一次全量 val（2,531）**并落档，作为 `validation 轨迹`（单点读数）。即 **validation 仍被观测、可事后审视，只是不承担选择职责**——这也让"到底有没有过拟合"保有证据。
> - **候选 ckpt 仍全量保留**：`save_steps=500` + `save_total_limit: null`。用途变为 ① Step 4 resume 验证取中间 ckpt ② 训练中断保险；**不再用于选择 best**。
> - **后果自担**：test 结果若不理想，"换 checkpoint"这条路在本阶段已关闭（重跑 test 即破坏预注册）。

**已否决的备选（备查；本阶段不再启用）**：

- **事后 sweep（原 B1）**：候选 × val → 按选择指标排序 → 冠军冻结 best。**否决理由**：全量 val 扫 11 候选 ≈ **36 h**（≈ 训练本身的 4 倍）；退到抽样扫描虽便宜，但样本量缩小会放大噪声（SE ∝ 1/√n），产生"相邻候选差距小于噪声 → 选错"的风险，而 **test 只跑一次、选错不可恢复**——确定性收益不抵成本。
- **训练内 callback（否决 B2）**：生成态切换需要 `eval()` + `gradient_checkpointing_disable()` + `config.use_cache=True`（见 `sft/train.py: prepare_model_for_generation`），在 Trainer 训练循环内反复切换与 gradient checkpointing / 状态机耦合，失败面大且会污染训练日志。
- **选择指标（备查，本阶段不生效，非 gate）**：主 = `object_f1_micro@0.5`；tie-break 依次 `speed_action_f1_macro` → `risk_factors_f1_macro` → **更小的 global_step**。

### 2.4 val 推理规模（本期 = 末点全量 val）

只对末点跑**全量 val 2,531**，落单点读数（§2.3）。**无抽样、无扫描。**

**side-effect 提示**：需核对 `evaluation/evaluator.py` 是否对 split 有硬绑定（`eval_v1.yaml` 有 `evaluation_split: test`）。若绑定，只加 `--split` 透传，**指标口径与 config sha 不动**。

### 2.5 训练规模（2026-10-07 拍板：1 epoch）

- train 22,941 anchors ÷ grad-accum 4 → **5,735–5,736 optimizer step / epoch**。
- 按 S11 实测 ~1.4 s/micro-batch 外推 → **1 epoch ≈ 8.7–9.0 h**（+ 反复读图与编码开销，上界按 ~10 h 排期）。
- `warmup_ratio=0.05` → warmup 约 287 step；cosine 在 1 epoch 内走完一遍退火。
- 候选 ckpt 节奏：`save_steps=500`（≈11 个候选 + 最终），`save_total_limit: null`（adapter 体积小）。**用途 = Step 4 resume 验证取中间 ckpt + 训练中断保险**（不再用于选择 best，§2.3）。

### 2.6 test 推理入口（adapter 支持）

给 `cli/base_benchmark.py` 增 `--adapter` 参数，加载路径复用 `sft/train.load_adapter_model`（新进程 PeftModel 重建 + `eval()`），其余推理路径（`serialize(ONE_FRAME)` → `generate_one` → 严格解析 → `--resume` / `--anchors-file` / `--image-transform`）**零改动**。

新增冻结配置 `configs/benchmark/sft_1f_v6.yaml`：与 `base_1f_v6.yaml` 同字段（contract v6 / policy=ONE_FRAME / greedy / seed / dtype），**只多 `adapter_path`**，且 `processor.max_pixels: 360000` **必须与训练侧逐值相同**（train/eval 同口径，否则 A1 的加速与视觉 token 口径在两端不一致）。

### 2.7 对比口径：paired-delta vs Base-1F v6

- 评测器 `cli/evaluate.py` **零改动**、`eval_v1.yaml` **零改动**（保与 S09 读数严格可比）。
- 新增 `cli/sft_eval_delta.py`：读 SFT 与 Base 的两份 `anchor_scores.jsonl` → `bootstrap_paired_delta`（**同一 cluster map、同一 seed、同一 B**）→ 输出逐指标 point / delta / delta CI。沿 S09 §3 与项目记忆的冻结口径。
- **必报项（S11 D3 登记）**：①**多数类常量基线**（SFT 的 `speed_action` 多数类占比，对比 Base 86.7%）；②**counterfactual `action_flip_rate`**（blank / shuffled）。这两项是"是否只是换了个常量预测器"的直接证据。

### 2.8 退化守卫预注册（**必须在看到 test 之前冻结**；2026-10-07 拍板）

> **本节只承诺一件事**：`SFT-1F` 没有靠"用怂换分"作弊。
> **达标判据**（`parse_rate` + P/Q）**不在这里**——它就是 §4 的 G4 那一行，不在本表重复抄写。

**为什么必须预注册（不是形式主义）**：Base 在动作轴上是个常量预测器（86.7% DECELERATE），且 `overconservative_yield_rate` 高到 0.834。SFT 完全可以靠"一律多让行 / 多报风险"把 `yield_required_f1`、`speed_action_f1_macro` 抬上去，而真实驾驶能力没有提升。若不在看到 test 结果**之前**冻结"什么算作弊"，等数字出来再定阈值，判据即失效。此为蓝图 Stage 9 Gate（"over-conservative rate 未越过预注册阈值"）与 S09 §3.1（"退化阈值在 S12 预注册"）的欠账，不可删。

**判据（2026-10-07 起用 CI 口径，不再用拍脑袋的绝对容忍值）**：

| 守卫 | 判据（触发即退化） | 方向 |
|---|---|---|
| `overconservative_yield_rate` | paired-delta 95% CI **上界 < 0** | 越大越坏 |
| `overconservative_stop_rate` | paired-delta 95% CI **上界 < 0** | 越大越坏 |
| `visual_dependence_action_flip_blank` | paired-delta 95% CI **下界 < 0** | 越小越坏（不看图了） |
| `visual_dependence_action_flip_shuffled` | paired-delta 95% CI **下界 < 0** | 越小越坏 |
| `visual_dependence_output_change_*` | paired-delta 95% CI **下界 < 0** | 越小越坏 |
| **常量预测器守卫** | SFT `speed_action` 多数类占比 ≤ 0.867 **且** `action_flip_rate` ≥ Base | 唯一保留绝对值的行 |

> **为什么常量预测器守卫不能用 CI**：它要判的是"是否只是把 Base 的常量换成了另一个常量"，这个语义 CI 表达不了（paired-delta 只能说"两系统有差异"），必须直接比多数类占比这个绝对值。它是 S11 D3 登记的必报项，也是本阶段最可能踩的坑。

> **口径说明（已知不对称，如实记录）**：test 主跑 n=5,468 / 150 scenes → CI 窄，**小的真实退化也会被判退化**（偏严，可接受）；反事实子集 n=200 → CI 宽，**只有大的退化会被抓到**（检验力低，属诚实反映）。若改用"point estimate 劣于 Base"会更宽松，但会漏掉与噪声不可分的退化；**建议保留 CI 口径**。

**选择指标**（§2.3 用于挑 best，非 gate）= 主指标 P，tie-break 依次 Q → `risk_factors_f1_macro` → 更小 global_step。

### 2.9 save/resume 验证口径（★ 待拍板）

蓝图要求"确认 optimizer、scheduler、global step 和随机状态恢复"。GPU 续训逐位复现不现实，**预注册为状态序列化层验证**（沿 G1 分层思路）：

- **硬层（状态保真）**：从中间 ckpt 新进程恢复，断言 optimizer state 的 key 集合与 shape 逐一对齐、`scheduler.last_epoch` 相等、`global_step` 相等、各 RNG state（torch/cuda/numpy/random）可加载。
- **功能冒烟层**：恢复后在固定样本上生成**合法可解析输出**（不比字节，容忍 kernel 非确定性）。

## 3. 模块拆分与执行流

### Step 0：规划快照

见 §0。快照后开工代码。

### Step 1：代码（CPU）

| 文件 | 变更 | 职责 |
|---|---|---|
| `src/drivealign/sft/dataset.py` | **扩展** | 新增惰性 map-style Dataset（`__getitem__` 内 load + `encode_frozen_sample`）；`build_training_sample` / `encode_frozen_sample` / parity 断言**不动**（§2.1 不变式） |
| `src/drivealign/cli/sft_train.py` | **改造** | `_load_dataset` 由 eager `HFDataset.from_list` 改为惰性 Dataset；新增 `--epochs` / `--save-steps` / `--resume-from-checkpoint` 覆盖；训练后仍落 `train_state_ref.pt` + 峰值显存 |
| `src/drivealign/sft/train.py` | **小改** | `make_trainer` 的 `save_steps`（现硬编码 64）与 `save_total_limit`（现 2）提升为参数——1 epoch 全量下 64 会产生 ~90 个候选且 limit=2 会删掉早期 ckpt；透传 `resume_from_checkpoint` |
| `src/drivealign/cli/sft_build_samples.py` | **扩展** | 新增 `train_full` 全量序列化（A 层 token 列表 + B 层 JSONL，沿 S11 D2 的 A/B 分层） |
| `src/drivealign/cli/base_benchmark.py` | **扩展** | 新增 `--adapter`，复用 `sft.train.load_adapter_model`；其余推理路径零改动 |
| ~~`src/drivealign/cli/sft_sweep.py`~~ | **本期不新增** | 属已否决方案（§2.3）：候选 ckpt × val → 逐候选指标 → 按选择指标排序 → 冠军 |
| `src/drivealign/cli/sft_eval_delta.py` | **新增** | 两份 `anchor_scores.jsonl` → `bootstrap_paired_delta` → paired-delta 报告 + 常量预测器读数 |
| `src/drivealign/cli/sft_resume_verify.py` | **新增** | §2.9 的 save/resume 状态验证（新进程，退出码 0/1） |
| `configs/sft/sft_1f.yaml` | **新增（冻结）** | 正式配置骨架：沿 `sft_smoke_1f.yaml` 全字段（v6 / ONE_FRAME / DUMP / LoRA 路径限定正则 / max_pixels 360000 / seed 0），训练段改 1 epoch、`save_steps=500`、`save_total_limit: null`、**`learning_rate: 5.0e-5`**（2026-10-07 拍板，自 1e-4 下调） |
| `configs/benchmark/sft_1f_v6.yaml` | **新增（冻结）** | base_1f_v6 全字段 + `adapter_path`，`max_pixels` 同值 |

**单测（`tests/unit/`）**：
- `test_sft_lazy_dataset`：惰性 Dataset 的 `__getitem__` 输出与 S11 `encode_frozen_sample` 逐张量一致（防装载策略漂移改变训练语义）；随机访问顺序不变性；长序列 pad 语义。
- `test_sft_eval_delta`：合成 `anchor_scores` 上的 paired-delta 数值与确定性（固定种子逐值断言）；相同输入两次逐字节一致。
- `test_sft_config_parity`：`sft_1f.yaml` 与 `sft_smoke_1f.yaml` 的 **face / DUMP / LoRA pattern / max_pixels / 生成参数**逐值相同（只允许训练规模类字段不同），防止"正式配置悄悄换了 face"。
- ~~`test_sft_sweep_select`~~（**本期不写**）：属已否决方案（§2.3）。

### Step 2：样本冻结（CPU，确定性）

- **train_full**：A 层 = train split 全量 token 列表（字典序）；B 层 = 按 v6 face 重打 `train_full.jsonl`（逐样本 `request_hash_1f` parity 由 `build_training_sample` 断言，不等即抛 `ValueError`）。落 `runs/S12_sft_1f/sft_samples/`。
- **反事实子集**：**沿用** `runs/S09_base_benchmark/counterfactual_subset.txt`（200 锚，与 Base 同构，保 paired 可比），不重新抽样。
- 逐样本 sha + manifest sha + git commit 登记。

### Step 3：全量训练（GPU，用户 tmux）

- 1 epoch（§2.5），`save_steps=500` 落候选，`| tee` 前 `set -o pipefail`。
- 训练结束：取峰值显存 → 生成 `train_state_ref.pt`（顺序不能反，`generate_one` 会 reset 峰值线，S11 已踩）。
- 落档：loss 曲线 / grad norm / step time / 峰值显存 / 吞吐 / GPU-hours。
- **B7 守卫自动生效**：`assert_text_only_lora_targets` 在开训前断言 adapter 504 张量、`visual.` 零命中。

### Step 4：save/resume 验证（GPU，新进程）

取中间 ckpt，跑 `cli/sft_resume_verify.py`，按 §2.9 判定。

### Step 5：val 读数 → 冻结 best（GPU）

`best := 末点 checkpoint`（§2.3 拍板，不再复议）。对末点跑**一次全量 val（2,531）**并落档（`validation 轨迹` = 单点读数）→ 冻结 best（`best/adapter_model.safetensors` + sha 登记）。**不写 `cli/sft_sweep.py`。**

**此步之前不存在任何 test 预测。**

### Step 6：test 一次性推理（GPU，best 冻结后）

- 主预测：全量 test 5,468，`cli/base_benchmark.py --adapter ...`，`--resume` 幂等。
- 反事实：**必须带** `--anchors-file runs/S09_base_benchmark/counterfactual_subset.txt`，blank / shuffled 各一遍，产物**物理分离**（S11 D3 教训）。
- 与主跑串行、有 `set -euo pipefail`、GPU 独占。

### Step 7：评测与对比（CPU，agent sandbox）

1. `cli/evaluate.py`（`eval_v1.yaml` 不变）→ `eval_report.{json,md}` + `anchor_scores.jsonl`。
2. `cli/sft_eval_delta.py` → paired-delta vs Base-1F v6 + 常量预测器 / action_flip 必报项。
3. 成本摘要（GPU-hours、latency、吞吐、峰值显存）。
4. 按 **G4 判据**（§4）与 **§2.8 退化守卫**逐项判定 G4 / G5。

### Step 8：文档回填与阶段关闭

- `S12_decision_log.md`（时间序裁定 + 参数冻结表 + 关闭核对）。
- `S12_run.md`（训练/评测/成本运行记录）。
- 配置 sha 登记后状态置 frozen；蓝图 §5.2 最小 run 目录（`resolved_config.yaml` / `metadata.json` / `metrics.json` / `train.log` / `checkpoints/`）补齐。
- 阶段关闭核对表清零 → commit。

## 4. Gate 总表

| # | Gate | 判据 | 判定 |
|---|---|---|---|
| G1 | 链路完好 | 全量训练无 NaN/Inf；adapter 张量数 = 504 且 `visual.` 零命中（B7 守卫）；策略 loss 曲线落档 | PASS/FAIL |
| G2 | 可恢复 | 新进程从中间 ckpt 恢复：optimizer state / `scheduler.last_epoch` / `global_step` / RNG state 对齐（§2.9 硬层）+ 生成合法输出（软层） | PASS/FAIL |
| G3 | 数据不变式 | train_full 逐样本 `request_hash_1f == v6 manifest`；样本集合 ⊆ train split（零 val/test 混入） | PASS/FAIL |
| G4 | 内容达标 | `parse_rate` ≥ max(0.99, Base 0.9965) **且** P/Q **至少一个**的 paired-delta 95% CI 下界 > 0 | PASS/FAIL |
| G5 | 无退化 | §2.8 退化守卫全部未触发（overconservative ×2 / visual dependence ×3 / 常量预测器守卫） | PASS/FAIL |
| G6 | 可复算 | test predictions 与 manifest 1:1（missing = 0 / extra = 0，哈希逐条一致）；`cli/evaluate.py` 重跑报告逐字节一致 | PASS/FAIL |
| G7 | 溯源与成本 | config / 样本 / ckpt 各 sha、eval_version、commit、GPU-hours、峰值显存全落档 | PASS/FAIL |

> **已知偏离（登记在案，非 FAIL）**：蓝图 Stage 9 Gate 要求"best checkpoint 完全由 validation 选择"。本期 `best := 末点`（§2.3），**不满足该条且不复议**（§6-J 已关闭）。阶段关闭时按"**接受偏离**"记录，但必须在 `S12_decision_log.md` 与 `S12_run.md` 中显式声明，并在 S13 交接说明中复述。

> **失败处理**：任一 Gate FAIL 时先报告原因，**不进入 S13**（蓝图 Stage 9 口径）；阈值调整走决策台账，不得就地改写预注册值。

## 5. 冻结物与登记清单

| 冻结物 | 位置 | 登记方式 |
|---|---|---|
| `configs/sft/sft_1f.yaml` | `configs/sft/` | sha256 → 决策台账 |
| `configs/benchmark/sft_1f_v6.yaml` | `configs/benchmark/` | sha256 → 决策台账 |
| train_full 样本 + manifest | `runs/S12_sft_1f/sft_samples/` | 逐样本 sha + manifest sha |
| 候选 ckpt / best adapter | `runs/S12_sft_1f/checkpoints/`（不进 git） | 大小 + sha |
| test predictions / 反事实 / eval 报告 / paired-delta | `runs/S12_sft_1f/` | sha + 运行记录回填 |
| 资源与成本报告 | `runs/S12_sft_1f/reports/` | 回填 `S12_run.md` |

## 6. 拍板记录（A–J 全部已定；2026-10-07）

- **A.（已拍板 2026-10-07 → 被 E 覆盖）** best 选择路线原定 = 事后 sweep（B1）。E 拍板后**本阶段否决 sweep**，作为备选记录在 §2.3。
- **B.（已拍板 2026-10-07）** 训练规模 = **1 epoch**（≈8.7–9.0 h）。
- **C.（已拍板 2026-10-07）** 数据管道 = **map-style 惰性 Dataset**（`__getitem__` 内 load + encode）；否决 arrow 落盘缓存。（§2.2）
- **D.（已拍板 2026-10-07）** 学习率 = **`5.0e-5`**（自 S11 smoke 的 `1e-4` 下调）。理由：`1e-4` 是为 32/128 条过拟合设定；全量 22,941 × 5,735 step 下过大易行为坍缩。（§2.5）
- **E.（已拍板 2026-10-07，不再复议）** `best := 末点 checkpoint`，只对末点跑**一次全量 val 2,531** 落单点读数。⚠️ **这是对蓝图"best checkpoint 完全由 validation 选择"的有意偏离，已登记、不复议**；候选 adapter 仍全量保留（供 Step 4 resume 验证）（§2.3 / §2.4）。
- **F.（已拍板 2026-10-07）** §2.8 退化守卫改用 **paired-delta CI 口径**（删掉 `+0.02` / `−0.05` 等绝对容忍值）；**常量预测器守卫保留绝对值**。（§2.8）
- **G.（已拍板 2026-10-07）** save/resume 判定 = **状态序列化层硬层 + 生成合法软层**，不做逐位续训复现。（§2.9）
- **H.（已拍板 2026-10-07）** 候选 ckpt 节奏 = `save_steps=500`（≈11 候选）/ `save_total_limit: null`。（§2.5）
- **I.（已拍板 2026-10-07）** G4 达标定义 = **`parse_rate` 达标 + P/Q 至少一个 paired-delta CI 下界 > 0**（不要求 P 与 Q 同时改善）。（§4 G4）

> **J.（已关闭 2026-10-07）** 不启用 sweep 重选 best——本阶段确定以末点作为 test checkpoint。**无遗留事项**：若未来阶段要改用 sweep，属新阶段的新决策，不由 S12 承接。（§2.3）

## 7. 交接（S13）

- 冻结 `SFT-1F` checkpoint、predictions、配置与成本；S13 从**同一 Base checkpoint、相同 anchors、匹配训练预算**起多帧 SFT。
- 4F 面出生即 v6（四字段，无 reasoning 历史包袱）；`sft` 包仅扩展 4F 政策分支（serializer 已就绪），序列化语义零变更。
- **S11 后置项**：4F smoke 需在 S13 开工前补跑。
- 本阶段若 G4/G5 FAIL，按蓝图先报告原因、不进入 S13；DPO/GRPO（S15/S16）继续后置。
