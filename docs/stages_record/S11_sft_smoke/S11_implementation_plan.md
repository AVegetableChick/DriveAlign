# S11 实施规划（SFT 计算 Smoke：1F 半边，2026-10-05，v2 修订）

> 状态：**规划已确认（2026-10-05 v2 修订），Step 0 进行中**。本版相对 v1 的核心变化：模型面升版 v6（删除 reasoning 字段），target 构造大幅简化，执行顺序按用户修订（0.1/0.2 → SFT 探索 → 0.3 夜间 → 0.4）。
> 裁定链：reasoning 字段监督方式经多轮讨论后裁定从模型面整体删除，完整论证与被否方案台账见 [S11_reasoning_supervision_decision.md](S11_reasoning_supervision_decision.md)；Step 0 过渡执行计划见 [S11_step0_v6_face_transition.md](S11_step0_v6_face_transition.md)。
> 链路裁定：S09 已关闭 → Step 0（v6 面 + Base 重跑）→ **S11 仅做 1F 半边**（32 条 train/save/reload + 128 条 overfit）→ S12 全量 SFT；S10 与 S11-4F 后置（4F smoke 留待 S13 前补跑）。
> 执行环境约定：沿用 S08/S09——命令先 `conda activate autovla_codeclean`；GPU 长跑用 tmux 且 `| tee` 前必加 `set -o pipefail`；物理盘（/root/autodl-tmp/datasets）写入由用户执行，workspace 内 `runs/` 写入可由 agent sandbox 执行。

## 0. 前置快照

三份规划文档（本文件、Step 0 计划、讨论总结）先提交一次快照再开工代码：

```
git add docs/experiment_record/S11_sft_smoke
git commit -m "[Docs] S11 plan v2 (v6 face transition + simplified SFT targets)"
```

## 1. 现状基线（v6 生效后口径）

- **数据与 GT 已冻结（S08）**：`data/dataset_v4`（contract v4 记录）零改动。SFT 训练目标 = record 内 `training_targets.expected_output` 的**四字段子集**（critical_objects、risk_factors、yield_required、speed_action）；`reasoning` 字段留作 GT provenance，训练侧不再消费。迭代入口 `data/dataset_v4/anchor_policy_manifest.json`（token → split/shard/line/record_hash/request_hash/anchor_image_relpath）。
- **模型面（Step 0 交付后）**：contract v6——4 字段 schema + v6 prompt（删 reasoning 指令），`DEFAULT_CONTRACT_VERSION="v6"`；Base 链 `serialize(ONE_FRAME)` → `generate_one`（`apply_chat_template(user=[image, prompt], add_generation_prompt=True)`，无 system）→ 严格解析；解码冻结 greedy、max_new_tokens=512。v6-face manifest 位于 `data/face_manifests/v6/`（T2 层，可复用派生视图，不属于任何实验）。
- **Base 基准（Step 0.3 交付后）**：`runs/S09_base_benchmark/eval_v6/` 为 S12 对比基准；S09 存档读数（v5 面）继续有效但**不得**与 SFT-v6 读数直接对比（双面标注规则）。
- **eval 栈**：eval_v1 指标定义不变（面引用升级至 v6，sha 重登记）；S11 自推理诊断复用 `parse_structured_output` 与错误 taxonomy。
- **Base checkpoint**：`models/Qwen2.5-VL-3B-Instruct`。
- **训练栈**：从零建 `src/drivealign/sft/` 包（§3 Step 1）。
- **沿用教训**：①图像 token 未正确传入会静默降级，需反事实验证（S04）；②输出 token 预算截断失败类（S09）——v6 输出更短，风险同步下降；③训练面 == 评测面 parity 纪律（S08）。

## 2. 核心设计裁定（v2）

### 2.1 reasoning 字段：模型面删除（v6）

不再存在 mask/stub/重排等训练侧方案。裁定链与被否方案台账见讨论总结文档。对 S11 的直接后果：**训练目标 = 四字段全监督**；原 v1 规划中的三段拼装、mask span、边界单测、预案 α/β 全部作废删除。

### 2.2 target 构造（简化版）

```python
# src/drivealign/sft/targets.py 核心逻辑（设计稿）
DUMP = dict(ensure_ascii=False, separators=(", ", ": "))   # 冻结进 sft config 并登记 sha

d = record.training_targets.expected_output
target = {k: d[k] for k in ("critical_objects", "risk_factors",
                            "yield_required", "speed_action")}   # 四字段，显式列举

full = json.dumps(target, **DUMP)          # 参照串：整串 dump

# 确定性钉子（v1 字节断言的收窄版，用途 = 防序列化漂移，不再承担 mask 职责）
assert json.dumps(target, **DUMP) == full           # 构造恒等，防御性保留
assert json.loads(full) == target                   # 语义 round-trip

ids_target = tok(full, add_special_tokens=False)
input_ids  = ids_prefix + ids_target + [IM_END_ID]
labels     = [-100] * len(ids_prefix) + ids_target + [IM_END_ID]   # 四字段全监督，无 mask
```

要点：①四字段**显式列举**而非字典推导过滤，防止未来 record 增字段时静默混入；②显式列举顺序即发射顺序（critical_objects → risk_factors → yield_required → speed_action），与 v6 prompt 字段清单一致；③`reasoning` 不进 target，模板文本零消费；④record 与契约零改动（训练侧消费逻辑，不触发 record hash 代际）。

### 2.3 模型面 parity（训练面 == 评测面）

`ids_prefix` 由与 [inference/base_runner.py](/root/autodl-tmp/drivealign_workspace/DriveAlign/src/drivealign/inference/base_runner.py) 完全相同的路径产生：同一 `apply_chat_template`、同一 `process_vision_info`、prompt 来自 `serialize(model_inputs, InputPolicy.ONE_FRAME)`（v6 面）。逐样本断言：SFT 样本 user 面文本与重算 `request_hash_1f` == `data/face_manifests/v6/anchor_policy_manifest.json` 值（S08 parity 纪律）。train/eval 唯一差异 = assistant 目标段存在。

### 2.4 训练技术路线（起点，§7 待拍板确认）

- **PEFT**：LoRA（blueprint 回退链第 4 级；3B + 消费级单卡下更低级回退作为后备记录）。
- **Trainer**：HF `Trainer` + 自定义 collator（逐样本 processor → `input_ids/pixel_values/image_grid_thw`；文本侧右 pad，`attention_mask=0`/`labels=-100`；`pixel_values` 沿 patch 维拼接；`remove_unused_columns=False`）。是否复用 pinned AutoVLA SFT checkout 见 §7-A。
- **训练纪律**：训练/优化器状态全程无 val/test split 参与；32 条与 128 条全部取自 train split。

### 2.5 风险水位（v6 后重估）

v1 预判的两大风险（JSON 安全、长度截断）在四字段全监督 + 输出缩短 ~40–60 tok 后坍缩为低水位，无预注册升级路径。残余风险与对策：①v6 面本身的行为漂移 → 由 Step 0.3 Base 重跑暴露，属 Step 0 范畴不属训练侧；②LoRA 超参失配 → Step 3/4 直接观测（loss 曲线 + overfit 自推理），超参调整走决策台账，不属于"方案升级"。

## 3. 模块拆分与执行流

### Step 0：v6 面过渡 + Base 重跑（独立文档，先于/并行于本阶段代码）

见 [S11_step0_v6_face_transition.md](S11_step0_v6_face_transition.md)。调度要点：**0.1/0.2 完成即解锁本阶段 Step 1**；0.3 独占 GPU（用户夜间执行），期间 Step 1 代码与 CPU 单测、Step 2 样本构建（CPU）可并行；Step 3 训练必须等 GPU 释放；Step 0.4 完成后本阶段方可关闭。

### Step 1：`src/drivealign/sft/` 包 + 单测（CPU，可与 Step 0.3 并行）

| 文件 | 职责 |
|---|---|
| `sft/targets.py` | 四字段显式列举 + 整串 dump + 断言（§2.2）；DUMP 配置注入 |
| `sft/dataset.py` | record → 训练样本：读 shard record → `serialize(ONE_FRAME)` v6 面 → chat 前缀（parity 断言）→ token ids/labels → 逐样本 sha256 |
| `sft/collator.py` | §2.4 collator；`pixel_values`/`image_grid_thw` 拼接 |
| `cli/sft_build_samples.py` | 样本冻结 artifact CLI：train manifest 子集 → `runs/S11_sft_smoke/sft_samples/{train32,overfit128}.jsonl` + `samples_manifest.json`（sha256、选取规则参数、git commit） |
| `sft/train.py` | LoRA 训练入口：读冻结样本 artifact → 训练 → save adapter → 新进程 reload → 固定输入输出复验 |
| `configs/sft/sft_smoke_1f.yaml` | 冻结运行配置：checkpoint 路径、contract v6、policy=ONE_FRAME、DUMP、LoRA 参数、batch/grad-accum/grad-checkpoint、seed、解码（沿 S09 冻结口径）、选取规则参数 |

单测（`tests/unit/`）：

- `test_sft_targets`：整串 dump == 断言；合成异常输入 fail-fast 语义；四字段显式列举防漏防混。
- `test_sft_dataset`：①`decode(input_ids)` 逐字节含四字段目标 JSON；②`decode(监督位)` == 目标 JSON + im_end（全监督，无 mask 位）；③面 parity（user 面文本 + request_hash_1f vs v6 manifest）；④确定性（两次构建 token ids 与逐样本 sha 逐字节一致）。
- `test_sft_collator`：批内 pad/labels 对齐、pixel_values 拼接形状、`attention_mask` 语义。

### Step 2：样本冻结 artifact（32 + 128，CPU）

采用 **A/B 分层**（2026-10-05 拍板，见决策记录）：

- **A）token 选取**（**可复用**，跨 contract 复用，只选一次）：`data/` 下可复用目录
  （默认 `sft_subsets/`）落 `train32_tokens.json` / `overfit128_tokens.json` +
  `sft_subsets_manifest.json`（选取参数 + sha + git commit）。它与模型 face 无关，
  换 contract 时**不重新选取**。
- **B）序列化样本**（**面绑定**，换 face 必须重打）：`runs/S11_sft_smoke/sft_samples/`
  落 `train32.jsonl` / `overfit128.jsonl` + `samples_manifest.json`。按当前 face
  的 anchor manifest 重打 prompt/target/request_hash。

- 落位原则：A 属可复用派生资产（非实验专属），B 属绑定 face + git commit 的实验
  artifact（实验专属）。**均不进 `data/dataset_v4/`**（那是 S08 冻结基座层，保持
  不可变不变式）。
- CLI：`cli/sft_build_samples.py` 拆成 `select`（A）与 `serialize`（B）两子命令。
- 逐样本面 parity 由 `dataset.build_training_sample` 在序列化时断言（GT 已冻结，
  只验读取与 hash 一致）。

### Step 3：1F 32 条 train/save/reload（GPU，Step 0.3 释放后）

- LoRA 训练 → adapter 保存 → **新进程** reload → 固定输入（训练 batch 原样本）输出复验（参数、global step、无 NaN/Inf）。
- 资源画像落档：loss/grad norm/step time/视觉 token 数/峰值显存/吞吐。

### Step 4：128 overfit + 自推理诊断（GPU）

- 128 条过拟合训练，记录 loss 曲线。
- 自推理诊断（解码沿 S09 冻结口径）：parse_rate（预期 ~1.0，全监督四字段）、输出 token 长度分布与截断率、四字段输出人工抽检表（S12 持续监控雏形）。
- `maxItems` 触顶率统计（S03-Q5 监控项，仅记录不决策）。
- 视觉依赖抽查：overfit 模型少量 blank-image 反事实（复用 `make_blank_image`），确认 SFT 后模型确实在用视觉。

### Step 5：文档回填与阶段关闭（前置 = Step 0.4 完成）

- 结果回填本目录 `S11_smoke_run.md` + `S11_decision_log.md`（时间序裁定、参数冻结表、阶段关闭核对，格式沿 S09）。
- 配置登记 sha 后状态置 frozen。

## 4. 样本选取规则（待拍板后冻结）

- **train32**：train split anchors 按 (location × time_of_day) 分层，每层字典序取前 k，凑满 32；先出 speed_action/yield 分布表再冻结，STOP/yield_true 样本 ≥ 各 4 条。
- **overfit128**：train32 的**超集**（同一分层规则扩至 128），符合蓝图"相同 anchors"口径。
- 确定性：选取只依赖 manifest 字典序 + 分层键 + 固定参数，无随机数；重跑逐样本 sha 一致。

## 5. 验证 Gate 总表

| # | Gate | 来源 | 判定 |
|---|---|---|---|
| G0.x | Step 0 全部 gate（面单测/manifest parity/重跑完整/双面标注/冻结登记） | Step 0 文档 §5 | PASS/FAIL |
| G1 | 1F forward/backward/save/reload 无 NaN/Inf；新进程 reload 后：权重逐张量可比（max\|Δ\|<1e-6）、global step 相等（**状态保真层，硬**）；固定输入 forward 输出可解析且四字段语义相等，容忍 token 层非确定性（**功能冒烟层，软**） | 蓝图 §4 + §7-I | PASS/FAIL |
| G2 | 128 overfit loss 明显下降（量化口径 = §7-D，Step 4 前冻结） | 蓝图 §4 | PASS/FAIL |
| G3 | 面 parity：逐样本 user 面文本与 request_hash_1f == **v6-face** manifest 值 | §2.3 | PASS/FAIL |
| G4 | target 构建断言全量通过 + 确定性重跑逐字节一致（config/seed/manifest/git commit 可追溯） | §2.2/蓝图 §4 | PASS/FAIL |
| G5 | `maxItems` 触顶率落档（S03-Q5，仅记录） | 蓝图 §6 | 记录 |
| G6 | 自推理诊断：parse_rate ≥ 0.99 且截断率 ≤ 1%（全监督下应为低水位 sanity 检查）；四字段抽检表落档 | §3 Step 4 | PASS/FAIL |
| G7 | 视觉依赖抽查：blank-image 反事实下 overfit 模型**离散四字段输出至少 1 项改变**（非全同输出） | §3 Step 4 + §7-J | PASS/FAIL |

## 6. 冻结物与登记清单

| 冻结物 | 位置 | 登记方式 |
|---|---|---|
| v6 面/manifest/eval 配置 | Step 0 交付 | Step 0 文档 §4 |
| `sft_smoke_1f.yaml`（DUMP/LoRA/选取规则参数） | `configs/sft/` | sha256 → 本目录决策台账 |
| 样本 artifact（train32/overfit128 + manifest） | `runs/S11_sft_smoke/sft_samples/` | 逐样本 sha256 + manifest sha |
| Smoke adapter（32 条）与 overfit checkpoint（128 条） | `runs/S11_sft_smoke/checkpoints/`（不进 git） | 大小 + sha → 运行文档 |
| 资源/吞吐报告 | `runs/S11_sft_smoke/reports/` | 回填 `S11_smoke_run.md` |

## 7. 待拍板清单（清零后开工 Step 3；Step 1 代码不受阻塞）

- **A. 训练栈（已拍板 2026-10-05）**：**HF `Trainer` + pinned Qwen2.5-VL 自建轻量链路**。理由：与现有推理栈同源（同 `apply_chat_template`/`process_vision_info`），满足 §2.3 训练面==评测面 parity；AutoVLA SFT 面向轨迹格式，适配成本高且不复用价值（§3 Step 1 代码不受此拍板阻塞）。
- **B. §4 选取规则（已拍板 2026-10-05）**：接受 **分层 × 字典序**；配额 **STOP ≥ 4 且 yield_true ≥ 4（两维互不重叠取并集下限各 4）**。分层键分布表先出再冻结进 config。
- **C. LoRA 起点参数（已拍板 2026-10-05）**：**r=16, alpha=32, dropout=0.05, target=q/k/v/o+gate/up/down, batch=1+grad-accum=4, lr=1e-4**；补 **optimizer=`AdamW`（bf16）、scheduler=`cosine`+`warmup_ratio=0.05`、`weight_decay=0.01`、`max_grad_norm=1.0`**。起点值，overfit 阶段允许独立配置微调，变更留台账。
- **D. G2 量化口径（已拍板 2026-10-05）**：主判 = 最终 loss ≤ 初始 loss × 0.3 且曲线单调（允许 ≤2 次回升 step）；**补 override：若初始 loss 异常高（>1.0），改判"下降 ≥75% 且单调"**，避免 0.3 相对量在低起点下过于苛刻/在爆炸起点下失真。
- **E. G6 阈值确认（已拍板 2026-10-05）**：**parse_rate ≥ 0.99、截断率 ≤ 1%**（全监督四字段下的 sanity 阈值，超限即查实现而非改方案）。
- **F（新增，已拍板 2026-10-05）. epochs / 轮数**：**train32 = 2 epoch（batch=1 → 64 step）；overfit128 = 3 epoch（→ 384 step）**，每 64 step 存一次 eval，取 loss 最低 step 复验。train32 仅验证链路通；overfit128 需足够步数跑出明显下降以支撑 G2。
- **G（新）. overfit dropout（已拍板 2026-10-05）**：overfit 阶段 **dropout = 0**（config 以 `overfit: {lr:…, dropout:0}` 覆盖起步值），消除 dropout 噪声对 G2 曲线与自推理的诊断干扰；确认过拟合阶段关 augmentation。
- **H（新）. seed 定值**：config `seed` 写死一个固定值（建议 `0`），避免逐环境（torch/cuda/numpy/random）随机性影响可复现性。
- **I（新）. G1 判定（已拍板 2026-10-05，分层，替代原"固定输入输出符合预期"）**：**状态保真层（硬，确定性）** = reload 权重逐张量 shape+dt 同构可比（容差 ∃ max\|Δ\| <1e-6）、global step 相等、无 NaN/Inf；**功能冒烟层（软，容忍 forward 随机性）** = 固定输入跑 forward 输出**可解析**且**四字段语义相等**（speed_action/yield 相同、critical_objects 集合相等，允许 token 层非确定性）。输出只当功能冒烟、不比字节（flashattention/kernel 与残留 dropout 可致逐 token 非确定）；如需逐位追平可另加 `torch.use_deterministic_algorithms(True)`+统一 seed+确定性 attention 后端，属可选开关，S11 不强制。
- **J（新）. G7 判定量级（已拍板 2026-10-05）**：blank-image 反事实下 Overfit 模型**离散输出四字段（speed_action / yield / critical_objects 集合）至少 1 项改变**，作为"确实在用视觉"的判定，避免纯 token 级噪声误判。

## 8. 交接

- **S12（全量 1F SFT）**：trainer 与数据序列化不得改变（蓝图 §5）；对比基准 = `runs/S09_base_benchmark/eval_v6/`（v6 面，**非** S09 存档 v5 读数）；四字段抽检表升级为持续监控；paired bootstrap 沿 eval_v1 冻结口径。
- **S13（4F）**：4F 面出生即 v6（4 字段，无 reasoning 历史包袱）；sft 包仅扩展 4F 政策分支（serializer 已就绪），序列化语义零变更；历史帧诊断（repeated-current / shuffled-history）按蓝图 §3.5 届时预注册。
- **reasoning 字段未来恢复**：如需真实场景 reasoning，属独立数据质量工程（scene-grounded 文本来源 + 新 contract 版本正门），见讨论总结文档 §5 边界声明。
