# S11 实施规划（SFT 计算 Smoke：1F 半边，2026-10-05，v2 修订）

> 状态（2026-10-07 更新）：**规划已确认（2026-10-05 v2 修订）；Step 0–5 全部完成，S11 关闭**。Step 0（G0.1–G0.5 全 PASS）；Step 1–2 完成；Step 3 已完成并冻结（D4）；Step 4 训练 + G1 已完成并冻结（D6），自推理诊断（G6）与视觉依赖抽查（G7）已完成；Step 5 文档回填完成（D7 + `S11_smoke_run.md`）。Gate 范围于 2026-10-07 收缩为 **2 项实跑（G6/G7）+ 4 项凭证/记录**（见 §5 与 D7）。运行结果汇总入口 = [S11_smoke_run.md](S11_smoke_run.md)。本版相对 v1 的核心变化：模型面升版 v6（删除 reasoning 字段），target 构造大幅简化，执行顺序按用户修订（0.1/0.2 → SFT 探索 → 0.3 夜间 → 0.4）。
>
> **执行期偏离本规划之处**（以决策台账/缺陷台账为准）：①LoRA 的 `target_modules` 由本文所述的裸名字写法改为**路径限定正则**（缺陷 B7，否则视觉塔 MLP 被连带训练）；②G1 软层基准由"vs 冻结 GT"改为"vs 训练端输出"（D4）；③"新进程 reload"从 `sft/train.py` 内部拆出为独立 CLI `cli/sft_reload_verify.py`（B5/B6 复盘后的结论）；④分辨率口径见 [S11_training_acceleration.md](S11_training_acceleration.md)。
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
| `sft/train.py` | LoRA 训练入口：读冻结样本 artifact → 训练 → save adapter → 落"状态保真参照"（adapter 权重 + global step + 训练端生成输出）。**不做同进程 reload**（理由见 §3 Step 3） |
| `sft/gate.py` | G1 判定纯函数：`adapter_state_dict`（按名字含 `lora_` 筛，**不能**用 `requires_grad`——推理态会被冻）/ `compare_state_dicts` / `nonfinite_parameter_names` / `expected_global_step` / `four_fields_equal` |
| `cli/sft_train.py` | 训练 CLI（GPU）：包装 `sft/train.py`；训练后**先取峰值显存再**生成参照（`generate_one` 会 reset 峰值线）并落盘 |
| `cli/sft_reload_verify.py` | G1 复验 CLI（**新进程**，GPU）：仅凭磁盘 artifact（配置 + adapter + 冻结样本）复现硬层/软层判定，退出码 0/1 |
| `configs/sft/sft_smoke_1f.yaml` | 冻结运行配置：checkpoint 路径、contract v6、policy=ONE_FRAME、DUMP、LoRA 参数（**`target_modules` = 路径限定正则**，见 B7）、batch/grad-accum/grad-checkpoint、seed、解码（沿 S09 冻结口径）、选取规则参数 |

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

**状态：已完成并冻结（2026-10-06，D4；因 B7 修复于同日重跑，重跑后 sha 见 D4）。**

- LoRA 训练 → adapter 保存 → **新进程** reload → 固定输入（训练 batch 原样本）输出复验（参数、global step、无 NaN/Inf）。
  > 落地偏离：**同进程 reload 已否决**——同进程会让 trainer 模型与第二份 base+adapter 双驻显存，
  > 且无法证明"只存在于磁盘的 adapter"能独立加载。故拆成 `cli/sft_train.py`（训练 + 落参照）
  > 与 `cli/sft_reload_verify.py`（新进程复验）。
- 资源画像落档：loss/grad norm/step time/视觉 token 数/峰值显存/吞吐。
  > 实测：`train_runtime=90.87 s`、`train_loss=0.7981`、`global_step=16`（32×2÷4）、
  > 峰值显存 allocated 10.10 / reserved 15.03 GiB（卡 31.48 GiB）。
- **G1 判定结果：PASS**（硬层 `n_adapter_tensors=504`、`worst_abs_diff=0.0`、step 16==16；
  软层四字段与训练端参照逐 token 相同）。

### Step 4：128 overfit + 自推理诊断（GPU）

**状态：已完成（2026-10-07）。** 训练 + G1 于 2026-10-06 完成并冻结（D6）；自推理诊断（G6）与视觉依赖抽查（G7）于 2026-10-07 完成，执行器 `cli/sft_closeout.py model`，结果见 `runs/S11_sft_smoke/reports/g67_model.json` 与 [S11_smoke_run.md](S11_smoke_run.md) §5–§7。

- [x] 128 条过拟合训练，记录 loss 曲线。
  > 实测：`train_runtime=525.42 s`、`train_loss=0.5732`、`global_step=96`（128×3÷4）、
  > 峰值显存 allocated 10.14 / reserved 13.90 GiB。**G1 判定 PASS**（硬层 504 张量、
  > `worst_abs_diff=0.0`、step 96==96；软层四字段与训练端参照逐 token 相同；
  > 信息项 `vs_gt` 四字段**全等**，即已过拟合到 GT）。
- [x] 自推理诊断（解码沿 S09 冻结口径）：parse_rate（预期 ~1.0，全监督四字段）、输出 token 长度分布与截断率、四字段输出人工抽检表（S12 持续监控雏形）。
  > **已完成（2026-10-07，G6）**：`parse_rate=1.0000`（128/128）、截断率 `0.0000`（0/128）、
  > 输出 token min/中位/max = **25/125/416**、未以 `}` 收尾 0。抽检表
  > `runs/S11_sft_smoke/reports/g67_model_table.md`。**判定 PASS**。
- [x] `maxItems` 触顶率统计（S03-Q5 监控项，仅记录不决策）。
  > **已完成（2026-10-07，G5，记录项）**：`critical_objects` 触顶 **6/128（4.69%）**、
  > `risk_factors` 触顶 0/128（分母只含解析成功的样本）。折入 `g67_model.json` 的
  > `g5_max_items_record` 字段。
- [x] 视觉依赖抽查：overfit 模型少量 blank-image 反事实（复用 `make_blank_image`），确认 SFT 后模型确实在用视觉。
  > **已完成（2026-10-07，G7）**：overfit128 前 32 条（= train32 集合）blank-image 反事实，
  > 离散字段至少 1 项改变者 **25/32（78.1%）**。**判定 PASS**（条件 ≥1 条改变）。

### Step 5：文档回填与阶段关闭（前置 = Step 0.4 完成）

**状态：已完成（2026-10-07，D7）。**

- [x] 结果回填本目录 `S11_decision_log.md`（时间序裁定 D1–D6、参数冻结表、阶段关闭核对）。
- [x] 结果回填 `S11_smoke_run.md`（**已创建**，2026-10-07）：运行结果汇总入口，含 Gate 一览、
  配置与冻结物、Step 3/4 资源画像、G2 读数、G3/G4、G5/G6/G7、复现命令、交付与交接。
- [x] 配置登记 sha 后状态置 frozen（`configs/sft/sft_smoke_1f.yaml` 的 `status: frozen` + D4/D6 登记 sha）。
- [x] 阶段关闭核对表清零（见 `S11_decision_log.md` 末尾，3 项已勾）。

## 4. 样本选取规则（已拍板并冻结）

- **train32**：train split anchors 按 (location × time_of_day) 分层，每层字典序取前 k，凑满 32；先出 speed_action/yield 分布表再冻结，STOP/yield_true 样本 ≥ 各 4 条。
- **overfit128**：train32 的**超集**（同一分层规则扩至 128），符合蓝图"相同 anchors"口径。
- 确定性：选取只依赖 manifest 字典序 + 分层键 + 固定参数，无随机数；重跑逐样本 sha 一致。

## 5. 验证 Gate 总表（2026-10-07 收缩后）

> **收缩裁定（2026-10-07，见 D7）**：原表 G2–G7 六项经逐项复核，真正需要**实跑**的只有
> **G6 + G7**；G2 降级为**读数记录**、G3 以"构造期已强制"为凭证、G4 保留一次纯 CPU
> 重跑作实证、G5 折入 G6 报告。判定执行器 = `cli/sft_closeout.py`（`disk` 子命令纯 CPU、
> `model` 子命令需 GPU）。

| # | Gate | 来源 | 处置（2026-10-07） | 判定 |
|---|---|---|---|---|
| G0.x | Step 0 全部 gate（面单测/manifest parity/重跑完整/双面标注/冻结登记） | Step 0 文档 §5 | 实跑（已完成） | PASS（Step 0） |
| G1 | 1F forward/backward/save/reload 无 NaN/Inf；新进程 reload 后：权重逐张量可比（max\|Δ\|<1e-6）、global step 相等（**状态保真层，硬**）；固定输入 forward 输出可解析且四字段语义相等，容忍 token 层非确定性（**功能冒烟层，软**） | 蓝图 §4 + §7-I | 实跑（Step 3/4 各一次） | PASS（D4/D6） |
| — | **G1 软层基准修订（2026-10-06 拍板，见 D4）**：比对对象为**训练端在同一固定样本上的输出**，**不是**冻结 GT。理由：32/128 条本就不可能学会全部 GT，拿 GT 当基准会把"训练不充分"误判成"G1 失败"。与 GT 的一致率降为报告信息项 `vs_gt`（不参与判定）。 | D4 | — | — |
| G2 | 128 overfit loss 明显下降（量化口径 = §7-D，Step 4 前冻结） | 蓝图 §4 | **降级为读数记录**（不再判 PASS/FAIL）。理由见下 | 记录 |
| G3 | 面 parity：逐样本 user 面文本与 request_hash_1f == **v6-face** manifest 值 | §2.3 | **构造期已强制**（`sft/dataset.build_training_sample` 不等即抛 `ValueError`）；`disk` 子命令另做独立复核 | PASS |
| G4 | target 构建断言全量通过 + 确定性重跑逐字节一致（config/seed/manifest/git commit 可追溯） | §2.2/蓝图 §4 | **实跑（纯 CPU）**：以同一条序列化路径重打 JSONL，逐字节比 sha256 | PASS |
| G5 | `maxItems` 触顶率落档（S03-Q5，仅记录） | 蓝图 §6 | **折入 G6 报告的记录字段**（原本就不是 PASS/FAIL） | 记录（附于 G6） |
| G6 | 自推理诊断：parse_rate ≥ 0.99 且截断率 ≤ 1%（全监督下应为低水位 sanity 检查）；四字段抽检表落档 | §3 Step 4 | **实跑（GPU）**：128 条 overfit128 全量自推理 | PASS/FAIL |
| G7 | 视觉依赖抽查：blank-image 反事实下 overfit 模型**离散四字段输出至少 1 项改变**（非全同输出） | §3 Step 4 + §7-J | **实跑（GPU）**：取 overfit128 前 32 条（= train32 集合） | PASS/FAIL |

> **G2 为什么降级而不是继续判**：§7-D 的阈值（最终 ≤ 初始 × 0.3，或初始 >1.0 时下降 ≥75%）
> 在本 run 的实测曲线上**会判 FAIL**——`train128.log` 首点 0.9882 → 末点 0.4606，仅降 53.4%。
> 但同一个 run：G1 双层判定 PASS，且信息项 `vs_gt.four_fields_equal=true`（128×3 epoch 已把
> 固定样本过拟合到 GT）。**阈值判失败、证据表明学得很好**，说明该阈值是噪声来源而非信号；
> 加之 loss 下降的实质信息已被 G1 + G6 覆盖，故只保留读数（落 `reports/g34_disk.json`）。
> 附带说明：`train32.log` 在 `logging_steps=10` / 16 step 下只有 **1 个**日志点，本就不构成曲线，
> 判据对它无从下手——这也是"该阈值不稳健"的另一个佐证。

> **判定进度（2026-10-07）**：**G0.x PASS**；**G1 PASS**（Step 3 见 D4、Step 4 见 D6，
> 两次均为 B7 修复后的重跑）。**G3 PASS / G4 PASS**（`cli/sft_closeout.py disk`，
> 160 条样本复核 + 重建逐字节一致，报告 `runs/S11_sft_smoke/reports/g34_disk.json`）。
> **G2 记录**、**G5 记录**。**G6/G7 见 `runs/S11_sft_smoke/reports/g67_model.json`**
> 与 [S11_smoke_run.md](S11_smoke_run.md)。

## 6. 冻结物与登记清单

| 冻结物 | 位置 | 登记方式 |
|---|---|---|
| v6 面/manifest/eval 配置 | Step 0 交付 | Step 0 文档 §4 |
| `sft_smoke_1f.yaml`（DUMP/LoRA/选取规则参数） | `configs/sft/` | sha256 → 本目录决策台账（**已登记**：D4/D6） |
| 样本 artifact（train32/overfit128 + manifest） | `runs/S11_sft_smoke/sft_samples/` | 逐样本 sha256 + manifest sha（**已登记**：D4/D6；**确定性已实证**：G4 重建逐字节一致） |
| Smoke adapter（32 条）与 overfit checkpoint（128 条） | `runs/S11_sft_smoke/checkpoints/`（不进 git） | sha → 决策台账（**已登记** D4/D6）；运行读数汇总见 **`S11_smoke_run.md`（已创建）** |
| 资源/吞吐报告 | `runs/S11_sft_smoke/reports/` | **已回填** `S11_smoke_run.md`；现有 `g1_smoke32.json` / `g1_smoke128.json` / `g34_disk.json` / `g67_model.json` / `g67_model_table.md` |

## 7. 原待拍板清单（A–J 已全部拍板；Step 3/4 已据此执行）

- **A. 训练栈（已拍板 2026-10-05）**：**HF `Trainer` + pinned Qwen2.5-VL 自建轻量链路**。理由：与现有推理栈同源（同 `apply_chat_template`/`process_vision_info`），满足 §2.3 训练面==评测面 parity；AutoVLA SFT 面向轨迹格式，适配成本高且不复用价值（§3 Step 1 代码不受此拍板阻塞）。
- **B. §4 选取规则（已拍板 2026-10-05）**：接受 **分层 × 字典序**；配额 **STOP ≥ 4 且 yield_true ≥ 4（两维互不重叠取并集下限各 4）**。分层键分布表先出再冻结进 config。
- **C. LoRA 起点参数（已拍板 2026-10-05）**：**r=16, alpha=32, dropout=0.05, target=q/k/v/o+gate/up/down, batch=1+grad-accum=4, lr=1e-4**；补 **optimizer=`AdamW`（bf16）、scheduler=`cosine`+`warmup_ratio=0.05`、`weight_decay=0.01`、`max_grad_norm=1.0`**。起点值，overfit 阶段允许独立配置微调，变更留台账。
  > **⚠️ 上句 "target=q/k/v/o+gate/up/down" 的写法已被否（B7，2026-10-06）**：这组**裸名字**
  > 在 peft 里按"精确相等或名字尾"匹配，会把视觉塔同名的 `visual.blocks.*.mlp.{gate,up,down}_proj`
  > 一并命中（ViT 被意外训练；adapter 696 张量而非 504），且**全程不报错**。
  > 现行写法是**路径限定正则**：
  > `'model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))'`
  > （peft 对 str 走 `re.fullmatch`）。并已加运行时守卫 `sft/train.py: assert_text_only_lora_targets`
  > 与回归单测 `tests/unit/test_sft_lora_targets.py`。详见 [S11_bug_log.md](S11_bug_log.md) B7。
- **D. G2 量化口径（已拍板 2026-10-05）**：主判 = 最终 loss ≤ 初始 loss × 0.3 且曲线单调（允许 ≤2 次回升 step）；**补 override：若初始 loss 异常高（>1.0），改判"下降 ≥75% 且单调"**，避免 0.3 相对量在低起点下过于苛刻/在爆炸起点下失真。
  > **补（2026-10-07，见 D7）**：本条阈值**已降级为读数记录**（不再判 PASS/FAIL）——
  > 阈值在本 run 实测曲线上会判 FAIL（0.9882 → 0.4606，仅降 53.4%），而同一 run 的 G1
  > 双层 PASS 且 `vs_gt.four_fields_equal=true`。理由详见 §5 "G2 为什么降级"。
- **E. G6 阈值确认（已拍板 2026-10-05）**：**parse_rate ≥ 0.99、截断率 ≤ 1%**（全监督四字段下的 sanity 阈值，超限即查实现而非改方案）。
- **F（新增，已拍板 2026-10-05）. epochs / 轮数**：**train32 = 2 epoch（batch=1 → 64 micro-batch → 16 optimizer step）；overfit128 = 3 epoch（384 micro-batch → 96 optimizer step）**，每 64 step 存一次 checkpoint，取 loss 最低 step 复验。train32 仅验证链路通；overfit128 需足够步数跑出明显下降以支撑 G2。
  > **数字修正**：原句把 overfit128 写成"→ 384 step"是把 **micro-batch 数当成了 optimizer step**。
  > 计 `gradient_accumulation_steps=4` 后，**optimizer step = 384 ÷ 4 = 96**；G1 硬层的
  > `expected_global_step` 判据用的正是 96（实测 `global_step=96 == expected`，见 D6）。
- **G（新）. overfit dropout（已拍板 2026-10-05）**：overfit 阶段 **dropout = 0**（config 以 `overfit: {lr:…, dropout:0}` 覆盖起步值），消除 dropout 噪声对 G2 曲线与自推理的诊断干扰；确认过拟合阶段关 augmentation。
- **H（新）. seed 定值**：config `seed` 写死一个固定值（建议 `0`），避免逐环境（torch/cuda/numpy/random）随机性影响可复现性。
- **I（新）. G1 判定（已拍板 2026-10-05，分层，替代原"固定输入输出符合预期"）**：**状态保真层（硬，确定性）** = reload 权重逐张量 shape+dt 同构可比（容差 ∃ max\|Δ\| <1e-6）、global step 相等、无 NaN/Inf；**功能冒烟层（软，容忍 forward 随机性）** = 固定输入跑 forward 输出**可解析**且**四字段语义相等**（speed_action/yield 相同、critical_objects 集合相等，允许 token 层非确定性）。输出只当功能冒烟、不比字节（flashattention/kernel 与残留 dropout 可致逐 token 非确定）；如需逐位追平可另加 `torch.use_deterministic_algorithms(True)`+统一 seed+确定性 attention 后端，属可选开关，S11 不强制。
  > **补（2026-10-06）**：软层"四字段语义相等"的**比对对象已明确为训练端输出**（而非冻结 GT），
  > 见 §5 表的 G1 修订行与 D4。另：硬层的 adapter 筛选**不能**用 `requires_grad`
  > （peft 推理态会冻结权重，会把 696/504 项全判为 missing），必须按名字含 `lora_` 筛
  > ——这是 Step 3 首跑的一次假 FAIL，已修。
  > **实现落点**：`sft/gate.py`（判定纯函数）+ `cli/sft_reload_verify.py`（新进程执行）。
- **J（新）. G7 判定量级（已拍板 2026-10-05）**：blank-image 反事实下 Overfit 模型**离散输出四字段（speed_action / yield / critical_objects 集合）至少 1 项改变**，作为"确实在用视觉"的判定，避免纯 token 级噪声误判。

## 8. 交接

- **S12（全量 1F SFT）**：trainer 与数据序列化不得改变（蓝图 §5）；对比基准 = `runs/S09_base_benchmark/eval_v6/`（v6 面，**非** S09 存档 v5 读数）；四字段抽检表升级为持续监控；paired bootstrap 沿 eval_v1 冻结口径。
- **S13（4F）**：4F 面出生即 v6（4 字段，无 reasoning 历史包袱）；sft 包仅扩展 4F 政策分支（serializer 已就绪），序列化语义零变更；历史帧诊断（repeated-current / shuffled-history）按蓝图 §3.5 届时预注册。
- **reasoning 字段未来恢复**：如需真实场景 reasoning，属独立数据质量工程（scene-grounded 文本来源 + 新 contract 版本正门），见讨论总结文档 §5 边界声明。
