# S12 单帧 SFT 阶段运行记录（S12_run.md）

> **状态**：进行中——Step 1–2 已完成并 commit，Step 3（全量训练）待用户执行。
> **规划稿**：[S12_implementation_plan.md](./S12_implementation_plan.md)（v1，2026-10-07 冻结）
>
> 本文件只登记**执行凭证**（命令 / sha / 产物路径 / 读数）。判据、口径、阈值一律以规划稿为准，
> 此处不重复抄写；出现分歧时以规划稿 + §6 决策台账为准。Step 8 做最终回填与阶段关闭。

---

## 0. 执行状态

| Step | 内容 | 状态 | 凭证 |
|---|---|---|---|
| 1 | 代码 + 冻结配置 + 单测（CPU） | 完成 | commit `ba93717` |
| 2 | train_full 样本冻结（CPU，确定性） | 完成 | commit `f1e60a6`；`runs/S12_sft_1f/reports/step2_freeze.json` |
| 3 | 全量训练（GPU，tmux） | **进行中**：诊断 smoke 已完成（§2.6），正式训练待重启 | 本文件 §2 / §2.6 |
| 4 | save/resume 验证（GPU，新进程） | 未开工 | — |
| 5 | val 读数 → 冻结 best（GPU） | 未开工 | — |
| 6 | test 一次性推理（GPU） | 未开工 | — |
| 7 | 评测与对比 | 未开工 | — |
| 8 | 文档回填与阶段关闭 | 未开工 | — |

---

## 1. 冻结物 sha256 与执行环境

### 1.1 冻结物（本文件登记，供 Step 8 复核）

| 物件 | 路径 | sha256 |
|---|---|---|
| 训练冻结配置 | `DriveAlign/configs/sft/sft_1f.yaml` | `06799e25b5e435587b275e28a4f381ff228b6cfb784c4a2c0a9b2fa3410c9b69` |
| 评测冻结配置 | `DriveAlign/configs/benchmark/sft_1f_v6.yaml` | `6db57767335597d405c881d2e71adbe7bb66a5882bf1a56056e91d652fc07e88` |
| 规划稿 | `DriveAlign/docs/.../S12_implementation_plan.md` | `6ce8110da4ae748ef22f8741d138d3a5e8ca951c1337f9fa7975ba3ea979b140` |
| train_full token 列表（A 层） | `data/sft_subsets/train_full_tokens.json` | `1166ff34fa14972bc472c55e59d57428a39ea0648cac8dcf64ef2d5b7d802647` |
| train_full 训练样本（B 层） | `data/sft_samples/v6_1f/train_full.jsonl` | `eb84e48ea5b05c2304ede0e8d1ef3b960ddab8082950d5ebc5f325008eed7514` |
| 样本 manifest | `data/sft_samples/v6_1f/samples_manifest.json` | `6b98b16e0f80cbc02d9c759b246357ae365844ed52bef26c63625a48703d90a1` |
| 反事实子集（沿用 S09） | `runs/S09_base_benchmark/counterfactual_subset.txt` | `55bc70cd…`（200 锚，详见 `step2_freeze.json`） |

Step 2 的 G3 独立复核结论（重读 v6 manifest 逐样本核，不依赖构造期断言）：22,941 行、0 重复、
**0 parity 不一致**、逐样本 split 全为 `train`、与 token 列表集合精确相等（missing 0 / extra 0）。
S12 冻结样本目录只有 `train_full.jsonl` + `samples_manifest.json`。

### 1.2 执行环境（训练与评测共用）

| 项 | 值 |
|---|---|
| 解释器 | `conda activate autovla_codeclean` |
| 工作目录 | `/root/autodl-tmp/drivealign_workspace`（**git repo 根在子目录 `DriveAlign/`**） |
| `PYTHONPATH` | `DriveAlign/src` |
| dataroot | `data/nuscenes/trainval` |

**必须从 workspace 根执行**：`runs/` 与 `data/` 都是它的子目录，且不在 git 内（运行产物天然不进 git）。

---

## 2. Step 3：全量训练启动命令

### 2.0 为什么不传 `--epochs` / `--save-steps`

Step 1 给 `sft_train` 加了 `--epochs` / `--save-steps` 覆盖开关，但**正式跑不要用**：不传时
训练规模完全由冻结配置 `sft_1f.yaml` 决定（`num_train_epochs: 1`、`lr: 5.0e-5`、
`save_steps: 500`、`save_total_limit: null`），这样"跑的是什么规模"只需核对一份配置 sha，
不会出现命令行与配置两份口径。该开关只服务于 Step 4 的 resume 验证与中断续跑。
另：`--overfit` 也**不要传**（那是 S11 smoke 的配方）。

### 2.1 开训前 pre-flight（一次性，秒级）

```bash
conda activate autovla_codeclean
cd /root/autodl-tmp/drivealign_workspace

# ① 卡是否空闲（沙箱内 nvidia-smi 会被拦，请在真实终端执行）
nvidia-smi

# ② 记下 commit（应为 f1e60a6367d6c83f07533bcbb77e0d0b121f042e）
git -C DriveAlign rev-parse HEAD

# ③ 核对冻结配置未被改动（应为 06799e25…，见 §1.1）
sha256sum DriveAlign/configs/sft/sft_1f.yaml

# ④ 核对训练样本未被改动（应为 eb84e48e…，见 §1.1）
sha256sum data/sft_samples/v6_1f/train_full.jsonl
```

### 2.2 正式训练（tmux，约 8.7–10 h）

```bash
# 1) 起会话（可随时 detach：Ctrl-b d；回来：tmux attach -t s12_train）
tmux new -s s12_train

# 2) 会话内执行。注意 `set -o pipefail` 必须写在 `| tee` 之前，
#    否则训练进程非零退出会被 tee 的 0 掩盖（S11 已踩）。
conda activate autovla_codeclean
cd /root/autodl-tmp/drivealign_workspace
set -o pipefail

PYTHONPATH=DriveAlign/src python -m drivealign.cli.sft_train \
  --config DriveAlign/configs/sft/sft_1f.yaml \
  --samples data/sft_samples/v6_1f/train_full.jsonl \
  --out runs/S12_sft_1f/checkpoints/sft_1f \
  --dataroot data/nuscenes/trainval \
  2>&1 | tee runs/S12_sft_1f/reports/step3_train.log
```

### 2.3 中断后续跑（可选；`--out` 必须与 §2.2 完全一致）

```bash
PYTHONPATH=DriveAlign/src python -m drivealign.cli.sft_train \
  --config DriveAlign/configs/sft/sft_1f.yaml \
  --samples data/sft_samples/v6_1f/train_full.jsonl \
  --out runs/S12_sft_1f/checkpoints/sft_1f \
  --dataroot data/nuscenes/trainval \
  --resume-from-checkpoint runs/S12_sft_1f/checkpoints/sft_1f/checkpoint-5000 \
  2>&1 | tee -a runs/S12_sft_1f/reports/step3_train.log
```

`--out` 一旦改成别的目录，optimizer / scheduler / RNG 状态与 adapter 就会分家，resume 失去意义。

### 2.4 预期读数与产物

| 项 | 预期 | 依据 |
|---|---|---|
| optimizer step | ≈ 5,736 / epoch（22,941 ÷ 4） | §2.5 |
| 时长 | ≈ 8.7–10 h | §2.5（S11 实测 ~1.4 s/micro-batch 外推） |
| 时长（**实测更正**） | **≈ 4.13 h**（5,735 × 2.59 s） | 本文件 §2.6；plan §2.5 已同步更正 |
| 中间 checkpoint | `checkpoint-500 … checkpoint-5500`，共 11 个，每个 ≈344 MB → 约 3.8 GB | §2.5 + S11 实测体积 |
| 磁盘余量 | 需 ≥ 5 GB（实测可用 196 GB，充裕） | `df -h` |
| 峰值显存 | 待记（作为 Step 8 资源报告输入） | §3 Step 3 |

两个易错点：

1. **末点 = `--out` 根目录里的 adapter**（`runs/S12_sft_1f/checkpoints/sft_1f/adapter_model.safetensors`），
   不是某个 `checkpoint-XXXX`。5,736 不是 500 的整数倍，HF `save_strategy=steps` 不会在最后一步
   自动落 checkpoint；末点由训练收尾的 `trainer.save_model(--out)` 落盘。Step 5 冻结 `best` 时即取这一份。
2. **收尾顺序不可反**：CLI 内先取训练峰值显存，再生成 `train_state_ref.pt`。因为生成参照内部会
   `reset_peak_memory_stats()`，顺序反了读到的就只是生成阶段的峰值（S11 已踩）。

### 2.5 开训即自动生效的守卫

- **B7（LoRA 落点）**：`assert_text_only_lora_targets` 在开训前断言 adapter 张量数 == 504、
  `visual.` 零命中——命中即抛错停机，不会静默训到 ViT。
- **冻结配置**：`sft_train.main` 校验 `status: frozen`，非冻结配置直接拒绝开跑。
- **A1（分辨率）**：训练侧 `max_pixels: 360000` 已由冻结配置钉死；评测侧
  `sft_1f_v6.yaml` 同值，由 `test_sft_config_parity.py` 钉住，两端不会漂移。

### 2.6 诊断 smoke（30 step 全量数据；**throwaway，不进 §5 冻结物**）

**动机**：§2.2 全量训练启动后，两项读数与 S11 smoke 对不上——

| 读数 | S11 `smoke128`（`runs/S11_sft_smoke/checkpoints/smoke128/train128.log`） | S12 全量（`runs/S12_sft_1f/reports/step3_train_aborted.log` 前 234 step） |
|---|---|---|
| 单步耗时 | 5.47 s/opt-step（525.42 s / 96 step） | **2.60 s/opt-step**（10:08 / 234 step） |
| 峰值 allocated | 10.14 GiB | 未落档（进程被中断） |
| 峰值 reserved | 13.90 GiB | 未落档；`nvidia-smi` 稳态 **28,090 MiB**（45 s 内两次采样完全不变 → 不涨） |

两份运行**同配置**（`sft_1f.yaml` 与 `sft_smoke_1f.yaml` diff 仅 4 处训练规模字段）、**同序列长度**
（train_full 均 1,123 tok vs smoke32 1,070 / smoke128 1,104，1.02–1.05×），故差异不能归因于数据规模。
处置：停掉全量训练（pid `621013`，跑到 234/5,735、10:08，**尚无 checkpoint 落盘**，`save_steps=500`），
改跑**同数据规模、只 30 step** 的诊断 smoke，把"数据规模"与"代码路径"两个变量分离。

#### 2.6.1 开训前 pre-flight（真实 collator 取 batch，batch=1 == 训练 micro-batch 口径）

| 项 | 实测 |
|---|---|
| `len(dataset)` | 22,941（= train_full 行数，装载完整） |
| `input_ids` / `labels` / `attention_mask` | `(1, 1314)` int64 |
| `pixel_values` | `(7168, 1176)` float32 — mean **−0.1087** / std **0.7186** / min **−1.7777** / max **2.1459** |
| `pixel_values` 全零？ | **否**（非全零、非 garbage，量级与归一化后的图像一致） |
| `image_grid_thw` | `[[1, 32, 56]]`（32×56 = 1,792 patch/图，与 `max_pixels=360000` 的 448×784 一致） |
| 监督位数 | 74–380（视目标长度） |

#### 2.6.2 视觉通路判定（同一 batch，仅改视觉输入；batch=1，4 条不同样本）

| 样本 | 真实 | 全零 | 随机噪声 | 删除视觉键 | real==zero | real==noise | real==no_vis |
|---|---|---|---|---|---|---|---|
| 0 | 1.725617 | 1.313969 | 1.350558 | 1.322114 | False | False | False |
| 1 | 1.808458 | 1.314243 | 1.368414 | 1.311629 | False | False | False |
| 2 | 1.868363 | 1.363640 | 1.401419 | 1.348366 | False | False | False |
| 3 | 0.980915 | 0.841019 | 0.847233 | 0.837646 | False | False | False |

**结论**：三种扰动（置零 / 换噪声 / 删键）都让 loss 变化 0.14–0.52 nats，**视觉通路正常**，
`pixel_values` 确实参与前向；不存在"图像被静默忽略、只训文本"的情形。

#### 2.6.3 30-step 全量 smoke 读数

命令（等价 §2.2 + `--max-steps 30`，产物落 `diag/` 与正式训练隔离）：

```bash
PYTHONPATH=DriveAlign/src python -m drivealign.cli.sft_train \
  --config DriveAlign/configs/sft/sft_1f.yaml \
  --samples data/sft_samples/v6_1f/train_full.jsonl \
  --max-steps 30 \
  --out runs/S12_sft_1f/diag/smoke30 \
  --dataroot data/nuscenes/trainval
```

| 项 | 读数 | 对照 S11 `smoke128` |
|---|---|---|
| `global_step` | 30 | 96 |
| `train_runtime_s` | **77.67**（**2.59 s/opt-step**） | 525.42（5.47 s/opt-step） |
| `train_loss`（均值） | 0.753 | — |
| loss 走势（step 10/20/30） | 1.0431 / 0.6409 / 0.5749 | — |
| `peak_allocated_gib` | **10.15** | 10.14 |
| `peak_reserved_gib` | **27.12** | 13.90 |
| `device_total_gib` | 31.48 | — |
| 生成参照（软层） | `speed_action=KEEP_SPEED`、`yield_required=false`、80 out-tok | — |

**关键点：全量数据下的单步耗时（2.59 s）与全量训练实测（2.60 s）逐值复现** → 耗时是**代码路径**的
属性，与数据规模无关，可外推 1 epoch ≈ 4.13 h。

#### 2.6.4 显存"翻倍"的归因：allocated 未变，变的是分配器预留池（碎片）

| 口径 | `peak_allocated_gib` | `peak_reserved_gib` |
|---|---|---|
| S11 `smoke128`（默认分配器） | 10.14 | 13.90 |
| S12 `diag/smoke30`（默认分配器） | **10.15** | **27.12** |
| S12 `diag/smoke30_expseg`（`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`） | 10.04 | **10.80** |

**结论**：

1. **真实张量占用没有变化**（10.14 → 10.15 GiB，±0.01）。`nvidia-smi` 报的 28.1 GiB 是
   **reserved**（分配器预留池，含碎片），不是活动张量。
2. 预留池翻倍源于**碎片化**：惰性管道每步现取现编码，序列长度（1,008–1,314）与各步 buffer
   尺寸逐步变化，caching allocator 难以复用，池子被撑大。**决定性证据**是
   `expandable_segments:True` 下 reserved 掉回 **10.80 GiB**（−60%）。
3. 该池**不增长**（45 s 内两次采样恒为 28,090 MiB），`allocated` 距卡上限 31.48 GiB 尚有 **21 GiB** 余量
   → **无 OOM 风险**，不是缺陷、不影响正式训练。
4. 若想让 `nvidia-smi` 数字好看/给同卡部署其他任务留空间，可在正式训练前加
   `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`（可选优化，非必需）。

#### 2.6.5 耗时"减半"的归因

S11 的装载路径是 `HFDataset.from_list` 全量预编码；HF Dataset 的**行索引会把张量列退化成嵌套
`list`**（S11_bug_log B5），collator 每 micro-batch 需用 `torch.as_tensor` 把 1792×1176 的嵌套
list 重转回张量（默认 `num_workers=0`，**串行、GPU 空转**）。CPU 微基准（本机实测）：

| 操作 | 耗时 / micro-batch |
|---|---|
| 嵌套 list → tensor（2,107,392 元素） | **321.3 ms** |
| 已是 tensor → tensor（短路） | 0.001 ms |
| 差值 × 4（accum=4） | **1.285 s / opt-step** |

S12 惰性 Dataset 直出张量省掉这段 → 可解释 2.88 s 差距中的约 1.29 s（45%）；其余约 1.6 s 属
HF Dataset 逐行 arrow↔Python 的其他开销，**未做逐项实测**（该路径已被 Step 1 的改动移除，
重建 A/B 不划算）。**属正当加速的副产品，不是异常**。

**产物**（throwaway，Step 8 不入 §5 冻结物）：
`runs/S12_sft_1f/reports/diag_smoke30.log`、`runs/S12_sft_1f/reports/diag_smoke30_expseg.log`、
`runs/S12_sft_1f/reports/step3_train_aborted.log`（被中断那次的 234 step 原始日志，重启前改名留档，
否则会被 §2.2 的 `tee` 覆盖）、
`runs/S12_sft_1f/diag/smoke30/`、`runs/S12_sft_1f/diag/smoke30_expseg/`。

**结论**：两项异常均**不是缺陷**。正式全量训练可重启（§2.2 命令原样，**不要**加 `--max-steps`），
预期待落档：`global_step=5,736`、`train_runtime ≈ 4.13 h`、`peak_allocated ≈ 10.2 GiB`、
`peak_reserved ≈ 27 GiB`。

---

## 3. Step 3 结束时要落档的读数（§3 Step 3）

训练进程 stdout 尾部会打一份 JSON；这些数再加日志里的曲线，一并登记进本文件（Step 8 汇总）：

- [ ] `global_step`（预期 5,736）
- [ ] `train_loss`（末段）
- [ ] `train_runtime_s` → 换算 GPU-hours
- [ ] `cuda_peak_memory`（训练阶段峰值，非生成阶段）
- [ ] `state_reference`（`train_state_ref.pt` 路径，Step 4 硬层比对要用）
- [ ] loss 曲线 / grad norm 曲线 / step time / 吞吐（从 `step3_train.log` 提取）
- [ ] `ls runs/S12_sft_1f/checkpoints/sft_1f/` 的实际 checkpoint 清单

---

## 4. 偏离登记（必载）

> 规划稿 §2.3 与 §4 明确要求下列偏离**原样落入** `S12_decision_log.md` 与 `S12_run.md`。
> `S12_decision_log.md` 于 Step 8 创建，届时需复述同一段；此处先按 §2.3 原文登记。

**⚠️ 偏离登记（§2.3）**

> 蓝图 Stage 9 Gate 与 [12_single_frame_sft.md](../../../stages_plan/12_single_frame_sft.md) §4 均要求"**best checkpoint 完全由 validation 选择**"。本期取末点 = **不满足该条**，属**有意偏离且不复议**。
>
> - **理由**：1 epoch + 22,941 条（smoke 数据量的 900 倍）下，单遍训练内过拟合概率极低，val 指标大概率单调改善、末点即最优。
> - **配套 val 义务（仍然执行）**：对末点跑**一次全量 val（2,531）**并落档，作为 `validation 轨迹`（单点读数）。即 **validation 仍被观测、可事后审视，只是不承担选择职责**——这也让"到底有没有过拟合"保有证据。
> - **候选 ckpt 仍全量保留**：`save_steps=500` + `save_total_limit: null`。用途变为 ① Step 4 resume 验证取中间 ckpt ② 训练中断保险；**不再用于选择 best**。
> - **后果自担**：test 结果若不理想，"换 checkpoint"这条路在本阶段已关闭（重跑 test 即破坏预注册）。

**已知偏离（§4，非 FAIL）**：阶段关闭时按"**接受偏离**"记录；§6-J 已关闭，不复议。

**待 Step 8 复述**：上述两段须在 `S12_decision_log.md` 中再次原样声明，并在 S13 交接说明中复述。

**Step 6b 附加登记（2026-10-07，batched_v1 path，另开面不影响上述冻结登记）**：

- 批量推理另开独立 path：`configs/benchmark/sft_1f_v6_batched.yaml`（sha `defba3c5…`，
  `batch_size: 16` 由配置承载、CLI 不覆盖）+ `batch_benchmark` CLI + `runs/S12_sft_1f/eval_v6_batched/`
  独立输出；**parity gate 经拍板延后**，该面读数一律标注"**待 parity 复核**"
  （bs=1 参照面 `eval_v6/` 于 164 行中断，待 `--resume` 续跑后做 CPU diff）。
  注：yaml 于 2026-10-07c 依会话记录重建（撤销事故），注释措辞略异、字段值一致，
  重建前 sha 为 `358b3aa5…`（见 run_summary 登记与 yaml 变更记录）。
- Bug 台账：[S12_bug_log.md](S12_bug_log.md)（B1 批量静默退化 bs=1 的双层故障链、B2 manifest 面误用被守卫拦截）。
- 速度记录：[S12_batched_inference_speed.md](S12_batched_inference_speed.md)
  （bs∈{1,2,4,8,16,32} 同批扫描，bs=16 定档 3.48×，GPU 利用率近饱和佐证）。
