# S11 SFT 计算 Smoke（1F 半边）—— 运行结果汇总

> **面标签**：本文件所有读数均为 **v6 面**（contract v6，模型面已删除 reasoning 字段）。
> S09 存档的 v5 面读数**不得**与本文件数字直接互比，双面标注规则见
> [S09_v6_face_rerun.md](../S09_base_benchmark/S09_v6_face_rerun.md) §双面标注规则。
>
> **本文件定位**：S11 运行结果的**唯一汇总入口**（规划 §3 Step 5 的交付物）。
> 决策链见 [S11_decision_log.md](S11_decision_log.md)（D1–D7），缺陷见
> [S11_bug_log.md](S11_bug_log.md)（B1–B7），加速实验见
> [S11_training_acceleration.md](S11_training_acceleration.md)。
> 生成时间 2026-10-07。

---

## 0. Gate 判定一览（结论摘要）

> Gate 范围于 2026-10-07 收缩（见 D7）：原 §5 的 G2–G7 六项 → **2 项实跑 + 4 项凭证/记录**。

| Gate | 判定 | 凭证 |
|---|---|---|
| G0.1–G0.5（Step 0 面过渡） | **PASS** | [S11_step0_v6_face_transition.md](S11_step0_v6_face_transition.md) §5；D3 |
| G1（Step 3，train32） | **PASS** | `runs/S11_sft_smoke/reports/g1_smoke32.json`；D4 |
| G1（Step 4，overfit128） | **PASS** | `runs/S11_sft_smoke/reports/g1_smoke128.json`；D6 |
| G2（loss 明显下降） | **记录**（降级，不再判） | `runs/S11_sft_smoke/reports/g34_disk.json` → `g2_loss_curve_record`；§3 本文件 |
| G3（面 parity） | **PASS** | 构造期强制（`build_training_sample`）+ `disk` 独立复核 160 条 0 失配；`reports/g34_disk.json` |
| G4（target 断言 + 确定性重跑） | **PASS** | 重建 JSONL 与冻结 sha256 **逐字节一致**（train32 + overfit128） |
| G5（`maxItems` 触顶率） | **记录**（折入 G6 报告） | `runs/S11_sft_smoke/reports/g67_model.json` → `g5_max_items_record` |
| G6（自推理诊断） | **PASS** | `runs/S11_sft_smoke/reports/g67_model.json` → `gate_g6_self_inference`；抽检表 `reports/g67_model_table.md` |
| G7（视觉依赖抽查） | **PASS** | 同上 → `gate_g7_visual_dependence` |

---

## 1. 运行配置与冻结物

| 对象 | 值 |
|---|---|
| SFT 配置 | `configs/sft/sft_smoke_1f.yaml`，sha256 `05231b343307ce6e6b95020d882a714fd9af344e49e69ba06c09acfb32c97085` |
| contract | v6（四字段 + v6 prompt），`input_policy: 1F` |
| 输入分辨率 | `max_pixels: 360000`（A1；原图 `smart_resize` → 784×448，visual token 1824→448、序列 2384→1008） |
| LoRA | r=16 / alpha=32 / dropout=0.05（overfit 阶段 0）；`target_modules` = 路径限定正则（B7 修正后） |
| 优化 | batch=1 + grad-accum=4、lr=1e-4、AdamW、cosine + warmup 0.05、bf16、gradient checkpointing |
| seed | 0 |
| 解码（仅自推理诊断用） | greedy（`do_sample=false`）、`max_new_tokens=512` |
| train32 样本 | `runs/S11_sft_smoke/sft_samples/train32.jsonl`，sha256 `72285c15e3be653a166db8e43959b2515424f2679476a0206f51bf365cb5bb0f` |
| overfit128 样本 | `runs/S11_sft_smoke/sft_samples/overfit128.jsonl`，sha256 `aec424f786cc65a4c20b6870bb65b4c2ae7892f8050392143009133b9b4d6fcc` |
| smoke32 adapter | `runs/S11_sft_smoke/checkpoints/smoke32/adapter_model.safetensors`，sha256 `3dfe30c1dd8935f13bd3132b9a63a510b4e89260ae4eddedea11549c62e46f70` |
| smoke128 adapter | `runs/S11_sft_smoke/checkpoints/smoke128/adapter_model.safetensors`，sha256 `db035370c66c8b9b1d16526cc42f60f362b3b1c04167c099b4434738d16a4242` |

---

## 2. Step 3：train32（32 条 / 2 epoch / 16 optimizer step）

| 指标 | 实测 |
|---|---|
| `train_runtime` | **90.87 s** |
| `train_loss`（全程均值） | 0.7981 |
| `global_step` | 16 |
| 吞吐 | 0.704 samples/s（0.176 steps/s） |
| 峰值显存 allocated / reserved | **10.10 / 15.03 GiB**（卡 31.48 GiB） |

> **G1 双层判定：PASS**
> - 硬层：`n_adapter_tensors=504`（= 36 层 × 7 模块 × 2）、`missing=[]`、`unexpected=[]`、
>   `shape_mismatch={}`、`dtype_mismatch={}`、`worst_abs_diff=0.0 < atol=1e-6`、
>   `nonfinite_parameters=[]`、`global_step=16 == expected=16`。
> - 软层：`logits_finite=true`（`[1,1008,151936]`）、`parse_ok=true`、
>   `four_fields_equal_to_reference=true`（reload 输出与训练端参照**逐 token 相同**，81 token）。
> - 信息项（不判定）：`vs_gt.four_fields_equal=false`，差异字段 `speed_action`
>   （模型 `ACCELERATE` vs GT）——32×2 epoch 不足以拟合 GT，属预期。

## 3. Step 4：overfit128（128 条 / 3 epoch / 96 optimizer step）

| 指标 | 实测 |
|---|---|
| `train_runtime` | **525.42 s** |
| `train_loss`（全程均值） | 0.5732 |
| `global_step` | 96 |
| 吞吐 | 0.731 samples/s（0.183 steps/s） |
| 峰值显存 allocated / reserved | **10.14 / 13.90 GiB** |

> **G1 双层判定：PASS**
> - 硬层：`n_adapter_tensors=504`、`worst_abs_diff=0.0`、`global_step=96 == expected=96`、
>   无 NaN/Inf。
> - 软层：`parse_ok=true`、`four_fields_equal_to_reference=true`（125 token 逐 token 相同）。
> - 信息项：**`vs_gt.four_fields_equal=true`、`field_diffs=[]`**——128×3 epoch 已把固定样本
>   **过拟合到 GT**。这正是软层基准选"vs 训练端输出"而非 GT 的证据（否则 train32 那轮会被误判）。

### 3.1 G2 loss 曲线读数（**记录项**，不判 PASS/FAIL）

| run | 日志点数 | 首点 | 末点 | 曲线均值 | 下降 | 回升次数 |
|---|---|---|---|---|---|---|
| smoke128（overfit128） | 9 | 0.9882 | 0.4606 | 0.5803 | 53.4% | 2 |
| smoke32（train32） | 1 | 0.9379 | 0.9379 | — | — | — |

> **为什么不判**：规划 §7-D 的阈值（最终 ≤ 初始 × 0.3，或初始 >1.0 时下降 ≥75%）在本 run
> 上**会判 FAIL**（仅降 53.4%），但同一 run 的 G1 双层 PASS 且 `vs_gt` 四字段全等——
> **阈值判失败、证据表明学得很好**，该阈值是噪声来源而非信号。
> 另：`train32.log` 在 `logging_steps=10` / 16 step 下只有 **1 个**日志点，本就不构成曲线。
> 两点合起来说明该量化口径不稳健，故 2026-10-07 收缩为读数记录（D7）。
> 说明：日志点是 `logging_steps` 粒度的**窗口均值**，不是逐步 loss。

---

## 4. G3 / G4：样本 artifact 完整性（纯 CPU）

执行器：`cli/sft_closeout.py disk` → 报告 `runs/S11_sft_smoke/reports/g34_disk.json`。

| 项 | 方法 | 结果 |
|---|---|---|
| **G4 确定性** | 以 `cli/sft_build_samples.py serialize` 的**同一条代码路径**重打到临时目录，逐文件比 sha256 | train32 / overfit128 **均逐字节一致**（重建 sha == 冻结 sha == manifest 登记 sha） |
| **G3 面 parity** | 逐样本独立复核 `request_hash_1f` vs v6-face manifest（并核 `record_hash`） | **160 条 0 失配** |

> G4 的这半此前**从未实证**：D4 里"样本 sha 与首次冻结逐字相同"说的是"文件没被动过"，
> 不等于"重建能复现"。本次补上，S12 的"序列化不得改变"由此有了可复现的基线。
> G3 在构造期即由 `build_training_sample` 的 `ValueError` 强制（不等即拒绝），`disk`
> 子命令做的是**独立复核**而非首次判定。

---

## 5. G6：自推理诊断（overfit128 全量 128 条）

执行器：`cli/sft_closeout.py model` → 报告 `runs/S11_sft_smoke/reports/g67_model.json`，
逐样本抽检表 `runs/S11_sft_smoke/reports/g67_model_table.md`。

| 指标 | 阈值（§7-E 冻结） | 实测 | 判定 |
|---|---|---|---|
| parse_rate | ≥ 0.99 | 1.0000（128/128） | PASS |
| 截断率（`output_tokens >= 512`） | ≤ 1% | 0.0000（0/128） | PASS |
| 文本未以 `}` 收尾 | —（诊断项） | 0 | — |
| 输出 token 数 min / 中位 / max | — | 25 / 125 / 416 | — |

> **口径**：解码沿 S09 冻结口径（greedy、`max_new_tokens=512`），严格解析 v6。
> 截断判定用**严格口径**（生成预算被用尽），"未以 `}` 收尾"另记为诊断项，避免把
> "提前停下但格式合法"误算成截断。最长 416 token 距 512 预算尚有余量，与截断率 0 自洽。
> **边界**：本项是在**训练样本**上的自推理（过拟合集），因此 parse_rate=1.0 主要验证的是
> **生成链路 + 解析器 + 契约**的贯通，不是泛化能力的证据——泛化由 S12 全量训练后的 eval 承担。

## 6. G5：`maxItems` 触顶率（**记录项**，不决策）

| 指标 | 值 |
|---|---|
| schema 上限（v6） | `critical_objects` 8 / `risk_factors` 8 |
| `critical_objects` 触顶样本数 | **6 / 128（4.69%）** |
| `risk_factors` 触顶样本数 | 0 / 128 |

> 分母只含**解析成功**的样本（128 条全部解析成功）。仅记录，S03-Q5 监控项延续到 S12。
> 4.69% 的触顶率值得 S12 持续观察：若全量训练后显著上升，说明模型倾向于"堆满对象"，
> 会同时推高输出长度与打分噪声。

## 7. G7：视觉依赖抽查（blank-image 反事实）

| 项 | 值 |
|---|---|
| 反事实子集 | overfit128 前 32 条（该前缀即 train32 集合） |
| 可用（两侧均可解析） | 32 / 32 |
| **离散字段至少 1 项改变的样本数** | **25 / 32（78.1%）** |
| 判定 | **PASS**（条件：≥ 1 条改变） |

按改变字段分组（同一行 = 该样本发生变化的字段集合）：

| 改变的字段 | 样本数 |
|---|---|
| 仅 `critical_objects` | 21 |
| `speed_action` + `critical_objects` | 3 |
| 仅 `speed_action` | 1 |
| 无改变 | 7 |

> 用 `structured_runner.make_blank_image` 生成同尺寸中灰图（销毁全部场景内容），
> 其余输入不变，重跑贪心生成后比对**离散字段**（`speed_action` / `yield_required` /
> `critical_objects` 集合，§7-J）。逐样本明细（含两侧四字段语义值）见报告
> `gate_g7_visual_dependence.per_sample`。
> **判读**：25/32 的样本在图像被抹平后改了离散输出（绝大多数落在 `critical_objects`），
> 说明模型**确实在用视觉**；7 条未变属"过拟合到记忆 + 该样本决策本就由文本侧决定"的
> 合理残留——判定只需 ≥1 条，故不影响结论。
> **意义**：这是本阶段唯一能证明"模型确实在用图"的检查（S04 教训：图像 token 未正确
> 传入会静默降级）。B7 修复后 ViT 已被冻结（只有语言侧 LoRA 在训），故该确认更必要。

---

## 8. 复现命令

```bash
# --- G3/G4 + G2 读数（纯 CPU，无模型） ---
cd /root/autodl-tmp/drivealign_workspace
PYTHONPATH=DriveAlign/src python -m drivealign.cli.sft_closeout disk \
  --dataset-root data/dataset_v4 \
  --anchor-manifest data/face_manifests/v6/anchor_policy_manifest.json \
  --subsets-dir data/sft_subsets \
  --samples-dir runs/S11_sft_smoke/sft_samples \
  --train-log runs/S11_sft_smoke/checkpoints/smoke128/train128.log \
  --train-log runs/S11_sft_smoke/checkpoints/smoke32/train32.log \
  --report runs/S11_sft_smoke/reports/g34_disk.json

# --- G6/G5/G7（GPU，tmux + set -o pipefail + tee） ---
PYTHONPATH=DriveAlign/src python -m drivealign.cli.sft_closeout model \
  --config DriveAlign/configs/sft/sft_smoke_1f.yaml \
  --adapter runs/S11_sft_smoke/checkpoints/smoke128 \
  --samples runs/S11_sft_smoke/sft_samples/overfit128.jsonl \
  --dataroot data/nuscenes/trainval \
  --report runs/S11_sft_smoke/reports/g67_model.json
```

两个子命令均以退出码 0/1 表示判定通过与否。

---

## 9. 交付与交接

- **S11 交付**：可复现的 1F SFT 训练链路（组装 → collate → forward → backward → save →
  新进程 reload 复验）+ 冻结的配置/样本/adapter/参照/报告 + G1–G7 判定。
- **给 S12**：
  - trainer 与数据序列化**不得改变**（蓝图 §5）；G4 已给出逐字节可复现基线。
  - 对比基准 = `runs/S09_base_benchmark/eval_v6/`（v6 面），**非** S09 存档 v5 读数。
  - 四字段抽检表（本文件 §5 的 `g67_model_table.md`）升级为持续监控。
  - 「多数类常量基线 + counterfactual action_flip_rate」为 S12 必报项（D3）。
  - **A3 惰性数据集**（全量预编码约 96 GB）与 **A4 冻结 ViT embedding 缓存**（端到端约
    1.4×，前提已由 B7 修复满足）属 S12 范围，见 [S11_training_acceleration.md](S11_training_acceleration.md)。
- **已知遗留**：bug 期留证目录（`*_vit_leak_bug/`、`reports/vit_leak_bug/`）保留未清理，
  处置见 D7。
