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

## 阶段关闭核对（待 Step 5 填写）

- [ ] 参数冻结表（SFT 关键参数与 sha）
- [ ] Step 2/3/4 交付物 sha 登记
- [ ] G1–G7 判定结果
- [ ] 双面标注（S09 存档 v5 vs eval_v6）