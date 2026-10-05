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

## 阶段关闭核对（待 Step 5 填写）

- [ ] 参数冻结表（SFT 关键参数与 sha）
- [ ] Step 2/3/4 交付物 sha 登记
- [ ] G1–G7 判定结果
- [ ] 双面标注（S09 存档 v5 vs eval_v6）