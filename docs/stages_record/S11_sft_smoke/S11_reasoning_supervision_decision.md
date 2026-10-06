# S11 讨论总结：reasoning 监督方式的裁定链（2026-10-05）

> 本文档记录 S11 规划期间关于"模板 reasoning 字段如何监督"的多轮讨论，终点裁定：**模型面升版 v6，删除 reasoning 字段**（执行见 [S11_step0_v6_face_transition.md](S11_step0_v6_face_transition.md)）。
> 讨论起点是 [S09_reasoning_supervision.md](../S09_base_benchmark/S09_reasoning_supervision.md) 预注册的路线 2（loss masking，"待 S11 规划确认"）；本文档即为该确认过程，结论是取代而非执行路线 2。closure note 于 Step 0.4 落入备案文档。

## 1. 讨论轮次（时间序）

### R1 开题裁定：路线 2 三段式 mask + 机制澄清

S11 规划开题时按备案预注册选择路线 2：target 按 JSON 序列化序切三段，`"reasoning": "..."` 的**字符串值域 token** 置 -100，JSON 结构键与其余四字段照常监督。同时确认两个语义要点：

1. **mask 移除的是 B 段自身 NLL，不是条件通路**——causal attention 下 action 字段的预测条件于 reasoning 段 hidden states，其梯度仍流经 B；"reasoning 计入上下文"在训练期的作用恰是"抄答案通路的条件源"，不承载正面监督信号；
2. 推理期 B 段是无训练信号的自由文本 → 预判两项工程风险：JSON 安全（parse 失败）与长度失控（512 预算截断）。

### R2 工程修正：字节级断言 gate（用户质疑触发）

用户质疑三段拼装（A/C 分别 dump 再切片）能否保证拼接为完整 JSON。结论：文本层面成立（JSON 序列化逐值上下文无关 + 键序由插入序保持），但**正确性不应靠论证**——改为构建期断言：`seg_a + reasoning + seg_c == json.dumps(expected_output)`（逐字节等于整串 dump）+ `json.loads` 语义 round-trip，替代原字符黑名单方案（黑名单对 `\t` 等控制字符有漏网风险）。三段拼装的唯一目的收窄为在 token 层提供精确 mask 边界，避开 BPE 在引号边界的偏移映射不稳定。

### R3 用户质疑 mask 有效性：确认训推分布不一致是路线 2 的新增缺陷

用户提出根本性质疑："mask 真的解决了吗？我甚至怀疑反而导致训推分布不一致。"逐层检验结论：

1. **原始质疑（备案否决路线 1）的理由**：模板 reasoning 是标签的确定函数（决策段与 action GT 值逐字锚定）→ ①捷径强化（action 头走"从 B 决策句抄答案"的最低阻力通路）；②归因污染（提升无法区分"学会看图"与"学会写范文+抄答案"）；③模板塌缩（表象级）。
2. **mask 只解决生产侧**：C 段的监督项与梯度压力在路线 1/2 完全相同（C 的 loss 一样、B 作为输入一样），条件通路上的泄漏强化原封未动——mask ≠ 切断通路。
3. **训推分布不一致成立**：路线 1 的错配是信息性的（推理期 B 流的是信念而非标签，表面分布 on-policy）；路线 2 叠加**分布性**错配（B 的生成分布从未被训练，action 头条件于 OOD 自由文本）。备案把错配当 feature（"打断捷径"）的论证不成立：给捷径输入注入不受控噪声是让 action 条件于 OOD，是失稳不是修复。
4. **判据重构**：同时满足"去泄漏 + on-policy"才有资格入选。五方案对比：路线 1（泄漏✗/on-policy✓）、路线 2（泄漏✗/on-policy✗/parse 风险高——三轴最差）、α 常量 stub（三项全过）、γ 证据型渲染（三项全过）、β 重排（三项全过）。

### R4 用户质疑 γ：影子 GT 定性，γ 出局

用户定性："γ 实际上相当于在 reasoning 模板中取出 action 相关内容，随后重构数据集。"确认成立：γ 让训练目标的 reasoning 内容 ≠ 冻结记录内容，属于**发明新 GT 内容**——要么走 contract 代际 + 数据集重跑（重活），要么形成训练侧"影子 GT"（账务分叉，可追溯性退化）。γ 出局。同轮确认 β 的关键洞见：**record canonical hash 为 sorted-keys JSON，字段序从来不是记录身份的一部分**，发射顺序是训练侧自由度（此洞见后来融入 v6 裁定）。

### R5 数据观感：模板塌缩的内容级实证，α 升主方案

用户查看 `data/dataset_v4/train/shard-0000.jsonl` 原始记录后判定模板 reasoning"驴唇不对马嘴"，不愿监督模型生成此类文本（担心降低推理能力），但仍需监督完整 JSON。分析：模板 reasoning 三段全部是确定性再渲染（感知/风险段=前文字段同义复读，决策段=action GT 同义复读），千帧一面；联立"不监督内容 + 必须完整 JSON"两条约束，**mask 无解**（未训练槽位推理期不可控 → JSON 安全无保障），唯一解为 **α 常量 stub**（reasoning 值 = 冻结常量占位，如 "See structured fields above."，全段监督）。α 升主方案。

### R6 两条路径厘清：v5+stub vs v6 删字段

用户问：v5 不变 + stub，或改 contract 彻底删字段？厘清：路径 2 的改动范围**不在数据集**（records 零改动，训练侧省略字段属于序列化选择），而在模型面/评测级联：prompt 重写 → request_hash 全变 → 新 manifest → parser v6 → **Base 重跑**（对比纪律：Base 与 SFT 必须同面）。当时建议推迟 v6 至 S12 后。

### R7 最终裁定：v6 现在（用户拍板）

用户裁定："v6 现在开始才是亡羊补牢为时未晚，等 v5 牵扯环节多起来再改就难了；现在就改 v6 + 重跑 1F Base，后面就顺畅了。"确认该判断优于"推迟"建议：推迟论低估了纠缠增长（SFT checkpoint 将成为 v5 面资产，之后改 = 重训或双面记账）与高估了重跑成本（纯 GPU 时，v6 输出更短反而更快）；**当前 S11 未开工，唯一沉没资产是 S09 v5 存档（作为 v5 面结果依然有效）**——现在确实是过渡的最便宜时刻。v5+stub 被取代，路线定为 v6。

### R8 执行修订（用户）

Step 0 顺序调整为：0.1（面定义）→ 0.2（manifest+parity）→ SFT Step 1 探索优先 → 0.3（Base v6 重跑，用户夜间 GPU）→ 0.4（冻结落档）。理由：Base 重跑耗时长，不阻塞 SFT 实现探索。

## 2. 被否方案台账（防未来重新翻案，每条含否决理由）

| 方案 | 内容 | 否决理由 | 轮次 |
|---|---|---|---|
| 路线 1 监督模板 | reasoning 照常监督（原序） | 捷径强化 + 归因污染（S09 备案机制分析，未被推翻） | R3 |
| 路线 2 mask | reasoning 置 -100（原序） | 条件通路泄漏仍在 + 分布性训推错配（OOD 条件）+ 无法保证完整 JSON | R3/R5 |
| γ 证据型渲染 | 感知+风险段重渲染（去决策段）并监督 | 影子 GT（发明数据集中不存在的内容）或需 v6 级数据集账务 | R4 |
| β 重排 | 冻结 reasoning 逐字保留、移至 JSON 末位并监督 | 技术成立，但 R5"不监督模板内容"约束下失去对象；"字段序=训练侧自由度"洞见被 v6 吸收 | R5 |
| α 常量 stub | reasoning 值=冻结常量，全段监督（v5 面内） | 曾为主方案；被 v6 取代——保留死槽位的边际收益 < 纠缠成本 | R7 |
| 推迟 v6 至 S12 后 | 先 v5+stub 跑通再过渡 | 纠缠成本单调增长，SFT 资产将锁死 v5 面；当前沉没资产最少 | R7 |

## 3. 最终裁定的完整理由（一句话版）

模板 reasoning 是标签的同义再渲染，对模型零信息量且携带捷径/归因风险；监督它没有任何一条成立的收益；保留该字段（无论 mask 还是 stub）都让模型面携带一个永不产出真实内容的死槽位；在其牵扯面扩大之前删除（v6 + Base 重跑）是总成本最低的时刻。

## 4. 影响范围

| 改动 | 不改动 |
|---|---|
| `configs/contracts/v6/`（prompt + output_schema 删 reasoning） | dataset_v4 records（逐字节冻结，reasoning 留作 GT provenance） |
| `contracts/versions.py`（v6 注册 + DEFAULT 切换） | v1–v5 contract 目录与 S09 v5 存档结果 |
| v6-face anchor manifest（派生进 runs/） | 锚集、评测指标定义、GT 侧资产（含 reasoning_templates.json） |
| parser/schema 校验（4 字段） | S08/S09 全部文档历史（补面标签即可） |
| 1F Base 重跑（含 blank/shuffled 反事实） | 评测脚手架结构（仅面引用升级） |

## 5. 遗留边界声明

未来若需要模型输出真实场景 reasoning（CoT 价值），那是独立的数据质量工程：需要 scene-grounded 文本来源（VLM caption / 人工标注），以新 contract 版本走正门回填；模板渲染方案不应复活。S12 预测的输出中将不再有 reasoning 字段，相关抽检/归因表述需同步更新。
