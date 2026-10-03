# S09 备案：reasoning 监督方式选项（S11/S12 设计输入）

> 状态：**备案**（2026-10-03），不改变任何 S09/S11/S12 v1 计划；路线 2 为**推荐路线**（机制分析见下节），待 S11 规划确认。
> 关联：`S09_metric_spec.md` §2.5（Eval v1 不评 reasoning）、S08 GT 资产（模板 reasoning）、S11 SFT 数据规划、S12 gate。

## 背景

pilot/全量产物显示：Base 未微调模型的 reasoning 输出格式自由、内容多样。而 S08 回填的 GT reasoning 是模板生成（3–5 变体、`hash(sample_token + segment salt)` 确定性选句、感知→风险→决策固定叙事序、枚举锚定、无距离数字），信息熵低。SFT 若逐 token 监督模板 reasoning，存在"模板塌缩"风险——reasoning 沦为填空回声。监督方式因此成为一个真实的设计权衡。

## 关键机制：模板监督的代价不在训练，在推理错配与归因污染

训练环节没有任何问题（teacher forcing 下上下文里的 reasoning 永远是 GT 模板，loss 完美收敛）。问题出在推理与评测归因两个环节：

```
训练:  图像 → [GT模板 reasoning] → action     reasoning = 标签的函数（同义编码），永远正确
推理:  图像 → [模型自写的 reasoning] → action  reasoning = 模型自身信念的函数
```

1. **推理错配**：schema 字段序中 reasoning 在 action 之前，且模板决策段与 `speed_action`/`yield_required` GT 值锚定（S08 硬约束）——训练时 action 头走"从上下文 reasoning 决策句抄答案"的最低阻力通路并被强化。推理时该通道里流的不再是标签信息而是模型自身视觉信念：图看对 → reasoning 对 → action 对；图看错 → 错误信念被 verbalize 成笃定承诺后照抄执行。
2. **归因污染**：Base→SFT 的结构化指标提升 = "学会看图" + "学会写范文+抄答案的格式机制"两部分混合，M09 指标无法区分。若提升主要来自后者，S12"更会看图"的核心结论不成立，而报表上看不出来。

普通 CoT 数据（人类解题过程）是**题目的函数**而非**答案的函数**，训练/推理时 reasoning 的信息性质一致，因此无此问题；这里的模板 reasoning 是**标签的确定函数**，才构成捷径。

## 四条路线

| # | 路线 | 优点 | 代价/风险 | 状态 |
|---|---|---|---|---|
| 1 | 模板 reasoning 监督（GT 现状，逐 token 计算 loss） | reasoning↔GT 字段双向一致可验；监督信号干净；训练期零异常 | **推理错配 + 归因污染**（机制见上节）：action 头在训练时被优化为"从 reasoning 决策句抄答案"（模板=标签的同义编码），推理时该通道流的是模型自身信念，错误视觉信念被 verbalize 成笃定承诺后照抄执行；S12 结构化指标提升 = 看图学习 + 格式机制学习的混合，M09 无法区分。模板塌缩（信息量趋零）只是表象 | GT 资产形态（S08 冻结），默认监督源 |
| 2 | **不监督 reasoning（loss masking）** | 模板梯度为零，从根上杜绝模板塌缩；reasoning 保持 Base 自由风格；实现近零成本——collator 对 reasoning 值域 token 置 loss 权重 0，schema 不变、模型推理时仍输出 reasoning 字段；train/infer 错配（action 头推理时看到 OOD 自由文本）反而**打断了抄答案捷径**，迫使 action 依赖图像与被监督的结构化字段，S12 归因干净 | reasoning 与结构化字段的一致性**无监督约束**，推理时可能出现文本与 `speed_action`/`yield_required` 矛盾的输出；reasoning 质量只能人工抽检 | **推荐路线（2026-10-03）**，待 S11 规划确认 |
| 3 | 自蒸馏 rejection sampling（STaR 式） | 兼得流畅性与可验证性：对每帧采样多条自由 reasoning，仅保留结构化字段与 GT 逐项一致的样本做监督（保留条件机械可查） | 需额外采样轮与过滤管线，v2 工作量 | 未来方向 |
| 4 | DriveLM 增强包（人工/大模型自由文本） | 语言多样性最高 | 覆盖 14.5%、~93 QA/帧需重过滤、与规则 GT 冲突需清洗；增强包当前关闭 | 备选 |

## 与评测侧的衔接

- Eval v1 对 reasoning 不设任何自动指标（`S09_metric_spec.md` §2.5），四种路线切换**均不影响 benchmark 与 gate**；
- S09 全量 predictions 已将 Base 自由 reasoning 全量留档（`raw_text` 字段），"自由性"基线已保存；
- 若未来要评 reasoning 一致性/多样性 → eval_v2，候选诊断：reasoning↔结构化字段一致率、distinct-n 多样性；
- 可证伪检验（针对路线 1 的归因质疑）：对路线 1 训出的模型跑 blank-image 反事实——若提升主要来自"写范文+抄答案"的格式机制，blank 图下仍会输出笃定的 reasoning 与 action（纯先验驱动），visual dependence 读数会异常地低。

## 待办（S11 规划拍板时明确）

若选路线 2（推荐），S11 文档需定义 loss masking 实现细节：
1. mask 范围 = reasoning **值域 token**（JSON 内 `"reasoning": "..."` 的字符串内容），JSON 结构键与其余四字段（critical_objects/risk_factors/yield_required/speed_action）照常监督；
2. collator 实现方式与单测（mask 边界的 tokenizer 偏移正确性）；
3. S12 人工抽检方案（reasoning↔结构化字段矛盾率抽检表）。
