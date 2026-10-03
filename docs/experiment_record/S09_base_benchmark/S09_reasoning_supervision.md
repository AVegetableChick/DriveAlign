# S09 备案：reasoning 监督方式选项（S11/S12 设计输入）

> 状态：**备案**（2026-10-03），不改变任何 S09/S11/S12 v1 计划；路线 2 为用户意向，待 S11 规划正式拍板。
> 关联：`S09_metric_spec.md` §2.5（Eval v1 不评 reasoning）、S08 GT 资产（模板 reasoning）、S11 SFT 数据规划、S12 gate。

## 背景

pilot/全量产物显示：Base 未微调模型的 reasoning 输出格式自由、内容多样。而 S08 回填的 GT reasoning 是模板生成（3–5 变体、`hash(sample_token + segment salt)` 确定性选句、感知→风险→决策固定叙事序、枚举锚定、无距离数字），信息熵低。SFT 若逐 token 监督模板 reasoning，存在"模板塌缩"风险——reasoning 沦为填空回声。监督方式因此成为一个真实的设计权衡。

## 四条路线

| # | 路线 | 优点 | 代价/风险 | 状态 |
|---|---|---|---|---|
| 1 | 模板 reasoning 监督（GT 现状，逐 token 计算 loss） | reasoning↔GT 字段双向一致可验；监督信号干净；3B 模型 SFT 对监督噪声敏感，模板是最稳的起点 | 模板塌缩：信息量趋零，"解释"能力退化 | GT 资产形态（S08 冻结），默认监督源 |
| 2 | **不监督 reasoning（loss masking）** | 模板梯度为零，从根上杜绝模板塌缩；reasoning 保持 Base 自由风格；实现近零成本——collator 对 reasoning 值域 token 置 loss 权重 0，schema 不变、模型推理时仍输出 reasoning 字段 | reasoning 与结构化字段的一致性**无监督约束**，推理时可能出现文本与 `speed_action`/`yield_required` 矛盾的输出；reasoning 质量只能人工抽检 | **用户意向（2026-10-03）**，待 S11 规划拍板 |
| 3 | 自蒸馏 rejection sampling（STaR 式） | 兼得流畅性与可验证性：对每帧采样多条自由 reasoning，仅保留结构化字段与 GT 逐项一致的样本做监督（保留条件机械可查） | 需额外采样轮与过滤管线，v2 工作量 | 未来方向 |
| 4 | DriveLM 增强包（人工/大模型自由文本） | 语言多样性最高 | 覆盖 14.5%、~93 QA/帧需重过滤、与规则 GT 冲突需清洗；增强包当前关闭 | 备选 |

## 与评测侧的衔接

- Eval v1 对 reasoning 不设任何自动指标（`S09_metric_spec.md` §2.5），四种路线切换**均不影响 benchmark 与 gate**；
- S09 全量 predictions 已将 Base 自由 reasoning 全量留档（`raw_text` 字段），"自由性"基线已保存；
- 若未来要评 reasoning 一致性/多样性 → eval_v2，候选诊断：reasoning↔结构化字段一致率、distinct-n 多样性。

## 待办（S11 规划拍板时明确）

若选路线 2，S11 文档需定义 loss masking 实现细节：
1. mask 范围 = reasoning **值域 token**（JSON 内 `"reasoning": "..."` 的字符串内容），JSON 结构键与其余四字段（critical_objects/risk_factors/yield_required/speed_action）照常监督；
2. collator 实现方式与单测（mask 边界的 tokenizer 偏移正确性）；
3. S12 人工抽检方案（reasoning↔结构化字段矛盾率抽检表）。
