# S05 导航输入设计要点（Stage 05 准备阶段的澄清记录）

## 1. 背景

在开始 Stage 05（AutoVLA 原生多帧预处理 Reference Smoke）前，对"导航输入（driving command）是否进入 DriveAlign"做了澄清。本记录固化结论，避免在 Stage 06 复用上游 preprocessing 时误把 future-derived command 纳入模型输入。

## 2. 核心结论

- **导航指令是可选/按需的因果输入**，不是"当前不用、未来再加"的简单开关。关键不在"何时加"，而在于"永远只能走因果来源路径"。
- **当前任务不需要导航输入**：DriveAlign 输出域是 `critical_objects / risk_factors / reasoning / yield_required / speed_action`（纵向速度动作），属于感知+风险+纵向车控任务。导航意图（第几个路口转弯、横向去哪）与这些输出无直接贡献，反而引入未来信息。
- **未来引入是有条件的**：当任务域扩展到横向轨迹/意图（trajectory 阶段，Stage 13/17–20）并且路线意图在决策时刻因果已知时，导航才作为**因果输入**按需加入。它应是**独立设计的因果导航信号**，绝不能复用 AutoVLA 的 future-derived command。

## 3. AutoVLA 的 driving command 为何被视为 future-derived / Reference-only

- AutoVLA preprocessing 从 **GT 未来自车轨迹**反推 command（读历史/当前/未来 pose 判定"这段路径是直行/转弯"），在运行时物理上不可知。
- 按项目硬约束第 2 条：未来轨迹、未来对象、future-derived command **只能进入 target/reward/evaluation oracle**，不能进 `model_inputs`。
- 因此它在 Stage 05 的标签为 **Reference only**，仅作流程参考，不自动转正为 DriveAlign 实验输入。
- 注意区分：command 本身（导航意图）不违法，违法的是"由未来轨迹计算得到的来源"。

## 4. 阶段对照

| 阶段 | 任务域 | 是否需导航输入 | 来源要求 |
|---|---|---|---|
| Stage 05–12（SFT / DPO / GRPO） | 感知 + 风险 + 纵向 speed | 不需要 | 上游 future command 标 Reference-only |
| Stage 13 / 17–20（trajectory / 动作） | 横向轨迹 + 意图 | 按需可选 | 仅因果路线信号，禁止未来轨迹反推 |

## 5. 落地方向

- Stage 05：原样跑 AutoVLA preprocessing，在 `references/autovla_preprocessing.md` 中将 future-derived command 单独登记为"来源=future trajectory → 判定=违规 → 处置=只进 target/oracle 或丢弃"。
- Stage 06：`build_record` 复用 CAM_FRONT 四帧 + processor/trainer 契约时，只挑因果合法字段进 `model_inputs`，导航指令不进入。
- 未来：如需要导航输入，独立设计因果导航信号（决策时刻已知的路由意图），不作为本阶段的扩展点。