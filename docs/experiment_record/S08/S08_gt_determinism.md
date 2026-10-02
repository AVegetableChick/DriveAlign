# S08 GT 字段确定性原则与代码审查记录（2026-10-02）

## 0. 目的与结论

S08 的 GT 完全由程序生成，**确定性是 GT 可信的前提**：同一帧数据 + 同一冻结配置，在任何时间、任何机器上重跑，五个训练目标字段必须逐字节相同。本文档逐字段列出确定性原则（"这个值是怎么唯一算出来的"），并记录 2026-10-02 的逐文件代码审查结论。

**审查结论：全链路 PASS，未发现任何破坏确定性的构造。**

| 字段 | 确定性来源 | 审查结果 |
|---|---|---|
| critical_objects | 纯几何筛选 + 显式排序（含 token 平手裁决） | PASS |
| motion_state | 纯阈值比较，四分支互斥完备 | PASS |
| risk_factors | 逐项封闭规则 + 封闭词表序输出 | PASS |
| speed_action | 纯 min/max 比较 + 固定优先级 | PASS |
| yield_required | corridor 族成员测试 | PASS |
| reasoning | sha256 定句 + 模板填充 + 双向对应 gate | PASS |

## 1. 六条全局确定性原则

整条 GT 生成链（`gt/backfill.py` → `pool` / `motion_state` / `risk` / `action_yield` / `render`）遵守：

1. **纯函数链**：全部派生函数只依赖显式入参（t0 状态、冻结配置、未来帧证据），无全局可变状态，无相互依赖的隐藏输入。
2. **三无**：无随机数、无系统时钟、无 Python 内置 `hash()`（内置 hash 对 str 受 `PYTHONHASHSEED` 影响，跨进程不稳定——已 grep 确认 gt 包零调用）。
3. **封闭序输出**：任何"收集到若干命中再输出"的字段，一律按预定义封闭顺序发射（risk_factors 按 `RISK_TAXONOMY_ORDER`），内部集合只做成员判断，绝不让 set 迭代顺序泄漏到输出。
4. **显式平手裁决**：所有"取前 N"的排序都带唯一 token 决胜键（`(round(distance_m, 3), ann_token)`），距离平手结果仍唯一。
5. **数值舍入边界**：所有序列化速度/距离在写入前 `round(., 3)`；比较在舍入后的值上进行（浮点噪声不进入标签判定与 canonical hash）。
6. **单一配置源**：全部阈值只从冻结的 `gt_rule_config.json` / `enum_phrase_map.json` / 模板库读取（JSON 文件序稳定），代码零硬编码数值。

## 2. 逐字段确定性规则

### 2.1 critical_objects（模型关注哪些对象）

```
可见性门控（投影几何纯函数，零面积 box 归 degenerate）
  → 幸存者 ∧（前向锥 30° ∪ 近距盘 15 m）∧ 类别距离上限（45/30/15 m）
  → 按 (round(距离, 3), ann_token) 升序取前 8
```

确定性保证：输入顺序无关（排序键含唯一 token）；MAX_POOL_SIZE=8 是契约常量。截断丢弃的是排序尾部的第 9+ 名，截断后果已量化并豁免（漏检帧 0.111%，见 [S08_parameter_freeze.md](./S08_parameter_freeze.md) §3）。

### 2.2 motion_state（四值运动标签）

纯比较链：速度 < 0.5 m/s → `stationary`；速度方向与本车航向夹角 θ ∈ [45°, 135°] → `crossing`；θ > 135° → `oncoming`；否则 → `same_direction`。四分支互斥且完备，任何输入有且仅有一个输出。输入角度由 `atan2` 从标注速度向量计算（同输入同输出）。

### 2.3 risk_factors（七项封闭词表）

每项一条独立规则，全部只消费池内对象 + ego t0 状态（visibility-first）：

| 项 | 判定规则 | 确定性要点 |
|---|---|---|
| pedestrian_crossing | 行人 motion_state=crossing 且反事实外推与走廊最近距 < 2.0 m（3 s / 0.25 s 步长采样） | 采样点数固定（n_steps=round(T/step)），外推纯算术 |
| vehicle_merging | 车辆类同上（d_risk 3.0 m） | 同上 |
| corridor_conflict | 池内任意对象反事实最近距 < 类别 d_risk | 同上 |
| lead_vehicle_braking | 本车道内最近前车窗口速度 min ≤ t0 速度 − 0.5 m/s | leads 按 (距离, token) 排序；命中即 break，与遍历起点无关 |
| stationary_obstacle | 本车道内 t0 与窗口全速度 ≤ 0.5 m/s 的对象 | 同上，pool 有序 |
| congestion | 本车道内 ≥ 3 辆车且最近一辆窗口速度 max ≤ 2.0 m/s | "最近一辆"= leads[0]，排序唯一 |
| oncoming_traffic | 车辆类 motion_state=oncoming 且距离 < 20 m | pool 有序，break |

输出装配：`tuple(r for r in RISK_TAXONOMY_ORDER if r in hits)` —— 封闭词表序，与命中收集顺序无关。

### 2.4 speed_action（四值动作标签）

由 ego **真实未来轨迹**反推（行为证据，非反事实），窗口 3.0 s = 6 帧，与 `oracle_only.future_ego_poses` 对齐：

1. 未来为空（场景末尾截断）→ `KEEP_SPEED`（规则兜底，可计数）
2. 窗口 min < 0.5 m/s → `STOP`
3. 窗口 min < 0.85 × v₀（δ=0.15，**相对值**）→ `DECELERATE`
4. 窗口 max > 1.15 × v₀ → `ACCELERATE`
5. 否则 → `KEEP_SPEED`

确定性保证：速度序列按未来帧 `next` 链时序构建（顺序固定）、逐段 round 3；min/max 与优先级链是纯比较，输入唯一则标签唯一。

### 2.5 yield_required（是否让行）

`any(r in CORRIDOR_FAMILY for r in risk_factors)` —— corridor 三项（corridor_conflict / pedestrian_crossing / vehicle_merging）任一成立即 True。纯成员测试；刻意不混入"ego 实际是否让了"的真实行为（自我抹除陷阱，P1.6 证据切分决策）。

### 2.6 reasoning（推理文本）

- 分段选句：`sha256("{sample_token}:{salt}")` 十六进制摘要 → 对 3–5 个变体取模。**hashlib 与进程/机器无关**，同 token 同 salt 永远同句。
- 模板填充只消费 GT 值（critical_objects、risk_factors、speed_action、yield_required）——上游确定则文本确定。
- 落盘前过**双向对应 gate**：reasoning 提及的风险与 risk_factors 逐项双向核对（遗漏/多余都报错，集合先 sorted 再比较），加上零矛盾 gate（reasoning 不得提及池外对象、交通灯不得出现）——不通过直接抛异常拒绝出标签，不存在"带病落地"。
- 空态（零对象/零风险）走固定 empty_variants 句式，同样经 sha256 选句。

## 3. 反模式清单（代码中确认不存在）

审查时逐项核对 gt 包（2026-10-02，覆盖 `config/observability/pool/motion_state/risk/action_yield/render/backfill` 八个模块）：

- 内置 `hash()` —— 零调用（grep 证据）
- `random` / `time.time()` / `datetime.now()` —— 零 import
- set/dict 迭代直接决定输出顺序 —— 零处（所有输出序 = 封闭序或显式 sort）
- 依赖文件系统遍历顺序（`os.listdir` 等）—— 零处（nuScenes 表序由 JSON 文件保证，devkit `get()` 幂等）
- 未舍入浮点参与标签判定或序列化 —— 零处（速度/距离统一 round 3）
- 数值硬编码 —— 零处（全部经 `load_rule_config` 读冻结 JSON）

## 4. 验证方式（原则之外的第二道保险）

1. **单测层**：`tests/unit/test_gt_backfill.py` 含重建逐字节一致性测试（同输入两次 backfill，JSON 逐字节相同）；`test_gt_pool.py` 含平手裁决测试（距离相同按 token 决胜）。
2. **冒烟层**：标定与截断验证 CLI 均在真实 trainval 上双跑过，汇总计数一致。
3. **全量层（Step 5 将执行）**：v4 数据集 gates —— pinned-commit 确定性重跑逐字节比对（v1 先例：38 文件全同）、request_hash parity、GT 100% 填全、零矛盾 gate 全量通过。数据集 manifest 与 record hash 均基于舍入后的 canonical 字段，跨机器可复算。

## 5. 边界声明

确定性保证的前提是三个输入也冻结：**冻结配置**（gt_rule_config.json `status: frozen`）、**冻结数据**（nuScenes 表只读）、**冻结代码**（pinned builder commit）。任一变动（如重开标定、升级 devkit）都会改变输出——这不是缺陷而是设计：变化必须走 contract 版本升级与台账记录，不允许静默漂移。
