# S13 Contract v7 — Risk Factors 判定方法详述（审查稿 v2.4）

- 状态：**审查稿 v2.4**（P-1 ~ P-8 全部确认；§1–6 与 [risk.py](../../../src/drivealign/gt/risk.py) / [motion_state.py](../../../src/drivealign/gt/motion_state.py) 实现逐行对齐。v2.1 清晰度修订：d_k=0 边界、θ 坐标系约定、braking"任一前车"语义、stationary 回退与 braking 跳过的不对称理由、oncoming 静止排除。v2.2 新增 P-6：braking / stationary 在路径条件统一为 x_c > 0 ∧ \|y_c\| ≤ Y，Y_brake < Y_st 标定，删除 static 的 firing 专设分支。v2.3 新增 §12 风险因子归因（决策台账 V7-A1 / D-A1-3：全量合格对象口径，§1–§6 命中判据不动）。**v2.4（2026-10-12）：§0.7 crossing 判据重写为到达时间口径并与 firing 解耦**（台账 V7-R1 §3.7）：两族同公式 `v_toward > 0.8 ∧ t_arrive ≤ 5.0`（`t_arrive = (|y_c| − 1.185)/v_toward`，带内记 0）；firing 引擎本体不动、只喂 yield 并集（§0.5–0.7）；config 新增键 `crossing_gate.t_arrive_max_s = 5.0`，`lateral_thr_mps` 角色变为 crossing 速度门槛）
- 决策依据：[S13_contract_v7_decision_log.md](./S13_contract_v7_decision_log.md) §3（R-OPT-1 ~ R-OPT-6 + 2026-10-11 审查确认 + 2026-10-12 crossing 到达时间修订 §3.7）
- 参数基线：[configs/contracts/v6/gt_rule_config.json](../../configs/contracts/v6/gt_rule_config.json)（下称 v6-config）；v7 变更处以 **[v7]** 标注
- 单位：距离 m，角度 deg，时间 s

---

## 0. 共用机制（走廊族因子与 yield 共用）

### 0.1 对象池（pool，v7 不变，仅分组调整）

入池条件（v6-config `pool.*`）：

1. 可见性门控：深度 ≥ `min_depth_m = 0.1`（可见性优先原则）；
2. 几何候选：ego 前向锥（半角 30°）内且距离 ≤ 组上限，**或** 近距盘 15m 内；
3. 组距离上限：vehicle 45 / pedestrian 30 / static 15；
4. 相关度排序取 **top-8**。

**[v7] 分组调整（P-1 已确认）**：类别组拆为

```text
vru      = [pedestrian, bicycle, motorcycle]     # 标签与 buffer 分组用
vehicle  = [car, truck, bus, trailer, construction_vehicle]
static   = [barrier, traffic_cone]
```

锥形距离上限：vru 组内 pedestrian 30m；**bicycle / motorcycle 保持 45m**（per-class 覆盖，速度特性接近车辆）。

对象字段：`ann_token`、类别、t0 全局中心 (x, y)、t0 速度向量 (vx, vy)、`distance_m`（**t0 传感器系中心距的 3D 范数**）、方位角、标注框尺寸 (L, W)、t0 朝向 θ_obj、未来轨迹 track、motion_state。

### 0.2 坐标系

ego 系：原点 = ego t0 位置，x 轴 = ego t0 航向前向，y 轴 = 左。对象中心 ego 系坐标记 (x_c, y_c)。**"前方"判定 = x_c > 0**（相对 ego 中心，非车头）。

### 0.3 ego 矩形与扫掠带

- ego 矩形：4.1 × **1.87** m（ego 车宽 1.87；2.37 是含 margin 的带宽，不是车宽）；
- 膨胀 `width_margin_m = 0.5`（每侧 0.25）→ **判定带半长 2.05m × 半宽 1.185m**；
- 走廊"列内"判据（`in_ego_corridor_column`，v6 基线）：**x_c > 0 且 |y_c| ≤ 1.185**（t0 中心点判据，无纵向 containment——前方 8m 的前车同样在列内）。**[v7] P-6**：braking / stationary 的"在路径"横向半宽参数化为 Y_brake / Y_st（§3 / §4），1.185 保留为扫掠带本身的半宽（crossing 判据 §0.7 的带内边界仍用它）；

### 0.4 采样与双侧 CV 外推

K = 13 个采样时刻 t_k ∈ {0, 0.25, …, 3.0}。相对运动框架下（`corridor_distance_curve` 实现）：

```text
Δp     = p_obj(t0) − p_ego(t0)
Δv     = v_obj(t0) − v_ego(t0)          # 双侧恒速（bilateral CV）
r_k    = Δp + Δv · t_k                  # 对象相对 ego 的 t_k 时刻位置
(local_x, local_y) = R(−ψ) · r_k        # 旋到 ego 系
```

即 ego 与对象各自按 t0 状态匀速直线外推、姿态保持不变——"反事实"= 双方均不反应。

### 0.5 中心距 → 投影修正 → 物理间隙

对每个采样 k：

1. **中心距** `d_k = rect_distance(local_x_k, local_y_k, 2.05, 1.185)`——点到矩形区域的距离（矩形内取 0）；
2. **间隙方向** û_k（仅当 d_k > 0 时需要）：从对象中心指向带矩形最近点的单位向量（决定 h 的投影方向，**随几何变化**：带侧方→横向，带远端前方→纵向，斜对角→斜向）；
3. **投影半延展**（支撑函数）：

   ```text
   h_k = 0.5·L·|cos(θ_obj − θ_ûk)| + 0.5·W·|sin(θ_obj − θ_ûk)|
   ```

   其中 θ_obj = 对象 t0 朝向角，θ_ûk = û_k 的方向角，两者取**同一坐标系**（全局系或 ego 系均可，公式只用差值）。几何含义：车身沿间隙方向伸出的长度。垂直横穿→半长主导；平行并入→半宽主导；斜向→前外角主导（连续插值，最大值 = 半对角线 @ Δθ ≈ atan(W/L)）。
4. **物理间隙** `clearance_k = d_k − h_k`；**若 d_k = 0（对象中心已在带内）则无需 û_k 与 h_k，该样本直接计为 firing**（中心已在扫掠带内，间隙必然 ≤ 0 < base）。

**firing [v7]（P-2 已确认：逐样本最小）**：

```text
clearance* = min over k of (d_k − h_k)      # 存在 d_k = 0 的样本时直接为 firing
firing(o)  ⟺ clearance* < base(o)
```

- base：组级标量（标定项，见 §9）；单 base 假设下全组同值，备择假设 base_vru / base_vehicle；分裂假设下 static 归 base_vehicle（v6 先例：`_d_risk_for` 中非 pedestrian 一律 3.0）——**firing 的唯一消费方是 yield 并集（§0.6 / §7）**：v2.4 起两 crossing 因子改用到达时间判据（§0.7）不再消费 firing，static 的 firing 自始即不参与任何因子标签；
- h 由标注框尺寸 + t0 朝向确定性计算，**零标定**。

### 0.6 firing 与 yield 的关系

- **yield_required = ∪{ firing(o) : o ∈ pool }**——与模型面标签完全解耦（P1.6 证据分离）；
- 无标签的 firing（同道接近、沿路行人等）依然贡献 yield=true；
- static 对象 firing 同样喂 yield；
- 注：t=0 是采样点之一——"t0 已贴近"的对象直接贡献 firing。实际影响有限：池受前向可见性约束（§0.1），贴身侧向对象通常不在池内；若未来扩池需重审该性质。

### 0.7 crossing 到达时间判据（vru_crossing / vehicle_crossing 共用；v2.4 重写）

1. 相对速度横向分量（ego 系）：`v_lat = (R(−ψ)·(v_obj − v_ego)).y`；v_ego 无横向分量，故等价于 v_obj 在 ego 系的横向分量；
2. 朝向走廊统一为正：带外（|y_c| > 1.185）`v_toward = −sign(y_c) · v_lat`；**带内边界规则（P-3 已确认）**：|y_c| ≤ 1.185（中心已在带内）时取 `|v_lat|`；
3. **到达时间**：带内 `t_arrive = 0`（中心已在带内，即时到达）；带外 `t_arrive = (|y_c| − 1.185) / v_toward`（v_toward ≤ 0 即远离走廊，视为不可达）；
4. **crossing ⟺ v_toward > thr ∧ t_arrive ≤ T**（config：`crossing_gate.lateral_thr_mps = 0.8`、`crossing_gate.t_arrive_max_s = 5.0`，两族同参数、两族同公式）。

v2.3 及以前的口径是 `firing ∧ v_toward > thr`（统一横向 gate）。v2.4 语义（2026-10-12 拍板，台账 §3.7）：**crossing 与 firing 解耦**——

- 证据类型不同：firing 是"3s 双侧 CV 反事实下物理间隙已不足"，回答"是否已经贴近"；到达时间是"沿当前速度多久进入走廊"的预测，回答"是否正在过来"。横穿但距离较远（全量快照 crossing 对象距离 p50 = 24.5 m）的对象在 3s CV 窗内不 firing，旧口径系统性漏标（vru_crossing 0.31% / vehicle_crossing 1.79%）；改为到达时间后检出升至 4.40% / 10.94%（全量 22,941 帧，thr = 0.8、T = 5s）。
- 反向保护：缓慢漂移（v_toward > 0.8 但 t_arrive > 5s，如远处斜向并行渐近）不再获得标签——这是旧 gate 完全没有的维度。
- firing 引擎本体（§0.4–0.5）与 yield 并集（§0.6 / §7）不动；带内边界 1.185 继续复用扫掠带半宽（§0.3）。

### 0.8 motion_state 判定（P1.4 口径，`classify_motion_state` 实现）

```text
speed < stationary_speed_mps(0.5)            → stationary
θ = |速度方向与 ego 航向的夹角| ∈ [0, 180]：
    45 ≤ θ ≤ 135                             → crossing
    θ > 135                                  → oncoming
    θ < 45                                   → same_direction
```

v7 中 motion_state 仅两处用途：① oncoming_traffic 判据（§6）；② 模型可见特征。**不再参与 crossing 标签 gate**（R-OPT-6）。

---

## 1. vru_crossing **[v7]**

| 项 | 内容 |
|---|---|
| 语义 | 易受伤害道路使用者正侧向进入 ego 路径（横穿、斜穿、擦边切入） |
| 对象族 | vru 组（pedestrian / bicycle / motorcycle） |
| 判定 **[v2.4]** | `crossing 判据(§0.7)`：`v_toward > thr ∧ t_arrive ≤ T`（与 firing 解耦） |

精确计算步骤（对每个池内 vru 对象）：

1. 按 §0.7 计算 (v_toward, t_arrive)（带内 t_arrive = 0；带外 (|y_c| − 1.185)/v_toward）；
2. v_toward > 0.8 ∧ t_arrive ≤ 5.0 → 该对象触发 vru_crossing；任一对象触发即因子置 true。

v6 对照：pedestrian_crossing = ped firing 无 gate、d_risk_ped 2.0；两轮车横穿被误标 vehicle_merging。

行为变化（v2.4 更新）：① 静止行人（v_lat ≈ 0）不进本因子，归属见 §4/§5；② 沿路行走 / 同道接近（v_lat ≈ 0）无标签；③ **与 firing 解耦（v2.4 主变更）**：3s CV 反事实窗外、但 5s 内将进入走廊的横穿对象获得标签（旧口径漏标主体，检出 0.31% → 4.40%）；④ 缓慢漂移（t_arrive > 5s）不再获得标签。

## 2. vehicle_crossing **[v7]**（原 vehicle_merging 改名）

| 项 | 内容 |
|---|---|
| 语义 | 车辆侧向进入 ego 路径（cut-in 与垂直横穿统一纳入） |
| 对象族 | vehicle 组（car / truck / bus / trailer / construction_vehicle） |
| 判定 **[v2.4]** | `crossing 判据(§0.7)`（与 §1 同一公式、同一 thr / T） |

计算步骤同 §1，仅对象族不同。两判定由类别分组天然互斥（同一对象只属一族）。

v6 对照：vehicle_merging = vehicle 族 firing 无 gate；误触发构成：对向通过 46.8%、同道接近 49.5%（stationary 38.0% + same_dir 11.5%）。

行为变化（v2.4 更新）：① 同道接近（v_lat ≈ 0）不获得标签——修复 R-OPT-1 指出的系统性误标；② 垂直横穿的汽车首次获得标签；③ 静止两轮车不进本因子（族归 vru 且 v_lat ≈ 0），由 §4/§5 承接；④ **与 firing 解耦（v2.4 主变更）**：远处横穿 / cut-in 对象（3s CV 窗外）获得标签（检出 1.79% → 10.94%），缓慢漂移（t_arrive > 5s）不再获得标签。

## 3. lead_vehicle_braking（判定逻辑 v7 不变；在路径放宽 + 族范围 P-5 已确认）

| 项 | 内容 |
|---|---|
| 语义 | **存在**列内前车在实际减速（真实动力学证据，非反事实）——按距离升序逐一检查，任一命中即 true；"首个命中即止"只是计算早停，不改变"任一"语义 |
| 在路径 **[v7]** | `x_c > 0 ∧ \|y_c\| ≤ Y_brake`（t0 中心点；Y_brake 标定项，建议网格 [1.185, 2.2]，1.185 = v6 现状） |
| 族范围（P-5 已确认） | 维持原 vehicle 组口径：vehicle ∪ {bicycle, motorcycle}——braking 是动力学证据，不随保护类别改变，对向骑行者减速同样是证据 |

精确计算步骤：

1. **候选"前车"**：池内 (vehicle ∪ {bicycle, motorcycle}) 族对象 ∧ 在路径（x_c > 0 且 |y_c| ≤ Y_brake，t0 中心点）；
2. 按 `(round(distance_m, 3), ann_token)` 升序排序；
3. 依序检查每个前车：
   - 无未来轨迹 track → 跳过；
   - t0 速度 ≤ `motion.stationary_speed_mps(0.5)`（静止"前车"不是制动）→ 跳过；
   - 取未来证据窗（`rel_times ≤ evidence_window_s = 3.0` 的速度序列，`ObjectTrack` 为**未来**窗）；窗内无速度样本 → 跳过；
   - `min(窗速度) ≤ v0 − speed_drop_mps(0.5)` → 命中，**首个命中即止（break）**；
4. 任一前车命中 → 因子置 true。

在路径放宽的动机（P-6）：v6 的 1.185m 半宽按"ego 扫掠带"设计，但前车中心略偏出带（如压线行驶，|y_c| ∈ (1.185, 2.2]）时其车身仍部分覆盖 ego 通道，减速同样构成有效证据；Y_brake 进标定网格，由数据裁决宽度。

## 4. stationary_obstacle **[v7] 在路径条件统一放宽（P-6）**

| 项 | 内容 |
|---|---|
| 语义 | 有东西停在我的路径上（或走廊近旁），现在停且将继续停 |
| 在路径 **[v7]** | `x_c > 0 ∧ \|y_c\| ≤ Y_st`（t0 中心点；Y_st 标定项，建议网格 [1.185, 3.0]，约束 Y_st ≥ Y_brake） |
| 对象族 | **全类别同一判据**（P-6：不再区分可移动 / 不可移动目标） |

精确计算步骤（对每个池内对象，**类别无关**，任一命中即止）：

1. **"停着"证据（两段式）**：
   - a. t0 速度 ≤ `max_speed_mps = 0.5`；不满足 → 跳过；
   - b. 未来证据窗（`rel_times ≤ evidence_window_s = 3.0`）内**最大**速度 ≤ 0.5（窗语义：**将继续停着**）；对象无 track 或窗内无速度样本 → 回退仅用 t0 速度判（即步骤 a）；
2. **"在路径"条件（v7 修改，P-6）**：x_c > 0 且 **|y_c| ≤ Y_st**（t0 中心点）。全类别同一几何判据——**删除原"static 族 = 列内 ∨ firing"的专设放宽分支**，firing 仅服务于 yield 并集（§0.6 / §7；v2.4 起 crossing 亦不消费 firing）；
3. 两条件同时满足 → 因子置 true。

澄清三个易混点（v1 审查意见）：

- 窗口方向：braking / stationary 的证据窗均为**未来窗**（t0 之后 3s），不是历史窗；
- 在路径判据：t0 **中心点**，要求在 ego 前方（x_c > 0）且 |y_c| ≤ Y_st；横向宽度本身就是车宽效应的参数化承担者（不再依赖 firing 的 h）；
- "停着"= t0 停 ∧ 未来 3s 全停（未来窗缺失时退化为仅 t0 判）。

与 §3 的两处不对称（均为有意设计）：

- **横向宽度**：Y_st > Y_brake——减速的邻道车与 ego 弱相关（"前车"身份要求贴近通道），而静止障碍不会自行移动，t0 位置即全部信息；走廊近旁（未入带）的静止物是规划必避的高价值证据，漏检代价（碰撞）远高于误检代价（多一分关注），支持更宽；
- **无 track 处理**：braking 对无 track 的对象**跳过**，本因子**回退仅用 t0 判**——减速是需正向证据才成立的事件，而"保持停止"是缺省状态，无未来证据时合理默认其延续（实现 [risk.py L320-325](../../../src/drivealign/gt/risk.py#L320-L325) 的 else 分支即此回退）。

v6 实测基线：9.0% 检出。Y_st 放宽后：检出上升幅度、以及 v6 corridor_conflict 的 1,720 帧 static 残余被 Y_st 覆盖的比例，均为标定必报项（Y_st = 2.5 时纯横向几何约覆盖 \|y_c\| ∈ (1.185, 2.5] 的残余，v6 firing 判据下该残余的 \|y_c\| 上界约 4.2–4.7m，覆盖率由数据报告）。

## 5. object_near **[v7 新增]**

| 项 | 内容 |
|---|---|
| 语义 | 有目标距 ego 很近（即时空间压力，不论运动状态与方向） |
| 判定 | `∃ 池对象：distance_m ≤ r` |
| distance_m | t0 传感器系中心距 3D 范数（`ObjectState.distance_m`，与池排序同源） |
| 参数 | r ∈ [3, 7] 标定，目标检出率 5–15% |
| 范围 | 全类别，无运动条件，纯 t0 空间证据，与走廊族不互斥 |
| 护栏 | 所选 r 下与 stationary_obstacle 共现 > 80% → 启用互斥变体（排除在路径 \|y_c\| ≤ Y_st 的对象）重标 |

## 6. oncoming_traffic（判定逻辑不变；族范围见 P-4 已确认）

精确计算步骤（对每个池内对象，任一命中即止）：

1. 族 ∈ vehicle ∪ {bicycle, motorcycle}（P-4：维持原 vehicle 组口径）；
2. `motion_state == oncoming`（按 §0.8：θ > 135°，即相对速度方向朝向 ego；速度 < 0.5 的对象先被判为 stationary，自然排除静止对象）；
3. `distance_m < oncoming.range_m(20)`；
4. → 因子置 true。

---

## 7. yield_required

```text
yield_required = ∪{ firing(o) : o ∈ pool }
```

- 与模型面标签完全解耦；无标签 firing 同样贡献 true；static firing 亦然；
- v7 净变化：buffer 结构（base + h）影响所有类别的 firing 集合，yield 分布可能升可能降——**标定必报**，不承诺不变量。

## 8. 移除项

| 移除 | 原因 | 去向 |
|---|---|---|
| corridor_conflict | 与两 crossing 构造性 100% 嵌套；唯一独立信息（static 残余 1,720 帧）由 §4 的 Y_st 横向放宽吸收（覆盖率标定必报，P-6） | firing 机制保留为内部引擎（§0），只喂 yield |
| congestion | 检出 0.27% 饥饿（±1.185m 窄带 × ≥3 辆 × 近停 2.0m/s 多条件乘法） | 槽位换语义为 object_near；"慢行前车残核"规模进标定必报 |

## 9. 标定参数汇总（train-only 网格，S08 式速度级重派生）

| 参数 | 网格 | 备注 |
|---|---|---|
| base | [1.7, 2.5] | 交互余量；假设检验：单 base vs base_vru/base_vehicle 分裂 |
| thr | 待定网格 | crossing 速度门槛（`v_toward > thr`；v2.4 起不再与 firing 联动，标定结论 0.8） |
| T | 5.0（固定） | crossing 到达时间上限（v2.4 冻结，非标定项；config `crossing_gate.t_arrive_max_s`） |
| r | [3, 7] | object_near 半径 |
| Y_brake | [1.185, 2.2] | braking 在路径横向半宽（v6 固定 1.185，现为标定项，P-6） |
| Y_st | [1.185, 3.0]，约束 ≥ Y_brake | stationary 在路径横向半宽（P-6）；网格为建议值，标定时可调 |
| h(θ_gap) | — | 确定性（标注尺寸 + t0 朝向），零标定 |

**Stage C 冻结（2026-10-12，台账 §3.8）**：base = 2.0、thr = 0.8、r = 5.5、Y_brake = 1.185、Y_st = 3.0、assumption = single，已回写 gt_rule_config（v7-rules-0.3.0，status = frozen）。Y_st 经 3.5 / 4.0 补扫后仍定 3.0：3.5 档挡路报警率 26.55% 破 25% 警戒线，且 3.0→3.5 新增约 1,317 帧报警中仅约 201 帧是 v6 漏报捞回（边际收益差，新增对象为邻道锥桶/护栏）；补扫脚本自校验通过（1.185 档复现 v6 基线 8.90%，2.5/3.0 档与报告原数一致）。

必报项：base 两假设检出率曲线；两族 crossing 检出率-thr 曲线；object_near 检出率-半径曲线与 stationary_obstacle 共现率；yield 分布净变化；firing 物理间隙直方图（擦肩残余规模）；braking 检出率-Y_brake 曲线与新增前车的 \|y_c\| 直方图（邻道/对向吸入诊断）；stationary 检出率-Y_st 曲线、放宽后检出上升与 static 残余（1,720 帧）覆盖率；慢行前车残核规模。

## 10. v6 → v7 参数对照

| v6 参数 | v6 值 | v7 去向 |
|---|---|---|
| corridor.d_risk_vehicle_m | 3.0 | base + h（vehicle 族） |
| corridor.d_risk_pedestrian_m | 2.0 | base + h（vru 族；v6 中两轮车按 3.0 判） |
| corridor.ego_width_m / width_margin_m | 1.87 / 0.5 | 不变（带半宽 1.185） |
| corridor.t_risk_s / time_step_s | 3.0 / 0.25 | 不变（13 采样） |
| braking.evidence_window_s / speed_drop_mps | 3.0 / 0.5 | 不变（未来窗） |
| stationary_obstacle.max_speed_mps / evidence_window_s | 0.5 / 3.0 | 不变（未来窗） |
| oncoming.range_m | 20 | 不变（严格 <） |
| 列内判据 \|y_c\| ≤ 1.185（braking / stationary / congestion 共用） | 1.185 | 拆为 Y_brake / Y_st 两个标定参数（P-6），congestion 已删 |
| motion.band_half_width_deg | 45° | 仅用于 §0.8 motion_state（模型特征 + oncoming），不再参与标签 gate |
| motion.stationary_speed_mps | 0.5 | 不变（§0.8 与 §3 非静止判据） |
| congestion.* | — | 删除 |
| pool.* | — | ⚠ P-1 已确认：vru 拆组，bicycle/motorcycle 锥距 45m per-class 覆盖 |

## 11. 审查点状态

| 编号 | 事项 | 结论 |
|---|---|---|
| P-1 | 池分组与两轮车锥距 | ✅ 已确认（2026-10-11）：vru 拆组；bicycle/motorcycle 保持 45m |
| P-2 | firing 判定口径 | ✅ 已确认：逐样本 min(d_k − h_k)；已回写台账 |
| P-3 | 横向 gate 带内边界 | ✅ 已确认：带内取 \|v_lat\| > thr |
| P-4 | oncoming 族范围 | ✅ 已确认：保留两轮车 |
| P-5 | braking"前车"族在拆组后是否含两轮车 | ✅ 已确认（2026-10-11）：维持原 vehicle 组口径（含 bicycle / motorcycle） |
| P-6 | braking / stationary 在路径条件 | ✅ 已确认（2026-10-11）：统一为 `x_c > 0 ∧ \|y_c\| ≤ Y`（t0 中心点），Y_brake < Y_st 均为标定参数；删除 static 的"列内 ∨ firing"专设分支，firing 服务 yield（v2.4 起 crossing 亦不消费 firing） |
| P-7 | 风险因子归因口径 | ✅ 已拍板（2026-10-11，V7-A1 / D-A1-3）：全量合格对象——命中判据不动，首中即止三因子仅去 break 改全收集；见 §12 与归因规则稿 |
| P-8 | crossing 语义与 firing 解耦 | ✅ 已拍板（2026-10-12，标定后修订，台账 §3.7）：两族同公式 `v_toward > 0.8 ∧ t_arrive ≤ 5.0`（带内 t_arrive = 0）；firing 只喂 yield；rules 升 v2.4，config 增 `crossing_gate.t_arrive_max_s = 5.0` |

---

## 12. 风险因子归因（v2.3 新增，V7-A1 / D-A1-3；v2.4 起 crossing 判据同步 §0.7）

§12 只定义「每个因子记录哪些触发对象」，归因机制（全收集、确定性顺序、因子存在 ⟺ 归因集合非空）自 v2.3 落地后不变；v2.4 仅 crossing 两行的命中判据随 §0.7 更新。完整设计（GT id 分配 / 模型面 schema / parser 语义校验 / token 预算）见决策台账 §6 与 [S13_v7_risk_attribution_rules.md](./S13_v7_risk_attribution_rules.md)。

| 因子 | 归因对象集合（全量合格） | 相对 v2.2 实现的改动 |
|---|---|---|
| vru_crossing | 池内全部满足 §0.7 到达时间判据 ∧ VRU 类别的对象 | 判据改到达时间口径（v2.4），归因机制不变 |
| vehicle_crossing | 同上（vehicle 五类） | 判据改到达时间口径（v2.4），归因机制不变 |
| lead_vehicle_braking | 全部满足 §3 判据的前车 | 去掉首中即止 break → 全收集 |
| stationary_obstacle | 全部满足 §4 两段式判据的在径对象 | 去掉首中即止 break → 全收集 |
| object_near | 全部 distance_m ≤ r 的池内对象 | 无（any → 全列） |
| oncoming_traffic | 全部满足 §6 判据的对向对象 | 去掉首中即止 break → 全收集 |

不变量：因子存在性 ⟺ 归因集合非空；factors 顺序（RISK_TAXONOMY_ORDER）与 `any_firing`（§7 yield 并集）逐字节不变；`derive_risk_factors` 返回扩为 `(factors, attribution, any_firing)`，attribution 顺序 = braking 距离升序（原循环排序键）、其余池序（距离升序 top-8）。
