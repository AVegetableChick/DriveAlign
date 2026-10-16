# S13 Contract v7 — Risk Factor 标定扫描方案

- 状态：**待审查**（对应 [S13_v7_risk_factor_rules.md](./S13_v7_risk_factor_rules.md) v2.2 §9 与 [决策台账](./S13_contract_v7_decision_log.md) §3.6）
- 单位：距离 m，速度 m/s，角度 deg，时间 s
- 数据口径：**train-only**（与 S08 相同的 train split 定义，约 23k anchors）；val 仅用于冻结后的 sanity 检查，test 全程不可见
- 沿袭 S08 惯例：扫描报告产出曲线与诊断 → **终值由用户审阅拍板** → 写入 config 冻结

---

## 1. 目标与总原则

1. **train-only 标定**：所有网格扫描、曲线、直方图仅在 train split 上统计；冻结后 val 上做一次分布 sanity（非调参）。
2. **速度级重派生**：GT 规则变更在速度层重算（record 层不可得），与 S08 同口径。
3. **中间量分解（本方案核心）**：所有标定参数都是对象级中间量的**后验比较阈值**——CV 外推、投影、窗口统计均不含标定参数。因此**参数无关中间量只提取一次**，任何参数组合的因子判定退化为轻量数值比较，全网格扫描无组合爆炸。
4. **可校验性**：提取的中间量必须能**无损重建 v6 全部 7 因子 + yield**（与现有 v6 GT 逐 anchor 100% 一致），证明分解无信息损失后扫描结果才可信。
5. **必报项全覆盖**：rules v2.2 §9 的必报项清单逐项映射到扫描块（§5）。

## 2. 标定参数与网格总表

| 参数 | 网格 | 点数 | 作用域 | 约束/备注 |
|---|---|---|---|---|
| base（单假设） | 1.7 → 2.5，步长 0.1 | 9 | 全组 firing（yield 侧含 static，归属 base_vehicle） | 对角线 View，见 §4-B1 |
| base_vru × base_vehicle（分裂假设） | 各 9 点全网格 | 81 | vru / vehicle+static 分别 firing | 单假设 = 网格对角线，一次扫描同时覆盖两假设 |
| thr（统一横向 gate） | 0.3 → 1.5，步长 0.1 | 13 | 两族 crossing 的 gate | 下界 ≈ 速度估计噪声（~0.3），上界 ≈ 行人横穿典型速度（~1.4） |
| r（object_near） | 3.0 → 7.0，步长 0.5 | 9 | object_near | 目标检出率 5–15% |
| Y_brake | 1.185 锚点 + 1.3 → 2.2 步长 0.1 | 11 | braking 在路径 | 1.185 = v6 现状（对照锚点） |
| Y_st | 1.185 锚点 + 1.3 → 3.0 步长 0.1 | 19 | stationary 在路径 | 约束 Y_st ≥ Y_brake（终选时检查） |
| h(θ_gap) / 窗口 3.0s / 步长 0.25s 等 | — | — | 确定性 | 零标定，不进网格 |

网格均为建议值，审查时可调。分解后各扫描块轻量组合总量约 1,500（§4），均为向量化比较，分钟级；**成本大头只有一次中间量提取**（§3，与 S08 全量重派生同量级）。

## 3. Stage A：中间量提取（一次性，参数无关）

### 3.1 对象级字段（每 anchor × 池内对象，v7 池定义）

| 字段 | 定义 | 消费方 |
|---|---|---|
| `ann_token`, `category`, `group_v7` | 类别与 v7 三组归属（vru / vehicle / static；bicycle/motorcycle 锥距 45m 覆盖已在池构建中生效） | 全部 |
| `x_c`, `y_c` | t0 ego 系中心坐标 | 在路径（Y_brake/Y_st）、gate 带内规则、v_toward |
| `v_lat` | t0 相对速度的 ego 系横向分量 | gate（thr 扫描） |
| `speed_mps` | t0 平面速度 | braking 非静止判据、stationary 停着判据 |
| `min_d_raw` | min over 13 采样的点到扫掠带距离 d_k（**不含 h**，v6 口径） | v6 因子重建、static 残余识别、CV 距离直方图 |
| `min_clearance` | min over 13 采样的 (d_k − h_k)（v7 口径；d_k=0 样本直接给负值） | v7 firing（base 扫描）、物理间隙直方图 |
| `min_win_speed`, `max_win_speed` | 未来证据窗（≤3.0s）速度序列的最小/最大；无 track 或空窗时置 NaN 并以 `has_track=False` 标记 | braking（min）、stationary（max，回退 t0）、congestion 重建 |
| `distance_m` | t0 传感器系 3D 中心距 | braking 排序、object_near、oncoming |
| `motion_state` | P1.4 四分类 | oncoming 重建、诊断 |

gate 所需的 `v_toward = −sign(y_c)·v_lat`（带内取 |v_lat|，P-3 规则）在扫描时由 `y_c, v_lat` 现算，不单独存储。

### 3.2 anchor 级字段

- anchor 标识（scene / sample token）与 split 标记；
- **v6 基线 join**：现有 v6 GT 的 `risk_factors`（7 因子）与 `yield_required`，用于 B0 基线统计、static 残余识别、yield 净变化对照。

### 3.3 执行方式

- 按 scene 分 shard 并行提取，输出 `runs/S13_v7_calibration/features/shard_*.parquet`；单 shard 失败可独立重跑（断点续跑）；
- 长 任务（与 S08 全量重派生同量级）→ **tmux 执行**；
- 环境与启动方式沿用项目惯例（`conda activate autovla_codeclean && PYTHONPATH=DriveAlign/src ...`）。

### 3.4 提取正确性校验（闸门）

用中间量按 v6 公式重建全部 7 因子 + yield，与 v6 GT 逐 anchor 对比（braking 排序 tie-break `(round(distance_m,3), ann_token)` 须复现）。**一致率 = 100% 才进入 Stage B**；不一致则修提取代码，不带偏扫描。

## 4. Stage B：扫描块与条件独立性

**可分块的依据**：五个参数各自只进入互不重叠的判定环节——`base` → firing（crossing 前置 + yield）；`thr` → gate（乘在 firing 之上，与 base 有交互故联合扫描）；`r` → 仅 object_near；`Y_brake` → 仅 braking；`Y_st` → 仅 stationary。braking / stationary / object_near 与 base / thr 完全独立。

### B0 v6 基线统计（无扫描，一次成表）

- v6 各因子检出率、v6 yield 率（后续所有曲线的对照列）；
- congestion 原"慢行前车"残核规模（v6 congestion=true 的 anchor 数，R-OPT-2 遗留量测）；
- **static 残余集合识别**：v6 corridor_conflict=true ∧ 所有 v6-firing 对象均为 static ∧ v6 stationary_obstacle=false 的 anchor 集合（对应此前 val 上测得的 1,720 帧，train 上重新量测）。

### B1 base 扫描（含分裂假设，81 组合）

- 每组 (base_vru, base_vehicle) 输出：vru 族 firing 检出率、vehicle 族 firing 检出率、yield 率（static 归 base_vehicle）；
- **对角线视图** = 单 base 假设曲线（检出率-base、yield-base）；
- **分裂增益** = 非对角最优 − 对角最优；> 1 个百分点（≈2σ，见 §6 噪声说明）才视为支持分裂；
- `min_clearance` 按族（vru / vehicle / static）直方图——base 取值依据与擦肩残余规模。

### B2 thr 扫描（与 base 联合，2 × 9 × 13 组合）

- vru_crossing、vehicle_crossing 各自的检出率-(base, thr) 热力图 + 固定 base 候选下的边际检出率-thr 曲线；
- firing 对象中 `v_toward`（带内为 |v_lat|）按族分布直方图——thr 网格取值依据；
- 诊断列：gate 结果与 motion_state=crossing 离散口径的一致率（仅报告，不作判据）。

### B3 r 扫描（9 点）

- object_near 检出率-r 曲线（目标带 5–15% 标注）；
- 与 stationary_obstacle 的共现率-(r, Y_st) 矩阵——护栏（>80% 触发互斥变体重扫）所需。

### B4 Y_brake 扫描（11 点）

- braking 检出率-Y_brake 曲线（1.185 锚点 = v6 对照）；
- **新增前车诊断**：每个 Y_brake 档位新吸入前车的 |y_c| 直方图，按 sign(y_c) 分解（同向 / 对向压线）——邻道吸入风险量化。

### B5 Y_st 扫描（19 点）

- stationary 检出率-Y_st 曲线与相对 v6（9.0%）的上升幅度；
- **static 残余覆盖率-Y_st 曲线**（B0 残余集合中 |y_c| ≤ Y_st 的比例）——Y_st 终值的主要依据；
- 覆盖对象的类别构成（barrier / traffic_cone / 其他 static）与 |y_c| 直方图。

## 5. 必报项 → 扫描块映射（全覆盖核对表）

| rules v2.2 §9 必报项 | 产出 | 块 |
|---|---|---|
| base 两假设检出率曲线 | 对角线曲线 + 分裂热力图 + 增益 | B1 |
| 两族 crossing 检出率-thr 曲线 | 热力图 + 边际曲线 | B2 |
| object_near 检出率-半径曲线 / 共现率 | 曲线 + (r, Y_st) 矩阵 | B3 |
| yield 分布净变化 | yield 率-base 曲线 vs v6 yield 率对照 | B1 |
| firing 物理间隙直方图（擦肩残余） | min_clearance 分族直方图 | B1 |
| braking 检出率-Y_brake 曲线 / 新增前车 \|y_c\| 直方图 | 曲线 + 分向直方图 | B4 |
| stationary 检出率-Y_st 曲线 / 上升幅度 / 残余覆盖率 | 曲线 + 覆盖率曲线 | B5 |
| 慢行前车残核规模 | v6 congestion=true 计数 | B0 |

## 6. 终选决策准则（推荐规则，终值由用户拍板）

**统计噪声基准**：n ≈ 23k，p = 5% 时 95% CI ≈ ±0.28%——曲线差异 < 1 个百分点不视为信号。

| 参数 | 推荐规则 | 拍板时同时审查 |
|---|---|---|
| base（单/分裂） | 单假设取检出率平台段中点；分裂增益 > 1pp 且 yield 分布改善才采用分裂 | min_clearance 直方图、yield 净变化 |
| thr | 两族检出率-thr 曲线拐点后的平台起点 | v_toward 分布、与离散口径一致率 |
| r | 进入 5–15% 目标带的最小 r；共现 > 80% → 互斥变体重扫 | 共现矩阵 |
| Y_brake | 新增前车 |y_c| 直方图上"本车道连续质量"与"邻道质量"分界；检出率平台起点 | 分向（同向/对向）构成 |
| Y_st | static 残余覆盖率达到平台（每 0.1m 边际覆盖 < 2%）的最小 Y_st；检出率不进入近饱和（> 25% 预警） | 覆盖率曲线、类别构成 |
| Y_st ≥ Y_brake | 终选组合硬约束；违反则以 Y_st = Y_brake 收敛 | — |

## 7. Stage C：冻结与验证

1. **终值写入** `configs/contracts/v7/gt_rule_config.json`（draft）；
2. **回归一致性验证**：正式 v7 `risk.py` 实现跑抽样 anchor（≥1,000，分层抽样保证各因子均有 true 样本），与"中间量分解判定"逐因子对比，**必须 100% 一致**；
3. **全量重派生**：train + val 速度级重建 dataset_v7（物理写入需 tmux，沿用既有流程）；
4. **val sanity 报告**：各因子检出率、yield 率与 train 对照（偏移 > 2pp 须解释）；不回调参数。

## 8. 计算预算与执行方式

| 阶段 | 载体 | 预算 |
|---|---|---|
| A 中间量提取 | tmux，按 scene shard 并行 | 与 S08 全量重派生同量级（一次性） |
| B0–B5 扫描 | 单机向量化比较，~1,500 组合 × 23k anchors | 分钟级 |
| 报告生成 | matplotlib 图表约 12 张 + 汇总表 | 分钟级 |
| C 冻结验证 | 抽样回归 + 全量重派生（tmux） | 回归分钟级；重派生同 A |

## 9. 交付物清单

| 交付物 | 路径 |
|---|---|
| 提取脚本 | `src/drivealign/gt/calibration_v7/extract.py` |
| 扫描与报表脚本 | `src/drivealign/gt/calibration_v7/sweep.py` |
| 中间量 | `runs/S13_v7_calibration/features/shard_*.parquet` |
| 扫描结果 | `runs/S13_v7_calibration/sweep_results/`（parquet + 图表） |
| 标定报告 | `docs/stages_record/S13_contract_and_eval_revision/S13_v7_calibration_report.md`（Stage B 后生成，供拍板） |
| 冻结产物 | `configs/contracts/v7/gt_rule_config.json`（Stage C） |

---

## 10. 待审查确认项

1. 网格取值与步长（§2，尤其 thr [0.3, 1.5] 与两个 Y 网格）；
2. 分裂假设采纳判据（增益 > 1pp）与近饱和预警线（25%）是否合适；
3. static 残余集合的定义口径（B0 第三条）是否符合你此前的 1,720 帧统计口径；
4. 终选流程：按本方案曲线拍板，还是需要增加自动推荐 + 人工确认两步走。
