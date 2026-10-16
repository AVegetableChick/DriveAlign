# S13 Contract v7 决策台账

## 0. 元信息与定位

- 文档定位：contract v7 契约修订的决策台账（改进点登记 → 选项讨论 → 拍板落档）。与同阶段 eval v2 口径台账 `S13_eval_v1_to_v2_metric_changes.md` 并列。
- 修订基准：contract v6 model face（`configs/contracts/v6/` 八件套 + `data/face_manifests/v6/anchor_policy_manifest.json`）；dataset 资产为 v4 代际（`data/dataset_v4`）。
- 状态：`provisional`（讨论中）。所有条目经用户拍板后才转为结论；未拍板项显式标注「待拍板 / 讨论中」。
- 数据证据：除注明外，帧数统计基于 v4 全量 30,940 帧（train+val+test）record 层复算（2026-10-09）。

## 1. 改进点清单总览

| 编号 | 主题 | 状态 | 结论/去向 |
|---|---|---|---|
| V7-B1 | critical_objects 新增 BEV 位置（ego 系） | 已登记，5 个设计点待拍板 | §2 |
| V7-R1 | risk taxonomy 治理（congestion 饥饿 / crossing 统一重构 / 缺失场景） | 已拍板并实施（6 项决议 + 2026-10-12 crossing 到达时间修订 §3.7，rules v2.4）；五槽位回写待 Stage C | §3 |
| V7-Y1 | yield_required 语义澄清 | 调查完成，建议维持现状 | §4 |
| V7-C1 | reasoning 资产退役（契约八件套 → 六件套） | 已拍板（2026-10-11 用户提出），资产已删，链路退役随 Step 2 T0 | §5 |
| V7-A1 | 风险因子归因（risk_factors ↔ critical_objects 关联） | 已拍板（2026-10-11，D-A1-1~3），设计稿 = 归因规则稿拍板稿 v1；随 Step 3 开工前落地 | §6 |
| V7-A2 | token 预算回炉（SFT 发射改紧凑 JSON + 生成预算 700） | 已拍板（2026-10-11，V7-A1 步骤 7 实测超限触发） | §7 |

## 2. V7-B1：critical_objects 新增 BEV 位置

### 2.1 动机

bbox_2d 是图像像素坐标，对规划无尺度/深度信息；规划决策需要目标相对 ego 的度量位置（纵向距离 + 横向偏移）。

### 2.2 可行性证据（GT 侧零新增依赖）

- backfill 池对象已携带 `global_x/global_y`（标注全局位姿）与 ego t0 全局位姿（`src/drivealign/gt/backfill.py`）。
- corridor 机制内部已实现全局→ego 系坐标变换（`risk.py` 的 local_x/local_y 计算），BEV GT 为确定性纯派生，不引入新的标定参数。
- 数据源 = nuScenes annotation × ego_pose，t0 时刻即完备，无未来信息泄漏。

### 2.3 代际影响（与 v6 对比）

- v6 是 dataless face（仅 manifest 层换戳，dataset 冻结不动）；v7 触碰 model face（output_schema/prompt/parser）+ training_targets 结构 → request hash 与 record hash 双代际变化：
  1. 新建 `configs/contracts/v7/` 八件套 + `versions.py` 更新（DEFAULT_CONTRACT_VERSION / MODEL_FACE_LINEAGE）；
  2. 全量重跑 GT 回填与数据集构建（Step 2/3，含 anchor manifest 重派生）；
  3. 新物理目录 `dataset_v7`（目录命名跟 contract 代际走）+ `data/face_manifests/v7/`；
  4. 物理盘写入由用户在 tmux 执行。

### 2.4 待拍板设计点

| 编号 | 决策点 | 候选 | 倾向（待确认） |
|---|---|---|---|
| D-B1-1 | BEV 表示法 | A) ego 系 [x_forward, y_left]，米制、0.1m 取整；B) (distance_m, azimuth_deg)；C) [depth, lateral] | A——对规划最直接，左右语义显式 |
| D-B1-2 | 参考点 | A) 目标中心（与现 distance_m 口径一致）；B) 包围盒最近点 | A |
| D-B1-3 | bbox_2d 去留 | A) 保留（2D grounding + 现有 IoU 指标链不动）；B) 移除 | A |
| D-B1-4 | 评测匹配键 | A) 维持 2D IoU 匹配，在匹配对上算 BEV 误差；B) 改 BEV 邻近匹配 | A |
| D-B1-5 | 新增指标族 | bev 纵向/横向误差 mean/p50/p90 + 距离分桶，命名按 `<field>_<statistic>[_<aggregate>]`，入 scene-cluster bootstrap | 按 eval v2 命名约定扩展 |

### 2.5 复核项

- token 预算：max_new_tokens=512 下 8 目标 × (bbox 4 数 + BEV 2 数) 的最坏占用需实测复核。**已实测并拍板（2026-10-11，实施规划稿 §1.4）**：Qwen2.5-VL-3B tokenizer 下最坏 case（8×construction_vehicle + 6 风险全选 + 最大位数坐标）初测紧凑 JSON 518 / 带空格 631 / 带 fence 635——512 预算下必截断。**拍板选 A+B**：A) v7 prompt 的 bbox_2d 教学改整数像素（已落地；复测最坏 case 紧凑 454 / 带空格 567 / fence 571，紧凑形态余 58）；B) v7 全部 run configs（benchmark base/sft）max_new_tokens=512→640，GPU 阶段创建 v7 run config 时落地（v6 系列历史 config 不动）。B 的佐证：SFT v6 parse failure 分析（qwen-1f-sft-v6）已记录 5/45（11%）512 截断案例——截断非纯长尾，640 为必要项而非双保险。
- 单目深度不适定性：有意保留的难度提升，`available_speed` 提供尺度线索；SFT 后需用反事实组验证模型确实读数而非复制先验。

## 3. V7-R1：risk taxonomy 治理（已拍板并实施）

### 3.1 现状分布（v4 全量 30,940 帧）

| factor | 帧数 | 占比 |
|---|---|---|
| corridor_conflict | 6,990 | 22.6% |
| vehicle_merging | 4,746 | 15.3% |
| stationary_obstacle | 2,775 | 9.0% |
| oncoming_traffic | 2,624 | 8.5% |
| lead_vehicle_braking | 934 | 3.0% |
| pedestrian_crossing | 672 | 2.2% |
| congestion | 82 | 0.27% |
| 无任何风险帧 | 20,203 | 65.3% |

### 3.2 已查明问题

**P-R1：vehicle_merging 实现 ≠ 文档语义（主问题）**

- 文档 gt_derivation："adjacent-lane vehicle future trajectory enters the ego corridor (cut-in)"；实现（`risk.py` vehicle 族判定分支）只检查 vehicle 族 CV 轨迹进入走廊，无邻道来源/横向运动条件。
- record 层归因分解（4,746 帧 merging；基于池内 vehicle 族 motion_state 组成，代理口径）：

| 池内构成 | 帧数 | 占比 | 判读 |
|---|---|---|---|
| 含 oncoming 车 | 2,222 | 46.8% | 对向车通过触发，非 cut-in |
| 仅 stationary 车 | 1,804 | 38.0% | ego 接近停车，与 stationary_obstacle 冗余 |
| 仅 same_direction 车 | 548 | 11.5% | 同道接近或温和 cut-in，record 层不可分 |
| 含 crossing 车 | 172 | 3.6% | 唯一明确横向语义桶 |

- 即 ≥85% 的 merging 标签在 cut-in 语义之外；真 cut-in 的精确定位需速度级重派生（record 层无速度字段，不可得）。

**P-R2：congestion 机理饥饿（0.27%）**

- 判定条件：≥3 辆 vehicle 族中心落在 ego 航向 ±1.19m 横向带内 + 最近前车 3s 窗口速度全 ≤2.0 m/s → 实际只覆盖"正前方正停队列"。
- 比 small_following_gap（0.64%，已删除先例）更稀有；prompt 语义 "dense slow traffic ahead" 与操作定义不匹配（慢而不停的密集车流不触发）。

**P-R3：两轮车语义噪声（次要）**

- 155 帧 merging 池内仅两轮车，其中 171 实例 motion_state=stationary（静止摩托/自行车挡路被标为"车辆切入"）。

**P-R4：行人近距场景无覆盖（候选新因子证据）**

- 池内有 crossing 行人但 pedestrian_crossing 未触发的帧：2,222（7.2%）——CV 外推下行人轨迹与 ego 走廊最近距离 ≥2.0m 的"擦肩"场景，当前无任何标签。
- 池内有行人但零风险的帧：4,957（16.0%）。

### 3.3 待讨论选项（逐条拍板后落档）

| 编号 | 议题 | 选项 |
|---|---|---|
| R-OPT-1 | vehicle_merging 修复方向 | A) 加横向相对速度条件恢复 cut-in 语义（标签大幅收缩，一般走廊冲突由 corridor_conflict 兜底）；B) 改名对齐实现（如 vehicle_corridor_conflict）；C) 维持现状 |
| R-OPT-2 | congestion 去留 | A) 删除；B) 放宽重标定（min_queue=2 / lead_speed 上调 / 横向带→车道半宽，train-only 标定）；C) 重定义为密度规则（距离内 ≥N 辆，不要求近停） |
| R-OPT-3 | 新增因子 | 候选：行人临近车道 / 两轮车穿插 / 转弯路口冲突（oracle 面派生）；硬约束：红绿灯类禁止；门槛：确定性派生 + 标定后不饥饿 |

### 3.4 判定逻辑复核（2026-10-09 应用户要求补充）

**vehicle_merging**：无邻道来源/横向运动条件。判定链 = 池内 vehicle 族（class_groups 含 bicycle/motorcycle）任一对象的 CV 走廊曲线最小距离 < d_risk_vehicle（3.0m）。ego 侧 = 4.1m × (1.87+0.5)m 矩形沿 t0 航向以 t0 速度匀速滑动 3.0s（0.25s 步长，13 采样含 t=0）；对象侧 = 中心以 t0 全局速度匀速外推，ego 系内取相对运动。典型非 cut-in 触发：对向车横距 ~3.5m 时带缘距 3.5−1.185=2.315m < 3.0m；ego 接近同道静止/慢车时距离归 0。

**congestion**：leads = 池内 vehicle 族中心位于 ego 前方且横向偏移 ≤ (1.87+0.5)/2=1.185m 的对象（无纵向距离上限，45m 锥内入池即可）。触发 = |leads| ≥ min_queue_vehicles(3) 且**最近前车**实际未来 3s 窗口速度全部 ≤ lead_speed_mps(2.0)。队列其余成员速度不检查；饥饿主因是窄带几何约束（3 中心同落 ±1.19m 带）叠加"近停"速度门槛（慢行 3–5 m/s 的密集车流不触发）。

### 3.5 拍板记录（2026-10-09）

- **R-OPT-1：已拍板选 A——vehicle_merging 恢复 cut-in 语义**。在 corridor firing 基础上增加横向相对速度条件（阈值 train-only 标定）；对向通过/同道接近场景剔除后由 corridor_conflict 兜底，冲突信息不丢失；需速度级重派生。副作用：P-R3 两轮车 stationary 误触发问题随之消解（该类场景改由 stationary_obstacle + object_near 覆盖）。
- **R-OPT-2：已拍板——congestion 槽位重定义为 object_near**（用户提出）。语义：存在距 ego 很近的目标；原"密集慢车流"概念退役。taxonomy 保持 7 项（congestion → object_near 换语义，非删除）。
  - 设计点（全部 train-only 标定，不预拍板）：
    1. 距离度量：建议沿用池构建的 3D 中心距离口径（sensor-frame center norm），零新增机制；
    2. 触发半径：候选 3–7m 网格，标定目标检出率约 5–15%——该语义有近饱和风险（城市场景近距目标常态存在），必须以检出率-半径曲线定参，防标签失去区分度；
    3. 类别范围：全部池内类别 vs 排除 static 组（traffic_cone/barrier 近距常驻，可能与工地场景强绑定）；
    4. 与走廊族是否互斥：建议不互斥（独立证据、共现可观测，保持各因子正交可派生）；
    5. 口径：池内对象（visibility-first 一致）；近距盘 15m 保证近距对象几乎必然入池，top-8 截断漏检用 gt_truncation_check 思路审计。
  - 独立风险评估（2026-10-09，助手意见，用户已采纳前提）：
    - 优点：判定确定性最强（纯 t0 空间证据，零未来依赖、零标定耦合）；填补 taxonomy 空白——现有因子全部面向"未来冲突"或"证据窗"，唯一缺失的正是"即时空间压力"。
    - 风险一（近饱和）：城市场景近距目标常态存在，半径取大则检出率冲高、标签失去区分度，SFT 可用近恒定策略刷 F1。护栏：检出率-半径曲线定参，目标 5–15%。
    - 风险二（信息冗余）：object_near 是唯一纯瞬时因子，可由对象列表 + BEV 距离直接派生（与 D-B1 叠加后更甚），对模型是低信息量标签。先例：oncoming_traffic 同为对象列表的纯函数（motion_state=oncoming + range 内），可接受。护栏：标定时报告与 stationary_obstacle 的共现率；若所选半径下与 stationary_obstacle 共现 >80%，则考虑"排除走廊列内对象"的互斥变体再标定，保住"近但不在走廊"的独特信号（邻道贴身、路侧近距等）。
    - 用户问询（2026-10-09）：object_near 与 stationary_obstacle 是否为包含关系？结论：**交叉非包含**——判别轴不同（"停 + 在路径上" vs "距 ego 半径"），且 stationary_obstacle 纵向无上限（可达池距离上限 ≈45m），object_near 仅覆盖半径 r（3–7m）。交集 = "停 + 在路径 + 近距"（如跟前 4m 停止车，双触发）；但 15m 外的停止车队只触发 stationary_obstacle（object_near 够不到），邻道 3m 行驶车只触发 object_near（不停 + 不在列）。重叠度由上述共现率护栏在标定中量化。
- **R-OPT-3：已拍板（2026-10-09 二次修订）——取消独立「行人临近」因子，双路吸收原擦肩人群**（用户裁定：pedestrian_near 并入 object_near + 放宽 pedestrian_crossing）：
  1. object_near 全类别含行人（近 ego 半径，R-OPT-2）；
  2. 放宽 pedestrian_crossing 判定：d_risk_ped 由 2.0m 上调（候选网格 3–6m，train-only 标定），原 2,222 帧（7.2%）擦肩人群（行人 CV 距离 ∈ [2.0, 上界)）由 crossing 直接吸收。
  - 连带后果：yield_required 分布随放宽上升（新覆盖帧 yield false→true，量级由标定的行人 CV 距离直方图给出）；R-OPT-4 的「yield 不变」仅指 corridor_conflict 移除本身，两项叠加后 yield 净变化以标定为准。
  - 语义稀释护栏：d_risk_ped 过大时 "crossing the ego path" 名不符实，prompt 措辞需同步（如 "entering or passing close to the ego path"）；标定时权衡检出率与语义纯度。
  - （原候选定义——独立行人因子 + 走廊带三级阶梯——已废弃，存档见 git 历史。）
- **R-OPT-4：已拍板选 A——corridor_conflict 从 model 面 taxonomy 移除**（2026-10-09，用户确认）。
  - 数据依据（record 层复算）：vehicle_merging（4,746 帧）与 pedestrian_crossing（672 帧）是其**构造性严格子集**（实现上二者就定义为对应类族的走廊冲突，100% 共现）；corridor_conflict 唯一独立信息 = static 族（barrier/traffic_cone）触发残余 1,720 帧（5.6%），其中 83.3% 的帧无其他任何风险标签（施工区锥桶线/护栏收窄 ego 路径场景）。lead_vehicle_braking 与 corridor 仅 30% 共现（70% 独立），为不同证据类型（实际减速 vs 反事实冲突），不受影响。
  - 处置：corridor 机制保留为**内部 GT 引擎**（不动）；static 残余由 stationary_obstacle 吸收——static 族成员几何从"走廊列内（横向 ≤1.185m）"放宽到"列内或 CV 走廊带 d_risk 范围内（≈4.185m）"，与 v6 prompt 既有措辞 "in or near the ego path" 对齐；yield_required 派生改为内部 firing 并集（vehicle ∨ ped ∨ static 任一触发），**该移除本身不改变 yield 分布**（6,990 帧 / 22.6% 保持）。
  - 连带：stationary_obstacle 检出率因吸收上升，幅度进标定确认；风险因子 taxonomy 7→6 项。
- **R-OPT-5：已拍板（2026-10-11，用户提出）——pedestrian_crossing 改名 vru_crossing 并扩类**：行人 + 自行车 + 摩托车（VRU = 易受伤害道路使用者）。
  - 动机：关闭 R-OPT-1 cut-in 修复留下的空隙——横穿两轮车（路口斜穿）横向相对速度大，既会被 cut-in 条件误判为 merging，又不是真并线；改归 VRU_crossing 后归属清晰，同时解决 v6 已发现的"自行车横穿被标成 vehicle_merging"怪象。
  - 判定阶梯（2026-10-11 用户问询后修正——行人/两轮车分支统一加 motion_state=crossing 门控）：**vru_crossing = VRU（行人/自行车/摩托车）firing 且 motion_state=crossing**。行人侧继承 d_risk_ped 放宽（2.0m→3–6m 标定，吸收横穿擦肩人群——证据人群本就是 crossing 行人，闭合无损）；两轮车 buffer 维持 vehicle 族 3.0m 不放宽。
  - 门控复用 motion_state 的三点收益：① 静止行人不再误入 crossing——查证 stationary_obstacle 为类别无关（risk.py L315 `for obj in pool`），静止行人列内已有归属，与 corridor_conflict 式冗余同构的问题提前消除；② GT 判定与模型可见特征共用同一离散词汇（motion_state 在 critical_objects 中输出），模型可从特征直接推理判定，降低学习难度；③ 零新增参数——原"crossing 航向带"标定项取消，对角角度归属继承 motion_state 既有判定规则。
  - VRU firing 的完整归属：crossing → vru_crossing；stationary → stationary_obstacle（列内）/object_near（近 ego）；same_direction（沿路走）→ 无标签仅 yield（规模进标定报告）。与 vehicle_merging 的边界 = motion_state 离散分类，天然互斥。
  - 静止两轮车不进 vru_crossing（motion_state≠crossing），仍由 stationary_obstacle（列内）/object_near（近 ego）覆盖——与 R-OPT-1 副作用条款一致。
  - 有意排除：汽车横穿冲突不进 vru_crossing（VRU 语义=保护类群体；此类场景 nuScenes 前视稀少，由 yield=true 承载冲突证据）。三汽横穿若未来标定显示规模可观可再议。
  - 不变量：标签重组不动 firing 并集，yield_required 分布不受 R-OPT-5 本身影响；模型面 taxonomy 仍 6 项（仅改名+扩类），maxItems=8 不变。
- **R-OPT-6：已拍板（2026-10-11，用户提出改名并要求抛开历史从零重构）——vehicle_merging 改名 vehicle_crossing，参数结构重建为 base+half_width 分解 + 统一 crossing gate**。
  - 改名效果：vru_crossing / vehicle_crossing 两族完全平行，语义统一为"侧向进入 ego 路径"；cut-in 与垂直横穿统一纳入 vehicle_crossing，**R-OPT-5 的"汽车横穿有意排除"条款作废**。
  - 参数结构（取代 v6 per-category buffer 与 R-OPT-3 的 ped 专项放宽）：
    - **firing buffer = base + h(θ_gap)**（2026-10-11 用户问询后修正）。h(θ_gap) = 对象带朝向矩形在最近贴近样本间隙方向上的投影半延展（支撑函数，由标注框尺寸 + t0 朝向确定性计算，零标定参数）；base = 交互余量，唯一组级标定标量。间隙方向决定投影：垂直横穿 → 对象半长（轿车 2.25m）、cut-in/同向横移 → 半宽（0.95m）、纵向追近 → 半长。半对角线为保守包络，因对齐情形虚胀 1.5 倍而弃用；v6 的 per-category 恒定值本质是两种几何的妥协常数，被投影公式取代。行人近似各向同性（0.25~0.30m）几乎无差；自行车/摩托车与垂直横穿汽车的 buffer 较固定半宽方案显著增大（更早触发，符合危险性排序）。
    - 验证：v6 的 ped 2.0 / vehicle 3.0 恰好 ≈ 同一 base（1.7~2.05）+ 各自半宽（0.3 / 0.95）——旧拆分本来就是该公式的粗糙近似，v7 把它显式化。
    - Q1（VRU 统一参数）：ped/bicycle/motorcycle 共享 base_vru，仅差确定性投影延展 h(θ_gap)，**零新增参数，不分开处理**。
    - Q2（vehicle 与 vru 参数关系）：两 base 是同一物理量，默认同值（单 base 假设）；"base_vru ≠ base_vehicle 分裂"作为备择假设进标定网格，由数据裁决。
    - 连带：**R-OPT-3 的 d_risk_ped 3–6m 放宽端点作废**——base 网格 [1.7, 2.5] 下 VRU buffer 上限 ≈2.8–3.0m，擦肩吸收收缩到 CV 距离 [2.0, ~3.0)；6m 端语义不成立（CV 距 5m 的行人不是冲突），残余擦肩规模由标定 CV 距离直方图报告，若可观再议新因子（本轮不加）。
  - 统一 crossing gate（取代 motion_state=crossing 离散门控与 R-OPT-1 的 merging 专属横向条件）：**横向相对速度朝走廊分量 > thr**，两族共用同一阈值。优点：cut-in（motion_state=same_direction 但横移）、垂直横穿、对角运动统一覆盖，无边界归属问题；motion_state 保留为模型可见特征不变。同道接近（无横移）仍不获标签——R-OPT-1 的反误标目标保留。
  - 标定参数最终清单：base（单/分裂两假设网格）、crossing gate 阈值 thr、object_near 半径 r、在路径横向半宽 Y_brake / Y_st（P-6 增补）；确定性参数：h(θ_gap) 投影延展（标注尺寸 + t0 朝向）。必报：firing CV 距离直方图、yield 分布净变化（buffer 变化影响所有类别）、擦肩残余规模。
- **修复后 taxonomy 预览（6 项，2026-10-11 R-OPT-6 终版 + P-6 修订）**：vru_crossing（VRU firing ∧ 统一 lateral gate）/ vehicle_crossing（原 vehicle_merging 改名；vehicle firing ∧ 同一 gate，含 cut-in 与垂直横穿）/ lead_vehicle_braking（独立证据；在路径放宽 Y_brake）/ stationary_obstacle（类别无关，在路径条件统一放宽 Y_st，全类别同一判据）/ object_near（全类别近 ego）/ oncoming_traffic。firing buffer 统一为 base + h(θ_gap) 方向投影延展。risk_factors maxItems=8 不变。
- **审查稿确认记录（2026-10-11，详述文档 [S13_v7_risk_factor_rules.md](./S13_v7_risk_factor_rules.md)）**：
  - **P-1 ✅**：池分组拆为 vru / vehicle / static 三组；bicycle / motorcycle 保持 45m 锥距（per-class 覆盖，速度特性接近车辆）。
  - **P-2 ✅**：firing 判定口径 = 逐样本 min(d_k − h_k) < base（逐样本投影，非单一最近样本）。
  - **P-3 ✅**：统一横向 gate 带内边界规则 = 中心已在带内（|y_c| ≤ 1.185）时取 |v_lat| > thr。
  - **P-4 ✅**：oncoming_traffic 族范围维持原 vehicle 组口径（含 bicycle / motorcycle）。
  - **P-5 ✅**（2026-10-11 用户确认）：lead_vehicle_braking"前车"族维持原 vehicle 组口径（含 bicycle / motorcycle）——braking 是动力学证据，不随保护类别改变。
  - **P-6 ✅**（2026-10-11 用户提出并确认）：braking / stationary 的"在路径"条件统一为 `x_c > 0 ∧ |y_c| ≤ Y`（t0 中心点），横向半宽参数化——**Y_brake（建议网格 [1.185, 2.2]）< Y_st（建议网格 [1.185, 3.0]，约束 ≥ Y_brake）**，均进标定网格。全类别同一几何判据，**不再区分可移动 / 不可移动目标**；删除原"static 族 = 列内 ∨ firing"专设分支（取代 R-OPT-6 的 static 带内放宽条款），firing 仅服务两个 crossing 因子与 yield。理由：前车中心略偏出扫掠带（压线）时其车身仍覆盖 ego 通道，减速仍为有效证据；静止障碍不会自行移动，t0 位置即全部信息，走廊近旁静止物是高价值必避证据，漏检代价高于误检代价，故 Y_st 更宽。必报增补：braking/stationary 检出率-Y 曲线、braking 新增前车 |y_c| 直方图（邻道/对向吸入诊断）、v6 corridor_conflict static 残余（1,720 帧）被 Y_st 覆盖的比例。
  - v2.1 清晰度修订（同日，用户要求复查各因子表述）：d_k=0 样本直接计 firing（无需定义投影方向）；θ_obj 与 θ_ûk 同坐标系约定；分裂假设下 static 归 base_vehicle（v6 先例）；braking 语义更正为"存在列内前车减速"（早停仅为优化）；stationary 无 track 回退 vs braking 跳过的不对称理由；oncoming 静止对象由 motion_state 排序自然排除。

### 3.6 待办

- 六项 GT 规则变更（R-OPT-1 / R-OPT-2 / R-OPT-3 修订 / R-OPT-4 / R-OPT-5 / R-OPT-6）统一进 S08 式 train-only 标定网格重跑（速度级重派生，record 层不可得）；标定产物更新 gt_rule_config v7 并冻结。
- 标定必报项（R-OPT-6 终版 + P-6 增补口径）：
  - base：单 base 假设 vs base_vru/base_vehicle 分裂两假设的检出率-取值曲线；
  - 统一 crossing gate 阈值 thr：vru_crossing / vehicle_crossing 两族检出率-thr 曲线；
  - object_near：检出率-半径曲线（目标 5–15%）、与 stationary_obstacle 共现率；
  - yield_required 分布净变化（buffer 结构变化影响所有类别的 firing 集合）；
  - firing CV 距离直方图（擦肩残余规模、base 取值依据）；
  - lead_vehicle_braking：检出率-Y_brake 曲线、新增前车的 |y_c| 直方图（邻道 / 对向吸入诊断）；
  - stationary_obstacle：检出率-Y_st 曲线、放宽后检出上升幅度、v6 corridor_conflict static 残余（1,720 帧）被 Y_st 覆盖的比例；
  - congestion 原"慢行前车"残核场景规模（R-OPT-2 遗留量测）。
- prompt / output_schema / risk_taxonomy.json / enum_phrase_map 同步更新（v7 八件套复制时一并处理）。

### 3.7 拍板记录（2026-10-12，标定后修订）：crossing 语义与 firing 解耦（到达时间判据）

- **拍板（用户提出）**：Step 3 标定扫描完成后复查 crossing 语义，确认 firing（3s 双侧 CV 反事实"是否已经贴近"）与 crossing（侧向进入路径"是否正在过来"）是两种证据类型，`firing ∧ gate` 联动口径系统性漏标"远而将达"的横穿对象。三轮实验（去 firing / 拉长 CV 窗 / 到达时间）后拍板**到达时间口径**，rules 升 v2.4（v2.3 已被 §12 归因条款占用）。
- **冻结公式（两族同参数）**：`crossing ⟺ v_toward > 0.8 m/s ∧ t_arrive ≤ 5.0 s`；`t_arrive = (|y_c| − 1.185) / v_toward`，带内（|y_c| ≤ 1.185）记 0；`v_toward` 定义不变（带内 |v_lat|、带外 −sign(y_c)·v_lat，P-3 带内规则保留）。
- **解耦范围**：firing 引擎本体（K=13、min(d_k − h_k) < base）与 `yield_required = ∪firing` 逐字节不动；firing 自 v2.4 起唯一消费方是 yield 并集；crossing 归因机制（V7-A1 全收集、池序）不变。
- **config 落地**：`crossing_gate` 新增 `t_arrive_max_s = 5.0`（固定值，非标定项）；`lateral_thr_mps = 0.8` 保留、角色变为 crossing 速度门槛（不再是"关闭标定"的联动槽位）；gt_rule_config 版本升 v7-rules-0.2.0-draft（仍 provisional，五槽位标定结论回写属 Stage C，本次不做）。
- **数据依据（全量 22,941 帧快照，thr = 0.8 / T = 5s）**：vru_crossing 0.31% → 4.40%（1,009 帧）、vehicle_crossing 1.79% → 10.94%（2,509 帧）、两族并集 14.53%；crossing∩firing 仅 0.6%（旧口径下绝大多数 crossing 对象因距离远而在 3s CV 窗内不 firing，被系统性漏标）；crossing 对象 26% 带内（t_arrive = 0）、距离 p50 = 24.5 m。
- **实施（同日完成并回归）**：risk.py 以 `crossing_kinematics` / `crossing_hit` 替换 `crossing_gate`（派生循环去 firing 门槛）；单测重写（到达时间方向性 / 带内 t=0 / thr 与 T 边界 / 解耦回归 / 族划分）；rules v2.4（§0.7 重写、§1/§2 判定行、§9 增 T 行、§11 增 P-8、§12 crossing 行同步）；归因规则稿 §3.2 两行同步。

### 3.8 拍板记录（2026-10-12，Stage C）：五槽位回写冻结

- **拍板（用户逐项确认标定报告 §4 六问）**：① 五槽位照单全收；② firing base 保持 single（分裂不被数据支持）；③ Y_st 先补扫 3.5 / 4.0 再定——补扫结果支持仍定 3.0；⑤ thr 标定关闭（随 rules v2.4 变为 crossing 速度门槛）；⑥ 执行 Stage C。
- **Y_st 补扫证据**（脚本 `runs/S13_v7_calibration/sweep_yst_extension.py`，复用 B5 机制与 B0 残余集合，自校验通过：1.185 档复现 v6 基线 8.90%、2.5/3.0 档与报告原数逐位一致）：3.5 档挡路报警率 26.55%（破 25% 警戒线）、捞回 89.27%；4.0 档 31.62% / 98.20%。3.0→3.5 新增约 1,317 帧报警仅约 201 帧为 v6 漏报捞回，新增对象为邻道锥桶/护栏涌入——边际收益差，**Y_st 终值 3.0**。
- **config 落地**：gt_rule_config `object_near.radius_m` 5.0→5.5、`stationary_obstacle.y_st_m` 2.5→3.0（base / thr / Y_brake 与占位一致不变），status provisional → **frozen**，版本 v7-rules-0.2.0-draft → **v7-rules-0.3.0**；description 改为冻结口径（含补扫否决理由）。
- **同步**：config.py docstring、alignment 单测（status=frozen + 五槽位精确值断言 + t_arrive_max_s=5.0）、dataset_build 单测（status 断言翻 frozen）、risk 单测三处边界取值改从 config 读取（r=5.5 / Y_st=3.0 下 y=3.0、distance 5.01 恰落带内的旧场景）；rules 文档 §9 追加 Stage C 冻结块；标定报告 §4 追加拍板结果。
- **遗留（已关闭，2026-10-12）**：报告问题 4——static 残余 1,221 vs 1,720 口径。复查结论：1,720 在所有落档产物（v6 GT 分布报告 / B0 基线 / 会话记忆）中均无出处；数十个候选口径复现未果（最近似"距离 ≤30 m 且 |y_c| ≤1.0 静止帧 = 1,720"，但邻域连续密集、判为巧合，且类别构成以 car 为主与静态漏报语境不符）；用户确认不记得此数来源。判定为无档案口头数字，**以 1,221（5.3%，v6 静态漏报残余，Y_st 扫描捞回分母）为准**，问题关闭，不阻塞任何决策。
- **待办 B（已完成，2026-10-12）**：smoke（2 场景/split，gates 7/7 过）→ 用户 tmux 全量重派生（`--out /root/autodl-tmp/datasets/drivealign_dataset/v7` + symlink `data/dataset_v7`，`--builder-commit 88ac5671…`，冻结 scene manifests 仍从 `data/dataset_v1` 只读读取，v1 未动）。产物：train 22,941 / val 2,531 / test 5,468 valid（隔离 659 全 BAD_GAP），gates 7/7 PASS，GT 回填 100%。**分布核对全部吻合**：vru_crossing 1,009（4.40%）/ vehicle_crossing 2,509（10.94%）与 rules v2.4 快照预期逐位一致；stationary_obstacle 4,774（20.81%）= Y_st=3.0 扫描报警率逐位复现（标定机器→生产代码对齐）；oncoming 1,998 / braking 595 与 v6 基线一致（公式未动）；yield 6,857（29.89%）= v6 23.77% + 6.1pp（标定预告吻合）；speed_action 与 motion_states 三 split 与 v1 GT 逐位一致（机制未动的硬性要求）。报告：`runs/S13_v7_calibration/full_reports/`。
- **Step 4 收尾（已完成，2026-10-13，G6 关闭）**：
  1. **face manifest v7**：`data/face_manifests/v7/`（v6 先例布局：anchor_policy_manifest.json + manifest_assets symlink staging + 两轮 run log + reports/）。派生 30,940 anchors（22,941/2,531/5,468），contract_version=v7，绑定 dataset_v7 dataset_manifest_sha256=534b97f4…，config builder_commit = dataset_builder_commit = 88ac5671…；**BYTE-IDENTICAL 复跑**通过（run1 vs run2：manifest + 两份 distribution report 三文件逐字节同）。
  2. **v7 专项 parity（全量 30,940，非抽样）**：新增 [v7_parity_check.py](../../../src/drivealign/cli/v7_parity_check.py)（S08 parity 口径的 v7 版：hash 双代际变化改判为**正向 gate**）。结果：model_inputs / oracle_only / metadata 逐值 0 diff；provenance 仅 builder_commit + contract_version 白名单漂移且 v7 戳正确；record hash 与 request hash（vs v6 face）100% 代际变化；v7 expected_output 四字段 100% 在位；11 gates 全 PASS。报告 `full_reports/v7_parity_report.{json,md}`。
  3. **dataset_verify 确定性复跑**：scratch 全量重建（build + manifest）+ 39 文件 byte 对比 differing=none，PASS（`full_reports/determinism_report.json`）。工具小适配：[dataset_verify.py](../../../src/drivealign/cli/dataset_verify.py) 新增可选 `--anchor-manifest`（face manifest 独立目录布局，缺省行为不变）。
  4. **val sanity（冻结后仅 sanity，不回调）**：7 项 >2pp——vru_crossing −2.03 / vehicle_crossing −4.42 / stationary_obstacle −2.00 / frames_with_risk −5.46 / yield +3.73 / KEEP_SPEED +2.92 / STOP −4.19。per-location 分解：location 混合贡献 ≤0.28pp（train/val 构成几乎一致），per-location rate 偏移方向不一致、小格子方差大（hollandvillage val 仅 257 帧、vru 0 事件）——**判定为 scene 级聚类抽样方差**（val 仅 70 scenes / 2,531 帧，二项 SE 低估真实方差），无系统性失准证据，不回调参数。报告 `full_reports/val_sanity_report.{json,md}`（test 参考取值随报，不 gate）。
  5. **BEV 审计（rules §6.2 GT 侧）**：全量 109,310 目标——底边出画（bbox 底边距图像下缘 ≤5px）6.04%（帧级 17.65% 锚点含 ≥1 个）；近距 d_gt<10m 14.39%（帧级 32.1%）。**出画目标 99.6% 落 [0,10) 桶，[10,25) 仅 25 个、[25,45) 为 0**——中远距分桶读数不受底边出画污染；GT BEV 由 3D 标注派生，不受图像裁剪影响。cut 目标因子归因集中 object_near（2,434）/ stationary_obstacle（1,895）。报告 `full_reports/bev_truncation_audit.{json,md}`。

## 4. V7-Y1：yield_required 语义澄清（调查完成）

### 4.1 调查结论

- 语义 = 「让行」（yield to another road user），不是 ego 变道；`vehicle_merging` 指他车切入，亦非 ego 变道。
- GT = corridor 族（corridor_conflict / pedestrian_crossing / vehicle_merging）任一触发的纯反事实信号，与 ego 实际行为（变道/刹车）刻意解耦——P1.6 证据分离设计，防止"正确让行擦除冲突证据"的自败陷阱（`action_yield.py` docstring 明示）。
- 数据佐证：STOP 帧 92.2% yield=false（停车多由已停住/前车制动/静止障碍驱动，而非走廊冲突）。

### 4.2 建议

维持现状，不改语义。若未来要承载「ego 需变道避让」类行为标签，应另立新字段而非改写本字段（待用户确认）。

## 5. V7-C1：reasoning 资产退役（2026-10-11 拍板）

- **提出与拍板**：用户提出「v7 可以把 reasoning 相关的 contract 移除，因为我们当前没有 reasoning 字段」——已拍板执行。
- **事实依据（代码核查 2026-10-11）**：
  1. 模型面自 v6 起无 reasoning（prompt/schema 不教不校验，`reasoning` 已从输出契约移除）；
  2. SFT 监督目标为四字段列举，不含 reasoning；
  3. GT 渲染链实际消费的是 **v4 资产**（`config.load_templates()` / `load_phrase_map()` 无参默认 v4）——v6/v7 目录中的这两个文件自建立起即零消费死资产；
  4. reasoning 文本仅存于 record 层 `expected_output`（GT provenance，v4 face 校验），eval / SFT / 报告均不消费。
- **决议内容**：
  1. v7 契约目录八件套 → **六件套**：`reasoning_templates.json` / `enum_phrase_map.json` 删除（已删，2026-10-11）；v7 pin 测试（test_contract_v7_alignment）断言两文件保持退役、其余六件存在；
  2. GT 渲染链随 v7 代际退役：[render.py](../../../src/drivealign/gt/render.py) 模块、[config.py](../../../src/drivealign/gt/config.py) 的 `load_templates` / `load_phrase_map`、[backfill.py](../../../src/drivealign/gt/backfill.py) 的渲染调用与 P4.5 zero-contradiction gate、[build_dataset.py](../../../src/drivealign/dataset/build_dataset.py) 的 GtAssets 挂载；
  3. `expected_output` 5 字段 → **4 字段**（去 reasoning）；`_validate_schema` 从硬编码 v4 切 v7——v7 output_schema 成为 GT / 模型统一校验器。
- **实施边界（原子批次，归 Step 2 T0）**：渲染链退役 + expected_output 4 字段 + 校验切 v7 + **BEV GT 就位**四者同批落地——v7 schema 的 critical_object required 含 bev_x/bev_y，校验切换依赖 GT 侧 BEV 字段生成。批次完成前渲染链与 v4 资产保持原样（现网构建不炸）。
- **不动的部分**：v5/v6 及更早契约目录保持八件套 byte-frozen（既有资产存在性 pin 不变）；dataset_v4 record 中的 reasoning 字段随冻结资产保留。
- **连带**：expected_output 结构变化 → record hash 代际变化——并入 v7 本就的代际重建（dataset_v7），无额外代价；`tests/unit/test_gt_render.py` 随链路退役删除；`versions.py` docstring 更新；enum_phrase_map 的「一表两消费者」约束（P4.2）随双消费者同灭而自然解除。
- **登记**：实施规划稿 §1.1 / §2 T0 / §7 红线同步修订。

## 6. V7-A1：风险因子归因（risk_factors ↔ critical_objects 关联）（2026-10-11 拍板）

- **提出与拍板**：用户提出「能不能在 risk_factor 当中，把每个出现的因素和对应的 critical object 关联起来」——三项设计点当场拍板。
- **拍板内容**：
  1. **D-A1-1 归因落点 = 模型面**：`risk_factors` 对象化为 `[{factor, objects}]`，模型输出归因——监督信号闭环（对象级 grounding ↔ 因子级语义互证），Step 5 可评归因准确率；
  2. **D-A1-2 引用机制 = 显式 id 字段**：`critical_object` 新增 `id`（0–7 整数），`objects` 引用该 id——位置计数是已知 LLM 失败模式，显式指针更鲁棒；代价每对象 +2~3 tokens；
  3. **D-A1-3 GT 归因口径 = 全量合格对象**：因子命中判据与阈值逐字不动，首中即止三因子（braking / stationary / oncoming）仅去掉 break 改全收集——归因与标定正交，G2 已通过判定不受影响。
- **代际定位**：v7 冻结前代内变更——request hash 变化被未冻结的 v7 face 吸收；dataset_v7 未建，零重建成本；冻结后再改 = v8 全代际重建。
- **设计稿**：[S13_v7_risk_attribution_rules.md](./S13_v7_risk_attribution_rules.md)（拍板稿 v1：GT 口径表 / schema 变更 / parser 五条语义校验 / token 预算估算 ~530 / 红线清单 / 9 步执行清单）。
- **登记**：规则稿升 v2.3（新增 §12 归因章）；实施规划稿 §2.9 随执行回填；归因指标族（factor-level P/R）留 Step 5 / eval 增补稿。

## 7. V7-A2：token 预算回炉——SFT 发射改紧凑 JSON + 生成预算 700（2026-10-11 拍板）

- **提出**：V7-A1 执行步骤 7（token 实测）超限触发归因规则稿 §6 预案的回炉条件——归因增量（8 × id 字段 + 对象化包装 + id 引用列表）把带空格/fence 形态最坏 case 从 567 推到 678–808 tokens（Qwen2.5-VL-3B 实测，三口径 × 三形态），640 预算全口径不足；SFT 监督文本 = 带空格形态 = 模型实际发射形态，紧凑形态 ≤640 救不了预算。
- **拍板内容**：
  1. **SFT 目标发射改紧凑 JSON**：[targets.py](../../../src/drivealign/sft/targets.py) `DUMP` separators 由 `(", ", ": ")` 改 `(",", ":")`——模型发射形态跟随 SFT 监督文本，紧凑化后三口径最坏 620；v6 冻结样本 artifact 的 target_text 已落盘不受影响（encode_frozen_sample 读冻结串）；
  2. **生成预算 640 → 700**：v7 run configs（GPU 阶段创建）统一写 700，紧凑发射下对抗口径 C（620）仍余 80。
- **不动的面**：v6 冻结 SFT config（sft_1f.yaml / sft_smoke_1f.yaml，target_dump 仍记录 v6 带空格口径）与其 sha 登记；契约面（schema/prompt/parser）零改动；v7 SFT config（GPU 阶段创建）写 `target_dump.separators: [",", ":"]` + `max_new_tokens: 700`。
- **代价**：输出文本可读性下降；SFT 监督面 v6（带空格）→ v7（紧凑）风格代际变化——被 v7 未冻结的代际面吸收，与 V7-A1 同定位。
- **实测记录**：归因规则稿 §6（三口径 × 三形态全表）、实施规划稿 §1.4（复测表 + 拍板记录）。

## 8. Step 5：eval v3 评测面增补（2026-10-13/14 拍板并实施，G7 通过）

- **提出**：v7 契约三变更在评测面无对应指标——V7-B1 BEV 字段（bev_x/bev_y）无定位误差指标、V7-A1 归因对象化（risk_factors 携 objects 引用）无"对象指得对不对"指标、V7-R1 词表收编（七枚举 → 六枚举）未进 eval 词表；eval_v2.yaml 已 frozen（2026-10-09）不可原地扩键 → eval v3 立项（实施规划 §8-4 拍板，§5 评审通过）。
- **拍板内容**（权威口径 = [S13_eval_v3_metric_additions_draft.md](./S13_eval_v3_metric_additions_draft.md)，已拍板定稿 2026-10-14）：
  1. **立项**：eval v3 = eval v2 + BEV 误差族 + risk 归因族 + 词表 v7 化；v2 byte-frozen 不动，新建 [eval_v3.yaml](../../../configs/evaluation/eval_v3.yaml)（v2 副本仅 eval_version/status + v7 词表两处差异）；BEV 与归因只在主阈值（IoU 0.5）匹配对上计算；
  2. **BEV 误差语义**（2026-10-13）：三族全非负（dx=|Δx|、dy=|Δy|、dxy=hypot）；带符号偏置信息另走方向分解键；
  3. **分桶**（2026-10-14）：四桶 [0,10) / [10,25) / [25,40) / 40+ 开放桶（用户裁定末桶放大并开放，吸收量程外尾巴），桶归属只看 GT 距离 d_gt；
  4. **signed 方向分解键**（2026-10-14）：现在加入——`bev_signed_{dx,dy}_{pos,neg}_mean` 四键；用户裁定原话："一定要把正误差和负误差分开统计平均，不要放在一起统计平均，那样就抵消了"——严禁代数均值，正/负子集分别求均值，零误差两侧都不进，空子集 0.0，不入 bootstrap；
  5. **归因口径**（2026-10-14）：conditional 双向——归因原子仅产生于两侧因子名集合之交（预测多报因子不计 / GT 因子未报不计），避免同一错误在因子指标与归因指标双计；id 桥接经 primary IoU 几何匹配，两侧独立编号空间严禁编号直比；
  6. **默认关闭两项**：cut/near 分层 BEV 读数走诊断脚本不入正式报告；counterfactual battery 不加 BEV/归因读数；
  7. **键数勘误**（2026-10-14）：§5 冻结口径按逐键清单枚举 = **51 标量键 + 1 嵌套块**（早前"50 标量键"系算术笔误，9+36+2+4=51）；bootstrap 标量键 22 → **27**（增量 +5；早前"10 → 15"基数有误，v2 实测 `subset_statistics_v2` = 22 标量 + 4 嵌套）。
- **不动的面**：eval_v2.yaml byte-frozen（sha `661f28f1c70f26893fe0259a91db2f6114bcd976fdb1fc03753af259cb816fc4`）；v1/v2 打分与报告路径行为零改变（v3 字段只由 `score_anchor_v3` 追加，键集 pin 钉死 v2 ⊆ v3）；bootstrap 三参数（n=2000 / seed=20261003 / 0.95，S09 预注册值）；T6 完整性硬前置；gate 取消裁定沿用（不设 PASS/FAIL）。
- **实现**：`evaluation/metrics_bev.py` + `evaluation/metrics_attribution.py`（纯函数）+ `evaluation/evaluator_v3.py`（`score_anchor_v3` / `aggregate_v3` / `subset_statistics_v3` / `run_evaluation_v3` + 独立 markdown 渲染）+ `configs/evaluation/eval_v3.yaml`（provisional，sha `e897e8c893c4e08133a5c8067a532f86a68b87a0d4e5b07f8111ef880328a496`）。
- **验证**：新增 50 测试（BEV 49 键形状 pin + (17.75, 0) 验算例 + 四桶左闭右开边界 + signed 对抗例；归因 conditional 双向 + 七行边界表 + id 桥接错位；evaluator 51+1 键集 pin + T6 + byte-identical 重跑 + bootstrap +5 键；config 逐值 pin 与词表同源 import 校验），全量回归 **492 passed**；30-anchor 真实数据冒烟 SMOKE PASS（byte-identical 重跑、signed 分离读数、归因 micro/macro=1.0、bootstrap v3 新键恰为 5 键）。
- **实测记录**：eval 台账 §10（v2→v3 变更登记 + 勘误）、实施规划 §5.5 执行记录、冒烟产物 `runs/S13_eval_v2_and_contract_v7/step5_smoke/`。

## 9. Step 6 CPU 准备（2026-10-14）：SFT 样本 v7_1f + v7 run configs 三件

- **定位**：实施规划 §6 的 CPU 可完成部分（GPU 阶段前置），无新拍板——全部数值沿用既有裁定（V7-A2 生成预算 700 + 紧凑发射、S12 训练超参冻结件、A1 max_pixels=360000）。
- **SFT 样本 v7_1f**：A 层（token 选取）实测确认与 v7 train 集合逐项相同（22,941）→ 跨 face 复用免跑；B 层 `serialize --subsets train_full` 重打 22,941 样本，逐样本 v7 face parity 断言全过；确定性复跑（两个 cwd）file_sha256 逐字节一致。产物 `data/sft_samples/v7_1f/`（train_full.jsonl sha `bafd2223…`，manifest sha `27ef5f2c…`，git_commit=88ac5671 与 builder_commit 同源）。
- **v7 run configs 三件**：`base_1f_v7.yaml`（**2026-10-14 开跑前冻结**，CLI 强制 frozen；冻结 sha256 `69cd4adda7d9b904eae19a76fad725f94231272e65ae82c9b84a8c910d1da94b`，config 内不放自引用 sha）/ `sft_1f_v7.yaml` 训练（紧凑 target_dump + 700，provisional 待训练前冻结，sha `440b88a3…`）/ `benchmark/sft_1f_v7.yaml` 评测（adapter_path 占位 GPU 阶段定稿，provisional，sha `03dc8b80…`）。v6 冻结 config 零改动。
- **Base v7 全量走 batch=16（2026-10-14 用户拍板）**：新建 [base_1f_v7_batched.yaml](../../../configs/benchmark/base_1f_v7_batched.yaml) 独立预注册面（S12 Step 6b 先例同款：`runner: batched_v1` + `batch_size: 16`，逐字段复制冻结面，frozen，sha `f9d986a3186d63bc6d3003155d66e663cbfcb4b7ca3e8a6b15b30690061efc7f`；bs=1 冻结面全量暂缓）。batched 冒烟 8 条一次通过（64s 含装载）；**bs=1 vs bs=16 同 8 token parity 预检：3/8 逐值一致、5/8 差异**（1 条 schema_failure↔json_parse_failure 翻转 + 4 条 output 数值差）——S09 备案预料的 padded attention 数值路径差异，无崩坏；本面读数标注"**待 parity 复核**"（S12 先例同款），不得与 S09/S11/S12 bs=1 冻结面读数混用；**S13 内部 paired-delta（base_v7 vs sft_v7）两侧同用 bs=16，runner 差异在配对中自洽**；受影响的仅是跨代绝对值对比。SFT v7 评测面建议同走 batched_v1（bs=16）保持两侧同 runner。
- **Base 面分辨率对齐 SFT（2026-10-14 开跑后口径变更，用户拍板）**：用户原话"S12 之所以这么做是因为时间紧急""必须把 base 的分辨率调整到和 sft 一致"——`base_1f_v7.yaml` 与 `base_1f_v7_batched.yaml` 追加 `model.processor.max_pixels: 360000`（与 SFT 评测面 A1 同值），v7 内部 base/sft paired-delta 两侧视觉 token 口径一致；跨代 base_v6（全分辨率）对比含分辨率因素，解读注明。旧口径（全分辨率）sha `69cd4add…`/`f9d986a3…` 作废，新冻结 sha：base `504c3070…`、batched `18167738…`；旧口径阶段产物（bs=1 313 条 + bs=16 288 条，2026-10-14 深夜应用户指令删除，不留归档）不入评测。新口径冒烟：tok_in 2458→1082（−56%）、bs=1 parse 6/8 无回归、batched gen 均值 6.5→4.9s（KV 变小步时延下降，bs=16 加速比改善）。
- **验证**：`TestSftV7ConfigParity` 5 测试（代际/预算/DUMP 同源/训练面逐值同 v6/A1 两侧同值），全量单测 497 passed。
- **Base v7 评测 bbox 坐标系修正（2026-10-15，eval v3 首跑暴露）**：base（未训）模型在 max_pixels=360000 下输出的 bbox_2d 是 **784×448 resize 空间**坐标（无训练教"原图绝对坐标"约定），eval 直接匹配 → 对象侧全零（object_f1=0、bev_pairs=0、归因全 0）。三证据链定案：S11 A1 节 processor 实测表（360000→784×448）；数据裁决（scale 1600/784, 900/448 下 400 anchor 产生 485 个 IoU≥0.5 匹配，identity/1596×896/896×504 候选为 0/0/26）；tok_in=1082=448 vtok+634 文本。修复分两阶段（第二阶段 2026-10-15 用户裁定升级，原话"你现在是手动进行了bbox的rescale，你其实应该把他们写在eval的过程当中吧"）：
  - **过渡态（当前 eval v3 读数的依据）**：CPU 后处理反变换（scale_x=2.040816、scale_y=2.008929，14770 框，round 1 位小数）→ `predictions_1f_rescaled.jsonl` 为评测输入；原 `predictions_1f.jsonl` 保留原始发射值。修复后读数：parse_rate=0.9164、object_f1_micro@0.5=0.3764、matched_iou_mean=0.7365、motion_state_f1_macro=0.2896、yield_required_f1=0.369（miss 0.3113 / overcons 0.6628）、stop_miss_rate=0.9973、BEV pairs=5757（dxy_mean=24.99m、p90=41.31m，未训模型量级）、risk_attribution micro/macro=0.2116/0.2144、bootstrap_ci 31 键（22+5+4，n=2000/seed=20261003/0.95/150 scenes 全预注册）。
  - **终态 = parse 端内联变换 + config 声明**：[output.py](../../../src/drivealign/contracts/output.py) `parse_structured_output` 加可选 `bbox_source_size`（output 构造后、semantic 检查前按 (image_w/784, image_h/448) 反变换——bounds 检查与 coarse_position 消费的都是原图坐标；给 bbox_source_size 不给 image_size 显式报错；缺省 None 零行为变化）；[structured_runner.py](../../../src/drivealign/inference/structured_runner.py) `SampleRequest` 加 `bbox_source_size` 字段透传，[batched_runner.py](../../../src/drivealign/inference/batched_runner.py) 同透传；[base_benchmark.py](../../../src/drivealign/cli/base_benchmark.py) `build_benchmark_request` PIL 实读 image_size（顺带启用此前未启用的 bounds 检查/coarse_position）+ 两个 benchmark CLI 从 config `bbox_output_space: [784, 448]` 读入。**predictions 文件约定自此 = 原图坐标**，evaluator 纯函数零改动。config sha 变更链：batched `18167738…`（全量 run 实跑口径）→ `a3480104…`、base `504c3070…` → `1fe77aea…`（`bbox_output_space` 是纯 parse 侧声明，不参与 request hash，生成行为零变化）；SFT 评测面 `benchmark/sft_1f_v7.yaml`（sha `03dc8b80…`）**不设此字段**。
  - **验证**：新增 5 单测（变换正确性 / 缺省零行为变化 / 缺 image_size 报错 / 变换先于 bounds 检查与 coarse_position / 变换后越界 semantic 失败），全量回归 **502 passed**（497+5）。
  - **冒烟等价性验证（2026-10-15，GPU bs=16 --limit 16，同批组合）**：16/16 raw_text 与全量 run 首批逐字节一致（批内确定性复证）；66/66 框 vs `predictions_1f_rescaled.jsonl` 误差 ≤0.05（最大 0.0500 = 手动版 round 1 位小数舍入）——内联变换 = 手动后处理严格等价。**冒烟同时暴露两项新启用行为**（build_benchmark_request 实读 image_size 首次激活 bounds 检查与 coarse_position）：① 真 OOB 拒判——全量 2/5468（0.04%）模型发射超源空间的 bbox 反变换后越原图界 → SEMANTIC_RANGE_FAILURE（S03 契约"从不静默修复"，正确行为；旧流水线不传 image_size 时该检查休眠，手动 rescale 文件保留了它们——当前 eval v3 报告读数含这 2 条，口径差异已在案）；② **边界浮点误判 bug 及修复**——乘法在前 `448*(900/448)=900.0000000000001>900` 把贴源空间底边的截断物体误判越界（冒烟 token 007c057c 实证）；修复 = 除法在前 `(v/src)*dst`（贴边精确映射，`(448/448)*900=900.0`），新增贴边回归 pin 测试，全量回归 **503 passed**；修复后对该 token 本地重 parse 成功且与 ref 全框吻合（≤0.049）。
  - **注意：SFT v7 评测不需要 rescale**（训练 target 教的就是原图坐标），paired-delta 时两侧评测输入口径按此条执行；counterfactual（blank/shuffled）GPU 产物在新口径下天然原图坐标，无需任何后处理。已跑的 blank/shuffled（各 200 条，config sha `18167738` 旧口径，bbox 仍为 resize 空间）建议新 config 重跑（各 ~10 分钟）以统一产物口径。
- **eval v3 定稿报告 + eval_v3.yaml 冻结（2026-10-15）**：eval_v3.yaml `provisional → frozen`（sha `fae363ad…`，eval 台账 §10.4 登记）；定稿报告 `runs/S13_base_1f_v7_batched/eval_v3_cf/`——主输入 rescaled 文件 + 新口径 counterfactual（blank/shuffled 各 200，config sha `a3480104`，bbox 天然原图坐标）。主指标与首跑逐值一致（同输入确定性复证）；**反事实读数（base 未训基线）**：blank output_change=1.0（200/200 输出全变）、action_flip=0.6977（86 可比对中）、shuffled output_change=1.0、action_flip=0.5515（165 可比对中）——blank 可比对仅 86/200（blank 图下 57% parse 失败），n_compared 差异本身即 base 不稳定性证据；正式的 BEV 视觉依赖判读留 SFT 后 paired-delta（BEV rules §6.1）。过程记录：用户重跑 counterfactual 覆盖同名文件为**追加写**（400 行 = 旧口径 200 + 新口径 200），触发 evaluator duplicate-token fail-fast（防线生效）；已去重保留每 token 最后一次出现（= 新 run），原 400 行文件未留档（旧口径产物本已作废）。首跑报告目录 `eval_v3/` 保留作过程记录，**定稿引用一律以 `eval_v3_cf/` 为准**。
- **SFT v7 训练前置：sft_1f_v7.yaml 冻结（2026-10-15，GPU 训练前）**：`configs/sft/sft_1f_v7.yaml` provisional → frozen（冻结 sha256 `a9eb6c2e391ed6434d77e4beb5b61566b4211e9ac23f2665a99691e07f354b5b`；上方条目的 provisional sha `440b88a3…` 作废）。开训前逐值核对完成：与 S12 冻结件 `sft_1f.yaml` 全字段逐值一致，唯一差异 = 预期代际字段（run_name / contract_version=v7 / target_dump.separators 紧凑 / max_new_tokens=700）；LoRA（r=16 / alpha=32 / dropout=0.05 / 路径限定正则 target_modules）、优化调度（batch/accum 1/4、lr 5e-5、1 epoch、cosine、warmup 0.05、wd 0.01、clip 1.0、gradient_checkpointing、save_steps 500 / limit null）、max_pixels=360000、seed=0 逐值同。样本复核：`data/sft_samples/v7_1f/train_full.jsonl` 22,941 行 sha `bafd2223…` + manifest sha `27ef5f2c…` 与 §9 SFT 样本条目登记逐位一致。parity pin 同步（test_status_gate_alignment：训练件 frozen / 评测件仍 provisional 待 adapter 定稿）；顺带修正既有 stale pin（test_eval_evaluator_v3 的 eval_v3.yaml status 仍钉 provisional——yaml 已同日冻结于 `fae363ad…`），全量回归 **503 passed**。训练 CLI `sft_train`（status gate 强制 frozen；不传覆盖开关即全走冻结值）；训练命令与 pre-flight 见实施规划 §6.2。SFT 评测面 `benchmark/sft_1f_v7.yaml` 保持 provisional（adapter_path 占位），adapter 定稿后冻结。
- **base v7 parse_failed 归因 + "格式不后处理挽救"裁决（2026-10-15）**：457 条失败全量归因完成（分析存档 [qwen-1f-benchmark-v7/parse_failure_analysis.md](../../experiment_record/qwen-1f-benchmark-v7/parse_failure_analysis.md)）：截断主导（308/5468 打满 700 预算，298 条 json 失败中 284 尾截 + 9 括号失衡 + 5 fence 提取缺口——`_FENCED_ENVELOPE` 要求闭合```，开 fence 无闭合的 5 条 strip 后 5/5 过 validator）；schema 155 条三类主因（自创 bev_z 45 锚点 / critical_objects 误带 objects 54 锚点 / risk_factors.objects 引用悬空 id 32 锚点——大 id 实证 = bbox 坐标串位 0dc3d1fb）；semantic 4 条（oncoming_traffic 引用未声明 id）；fence 包裹是 base 惯性（69.3% 成功样本也被提取），不致命。**用户裁决：格式问题不走后处理挽救**——parse 侧零改动（提取正则不放宽、附加属性不清洗、悬空 id 不重映射），S13 冻结 parse/eval 面 byte-frozen；格式纪律由 SFT 监督文本直接覆盖（紧凑 separators + 精确 schema 形状 + 归因引用 id），预期 SFT parse_rate 0.9164→≈1.0，parse 修复构成 paired-delta 收益的一部分。700 预算维持 V7-A2 冻结值（base 5.6% 打满 = 未训惯性；SFT target 最坏 620，SFT 评测后复核打满率≈0，若显著非零升级台账）。
- **SFT v7 训练完成 + 评测面冻结（2026-10-15）**：1-epoch 训练收口——global_step **5,735**（=22,941/4 上取整）、epoch 1.0、train_loss **0.4499**（尾段 0.37–0.46 无 NaN，日志 0 error）、wall 13,933 s（**2.43 s/step**，S12 2.59）、峰值显存 10.65/30.54 GiB；LoRA 落点断言通过（adapter_config.json r=16/alpha=32/dropout=0.05/文本层正则与冻结件逐字一致，visual. 零命中）。**best 冻结裁定：best ≡ 末点 adapter**（项目从未执行 S12 Step 5 val 选 best，1-epoch 训练不做 checkpoint 选择；`checkpoint-5735` 与 `--out` 根 adapter 同轮落盘）——best/ 为末点字节拷贝 + `best_freeze.json` 溯源，adapter sha **`7facb03d…`**。训练收尾自测（reference_generation，00014584…）：紧凑 separators / 无 bev_z 与 objects 串位 / 归因引用 id / bbox_2d 原图坐标（904–1020 ∈ 1600×900）、out_tok=142 ≪ 700——[parse_failure_analysis.md](../../experiment_record/qwen-1f-benchmark-v7/parse_failure_analysis.md) 的三条 SFT 覆盖预期全部成立。`configs/benchmark/sft_1f_v7.yaml` provisional → frozen（冻结 sha **`c462db1d…`**；adapter_path 定稿 `runs/S13_sft_1f_v7/checkpoints/sft_1f_v7/best`，SFT 面无 `bbox_output_space` 字段）；parity pin 同步（三件全 frozen），全量回归 **503 passed**。评测命令与预计耗时见实施规划 §6.3（主 5,468 bs=16 batched_v1 同 runner + CF blank/shuffled 各 200 同子集）。
- **SFT v7 训练-推理上下文错位（ctxbug）修复 + 重训裁定（2026-10-15，用户拍板"先修复"）**：SFT smoke8（8 锚，frozen config `c462db1d`，adapter `7facb03d`）7/8 parse_ok，唯一失败 0018da40 打满 700 预算——**发射格式为 `json.dumps(indent=4)` 美化 JSON，非训练监督的紧凑单行**，内容（键序/字段集/id 引用/无 fence）全对。根因三步实锤：①[sft/dataset.py encode_chat](../../../src/drivealign/sft/dataset.py) 旧实现用 `apply_chat_template(add_generation_prompt=False)` 渲染上下文后直接拼 target——训练态边界 = `...<|im_end|>\n{target}`，**22,941 条样本上下文从未出现 `<|im_start|>assistant\n` 开仓标记**；②推理端 `generate_one`/batched_runner 均 `add_generation_prompt=True`，模型站在训练未见过的状态上续写，格式回退 base 先验；③旧 NOTE"误用 True 会目标位错位"的论证不成立（True 渲染恰好止于开仓标记，target 接在其后位置语义正确）。影响量化：缩进发射 8 对象 ≈720–760 tok > 700，GT 7–8 对象锚点 698/5468（12.8%）→ 主跑 parse_rate 预期 ~0.87–0.93 且失败偏繁忙场景，paired-delta 被污染。**修复**：encode_chat 渲染改 `add_generation_prompt=True`（target 接在开仓标记后，labels 监督位 = target + im_end 不变）+ NOTE 更正 + 新回归钉 `test_encode_chat_context_alignment_with_inference`（断言上下文段以开仓标记收尾 + 监督位结构），全量回归 **504 passed**。**处置**：buggy 产物整目录归档 `runs/S13_sft_1f_v7/checkpoints/sft_1f_v7_ctxbug/`（adapter sha `7facb03d…` 作废，不入评测；smoke8 predictions 留根目录作诊断证据）；重训 fresh 落回原路径 `checkpoints/sft_1f_v7`——**`benchmark/sft_1f_v7.yaml` 的 adapter_path 指向不变，冻结 sha `c462db1d…` 仍有效；训练 config `sft_1f_v7.yaml`（`a9eb6c2e…`）与样本（`bafd2223…`）零改动**。样本 JSONL 不含编码结果，不受本修复影响（parity 钉不受影响）。备注：S12 v6 adapter 带同一隐患（从未评测，无既有结论被推翻；如未来评测 v6 需以修复后编码重训）。**归因对比实验定案（2026-10-15，用户质疑后执行）**：三臂 A/B（推理态/训练态/训练态+强制首 token，同一 buggy adapter × smoke8 八锚，判读标准跑前预注册）——A 臂 8/8 美化发射（含逐锚复现 smoke8 token 数）、B 臂 8/8 紧凑单行，0018da40 同 8 对象内容训练态 515 tok 完整收尾 vs 推理态 700 截断，内容跨臂一致仅格式翻转 → **H1（上下文错位归因）成立**，H2（LoRA 欠拟合格式 token）排除，重训按原方案执行；证据与全文见 [S13_v7_ctxbug_ab_experiment.md](S13_v7_ctxbug_ab_experiment.md)（原始输出归档 `runs/S13_sft_1f_v7/ctxbug_ab_evidence/`）。**归档精简转移（2026-10-15，用户拍板）**：buggy 产物自 `runs/S13_sft_1f_v7/checkpoints/sft_1f_v7_ctxbug/` 整体移至 workspace 根 `archive/sft_1f_v7_ctxbug/`（best/ + 根 adapter + train_state_ref 等溯源件保留，11 个中间 checkpoint-\* 删除释放 ~4.1G；best adapter sha `7facb03d…` 移后复验无损）；同轮 `reports/step3_train.log` 改名 `step3_train_ctxbug.log`（保 buggy 训练日志，防重训 `tee` 覆盖）。
- **SFT v7 重训收口 + best v2 冻结（2026-10-16）**：ctxbug 修复后 fresh 重训按 §6.2 命令执行完毕——global_step **5,735** / epoch 1.0 / train_loss **0.4459** / 13,935 s（2.43 s/step）/ 峰值 10.65 GiB，日志零异常；训练 config（`a9eb6c2e…`）与样本（`bafd2223…`）sha 与首训逐位一致（冻结面零改动）。新末点 adapter sha **`da1971e4…`**；训练收尾自测（reference_generation，00014584）out_tok=82 紧凑单行（训练态续写，与 A/B 实验 B 臂行为一致）。**best v2**：`checkpoints/sft_1f_v7/best/` 重建为末点字节拷贝，`best_freeze.json` 标 `v2（ctxbug 修复重训）`（含 code_note 指向回归钉）；`benchmark/sft_1f_v7.yaml` adapter_path 指向不变，冻结 sha `c462db1d…` 仍有效，评测面零改动。buggy smoke8 产物改名保留：`predictions_smoke8_ctxbug.jsonl`（A/B 实验对照基准，实验文档 §6 已同步）+ `reports/step4_smoke8_ctxbug.log`；smoke8 将按 §6.3 原命令 fresh 重写 `predictions_smoke8.jsonl`。**smoke8 复验通过（2026-10-16，wall 34.8 s）**：8/8 parse_ok、8/8 紧凑单行、fence 零命中、无打满（max out_tok=515——0018da40 上轮 700 截断本轮完整收尾；输出 bbox 为原图坐标，如 0018da40 首框 [1439.1, 478.0, 1600.0, 633.4] 含 x2=1600 图缘，符合 SFT 面无 rescale 预期）；且 7/8 锚点 token 数与 A/B 实验 B_free 臂逐位一致（277/328/515/94/440/513/276；仅锚 8 为 342，两臂 adapter 不同）——重训模型推理态行为已对齐训练态，ctxbug 修复效果直达。日志仅存既有无害告警（temperature+do_sample、tokenizer fork），与 S12 同。
- **SFT v7 正式评测 + paired-delta + BEV 反事实验证（2026-10-16）**：三跑产物齐（主 5,468 + CF blank/shuffled 各 200，CF 按 §6.3 注册命令 `--anchors-file` base 同子集）后 eval v3 定稿（frozen `fae363ad` 口径）→ 存档 [qwen-1f-sft-benchmark-v7/](../../experiment_record/qwen-1f-sft-benchmark-v7/README.md)。**核验**：主跑 parse_ok 5,465/5,468（99.95%，3 条失败全为 bbox x1>x2 倒置 schema 拒绝）、打满 0（max 563）、紧凑单行 5,465/5,465、fence 零；CF blank 200/200 且仅 3 种空对象输出。**G4 通过**：P delta +0.2301 [CI +0.2087,+0.2504]、Q delta +0.4283 [CI +0.4003,+0.4564] 双显著为正。**headline**：object F1 micro@0.5 0.376→0.606、speed_action F1 0.144→0.573、yield F1 0.369→0.701、BEV dxy_mean 24.99m→1.72m（分桶 0.65/1.40/2.62/5.57 单调条件化）、empty-scene hallucination 1.0→0.057。**6 条预注册守卫机械触发，逐条解读为退化基线伪影**（base blank 57% parse 失败抬高的 flip、base 恒不 STOP 致 overcons 恒 0、base 垃圾输出贡献的 100% output_change；SFT 非常量预测器有独立证据）——S13 语境下守卫判据沿用 S12 预注册表，机械触发不构成阶段阻断，最终阶段判定留人工复核。**BEV rules §6.1 通过**：blank 物体归零 + shuffled 响应 76.5% + 误差随 GT 距离单调条件化，否定先验抄近路。**过程事件（如实登记）**：CF 首跑两次命令交付错误——①参数名 `--counterfactual`（实为 `--image-transform`，argparse 前置拒绝无副作用）；②误用 `--limit 200` 而非注册的 `--anchors-file` 子集 → 锚集与 base 不可配对，mis-caliber 产物改名 `*_limit200` 保留、eval 重跑定稿。遗留：BEV §6.2 截断/近距审计；SFT 端到端 ~45-50 s/批数据管道开销插桩（两侧同款，非本阶段）。
- **S13 阶段判定与收官（2026-10-16，用户拍板）**：**SFT v7 评测判定通过**——依据：G4 双指标 CI 显著为正（P +0.2301 / Q +0.4283）、6 条守卫触发逐条解读为退化基线伪影（SFT 非常量预测器有独立证据）、BEV rules §6.1 反事实验证通过（blank 物体归零 + shuffled 响应 + 误差随距离单调条件化）、**§6.2 截断/近距审计通过**（SFT：0 截断，近距出画 864 锚 dxy 1.28m / 近距 1,879 锚 1.44m 均低于全局 1.77m 且配对密度更高，§6.2 "底边线索失效误差偏高"预期证伪；base：308 截断锚仅 8 个有 bev 配对——截断危害是配对缺失而非误差膨胀；两侧均不污染整体读数，审计已回填两份存档 README）。**单测终回归 504 passed**。S13 全链收口：contract v7（标定/冻结/数据重建）→ eval v3 → base 评测 → parse 归因 → SFT 前置 → ctxbug 发现/A/B 实验/修复/重训 → smoke8/主跑/CF → eval v3 定稿 → paired-delta → §6.1/§6.2 验证 → 存档互链。**移交下阶段候选**：①数据管道插桩（~45-50s/批，评测 5h→20min 级）；②STOP 召回（stop_miss 52.4%）；③v6/v7 受控对照（隔离 ctxbug 因素）；④4F face 训练扩展。
- **实测记录**：实施规划 §6.1 执行记录。
