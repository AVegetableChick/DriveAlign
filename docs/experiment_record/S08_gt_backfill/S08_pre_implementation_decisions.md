# S08 实现前决策台账（2026-10-01）

> 状态：范围对齐完成；P2 / P3 / contract v4 **已决**；**P1 结构与 P4 结构均已确认冻结（2026-10-02）**——P1 剩余全部为数值（归 P1.2 train-only 标定），P4 剩余为模板句式撰写（实现细节）；P5 框架已定、物流细节随实现规划。
> 定稿后本文件即 S08 构建的规则来源；实现时数值以 `gt_rule_config` / `reasoning_templates` / `enum_phrase_map` 三件资产落 `configs/contracts/v4/`。

## 1. 范围决议（已决）

| 决议 | 内容 |
|---|---|
| S08 范围 | 填全 train/val/test 全部 valid 记录 GT（train 22,941 / val 2,531 / test 5,468 = **30,940**；quarantine 659 条不回填）+ 同一确定性规则引擎冻结 M09 Evaluation Oracle。方案文档已对齐：`project_stages/08_m09_oracle.md`、`project_stages/README.md`（索引行 + 执行位置） |
| P2 相机口径 | **只含 CAM_FRONT**，扩视角留待未来 contract 级 ablation。视锥外关键对象由 observability 门控排除——**门控在前、top-8 在后**（反序会让不可见对象挤占名额缩小打分集合）；"相关但不可见"比率进 observability/coverage 报告按 split 量化，记为已知限制。`select_objects(rule_config)` 参数化预留相机集合扩展 |
| P3 DriveLM 增强包 | **关闭**，后续是否加入未定。`language_reference` 恒 None（字段零成本保留，彻底否决后可在下一代修剪）；D5/D6 冻结设计继续封存，不执行 |
| Contract v4 | **新建**。模型面（prompt / schema / parser / serializer）零改动 → v4 重建后 `request_hash_1f/4f` 必须与 v3 anchor manifest **逐值相同**（回归 gate）；`record_version` 不动（training_targets 是 Record v1 预留字段，填值不改 schema）；S03 §9 的 schema v4 候选（coarse_distance / 信号灯类）不进本次 v4，信号灯类在 P2/P3 前提下实质永久关闭 |
| 数据资产 | 按 D3 既有约定：v4 数据集建**新物理目录** + symlink 切换，v1 只读保留（request_hash parity diff 与回滚依赖）；物理盘写入由用户在 tmux 执行 |

## 2. P1：GT 侧规则（暂定，待批注）

### P1.1 候选池几何（纯几何两步，附门控顺序）

1. **observability 门控在前**：CAM_FRONT 视锥内判定（规则见 P1.3），视锥外 / behind → unscorable，排除出候选池
2. **几何相关过滤**：ego heading **前向锥**（偏角 ≤ ±30° 且距离 < 类别阈值）∪ **全向近距盘**（距离 < 10m，覆盖图像边缘可见的侧向近距对象）
3. **距离升序 top-8**（与模型 `maxItems: 8` 对齐；截断与打分集合同一实现；距离平手按 sample_token 字典序决胜——确定性要求）

- **可见性优先原则（2026-10-02 用户确认，撤回同日早先的"corridor 冲突第三准入通道"提案）**：先用门控过滤 front 可见对象，再用这些对象决定池与 risk_factors——**视野外的对象不产生任何 risk factor**。corridor 冲突测试只在池内对象上运行，"冲突对象在池外"的矛盾按定义不存在。设计含义：benchmark 以模型传感器视野为参照系（agent-relative risk），模型看不见的风险不是模型的职责，与 visual dependence 评测哲学同源。已知限制：front 视野外的真实冲突（如转弯时侧前方横穿）不被记录——该限制随未来视角扩展（前向三视角）自然缓解，届时只扩展门控模块、几何规则不变
- 备选（**不进 v4**）：车道级方案（NuScenesMap lane 归属）语义最忠实，但路口归属模糊需 fallback 链、规则复杂化——列为未来精化
- **全向近距盘的作用域澄清（2026-10-01 讨论）**：侧后方完整不可见对象在 ① 已被剔除，盘不"捞回"任何看不见的对象。全向定义的有效作用域 = **门控幸存者的角度豁免**：(a) 前视 hfov（≈±35°）与前向锥（±30°）之间的边缘带近距对象；(b) 大型目标侧前方贴身、车身一角部分入画（inferable，clip 后可评）的场景。设计理由：相关性轴（几何规则）与可见性轴（门控）分离——规则不含相机信息，未来扩视角 ablation 只改门控不改几何规则；写死前向半盘（±90°）会把相机配置烤进几何规则
- 待拍板：锥角 ±30°、近距盘 10m、类别距离阈值量级（标定输入参考 LeapAD：车辆 20 m / 本车道 60 m、行人 40 m）

### P1.2 阈值标定程序

- **只用 train split**（硬约束：val/test 不参与任何 fitted state），标定后冻结、全 split 应用
- 标定目标：top-8 **截断帧占比 < 5%**；池内 observable 对象数分布 P50/P90 报告；不可见率报告（P2）；防退化参数（P1.3）
- 标定参数扩至 corridor 族：T_risk 外推时域、每类 buffer d_risk、corridor 宽度余量（全部 train-only）
- 数值冻结进 `gt_rule_config`，每条 Derived GT 记录 rule version / thresholds（M09 Gate 溯源要求）

### P1.3 门控与 clip

- 沿用 S04 已定规则：8 角点投影 → 2D AABB（min/max + clip 图像界）；**含 behind 角点 → 弃用 / unscorable**
- 精确三分类（2026-10-02 精确化）：(a) **behind 检查**——任一角点深度 ≤ ε → unscorable 整框丢弃（防跨相机平面的退化投影）；(b) 全角点在相机平面之前：AABB 与图像不相交 → unscorable；AABB 完整在界内 → observable（bbox = AABB）；AABB 越界但相交 → **inferable**（bbox = clip 后可见部分）。**部分可见的精确定义 = 全角点在相机平面之前 + 投影框越过图像边界**；与 ego 并排、尾部角点在平面之后的近距大目标被 behind 规则挡住（保守合理：退化框 clip 无意义）
- GT 两侧同源：expected_output 的 bbox_2d 与 M09 IoU 打分均用同一个 clip 后框（同一实现）
- 可选防退化参数（进 P1.2 标定候选）：clip 后有效面积占比 / 可见角点数下限（devkit vis_threshold 同思路），不达标判 unscorable
- **observability 三态（2026-10-02 用户确认，由四态简化）**：observable / inferable / unscorable。原 privileged 态（"需未来信息方可判定关键性"的豁免通道）在 GT 派生全面改为 t0 证据口径后恒为空集，删除；`project_stages/08_m09_oracle.md`、`codebase/05` 蓝图已同步修订。功能定位：该标签 = 评测分母资格审查（unscorable 不进分母，防"考不可能看到的对象"）+ 训练标签可学性过滤（behind 对象无 bbox_2d 可写）+ 每帧剔除原因审计（不可见率报告的数据源）

### P1.4 motion_state GT 数值（示例值，train-only 标定后冻结）

v3 `motion_vocab.json` 四值枚举不变；GT 派生规则修订（2026-10-02 用户确认）：

| 枚举 | 暂定规则 |
|---|---|
| `stationary` | box speed < 0.5 m/s（t0 速度） |
| `same_direction` | 运动方向与 ego heading 偏角 ∈ [0°, 45°) |
| `crossing` | 偏角 ∈ [45°, 135°]（**纯运动学**，不再要求 corridor 相交） |
| `oncoming` | 偏角 ∈ (135°, 180°] |

- **三带铺满（2026-10-02 确认）**：角度带铺满 [0°,180°]，任何非 stationary 运动对象恰落入一带——消除原示例带（30°/60°）在 30°–60°、120°–150° 留下的斜插对象死角（如 45° 斜穿的电动车）；带边界数值仍可经标定微调（保持对称：same <θ、crossing [θ, 180−θ]、oncoming >180−θ）

- **修订理由**：(a) 枚举死角——v3 gt_derivation 要求 crossing = 横向 ∧ corridor 相交，则"横向移动但到不了 corridor"的对象（如人行道横走行人）四值无处安放；(b) 职责分离——motion_state 管"怎么动"，risk_factors 管"危不危险"，corridor 相交条件由 risk 层承担（`pedestrian_crossing` / `corridor_conflict` 本就含该条件）；(c) v3 crossing 同义词表自带 "lateral movement"、"cutting across"，运动学读法与词表自洽
- 判定顺序（先特殊后一般）：stationary → crossing → oncoming → same_direction
- **偏离记录**：v3 gt_derivation 字符串（"future displacement"）为 GT 侧指引、不在 prompt 中；修订进 `gt_rule_config` 记 rule version，模型面零改动

### P1.5 risk GT

8 项 gt_derivation 已冻结于 `risk_taxonomy.json` v3；S08 = 实现规则 + 补齐 preregistered 数值：

- **`small_following_gap` 改 t0 双侧 CV 口径（2026-10-02 用户确认）**：t0 间距与双侧 CV 外推的窗内最小间距 < 阈值即成立——v3 原文 "future 最小距离" 存在与 corridor 同款的自毁陷阱（ego 减速拉开车距 → 实际 future gap 变大 → 漏检跟车过近）；偏离记录同 corridor。**其余项自毁排查结论**：`congestion`（lead 低速 + 排队数）、`stationary_obstacle`（零速窗）、`lead_vehicle_braking`（制动证据窗）均为直接观测量、不因 ego 反应而消失，维持实际轨迹口径；`oncoming_traffic` 朝向规则（不用 corridor，不变）
- **corridor 冲突判定重定义（2026-10-02 用户确认）**，作用于 `corridor_conflict` / `pedestrian_crossing` / `vehicle_merging`，**且只在池内对象上运行**（可见性优先原则，见 P1.1）：
  1. **反事实 corridor**：ego 侧取 **t0 状态恒速（CV）直线外推** T_risk 秒的行驶带（宽 = 车宽 + 余量），**不使用实际未来轨迹**——反应性自毁陷阱：ego 检测到横穿而减速时，实际未来 corridor 变短、不再与行人轨迹相交，risk GT 会与驾驶质量负相关（越会开车 GT 越漏检）。CV 外推 = "若 ego 保持 t0 状态将扫过的空间"，是决策时刻的冲突证据，不被 ego 自己的规避行为抹除
  2. **对象侧对称 CV 外推**：对象轨迹同样取 t0 状态 CV 外推（nuScenes box.velocity 可得），而非实际未来轨迹——对称循环：前车因让行中止 cut-in / 制动，其"实际"轨迹同样自毁冲突证据。整体语义 = "若双方均不反应，T_risk 内是否冲突"（安全分析的标准冲突定义）
  3. **"可能进入"而非"事实相交"**：两条 CV 行驶带在 T_risk 内的**最近距离 < 每类 buffer d_risk**（行人 / 车辆分设，train-only 标定）即判冲突——缓冲吸收定位噪声、CV 近似误差与险些相撞（near-miss）仍需反应的场景
  4. **退化语义**：ego 静止（v≈0）时 corridor 退化为自车框 + buffer——近前横穿仍检出、远距不检出，符合"双方不反应则是否冲突"的反事实语义
- **偏离记录**：v3 gt_derivation 的 "future trajectory / occupancy" 表述被上述 t0 双侧 CV 外推替代；GT 侧指引、非模型面，修订进 `gt_rule_config` 记 rule version
- 同样 train-only 标定，数值进 `gt_rule_config`

### P1.6 speed_action / yield_required GT（待实现规划细化）

- 未来物理派生，供 expected_output 与 M09 action/yield F1 同源使用；具体规则与窗口长度在实现规划中定稿
- **证据分工预注记（2026-10-02）**：实际未来 ego 轨迹只用于"ego 实际做了什么"（speed_action 行为证据）；yield 的**冲突证据**必须用 P1.5 反事实 corridor——否则与 risk 层同一自毁陷阱（ego 让行了 → 冲突消失 → yield_required 判 false）

## 3. P4：reasoning 模板渲染（2026-10-02 用户确认，结构冻结）

| # | 子决策 | 方案 |
|---|---|---|
| P4.1 | 模板库 | 感知 / 风险 / 决策三段，各 **3–5 个句式变体**；`hash(sample_token + 段 salt)` 确定性选变体；**变体只换措辞、不换信息量**（信息完全由枚举 GT 决定） |
| P4.2 | 枚举→短语映射表 | speed_action 4 枚举（ACCELERATE / KEEP_SPEED / DECELERATE / STOP）、yield_required 2 值、risk 8 项、motion_state 4 值 → 固定短语；**一份映射表 P1 规则与 P4 模板共同消费**（联动核心，进 `enum_phrase_map.json`） |
| P4.3 | 空态句式 | 零对象、零风险、四种 speed_action 均有自然句式（渲染非空，schema minLength 1）——实现细节随模板库定稿 |
| P4.4 | 信号灯 | 模板**零提及**（确认抛弃）——词表无此类、taxonomy 无此 risk 项，不提即无矛盾，锚定约束自动满足 |
| P4.5 | 零矛盾校验 gate | 构建 gate + 单测：reasoning 提及对象 ⊆ critical_objects（类别名匹配）；决策 / 风险措辞 ↔ 枚举**双向精确对应**（risk_factors 每一项必须以其固定短语出现在风险段，多余或缺失均 gate 失败）；traffic light 词族正则零命中；渲染非空 |

- **模板语言（确认）**：英文——与 v3 prompt、schema 描述、AutoVLA 血统一致；SFT 目标分布与输出契约同语言
- **定性描述（确认）**：文本不出现距离 / 速度的具体数字（无 "15 m ahead"），只用定性词（ahead / close / crossing 等）——渲染器无需数字格式化，零矛盾校验面更小；量化信息全部由结构字段承载

## 4. 联动与执行顺序

1. P1 数值定稿（用户批注 + train split 标定）
2. P4 模板 / 映射消费同一份枚举
3. 三件资产（`gt_rule_config` + `reasoning_templates` + `enum_phrase_map`）落 `configs/contracts/v4/`（v3 五文件原样继承）
4. 一次性 v4 重建（tmux；builder 扩展未来帧读取——action/yield 与 gap/braking 类 risk GT 依赖实际未来轨迹；motion_state 与 corridor 冲突判定只用 t0 运动学状态）
5. Gate：request_hash parity（v4 == v3 逐值）+ GT 填全 100% + 确定性重跑逐字节一致（复用 S07 dataset_verify 模式）

## 5. 待用户拍板项

- P1.1：锥角 ±30°、近距盘 10m、类别距离阈值量级（可见性优先原则已确认，无第三通道）
- P1.2：截断目标 < 5%；corridor 族参数（T_risk、d_risk、宽度余量）
- P1.4：motion_state 数值示例（crossing 纯运动学 + 三带铺满**已确认**；带边界待标定）
- P4.1：模板库规模 3–5 变体
- P1.6：speed_action / yield 派生规则（实现规划细化；证据分工预注记已定）
