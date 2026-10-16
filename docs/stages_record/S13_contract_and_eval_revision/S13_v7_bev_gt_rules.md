# S13 Contract v7 — Critical Objects BEV GT 定义（拍板稿 v1）

- 状态：**拍板稿 v1**（2026-10-11 五项设计点全部拍板；实现级口径随 v7 八件套落地复核）
- 决策依据：[S13_contract_v7_decision_log.md](./S13_contract_v7_decision_log.md) §2（V7-B1，D-B1-1 ~ D-B1-5，2026-10-11 拍板）
- 机制参照：[S13_v7_risk_factor_rules.md](./S13_v7_risk_factor_rules.md)——本文共用其 §0.1 对象字段与 §0.2 坐标系定义
- 参数基线：contract v6 model face + dataset v4 代际；BEV GT 为 t0 纯派生（nuScenes annotation × ego_pose，见 [backfill.py](../../../src/drivealign/gt/backfill.py)），零新增标定参数
- 单位：距离 m，角度 deg

---

## 1. 拍板总览（D-B1-1 ~ D-B1-5）

| 编号 | 决策点 | 拍板结论 | 拍板日期 |
|---|---|---|---|
| D-B1-1 | BEV 表示法 | ego 系 [x_forward, y_left]，米制，0.1m 取整 | 2026-10-11 |
| D-B1-2 | 参考点 | 包围盒最近点（推翻台账倾向的中心点） | 2026-10-11 |
| D-B1-3 | bbox_2d 去留 | 保留——2D grounding 与既有 IoU 指标链不动 | 2026-10-11 |
| D-B1-4 | 评测匹配键 | 维持 2D IoU 匹配，在匹配对上计算 BEV 误差 | 2026-10-11 |
| D-B1-5 | 新增指标族 | dx / dy / dxy 三族，mean/p50/p90 + 距离分桶，入 scene-cluster bootstrap | 2026-10-11 |

五项合起来 = critical_objects BEV GT 的完整定义：**GT 算什么**（§3）、**模型面长什么样**（§4）、**怎么评**（§5）。

---

## 2. 参考点拍板理由（D-B1-2：最近点，非中心点）

### 2.1 中心点是 amodal 量，最近点锚在可见几何上

预测中心点 = 预测最近可见面 + 推断不可见半边的延展（amodal 补全），多出一项纯推断误差源：

```text
Var(中心点估计) ≈ Var(最近点估计) + Var(尺寸/朝向推断)
```

最近点就是前车车尾、近侧车缘这些直接可见的面，不需要补全隐藏部分。对行人/小目标两者几乎无差，差异集中在大车（nuScenes 前视常见）。

### 2.2 bbox_2d 底边提供近确定性学习通路

模型对同一对象先输出 bbox_2d；其底边中点像素对落地物体可经地面单应反投影出 BEV 位置，该反投影点 ≈ BEV 最近点（nuScenes 驾驶场景目标几乎全部落地）。即新字段存在一条从已输出字段出发的学习通路，模型无需显式估计 3D 尺寸/朝向。中心点的对应通路需叠加目标高度先验（卡车 vs 轿车差异大），噪声显著更大。

单目 3D 检测文献的普遍结论一致：目标中心深度是最大误差来源，主流方法转向底边/地面先验类线索。

### 2.3 连续性与系统偏差

- 最近点映射是凸集投影：坐标随相对位姿连续变化，不存在「最近面切换导致标签跳变」（切换的只是哪条面最近，坐标连续）；
- 高栏车/悬挂式车尾的底边像素与盒体最近点存在小偏移，属类别条件性系统偏差，可学习，不构成阻塞。

### 2.4 评测收益

dx/dy 度量「近表面定位精度」，对规划安全余量比中心误差更有意义；与 D-B1-4 维持 2D IoU 匹配的组合自洽（匹配用 2D、误差用 BEV 最近点，各司其职）。

---

## 3. GT 定义（精确口径）

### 3.1 输入（零新增依赖）

全部来自池内对象既有字段（risk_factor_rules §0.1）：

| 输入 | 来源 | 说明 |
|---|---|---|
| 中心 ego 系坐标 (x_c, y_c) | t0 全局位姿经全局→ego 变换 | 复用 corridor 机制同一变换（[risk.py](../../../src/drivealign/gt/risk.py)） |
| 标注框尺寸 (L, W) | nuScenes annotation | 长沿盒朝向、宽垂直 |
| 盒朝向 θ_obj | nuScenes annotation t0 yaw | 相对 ego x 轴 |

数据源 = nuScenes annotation × ego_pose，t0 时刻即完备，无未来信息泄漏（decision log §2.2）。

### 3.2 坐标系

同 risk_factor_rules §0.2：ego 系原点 = ego t0 位置，x 轴 = 航向前向，y 轴 = 左。

### 3.3 最近点计算

定义：**3D 标注盒在 BEV 平面的投影矩形上，距 ego 原点最近的点**。

```text
输入：(x_c, y_c)、(L, W)、θ_obj
1. 盒局部系下 ego 原点坐标：
     p_local = R(−θ_obj) · (−x_c, −y_c)ᵀ
   盒区域 = [−L/2, L/2] × [−W/2, W/2]
2. 盒上最近点（局部系）：
     q = ( clamp(p_local.x, −L/2, L/2), clamp(p_local.y, −W/2, W/2) )
3. 映回 ego 系：
     p_near = (x_c, y_c) + R(θ_obj) · q
4. 取整输出（D-B1-1）：
     bev_x = round(p_near.x · 10) / 10
     bev_y = round(p_near.y · 10) / 10
```

其中 R(θ) = [[cosθ, −sinθ], [sinθ, cosθ]]。p_local 是 ego 原点在盒坐标系中的位置；clamp 将其投影到盒矩形上，即得盒上距 ego 原点最近点。验算例：前车正前 (x_c=20, y_c=0)、θ_obj=0、L=4.5 → p_near=(17.75, 0)，即车尾中点，符合直觉。

### 3.4 取整口径

GT 与模型输出统一按 0.1m 取整，BEV 误差在取整后口径上计算——两侧位于同一离散空间，避免 0.05m 量级的口径性系统误差进入指标。

### 3.5 边界情况

- **ego 原点落入盒投影内**（物理上不应发生）：clamp 无收缩，p_near = 原点，即 bev = (0, 0)、距离 0——防御性定义，保证计算完备；
- **amodal 语义**：nuScenes 标注盒含被遮挡部分，最近点可能落在不可见面上——模型侧需 amodal 推断，但推断量级小于中心点（§2.1），GT 侧无歧义。

---

## 4. 模型面（D-B1-3 落地）

### 4.1 critical_objects 字段构成

每对象 = 既有字段全保留（bbox_2d 4 数、distance_m 等）+ 新增 `bev_x` / `bev_y` 2 个标量（建议字段名，v7 八件套复制时落定）。bbox_2d 保留的理由：2D grounding 能力承载 + 既有 IoU 指标链（matched_iou_* / object_f1_*）完全不动。

### 4.2 distance_m 并存口径

- `distance_m`：保持 v6 口径不动（t0 传感器系中心距 3D 范数）——object_near 标定与既有语义依赖它；
- `bev_x` / `bev_y`：最近点口径（ego 系坐标，非距离标量）。

两个「距离语义」并存，prompt/schema 必须显式区分：distance_m = 目标中心离 ego 多远；bev = 目标包围盒上离 ego 最近的点在哪。

### 4.3 prompt 措辞要点

bev 字段语义 = "position of the nearest point of the object's footprint relative to ego, as [x_forward, y_left] in meters, 0.1 m granularity"。措辞随 v7 八件套复制时定稿。

---

## 5. 评测面（D-B1-4 + D-B1-5 落地）

### 5.1 匹配键（D-B1-4）

维持 2D IoU 贪心匹配（eval 台账 §2.1 既有匹配计数链路，主阈值 τ=0.5），在匹配对上计算 BEV 误差。不引入 BEV 邻近匹配：会新增匹配阈值超参，且单目深度误差大时 BEV 匹配不稳定；2D 匹配成熟且与既有指标链共用 counting helper。

### 5.2 误差定义（D-B1-5）

匹配对上、取整后口径：

| 量 | 定义 | 符号语义 |
|---|---|---|
| `bev_dx`（纵向误差） | bev_x_pred − bev_x_gt（带符号） | 正 = 预测最近点更靠前 |
| `bev_dy`（横向误差） | bev_y_pred − bev_y_gt（带符号） | 正 = 预测最近点更偏左 |
| `bev_dxy`（合成误差） | sqrt(bev_dx² + bev_dy²)（逐帧合成） | 非负 |

### 5.3 统计量与距离分桶

- 每族统计 mean / p50 / p90；
- `bev_dxy` 独立逐帧合成后统计——**p90(dxy) ≠ 由 p90(dx)、p90(dy) 合成**，实现按独立序列处理；
- 距离分桶键 = **GT 最近点距离** d_gt = hypot(bev_x_gt, bev_y_gt)（与误差定义同口径；不用 distance_m 中心距，避免跨口径分桶）；建议桶 [0, 10) / [10, 25) / [25, 45)，边界随 eval 实现确认；
- 分桶后每桶独立出 mean/p50/p90，桶标识作为命名 aggregate 段。

### 5.4 指标命名与 bootstrap

- 命名按 eval v2 约定 `<field>_<statistic>[_<aggregate>]`（eval 台账 §0）：如 `bev_dx_mean` / `bev_dy_p90` / `bev_dxy_p50`，分桶键名带桶段；
- 三族全部入 scene-cluster bootstrap（同法：n=2000、seed=20261003、0.95，条目八字段）；
- 指标登记进 eval 台账，类别 = 新增；版本归属（eval v2 增补 vs v3）待 eval 侧拍板——`eval_v2.yaml` 已冻结（2026-10-09），本指标族属 v7 model-face 代际，倾向 v3 立项时收编。

---

## 6. 可学习性验证与审计

### 6.1 反事实验证（SFT 后）

目的：验证模型确实使用底边/地面线索，而非类别先验抄近路。判据：误差分布应随图像内容（目标实际位置/朝向）条件化——若误差分布与类别先验完全一致、且在反事实组（blank / shuffled，与 eval 台账 §6 visual_dependence 同思路）下输出不变，即判先验抄近路。

### 6.2 截断与近距审计

截断目标 / 近距目标底边出画的锚点，沿用 gt_truncation_check 思路审计其 BEV 误差贡献（这些锚点上底边线索失效，误差预计偏高，需量化确认不污染整体读数）。

---

## 7. 复核项与待办

| 项 | 内容 | 时点 |
|---|---|---|
| token 预算实测 | max_new_tokens=512 下 8 目标 × (bbox 4 数 + BEV 2 数) 最坏占用（decision log §2.5 遗留项） | v7 八件套定稿前 |
| 字段名与 prompt 定稿 | bev_x/bev_y 命名、§4.3 措辞、schema 更新 | v7 八件套复制时 |
| GT 重派生 | BEV 字段进 record 层，全量重跑 Step 2/3（随 decision log §2.3 代际计划，物理盘写入由用户 tmux 执行） | dataset v7 构建时 |
| eval 指标实现 | §5 全部 + 登记进 eval 台账（版本归属待拍板） | eval v7 增补 / v3 |
| 反事实验证 | §6.1 | SFT 后 |
