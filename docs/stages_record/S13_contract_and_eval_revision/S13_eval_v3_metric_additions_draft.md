# S13 eval v3 指标新增草案（BEV 误差族 + risk 因子归因族）

- 状态：**已拍板定稿（2026-10-14）**——§6 全部拍板/默认关闭（分桶四桶、归因 conditional 双向、signed 方向分解键现在加、分层读数走诊断脚本、counterfactual 不加）；立项拍板 2026-10-13 用户同意 eval v3 立项（Step 5 评测面增补，实施规划 §5 / G7）
- 设计依据：[S13_v7_implementation_plan.md](./S13_v7_implementation_plan.md) §5、[S13_v7_bev_gt_rules.md](./S13_v7_bev_gt_rules.md)（GT 侧 BEV 定义）、[S13_v7_risk_attribution_rules.md](./S13_v7_risk_attribution_rules.md)（V7-A1 GT 侧归因生成）、决策台账 §3.9（BEV GT 侧审计基线）
- 修订基准：eval_v2（[configs/evaluation/eval_v2.yaml](../../../configs/evaluation/eval_v2.yaml)，status: frozen 2026-10-09，byte-frozen 不动）
- 关系先例：本文之于 eval_v3，等同 [S13_eval_v1_to_v2_metric_changes.md](./S13_eval_v1_to_v2_metric_changes.md) 之于 eval_v2

---

## 1. 立项范围与不变量

**eval v3 = eval v2 + BEV 误差族（§2）+ risk 归因族（§3）+ risk 词表 v7 化（§4）**，其余全部不变：

| 项 | eval v3 取值 | 说明 |
|---|---|---|
| matching.iou_thresholds | [0.3, 0.5, 0.7]，主阈值 0.5 | **不变**（D-B1-4）；BEV 与归因都只在主阈值匹配对上计算 |
| bootstrap 三参数 | n=2000 / seed=20261003 / 0.95 | **不变**（S09 预注册值，改动即失去与 v1/v2 报告 CI 可比性） |
| counterfactual battery | blank / shuffled，subset 200 | **不变**；首版**不加** BEV / 归因读数（子集小、空输出占比高，见 §6 待拍板 5） |
| max_items / risk_max_items | 8 / 8 | 不变 |
| gate_thresholds | 不设 | 沿用 v2 裁定（报告不含 PASS/FAIL） |
| evaluation_split | test | 不变 |

- 新建 `configs/evaluation/eval_v3.yaml`（v2 byte-frozen 不动；yaml 冻结于正式报告前）
- 键命名沿用惯例 `<field>_<statistic>[_<aggregate>]`，分桶后缀 `[_bucket_<lo>_<hi>]`
- 实现落点建议：新增 `evaluation/metrics_bev.py` 与 `evaluation/metrics_attribution.py`（纯函数），`evaluator_v2.py` 的聚合扩展为 v3 聚合（新 `evaluator_v3.py` 或 v2 内开 v3 段，实现时定；v2 报告路径行为不得改变）

---

## 2. BEV 定位误差族

### 2.1 定义域与前置条件

- **计算对象 = 匹配对**：预测 critical_object 与 GT critical_object 经 2D IoU（主阈值 0.5）贪心一一匹配后，才有 BEV 读数；未匹配对象不产生 BEV 样本（检测层面的漏检/幻觉由 object_f1 族与空集指标量，不向 BEV 族泄漏）。
- **同离散空间**：GT 侧 `(bev_x, bev_y)` 为最近点对、0.1 m 粒度（BEV rules §4.1）；预测侧 parser 同样 0.1 m 取整（Step 1 已落地 `contracts/output.py`）。误差定义在**两侧同一离散网格**上，不存在量化不对称。
- **GT 值来源**：`(bev_x, bev_y)` 由 nuScenes 3D 标注框几何派生（BEV rules §3），与图像裁剪无关——Step 4 审计（台账 §3.9）已确认出画目标 99.6% 落在 [0,10) 桶、[25,45) 桶为 0，**中远距分桶读数不受底边出画污染**。

### 2.2 三族定义（全非负误差语义，2026-10-13 拍板）

设匹配对中预测最近点为 `(x_p, y_p)`、GT 最近点为 `(x_g, y_g)`（BEV 坐标系：x 前向、y 左向，单位 m）：

| 族 | 公式 | 含义 |
|---|---|---|
| `bev_dx` | `abs(x_p - x_g)` | **纵向定位误差**（沿 ego 前向轴的偏差大小） |
| `bev_dy` | `abs(y_p - y_g)` | **横向定位误差**（垂直于前向轴的偏差大小） |
| `bev_dxy` | `hypot(x_p - x_g, y_p - y_g)` | **定位距离误差**（平面欧氏距离，与轴向无关的总体读数） |

**拍板说明（全非负）**：三族均为误差量，语义统一为非负。早期草案中 dx/dy 曾保留符号（带符号 mean 可诊断系统性偏置），但与"误差"直觉冲突且带符号 p90 语义含混；2026-10-13 拍板全取非负。带符号偏置信息走 §6-3 拍板的方向分解诊断键（2026-10-14 加入，49–52）：**不得使用代数均值**（正负偏差在均值中相互抵消——+2m/−2m 对半分布代数均值为 0，偏置全丢），须正/负子集分别求均值。

### 2.3 统计量与键集

**顶层 9 键**（全量匹配对）：

```
bev_dx_mean   bev_dx_p50   bev_dx_p90
bev_dy_mean   bev_dy_p50   bev_dy_p90
bev_dxy_mean  bev_dxy_p50  bev_dxy_p90
```

- p50/p90 为样本分位数（实现与 `matched_iou_p90` 同一 `_percentile` 线性插值口径）；
- 空集约定：报告切片内无匹配对时键置 `0.0`（沿用 aggregate_v2 空集口径，不置 null）。

**分桶 36 键**（键 = **GT 最近点距** `d_gt = hypot(x_g, y_g)`；**分桶拍板 2026-10-14：四桶 [0,10) / [10,25) / [25,40) / 40+ 开放桶**——用户裁定末桶放大并开放以吸收量程外尾巴，替代早前 [0,10)/[10,25)/[25,45) + 45+ 尾巴两案）：

| 桶 | 键后缀 | 键数 | GT 样本量（v7 全量实测） |
|---|---|---|---|
| [0, 10) m | `_bucket_0_10` | 3 族 × 3 统计 = 9 | 15,729（14.39%） |
| [10, 25) m | `_bucket_10_25` | 9 | 46,241（42.30%） |
| [25, 40) m | `_bucket_25_40` | 9 | 39,072（35.74%） |
| [40, ∞) m | `_bucket_40p` | 9 | 8,268（7.56%；其中 [40,45) 8,147 + ≥45 121） |

分桶键命名示例：`bev_dxy_mean_bucket_25_40`。桶归属只看 `d_gt`（GT 距离），与预测误差无关——保证分桶条件量在模型间可比。

**样本量背书（2026-10-14 实测，v7 三 split 109,310 个 GT critical_objects）**：四桶全有统计支撑（40+ 开放桶 7.56%，train 6,139 / val 716 / test 1,413）；早前待拍板的 45+ 尾巴（≥45 共 121 个，0.11%，上限 45.5，pool 量程几何外溢）被开放桶整体吸收，**§6-1 关闭**；[0,10) 安全攸关段保留独立桶位（近距 yield/braking 判定距离段读数不稀释）。

### 2.4 含义与读数指南

- `bev_dxy_p90`（顶层）= "90% 的匹配对定位误差不超过 X m"——**主读数**（对长尾敏感，SFT 质量评比的区分度高于 mean）；
- `bev_dx_*` / `bev_dy_*` 分解纵向/横向：纵向（深度）误差预计显著大于横向（BEV rules §6.2 预期），两族分开才能暴露深度估计短板；
- 分桶 = **距离条件误差**：近距 [0,10) 桶承载关键决策目标（yield/braking 判定距离段），其 p50/p90 是安全攸关读数；[25,45) 桶是远景分辨率下限参照；
- 与 `object_f1` 的关系：f1 度"找没找到"，BEV 族度"找到了之后位置多准"——两族正交，组合成完整的检测定位评估。

### 2.5 bootstrap（+3 键）

`bev_dx_mean` / `bev_dy_mean` / `bev_dxy_mean` 入 scene-cluster bootstrap（聚类键与 v2 相同：scene token），点估计 + 0.95 CI；`bootstrap_paired_delta` 天然支持两 run 对比（v7 vs v6 模型的 BEV 增量 CI）。p50/p90 与分桶键**不入** bootstrap（分位数的重采样统计昂贵且读数稳定，首版只对 mean 出 CI）。

---

## 3. risk 因子归因族

### 3.1 任务定义

v7 契约下 `risk_factors` 对象化（V7-A1）：每条 = `{factor, objects: [critical_object 的 id]}`——模型不仅报"存在 vru_crossing 风险"，还要指认"是哪几个对象"。GT 侧同构（[S13_v7_risk_attribution_rules.md](./S13_v7_risk_attribution_rules.md)）。eval v2 只有因子名层面的 `risk_factors_f1_macro`，**没有衡量"对象指得对不对"的指标**——本族补齐。

**原子 = `(factor, object)` 对**：一个因子的每个 objects 引用是一个独立判定单位。

### 3.2 id 桥接（关键设计：几何匹配，非编号直比）

预测与 GT 的 object id 是**两个独立的局部编号空间**（各自 critical_objects 数组内顺序 id），数量稍有不齐编号即全体错位——**严禁编号直比**。桥接规则：

```
预测引用 (f_p, id_p)
  → 取 id_p 的预测 bbox
  → 与该帧全部 GT critical_objects 做 2D IoU τ=0.5 贪心匹配
    （与 object_f1 族同一匹配结果，一次计算两处复用）
  → 匹配成功 → 归因对 (f_p, →gt_id)
  → 匹配失败（IoU < 0.5，幻觉/严重偏位对象）→ 归因对无 GT 落点
```

漏检**不产生串位**：GT 对象没有预测对象匹配到它时，仅该对象上产生 FN，其他对象的一对一匹配不受影响（贪心匹配是 bbox 对 bbox，每个 GT 对象至多被一个预测对象匹配）。

### 3.3 计算方法（conditional 双向口径，2026-10-14 拍板）

对每帧、每对（预测因子条目 `f_p`，GT 因子条目集合）：

1. **因子名闸门（双向）**：只有**两侧因子名集合之交**内的因子才产生归因原子——
   - 预测多报 `f_p ∉ GT 因子集`：该条目不计入（因子误报已由 `risk_factors_f1_macro` 计 FP，归因侧不重复惩罚）；
   - GT 有而预测未报 `f`：`O_gt(f)` 不计入（因子漏报同理由因子指标计 FN）。
   - 这就是 conditional（双向）：归因读数 = "因子被正确检出的条件下，对象指得准不准"；unconditional 则把误报因子条目的每个 objects 引用计 FP、漏报因子的 `O_gt(f)` 计 FN——同一错误在两个指标中双计，且 FP/FN 量随引用数耦合（**已拍板 conditional，2026-10-14**）；
2. `f_p` 命中：取 GT 侧 `f_p` 的引用集 `O_gt(f_p)`；对 `f_p` 引用的每个预测对象 id 经 §3.2 桥接：
   - 落点 `g ∈ O_gt(f_p)` → **TP**
   - 落点 `g ∉ O_gt(f_p)` 或无落点 → **FP**
3. `O_gt(f_p)` 中未被任何预测引用（经桥接）覆盖的对象 → **FN**（含两种来源：对象被检出但没被引用；对象漏检——预测侧无从区分，匹配桥接的信息边界）；
4. 汇总：micro = 全原子池直接计 P/R/F1；macro = 六因子各自 F1 的均值（该因子无任何原子则跳过）。

**边界规则表**：

| 情形 | 判定 |
|---|---|
| 预测多报因子（`f_p ∉ GT 因子集`） | 不计入（conditional 双向闸门；unconditional 下每个引用计 FP） |
| GT 因子存在但预测未报 | 不计入（conditional；unconditional 下 `O_gt(f)` 全计 FN） |
| 预测报因子但 objects 为空 | `O_gt(f)` 全体计 FN（"报了因子没指对象" = 全漏归因） |
| 预测引用的对象是幻觉 / IoU<0.5 | FP |
| GT 对象漏检（无预测落点） | 该对象计 FN |
| GT 侧 objects 为空（理论上不发生：因子由对象触发） | 预测引用全 FP（防御口径） |
| 两侧 objects 均空 | 跳过该因子 |

**conditional vs unconditional（待拍板，§6-2）**：unconditional 把因子误报条目的引用也计入 FP（与因子 P/R 耦合，同一错误双重计数）。**倾向 conditional**（解耦、读数语义干净："在因子被正确检出的条件下，对象归因的准确率/召回率"）。

### 3.4 键集

```
risk_attribution_f1_micro          # 全原子微平均 F1（主读数）
risk_attribution_f1_macro          # 六因子 F1 的 macro
risk_attribution_per_factor        # 嵌套块：六因子各自
                                   # {tp, fp, fn, precision, recall, f1}
```

含义读数：`f1_micro` 度整体归因质量（大因子主导）；`per_factor` 暴露结构短板——典型预期：vru_crossing 的 objects 引用最难（小目标 + 引用链最长），若其归因 F1 显著低于 object_near，说明短板在"指认"而非"检出"。

### 3.5 bootstrap（+2 键）

`risk_attribution_f1_micro` / `risk_attribution_f1_macro` 入 scene-cluster bootstrap（同 §2.5 口径）。`per_factor` 嵌套块不进 bootstrap（诊断读数）。

---

## 4. risk 词表 v7 化（eval v3 必改项，非新增指标）

| 项 | eval v2（v6 词表，7 枚举） | eval v3（v7 词表，6 枚举） |
|---|---|---|
| taxonomies.risk_factors | pedestrian_crossing / vehicle_merging / lead_vehicle_braking / corridor_conflict / stationary_obstacle / congestion / oncoming_traffic | **vru_crossing / vehicle_crossing / lead_vehicle_braking / stationary_obstacle / object_near / oncoming_traffic** |

- metrics 默认常量 `_RISK_FACTORS` 与 pin 测试同步（config↔常量逐值钉死，v2 先例）；
- 因子名层面消费（`risk_factor_counts`）在 v7 双形状记录上已经过兼容（metrics `_risk_names` 提取）；归因族直接消费对象化形状；
- **代际红线**：v7 记录四字段 GT + v6 词表不可混用——eval_v3.yaml 的词表必须与 v7 契约 pin 测试同源（import 契约资产校验，防漂移）。

---

## 5. 新增键清单汇总

| # | 键 | 族 | 入 bootstrap |
|---|---|---|---|
| 1–9 | `bev_{dx,dy,dxy}_{mean,p50,p90}` | BEV 顶层 | mean 3 键 ✓ |
| 10–45 | `bev_{dx,dy,dxy}_{mean,p50,p90}_bucket_{0_10,10_25,25_40,40p}` | BEV 分桶 | ✗ |
| 46 | `risk_attribution_f1_micro` | 归因 | ✓ |
| 47 | `risk_attribution_f1_macro` | 归因 | ✓ |
| 48 | `risk_attribution_per_factor`（嵌套块） | 归因诊断 | ✗ |
| 49–52 | `bev_signed_{dx,dy}_{pos,neg}_mean`（方向分解偏置，§6-3 拍板 2026-10-14 加入） | BEV 偏置诊断 | ✗ |

冻结口径：**51 标量键 + 1 嵌套块**（按上方逐键清单枚举为准；四桶 + signed 拍板 2026-10-14 后锁定。**2026-10-14 勘误**：早前"50 标量键"系算术笔误，1–47 加 49–52 共 9 + 36 + 2 + 4 = 51，用户裁定按清单枚举）；bootstrap 标量键 22 → **27**（增量 +5：bev 三 mean + 归因 micro/macro。**2026-10-14 勘误**：早前"10 → 15"基数有误——v2 实际 bootstrap 标量基数为 22（`subset_statistics_v2` 空集键数实测钉死：22 标量 + 4 嵌套），v3 追加 5 键后为 27 标量 + 4 嵌套 = 31 bootstrap 键）。

---

## 6. 待拍板项汇总

| # | 事项 | 选项 | 倾向 |
|---|---|---|---|
| ~~1~~ | ~~BEV 45+ 尾巴~~ | **已关闭（2026-10-14）**：分桶改四桶 [0,10)/[10,25)/[25,40)/40+ 开放桶，尾巴被 40+ 桶吸收 | 拍板 ✓ |
| ~~2~~ | ~~归因口径~~ | **已拍板（2026-10-14）**：conditional 双向口径 | 拍板 ✓ |
| ~~3~~ | ~~带符号偏置诊断键~~ | **已拍板（2026-10-14）**：现在加入。方向分解口径（用户裁定）：严禁代数均值（正负抵消），正/负子集分别求均值——`bev_signed_{dx,dy}_{pos,neg}_mean` 四键（pos 恒 >0 = 推远量，neg 恒 <0 = 推近量），空子集置 0.0，不入 bootstrap；占比键 `*_pos_share` 可选后补 | 拍板 ✓ |
| 4 | cut/near 分层 BEV 读数入正式报告 | 维持默认（2026-10-14）：诊断脚本（Step 4 审计已备基线），不入正式报告 | 默认 ✓ |
| 5 | counterfactual battery 加算 BEV/归因 | 维持默认（2026-10-14）：不加（机制不变） | 默认 ✓ |

---

## 7. 验收口径（G7 成员）

1. **BEV 验算例**：前车 20 m 正前 → GT 最近点 (17.75, 0)（BEV rules §4.2 既有算例）；构造预测点验证 `bev_dxy = hypot` 与分桶归属；
2. **pin 测试**：eval_v3.yaml 六枚举 ↔ 契约 v7 词表逐值同源；新增键集与本文 §5 清单逐键一致（仿 v2 键集 pin 先例）；
3. **空集口径**：无匹配对切片全 0.0；`per_factor` 空因子跳过不置 null；
4. **bootstrap 复跑确定性**：同种子两跑 CI 逐值同（S09 先例）；
5. eval 台账登记：本文定稿后增补进 [S13_eval_v1_to_v2_metric_changes.md](./S13_eval_v1_to_v2_metric_changes.md) 同款变更登记（v2→v3 段）。
