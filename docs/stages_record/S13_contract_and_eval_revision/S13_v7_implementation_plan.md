# S13 Contract v7 实现流程（实施规划稿）

- 状态：**Step 1 已完成并通过 G1**（2026-10-11，执行记录见 §1.5；token 预案 A+B 已拍板，见 §1.4；**V7-C1 拍板后 v7 收敛为六件套**，见决策台账 §5）；**Step 2 已完成并通过 G2**（2026-10-11，执行记录见 §2.8；T0 reasoning 渲染链退役随批原子落地）；**Step 3 已完成并通过 G3/G4/G5**（2026-10-12，Stage C 冻结 v7-rules-0.3.0，见决策台账 §3.7/§3.8）；**Step 4 已完成并通过 G6**（2026-10-13，执行记录见 §4 执行记录 + 决策台账 §3.9）；Step 5（BEV 指标族，eval v3 立项待用户拍板）起待开工。
- 设计依据：[S13_contract_v7_decision_log.md](./S13_contract_v7_decision_log.md)（V7-B1 / V7-R1 / V7-Y1 全部拍板）、[S13_v7_risk_factor_rules.md](./S13_v7_risk_factor_rules.md)（v2.2 审查稿）、[S13_v7_bev_gt_rules.md](./S13_v7_bev_gt_rules.md)（拍板稿 v1）、[S13_v7_calibration_plan.md](./S13_v7_calibration_plan.md)（待审查）
- 代码基线：contract v6 model face（`configs/contracts/v6/` 八件套）+ dataset v4 代际；GT 引擎 [src/drivealign/gt/](../../../src/drivealign/gt/)
- 单位：距离 m，角度 deg，时间 s；环境惯例：`conda activate autovla_codeclean && PYTHONPATH=DriveAlign/src ...`，长任务 tmux + `set -o pipefail`

---

## 0. 流程总览

```text
Step 1  契约脚手架（v7 八件套 + versions.py + pin 测试 + token 预算实测）── V7-C1 后收敛为六件套
   │
Step 2  GT 引擎改造（T0 渲染链退役 → 六因子重构 + yield 解耦 + BEV 字段）──┐
   │                                                            │ 平行推进
Step 3  标定扫描（Stage A 中间量 → Stage B 网格 → 报告拍板）←────┘
   │        └─ 闸门：中间量无损重建 v6（100%）
Step 4  冻结 + 全量重建（gt_rule_config frozen → 回归一致性 → dataset_v7 → face manifest v7）
   │
Step 5  评测面增补（BEV 指标族 + parser 适配 + eval 台账登记）
   │
Step 6  下游与收尾（单测全量、文档回填、下游实验交接）
```

依赖关系与平行性：

- **Step 2 与 Step 3 可交叉并行**：Stage A 中间量提取只依赖 evidence 收集层（[backfill.py](../../../src/drivealign/gt/backfill.py) 的采集逻辑），不依赖 v7 判定代码；BEV GT 零标定参数，与标定完全正交。
- **Step 4 是唯一串行咽喉**：gt_rule_config 冻结（依赖 Step 3 拍板）→ 回归一致性 → 全量重建。dataset_v4 全程 byte-frozen 不动。
- **代际红线（decision log §2.3）**：v7 触碰 model face（prompt/schema/parser）+ training_targets 结构 → **request hash 与 record hash 双代际变化**，故必须新建物理目录 `dataset_v7` + `data/face_manifests/v7/`（目录命名跟 contract 代际走的既有裁定）。

### 全局 Gate 表

| Gate | 时点 | 判定 |
|---|---|---|
| G1 | Step 1 末 | 八件套齐备；pin 单测通过（category/motion vocab 与 v6 逐字节同）；token 预算实测 ≤ 512 |
| G2 | Step 2 末 | 单测全绿；BEV 验算例通过（前车 20m 正前 → (17.75, 0)）；v6 函数与 v7 判定边界清晰 |
| G3 | Stage A 末 | 中间量按 v6 公式重建 7 因子 + yield 与 v6 GT 逐 anchor **100% 一致**（calibration plan §3.4） |
| G4 | Stage B 末 | 必报项 → 扫描块映射全覆盖（calibration plan §5 核对表）；报告产出供拍板 |
| G5 | Stage C | 抽样 ≥1,000 anchor 回归：正式 v7 [risk.py](../../../src/drivealign/gt/risk.py) 判定与中间量分解逐因子 **100% 一致**；config 置 frozen 并登记 sha |
| G6 | Step 4 末 | build_dataset 三 gate PASS（parity / determinism / gt_backfill_full）；GT 分布表产出；val sanity 偏移 ≤ 2pp（超出须解释，不回调参数） |
| G7 | Step 5 末 | BEV 三族指标入报告与 bootstrap；eval 台账登记完成 |

---

## 1. Step 1：契约脚手架（模型面 + GT 规则配置草稿）

### 1.1 v7 八件套建立（V7-C1 拍板后收敛为**六件套**）

`configs/contracts/v7/` 由 v6 复制，逐一修改。**V7-C1（决策台账 §5）裁定 reasoning 资产退役**：`reasoning_templates.json` / `enum_phrase_map.json` 已删除（模型面 v6 起无 reasoning、SFT 监督四字段、渲染链仅消费 v4 资产——两文件零消费）；v7 资产 = 下表前六行：

| 文件 | 变更 | 依据 |
|---|---|---|
| `prompt.txt` | ① risk_factors 枚举行改 6 项新措辞（vru_crossing / vehicle_crossing / lead_vehicle_braking / stationary_obstacle / object_near / oncoming_traffic）；② critical_objects 条目教学新增 `bev_x` / `bev_y` 字段说明（措辞按 BEV rules §4.3 定稿）；③ object_near 语义句（"an object very close to ego"）；④ bbox_2d 教学改整数像素（token 拍板 A，§1.4） | rules §1–§5；BEV rules §4.3 |
| `output_schema.json` | ① risk_factors enum 7→6；② `$defs.critical_object` required + properties 新增 `bev_x` / `bev_y`（number，0.1m 粒度；是否设 min/max 由定稿拍板，倾向不设硬界仅 parser 语义检查）；③ maxItems=8 不变；④ description 更新 | rules §1–§6 + maxItems 不变量；BEV rules §4.1 |
| `risk_taxonomy.json` | 全量重写：6 项 canonical + synonyms + gt_derivation（vru_crossing 合并原 pedestrian_crossing 同义词并扩两轮车；vehicle_crossing 继承 cut-in 同义词并新增垂直横穿；object_near 新建；corridor_conflict / congestion 条目删除）；`schema_ref` 指向 v7 | decision log §3.5 R-OPT-4/5/6 |
| `gt_rule_config.json` | **status: provisional** 草稿（终值 Step 4 冻结）：结构见 §1.2 | rules v2.2 §9/§10 |
| `category_vocab.json` / `motion_vocab.json` | **逐字节不变**（HASH_FACE_FILES 成员，进 request hash 的词表不动；BEV 数值字段不涉词表） | [versions.py](../../../src/drivealign/contracts/versions.py) `HASH_FACE_FILES` |

v6 及更早目录保持八件套 byte-frozen 不动（含 reasoning_templates / enum_phrase_map 两文件及其既有 pin）。

### 1.2 gt_rule_config v7 草稿结构（rules v2.2 §9/§10 落 JSON）

```text
pool:    三组拆分 vru/vehicle/static + per-class 锥距覆盖（vru 组内 ped 30 / bicycle 45 / motorcycle 45）
motion:  不变（stationary_speed 0.5 / band 45°——仅 §0.8 motion_state 消费）
corridor: 保留 t_risk_s=3.0 / time_step_s=0.25 / ego 4.1×1.87 / width_margin 0.5；
          删除 d_risk_vehicle_m / d_risk_pedestrian_m（由 firing base 取代）
firing:  { "base_m": <标定, 草稿 2.0>, "assumption": "single|split", "base_vru_m": <备选> }
gate:    { "lateral_thr_mps": <标定, 草稿 0.8>, "band_half_width_m": 1.185 }
braking: { evidence_window_s 3.0 / speed_drop_mps 0.5 不变 + "y_brake_m": <标定, 草稿 1.185> }
stationary_obstacle: { max_speed 0.5 / window 3.0 不变 + "y_st_m": <标定, 草稿 2.5> }
object_near: { "radius_m": <标定, 草稿 5.0> }
oncoming: { range_m 20 不变 }
congestion: 删除
action:  不变
```

草稿数值仅作代码开发占位；所有标定槽位以 `calibration_targets` 显式登记（base / thr / r / Y_brake / Y_st），与 [calibration plan §2](./S13_v7_calibration_plan.md) 网格一致。

### 1.3 versions.py 与 pin 测试

- [versions.py](../../../src/drivealign/contracts/versions.py)：`AVAILABLE_CONTRACT_VERSIONS += "v7"`；`DEFAULT_CONTRACT_VERSION = "v7"`；`MODEL_FACE_LINEAGE["v7"] = "v7"`（v7 自有 model face，request hash 与 v5/v6-face 历史不可比——docstring 补 v7 段说明 BEV 字段 + taxonomy 重构双变更）。
- 新增 `tests/unit/test_contract_v7_alignment.py`（仿 [test_contract_v6_alignment.py](../../../tests/unit/test_contract_v6_alignment.py)）：八件套存在性、category/motion vocab 与 v6 逐字节相同、schema enum 与 risk_taxonomy terms 一致、BEV 字段存在于 schema required、`MODEL_FACE_LINEAGE` pin。
- 注意时序：`DEFAULT_CONTRACT_VERSION` 切 v7 后，[serializer.py](../../../src/drivealign/records/serializer.py) 的 request 戳与 [build_dataset.py](../../../src/drivealign/dataset/build_dataset.py) / [manifest.py](../../../src/drivealign/dataset/manifest.py) 的 manifest 戳随 `model_face_version()` / 常量自动走 v7——**dataset_v4 与 v5/v6 face manifests 是冻结资产不会被触碰**，但依赖 DEFAULT 常量的单测（test_dataset_build_v4 / test_records_serializer 等）需同步更新期望值。

### 1.4 token 预算实测（D-B1 复核项，G1 成员）——已实测，超限待拍板

最坏 case（8 目标全满，category 全取最长枚举 `construction_vehicle`，motion_state 全取 `same_direction`，bbox 取 1600×900 内最大位数像素坐标，bev 取 45m 量程 1 位小数含负号，risk_factors 6 项全选）在 Qwen2.5-VL-3B tokenizer 下的实测（2026-10-11）：

| 输出形态 | chars | tokens | vs max_new_tokens=512 |
|---|---|---|---|
| 紧凑 JSON（separators 无空格） | 1257 | **518** | 超 6 |
| 带空格 JSON | 1372 | **631** | 超 119 |
| 带 \`\`\`json fence 包裹 | 1384 | **635** | 超 123 |

结论：512 预算下最坏 case 必截断。预案（按序）：

- **A. bbox_2d 教学改整数像素**（schema number 不动，prompt 加 "integer" 字样；GT bbox 本就是整数像素网格）；
- **B. 推理预算 512 → 640**（SFT/benchmark config 的 max_new_tokens，GPU 阶段生效；不动契约面）；
- **C. 接受长尾**（仅作对照）。

**拍板（2026-10-11）：A+B 双保险。** A 已落地（v7 prompt + schema description 同步），复测：

| 输出形态 | 复测 tokens | 512 预算 | 640 预算（B 落地后） |
|---|---|---|---|
| 紧凑 JSON | **454** | 余 58 ✓ | 余 186 ✓ |
| 带空格 JSON | 567 | 超 55 | 余 73 ✓ |
| 带 \`\`\`json fence | 571 | 超 59 | 余 69 ✓ |

B 的佐证：SFT v6 parse failure 分析已记录 5/45（11%）512 截断案例——截断非纯长尾，640 为必要项。B 的落地时点：GPU 阶段创建 v7 run configs 时统一写 640（v6 系列历史 config 不动），登记进 §6 交接清单。

**V7-A1 归因增补复测 + V7-A2 回炉拍板（2026-10-11，G2 delta 执行时点）**：risk_factors 对象化（V7-A1，见归因规则稿 §6）后按本节同款口径复测（8 目标全满 + 6 因子全选，三口径 × 三形态，Qwen2.5-VL-3B）：

| 口径 | 紧凑 | 带空格 | fence |
|---|---|---|---|
| 每因子引用 1 id | 536 | 678 | 682 |
| GT 可达最坏（3 因子 × 全 8 id） | 545 | 699 | 703 |
| schema 对抗最坏（6 因子 × 全 8 id） | 620 | 804 | 808 |

带空格/fence 全口径超 640（SFT 监督文本 = 带空格 = 模型实际发射形态）。**V7-A2 拍板：SFT 目标发射改紧凑 JSON（targets.py `DUMP` separators `(", ", ": ")` → `(",", ":")`，v6 冻结 artifact/config 不动，v7 SFT config 写 `target_dump.separators: [",", ":"]`）+ 生成预算 640 → 700**（v7 run configs GPU 阶段统一写 700；紧凑发射下三口径最坏 620，余 80）。详见归因规则稿 §6、决策台账 §6 V7-A2。

### 1.5 Step 1 执行记录（2026-10-11）

- **八件套**：`configs/contracts/v7/` 建立——prompt.txt / output_schema.json / risk_taxonomy.json / enum_phrase_map.json / gt_rule_config.json（provisional 草稿，五个标定槽位 `calibration_targets` 登记）/ reasoning_templates.json 六项修订；category_vocab.json / motion_vocab.json 由 v6 逐字节复制（sha 与 v6 一致，已 pin）。
- **versions.py**：`AVAILABLE_CONTRACT_VERSIONS += v7`、`DEFAULT_CONTRACT_VERSION = "v7"`、`MODEL_FACE_LINEAGE["v7"] = "v7"`，docstring 补 v7 段。
- **新增 pin 测试**：`tests/unit/test_contract_v7_alignment.py`（14 项）——含一项超出原计划的发现：parser 同义词归一化把 `pedestrian_crossing`（下划线/空格变体）合法映射为 `vru_crossing`（v7 扩类使其成为合法同义词），已按 normalized_terms pin；`vehicle_merging` / `corridor_conflict` / `congestion` 保持 unmapped→fail。
- **parser 提前适配（计划外必要项）**：[output.py](../../../src/drivealign/contracts/output.py) `CriticalObject` 增加可选 `bev_x` / `bev_y`（默认 None，to_dict 仅非 None 时输出——保持 v1–v6 dict 形状不变），v7 面下解析并 **0.1m 取整**（两侧同离散空间）。否则 v7 schema 校验通过但数据被静默丢弃（pin 测试暴露）。
- **既有测试同步（DEFAULT 切换连锁面，23 处）**：v5/v6 对齐测试的"当前 default"pin 更新 + v6 面 parse 调用显式传 `contract_version="v6"`；test_contracts_output 新增 v7 默认 parse 样本（v6 样本保留走显式 v6 面）；test_structured_runner VALID_SAMPLE 补 bev 字段；test_base_benchmark_requests 全部 "v6" 字面量 → "v7"；test_eval_evaluator_v2 的 eval_v2.yaml status pin 由 provisional 更正为 frozen（yaml 已于 2026-10-09 冻结，属既有 stale pin，非本次引入）。
- **G1 判定（终）**：八件套 ✓ / pin 测试 ✓ / 全量单测 **412 passed** ✓ / token 预算实测超限 → **拍板 A+B 并落地 A**（复测紧凑 454，§1.4）→ **G1 通过**。
- **V7-C1 补充（2026-10-11）**：用户拍板 reasoning 资产退役——v7 目录删除 `reasoning_templates.json` / `enum_phrase_map.json`（Step 1 曾建八件套，收敛为**六件套**），pin 测试同步（三方一致性 → 双方一致性 + 退役断言）；渲染链退役归 Step 2 T0（决策台账 §5）。全量单测复绿 412 passed。

---

## 2. Step 2：GT 引擎改造（gt/ 包 + BEV）

设计依据 = rules v2.2 §0–§8 + BEV rules §3 + 决策台账 V7-C1。改造按文件列出（每项含单测要求）：

### 2.0 T0：reasoning 渲染链退役（V7-C1，与 BEV 同批原子落地）

v7 资产已删（Step 1 补做），代码链路在本批退役——四项原子绑定（v7 schema critical_object required bev → 校验切换依赖 GT 侧 BEV 生成）：

1. [backfill.py](../../../src/drivealign/gt/backfill.py)：删渲染调用（L322–336）与 `check_zero_contradiction` gate（L344）、删 `load_templates` / `load_phrase_map` / render import；`expected_output` 5 → **4 字段**；`_validate_schema` 硬编码 v4 → **v7**（与 §2.5 的 schema 解析合并实现）；
2. 删除 [render.py](../../../src/drivealign/gt/render.py) 整模块 + [config.py](../../../src/drivealign/gt/config.py) 的 `TEMPLATES_FILE` / `PHRASE_MAP_FILE` / `load_templates` / `load_phrase_map` + `gt/__init__.py` 对应导出；
3. [build_dataset.py](../../../src/drivealign/dataset/build_dataset.py)：`GtAssets` 摘除 templates / phrase_map 挂载（L155–156）；
4. 测试：删除 `test_gt_render.py`；`test_gt_backfill.py` 去 reasoning 断言（expected 4 字段）；`versions.py` docstring 更新。

红线：v4/v5/v6 目录与 `load_*` 的 v4 默认行为退役后消失；`expected_output` 结构变化并入 v7 record hash 代际（无额外代价）。

### 2.1 [config.py](../../../src/drivealign/gt/config.py)

- `PoolConfig`：三组拆分（`vru` / `vehicle` / `static`）+ per-class 锥距覆盖机制（`cone_range_by_class_m` 可选覆盖，bicycle/motorcycle = 45，其余随组；P-1 已确认）。
- `CorridorConfig` 删 `d_risk_*`；新增 `FiringConfig(base_m, assumption, base_vru_m)`、`GateConfig(lateral_thr_mps, band_half_width_m=1.185)`、`ObjectNearConfig(radius_m)`；`BrakingConfig += y_brake_m`；`StationaryObstacleConfig += y_st_m`；删 `CongestionConfig`。
- `load_rule_config(version)` 的校验链同步（含 `assumption` 枚举、Y_st ≥ Y_brake 约束在加载时断言——终选硬约束前移为加载期检查）。
- `NUSCENES_CATEGORY_MAP` 不动。

### 2.2 [pool.py](../../../src/drivealign/gt/pool.py)

- `select_objects` 消费新分组与 per-class 锥距覆盖；top-8 + `(round(distance_m,3), ann_token)` tie-break 不变；`MAX_POOL_SIZE = 8` 不变。
- `PoolCandidate` 增补 BEV 所需几何：`box_length_m` / `box_width_m` / `box_yaw_ego_deg`（t0 盒朝向，ego 系 x 轴基准）——由采集层填充。

### 2.3 [risk.py](../../../src/drivealign/gt/risk.py)（核心重构）

- **firing 引擎（rules §0.4–0.5）**：`corridor_distance_curve` 扩展为逐采样 `(d_k, h_k)`：d_k 沿用 `_rect_distance`；d_k > 0 时按间隙方向 û_k 算投影半延展 `h_k = 0.5·L·|cosΔθ| + 0.5·W·|sinΔθ|`（θ_obj 与 θ_ûk 同坐标系，只用差值）；`clearance_k = d_k − h_k`；**d_k = 0 样本直接 firing**。`firing(o) ⟺ min_k clearance_k < base(o)`；base 按 `assumption` 取单值或按组分裂（static 归 base_vehicle，仅消费于 yield）。
- **两 crossing 因子（§0.7 + §1/§2）**：`v_lat = (R(−ψ)(v_obj − v_ego)).y`；`v_toward = −sign(y_c)·v_lat`；带内（|y_c| ≤ 1.185）取 `|v_lat|`；gate ⟺ `v_toward > thr`。`vru_crossing = vru 族 firing ∧ gate`；`vehicle_crossing = vehicle 族 firing ∧ gate`（同一 thr，类别分组天然互斥）。
- **lead_vehicle_braking（§3）**：族 = vehicle ∪ {bicycle, motorcycle}（P-5）；在路径改 `x_c > 0 ∧ |y_c| ≤ Y_brake`；排序 `(round(distance_m,3), ann_token)` 与"任一命中"语义不变；无 track 跳过、静止跳过不变。
- **stationary_obstacle（§4）**：全类别统一判据；在路径 `x_c > 0 ∧ |y_c| ≤ Y_st`；两段式停着证据（t0 ≤ 0.5 ∧ 未来窗 max ≤ 0.5；无 track 回退仅 t0）；删除 v6 的 `for obj in pool` 类别无关列内判据中的 static 专设分支（P-6：firing 不再服务本因子）。
- **object_near（§5 新增）**：`∃ o: distance_m ≤ r`，全类别、无运动条件、与走廊族不互斥。
- **oncoming_traffic（§6）**：判定不变；族范围实现按**类别集合**（vehicle 五类 ∪ bicycle ∪ motorcycle，P-4）——拆组后不得写成"vehicle 组"字面。
- `RISK_TAXONOMY_ORDER` 更新为 6 项（输出顺序 = schema enum 顺序，确定性输出不变）。
- v6 专用函数（`corridor_conflict` / `point_in_ego_band` / `in_ego_corridor_column` / 旧 `derive_risk_factors`）处置：**v6 复现闸门由 Step 3 的 extract.py 自包含实现（内嵌 v6 公式 + 读冻结 v6 config），risk.py 直接重构为 v7 单路径**，避免双路径长期共存；legacy 测试改写为 v7 语义测试。

### 2.4 [action_yield.py](../../../src/drivealign/gt/action_yield.py)

- `derive_yield_required` 签名改造：入参从 `risk_factors` 改为 firing 并集布尔（`any firing`，含 static firing 与无标签 firing）——P1.6 证据分离的实现载体从"corridor 族标签"改为"firing 引擎"（rules §7）；`CORRIDOR_FAMILY` 常量退役。speed_action 逻辑不动。
- 连带：`backfill.py` 组装顺序改为先算 firing（risk 引擎返回 `(risk_factors, any_firing)` 两元组），再出 yield。

### 2.5 BEV GT（[backfill.py](../../../src/drivealign/gt/backfill.py) + 新派生函数）

- **采集层扩展**：`AnnRecord` 增补标注框尺寸 (L, W)（nuScenes `size = [w, l, h]`，长沿盒朝向）与全局朝向 yaw；`collect_anchor_evidence` 从 `sample_annotation` 直读，零新增依赖（annotation × ego_pose，t0 即完备）。
- **BEV 计算**（BEV rules §3.3 精确口径）：盒局部系下 ego 原点坐标 → clamp 到盒矩形 → 映回 ego 系 → **0.1m 取整**输出 `bev_x` / `bev_y`。独立纯函数（如 `bev_nearest_point(x_c, y_c, L, W, theta_obj) -> (x, y)`）便于单测验算例（前车 20m 正前 L=4.5 → (17.75, 0)）。
- **critical_objects 组装**：每对象在既有 category/bbox_2d/motion_state 之上追加 `bev_x` / `bev_y`；`distance_m` 口径不动（两者语义并存，prompt 显式区分——BEV rules §4.2）。
- `_validate_schema` 硬编码的 `"v4"` 改为按当前 DEFAULT version 解析（否则 v7 schema 的新增 required 字段会被旧 schema 拒绝，fail-fast 落在错误位置）。

### 2.6 [render.py](../../../src/drivealign/gt/render.py) 与 GT 分布

- 渲染消费 6 项新短语（enum_phrase_map v7）；`check_zero_contradiction` 的双向对应清单随 taxonomy 更新。
- [gt_distribution.py](../../../src/drivealign/dataset/gt_distribution.py) 累计器改 6 因子 + BEV 字段分布（可留待 Step 4 前完成，构建时生效）。

### 2.7 单测（G2）

- 各因子判定与边界：d_k=0 直接 firing；带内 gate 取 |v_lat|；无 track 的 braking 跳过 vs stationary 回退的不对称；tie-break 复现；Y_st ≥ Y_brake 加载断言；BEV 验算例 + 取整口径；yield = firing 并集（static-only firing 帧 yield=true 而 risk_factors 空）。

### 2.8 Step 2 执行记录（2026-10-11）

**落地清单（逐文件）**：

1. [config.py](../../../src/drivealign/gt/config.py)：按 §2.1 重写——`PoolConfig` 三组拆分 + `cone_range_by_class_m`（`cone_range_for()` per-class 优先回落组值）、`CorridorConfig` 删 `d_risk_*`、新增 `FiringConfig` / `GateConfig` / `ObjectNearConfig`、`BrakingConfig += y_brake_m`、`StationaryObstacleConfig += y_st_m`、删 `CongestionConfig` 与模板/短语表资产；加载断言（status 枚举、组键、per-class 键 ⊆ 已知类别、`assumption` 枚举、Y_st ≥ Y_brake、`band_half_width_m == (ego_width+margin)/2`）。
2. [pool.py](../../../src/drivealign/gt/pool.py)：`select_objects` / `truncated_count` 切 `cone_range_for`；`PoolCandidate` 增 `box_length_m` / `box_width_m` / `box_yaw_ego_deg`（默认 0.0 保持 CLI 构造兼容）。
3. [risk.py](../../../src/drivealign/gt/risk.py)：按 §2.3 重写为 v7 单路径——`clearance_curve` 返回 `(t_k, d_k, h_k)` 样本（K=13 双侧 CV；d_k=0 直接 firing 记 `(t,0,0)`）、`object_firing`（严格 `< base`）、`support_half_extent`（公开原语，Step 3 extract.py 复用）、`crossing_gate`（统一横向 gate，带内 |v_lat| / 带外 v_toward）、`in_path`、`firing_base_for`（split 下 vru 用 base_vru）、`derive_risk_factors` 返回 `(factors, any_firing)` 二元组；族口径全部按类别集合常量（`BRAKING_AND_ONCOMING_CATEGORIES` / `VRU_CATEGORIES` / `VEHICLE_CROSSING_CATEGORIES`），无"vehicle 组"字面；`RISK_TAXONOMY_ORDER` 6 项。
4. [action_yield.py](../../../src/drivealign/gt/action_yield.py)：`derive_yield_required(any_firing: bool)`；`CORRIDOR_FAMILY` 退役。
5. [backfill.py](../../../src/drivealign/gt/backfill.py)：T0（删渲染 + gate + 模板/短语表 import，4 字段，`_validate_schema` 用 DEFAULT version）与 §2.5 BEV（`AnnRecord` 增 `size_w_m` / `size_l_m` / `yaw_global_rad`；`bev_nearest_point` 纯函数；critical_objects 追加 bev 字段，`x_c/y_c` 经共享原语 `ego_frame_xy`）同批原子落地。
6. **删除** [render.py](../../../src/drivealign/gt/render.py) 与 `tests/unit/test_gt_render.py`；[gt/\_\_init\_\_.py](../../../src/drivealign/gt/__init__.py) 收敛为只导出 `RuleConfig` / `load_rule_config`。
7. **legacy_v4.py 新建**（用户拍板保留 S08 CLI 的结构化落点）：自包含 v4 数据类树 + `load_v4_rule_config()` + v4 走廊/池函数，与 v7 只共享 `EgoState` / `ObjectState` / `PoolCandidate` 容器（v4 只读旧字段）；docstring 明示"唯一消费者 = 两个 S08 审计 CLI，勿扩展"。[gt_calibration.py](../../../src/drivealign/cli/gt_calibration.py) / [gt_truncation_check.py](../../../src/drivealign/cli/gt_truncation_check.py) import 全部切至 legacy_v4。
8. 连带 v7 化：[build_dataset.py](../../../src/drivealign/dataset/build_dataset.py)（`GtAssets` 只剩 `rule_config`；docstring 4 字段）、[gt_distribution.py](../../../src/drivealign/dataset/gt_distribution.py)（删 reasoning 统计；词表读 v7 契约资产；risk_frames 0..6）、[dataset_parity_check.py](../../../src/drivealign/cli/dataset_parity_check.py)（`EXPECTED_OUTPUT_KEYS` 5 → 4，计划外发现的连带点）、[versions.py](../../../src/drivealign/contracts/versions.py)（v7 六件套 docstring）。

**实现要点**：

- BEV 验算例口径：20 m 正前 L=4.5 的取整前最近点 = 17.75（G2 表述值），GT 输出经 `round(v, 1)`（与 parser 严格同款，banker's）= **17.8**；侧向情形的 `-0.0` 以 `+ 0.0` 归一。
- braking 的 v0 取 `ObjectState` t0 速度（`speed_mps` property），证据窗判据 `min(speeds) ≤ v0 − drop`（≤ 边界）。
- stationary 两段式的"无 track 回退仅 t0"与 braking 的"无 track 跳过"不对称性按 rules §4 冻结。

**单测（G2 证据）**：全量 `pytest DriveAlign/tests/unit -q` → **417 passed**（删除 test_gt_render.py 后重算基数）。分文件：risk 24 / backfill 12（含 BEV 验算例、旋转盒、−0.0 归一、4 字段 shape、mini 集成）/ pool 8（含 per-class 45 m 覆盖）/ action_yield 9 / dataset_build 12（v7 四字段 pin、分布累计 6 因子、parity CLI）。回归锚点：self-defeating trap（t0 双侧 CV 在 ego 制动下仍 firing，内联实际轨迹对照）、firing 严格边界（x=6.3 不触发 / 6.29 触发）、带内直接 firing、gate 带内/带外、split 假设边界（vru clearance ∈ (base_vru, base) 区分 single/split）。

**调试插曲（不影响引擎正确性）**：risk 测试助手 `_obj` 默认 `vy=2.0` 污染全部"静止/同向"场景（9 处假失败，曾疑似环境问题——直接调用与 pytest 结果不一致实为显式传参差异），已逐场景显式 `vy=0.0` 修复；另修两处测试设计错误（225° 支撑角断言应等于 45° 值；split 区分度必须用 vru 对象，car 的 split base 仍是 base_m 无区分度）。

**G2 判定：通过（2026-10-11）**——单测全绿；BEV 验算例通过；v6 函数与 v7 判定边界清晰（v6 走廊/池逻辑自包含于 legacy_v4 冻结层，仅两个 S08 CLI 消费；v7 risk.py 无 v6 残留）。

### 2.9 G2 delta 执行记录：V7-A1 风险因子归因 + V7-A2 token 回炉（2026-10-11）

用户提出「risk_factor 与 critical object 关联」→ 三项拍板（D-A1-1 模型面归因 / D-A1-2 显式 id / D-A1-3 全量合格对象）+ 执行中回炉拍板 V7-A2（token 超限 → 紧凑发射 + 700 预算）。设计稿 = [S13_v7_risk_attribution_rules.md](./S13_v7_risk_attribution_rules.md)（拍板稿 v1 + §6 实测回填）；台账 = 决策台账 §6 / §7。

**落地清单（逐文件，按归因规则稿 §8 九步）**：

1. [output_schema.json](../../../configs/contracts/v7/output_schema.json)：`critical_object.required` + `"id"`（integer 0–7）；`risk_factors` 条目对象化 `{factor(enum 6), objects(1..8 unique int 0–7)}`，maxItems 8→6；description 补 V7-A1。
2. [prompt.txt](../../../configs/contracts/v7/prompt.txt)：critical_objects 教学加 id 句；risk_factors 教学对象化（因子枚举语义句不变 + objects 引用说明）。
3. [contracts/output.py](../../../src/drivealign/contracts/output.py)：`CriticalObject.id`（None 不发射，v1–v6 形状不变）；新增 `RiskFactorAttribution` dataclass；`StructuredDrivingOutput.risk_attribution` 旁路字段（`risk_factors` 名字元组保留 = evaluator/gt_distribution 版本无感）；to_dict 按版本发射归因形状/裸名列表；`_semantic_errors` 新增四条归因校验（重复对象 id / 重复因子 / 条目内重复 id / 未知引用）。
4. [gt/risk.py](../../../src/drivealign/gt/risk.py)：`derive_risk_factors` 三元组返回 `(factors, attribution, any_firing)`；braking/stationary/oncoming 三处 break 删除改全收集；object_near 收集 `radius_m` 内全部对象；attribution 顺序 = braking 距离升序（原循环排序键）、其余池序。
5. [gt/backfill.py](../../../src/drivealign/gt/backfill.py)：`id_by_token` 池序下标分配；critical_objects 落 `id`；`risk_factors` 落 `[{factor, objects(sorted)}]`。
6. 下游适配：[gt_distribution.py](../../../src/drivealign/dataset/gt_distribution.py)（条目取 `entry["factor"]`）；[sft/gate.py](../../../src/drivealign/sft/gate.py)（semantic_fields 双形状兼容）；[evaluation/metrics.py](../../../src/drivealign/evaluation/metrics.py)（`_risk_names` 双形状提取——计划外必要项：GT/预测侧归因 dict 条目直入 `set()` 会 TypeError）；[sft/targets.py](../../../src/drivealign/sft/targets.py)（V7-A2：`DUMP` separators `(", ", ": ")` → `(",", ":")` 紧凑发射）。
7. 单测（步骤 6）：[test_gt_risk.py](../../../tests/unit/test_gt_risk.py)（`_run` 保二元组视图 + 新增 3 归因测试）；[test_gt_backfill.py](../../../tests/unit/test_gt_backfill.py)（`_assert_v7_shape` 加 id + 归因条目断言）；[test_dataset_build_v4.py](../../../tests/unit/test_dataset_build_v4.py)（`_eo` 夹具对象化）；[test_contract_v7_alignment.py](../../../tests/unit/test_contract_v7_alignment.py)（enum 路径移位 + 新增 4 归因测试：schema 形状 / 解析发射 / schema 硬界 / 语义校验）；[test_contracts_output.py](../../../tests/unit/test_contracts_output.py) / [test_structured_runner.py](../../../tests/unit/test_structured_runner.py)（v7 样本加 id）；[test_contract_v5_alignment.py](../../../tests/unit/test_contract_v5_alignment.py)（`_risk_items` 剔除 factor/objects 包装键）。

**token 实测（步骤 7）**：三口径 × 三形态全表见归因规则稿 §6 / 本稿 §1.4——带空格/fence 全口径超 640 → **V7-A2 拍板**：DUMP 紧凑发射（v6 冻结 artifact/config 不动；v7 SFT config GPU 阶段写 `separators: [",", ":"]` + `max_new_tokens: 700`）。

**G2 delta 判定：通过（2026-10-11）**——全量单测 **424 passed**（417 基数 + 7 项新增归因测试，零删除）；归因不变量钉死（factors 顺序 / any_firing 逐字节不变，五标定槽位零触碰）；v1–v6 面回归零影响。

---

## 3. Step 3：标定扫描（calibration plan 落地）

按 [S13_v7_calibration_plan.md](./S13_v7_calibration_plan.md) 执行（该文档为方案基准，此处只登记实现顺序与代码落点）：

1. **Stage A**：`src/drivealign/gt/calibration_v7/extract.py` —— 一次提取 §3.1 对象级中间量（含 BEV 复用的 L/W/θ_obj；`min_clearance` 内嵌 h_k 计算）+ §3.2 anchor 级 v6 基线 join；按 scene 分 shard 出 `runs/S13_v7_calibration/features/shard_*.parquet`；tmux 长任务。**闸门 G3**：v6 公式重建（内嵌 v6 常量 + braking tie-break 复现）与 v6 GT 100% 一致后才进 Stage B。
2. **Stage B**：`sweep.py` —— B0 基线统计（含 static 残余集合识别、慢行前车残核计数）→ B1 base（81 组合含分裂假设 + 对角线视图）→ B2 thr（联合 2×9×13）→ B3 r → B4 Y_brake → B5 Y_st；产出 `runs/S13_v7_calibration/sweep_results/`（parquet + 约 12 张图）。
3. **报告**：`S13_v7_calibration_report.md`（本目录，Stage B 后生成）——必报项全覆盖核对 + 推荐规则预填（calibration plan §6），**终值由用户审阅拍板**（含分裂假设采纳、thr 拐点、r 目标带、Y 分界、共现护栏触发与否）。
4. 拍板后回写：决策台账登记标定决议；gt_rule_config v7 草稿数值更新（status 仍 provisional，Step 4 冻结）。

标定纪律：train-only（val 仅冻结后 sanity，test 全程不可见）；参数无关中间量只提取一次，扫描均为向量化比较（~1,500 组合，分钟级）。

---

## 4. Step 4：冻结 + dataset_v7 全量重建

1. **冻结**：gt_rule_config v7 `status → frozen`，登记 sha256（决策台账 + 报告双登记）；[build_dataset.py](../../../src/drivealign/dataset/build_dataset.py) 的 `status != frozen` 硬检查随即放行。
2. **回归一致性（G5）**：正式 v7 risk.py 跑 ≥1,000 抽样 anchor（分层保证各因子含 true 样本），与中间量分解判定逐因子对比 100% 一致。
3. **全量重建（Step 2/3，用户 tmux 执行物理盘写入）**：
   - `python -m drivealign.dataset.build_dataset --out /root/autodl-tmp/datasets/drivealign_dataset/v7 ...`（沿用 v4 命令形态 + `set -o pipefail`；GT 分布表随构建产出至 `runs/S13_v7_calibration/reports/` 或按惯例命名目录）；
   - symlink：`ln -s .../v7 data/dataset_v7`；
   - Step 3 manifest 派生：`python -m drivealign.dataset.manifest --assets data/dataset_v7`，产物迁至 `data/face_manifests/v7/`（anchor_policy_manifest.json + manifest_assets symlink staging，v6 先例），并做 BYTE-IDENTICAL 复跑验证。
4. **验证（G6）**：构建三 gate PASS；`dataset_verify` 确定性复跑；**v7 专项 parity 设计点**——request hash 与 record hash 均为预期代际变化（不可与 v6 对齐），parity 对象改为：model_inputs / oracle_only / 图像路径等非 GT 字段对抽样 anchor 与 v4 记录逐值相同 + v7 戳正确；val sanity 报告（各因子检出率 / yield 率 train-vs-val 偏移 >2pp 须解释，不回调参数）。
5. **BEV 审计（随构建）**：截断/近距目标（底边出画锚点）按 gt_truncation_check 思路出 BEV 误差贡献预审计（BEV rules §6.2 的 GT 侧部分）。

### 4.6 Step 4 执行记录（2026-10-13，G6 通过）

- 冻结回写 + 全量重建 + 分布核对已于 2026-10-12 完成（决策台账 §3.8 待办 B）；本日收尾四项：
  1. **face manifest v7**：派生至 `data/face_manifests/v7/`（v6 先例布局），30,940 anchors 绑定 dataset_v7 manifest sha `534b97f4…`；BYTE-IDENTICAL 复跑通过（三文件逐字节同）。
  2. **v7 专项 parity（全量，超抽样下限）**：新增 [v7_parity_check.py](../../../src/drivealign/cli/v7_parity_check.py)，model_inputs / oracle_only / metadata 逐值 0 diff、provenance 白名单漂移 + v7 戳正确、hash 双代际变化 100%（正向 gate）、四字段 GT 100% 在位——11 gates 全 PASS。
  3. **dataset_verify**：scratch 全量重建 + 39 文件 byte 对比 differing=none，PASS；[dataset_verify.py](../../../src/drivealign/cli/dataset_verify.py) 增可选 `--anchor-manifest`（布局适配，缺省行为不变）。
  4. **val sanity + BEV 审计**：7 项 >2pp 均判 scene 级抽样方差（location 混合贡献 ≤0.28pp），不回调参数；BEV 审计出画目标 99.6% 落 [0,10) 桶、[25,45) 桶为 0，中远距读数不受污染。
- 产物：`runs/S13_v7_calibration/full_reports/{v7_parity_report, determinism_report, val_sanity_report, bev_truncation_audit}.{json,md}`；判定细节见决策台账 §3.9。

---

## 5. Step 5：评测面增补

1. **parser 适配**（[contracts/output.py](../../../src/drivealign/contracts/output.py)）：bev_x/bev_y 数值解析 + 0.1m 取整归一（GT 与模型同离散空间，BEV rules §3.4）+ 语义校验（有限数值；可选 |bev| ≤ 量程合理性检查）；bbox 语义检查链不动。
2. **BEV 指标族**（[evaluation/](../../../src/drivealign/evaluation/)）：[matching.py](../../../src/drivealign/evaluation/matching.py) 的 2D IoU 贪心匹配复用（τ=0.5 主阈值，D-B1-4）；[metrics_v2.py](../../../src/drivealign/evaluation/metrics_v2.py) 在匹配对上算 `bev_dx/dy/dxy`（带符号 / 逐帧合成），三族 × mean/p50/p90 + 距离分桶（键 = GT 最近点距 d_gt，建议 [0,10)/[10,25)/[25,45)，边界随实现确认）；`p90(dxy)` 独立序列统计；三族入 [bootstrap.py](../../../src/drivealign/evaluation/bootstrap.py) scene-cluster bootstrap（n=2000 / seed=20261003 / 0.95，八字段）。
3. **eval 台账登记**：`S13_eval_v1_to_v2_metric_changes.md` 增补 BEV 族（类别=新增）；**版本归属待拍板**——`eval_v2.yaml` 已冻结（2026-10-09），倾向 eval v3 立项收编（本指标族属 v7 model-face 代际）。
4. 命名按 `<field>_<statistic>[_<aggregate>]`（如 `bev_dx_mean` / `bev_dxy_p50_bucket_10_25`）。

### 5.5 Step 5 执行记录（2026-10-14，G7 通过）

- **口径拍板**（2026-10-13/14）：草案 [S13_eval_v3_metric_additions_draft.md](./S13_eval_v3_metric_additions_draft.md) 定稿——BEV 全非负语义、四桶 [0,10)/[10,25)/[25,40)/40+ 开放桶、conditional 双向归因、signed 方向分解键现在加（正/负子集分开求均值）、分层读数与 counterfactual 维持默认关闭；§5 勘误两项：51 标量键按清单枚举（"50"系笔误）、bootstrap 基数 22 → 27（"10 → 15"基数有误，实测钉死）。
- **实现落点**（v1/v2 路径行为零改变，v3 全部以追加方式落地）：
  1. [metrics_bev.py](../../../src/drivealign/evaluation/metrics_bev.py)：`collect_bev_pairs`（匹配对 → (x_p, y_p, x_g, y_g)，任一侧 bev 缺失跳过——缺样本 ≠ 零误差）+ `bev_statistics`（49 键：顶层 9 + 四桶 36 + signed 4；p50/p90 与 `_percentile` 线性插值同口径；分桶左闭右开只看 d_gt；空集全 0.0）；
  2. [metrics_attribution.py](../../../src/drivealign/evaluation/metrics_attribution.py)：conditional 双向逐帧计数（因子名交集闸门 + id 桥接落点判定）+ `attribution_totals`/`aggregate_attribution`（micro/macro/per_factor，聚合与 bootstrap 共用一份定义）；
  3. [evaluator_v3.py](../../../src/drivealign/evaluation/evaluator_v3.py)：`score_anchor_v3`（v1 打分后追加 bev_pairs + risk_attribution_counts，parse 失败原样返回）+ `aggregate_v3`（= aggregate_v2 全量键 + BEV/归因增量 + denominators.bev_pairs）+ `subset_statistics_v3`（v2 统计传 v7 词表 + 5 个 bootstrap 新标量）+ `run_evaluation_v3`（T6 fail-fast、byte-identical 编排、meta.eval_version=eval_v3）+ 独立 markdown 渲染（新增 BEV / Risk attribution 两段）；
  4. [eval_v3.yaml](../../../configs/evaluation/eval_v3.yaml)：v2 副本仅 eval_version/status 与 v7 词表两处差异（provisional，sha `e897e8c893c4e08133a5c8067a532f86a68b87a0d4e5b07f8111ef880328a496`）。
- **测试（G7 成员 1–4）**：新增 50 测试全绿——BEV 49 键形状 pin + (17.75, 0) 验算例 + 四桶左闭右开边界（d_gt=10 → 10_25、40 → 40p）+ 开放桶吸收 + signed 对抗例（+0.2/−0.2 代数均值归零）；归因 conditional 双向三例 + 七行边界表 + id 桥接错位（GT 0/1 vs pred 3/7）+ micro 池化/macro 跳无原子；evaluator 51+1 键集 pin（v2⊆v3、v1 无 v3 字段、两空集口径）+ T6 + byte-identical 重跑 + bootstrap +5 键且分桶/signed/p50/p90/per_factor 不入；eval_v3.yaml 逐值 pin 与词表同源 import 校验。全量回归 **492 passed**（原 442 + 50）。
- **冒烟**：30 真实 test anchor（GT 对象数最多）+ GT 确定性变换合成预测（id 反转 7−id、bev 偏移 (+0.3, −0.2)、因子引用同步重映射）——SMOKE PASS：byte-identical 重跑 PASS；bev_pairs=240；dx_mean=0.3000 / signed 分离（dx_pos=0.3000、dy_neg=−0.2000）/ dxy_mean=0.3606=hypot(0.3, 0.2)；四桶全有读数；归因 micro/macro=1.0（id 桥接错位验证）；bootstrap v3 新键恰为 5 键。产物 `runs/S13_eval_v2_and_contract_v7/step5_smoke/`。
- **台账登记（G7 成员 5）**：eval 台账 §10（v2→v3 变更登记 + 勘误）、决策台账 §8、本节。
- **G7 判定**：BEV 三族指标入报告与 bootstrap ✓（49 标量 + 1 嵌套块入报告，5 键入 bootstrap，实测读数齐）；eval 台账登记完成 ✓（§10）。**通过**。
- **遗留**：eval_v3.yaml 冻结（provisional → frozen + sha256 重登记）待正式报告前执行；Step 6 下游适配（SFT 面 parity 重建 / run configs max_new_tokens 700）未动。

---

## 6. Step 6：下游适配与收尾

| 项 | 内容 | 时点 |
|---|---|---|
| 全量单测 | tests 全绿（含 v7 pin、gt 引擎重写测试、评测新键） | 每步随做，Step 6 汇总 |
| 文档回填 | 决策台账标定决议 + 本文档 Gate 判定 + 标定报告 + S13 run 记录（含偏离登记） | 各 Gate 后增量 |
| token 预算复核闭环 | G1 实测值 + 若有预案触发则登记 | Step 1 → Step 6 |
| SFT 面确认 | [sft/dataset.py](../../../src/drivealign/sft/dataset.py) 的 parity 校验按 v7 face 重建（四字段列举含 BEV 自动进监督目标；`v7_1f` 样本目录命名）；**v7 run configs（benchmark base/sft）max_new_tokens 统一写 700**（token 拍板 B + V7-A2 回炉 640→700，§1.4）；4F smoke（S11 遗留）在 GPU 重跑前补跑 | 下游实验前 |
| Base/SFT 重跑交接 | Base v7 benchmark（S09 式）+ eval v2/v3 报告；SFT（S12 式）；BEV 反事实验证（BEV rules §6.1，SFT 后） | 交接给后续 GPU 阶段 |
| 台账遗留项关闭 | V7-Y1「若承载变道避让语义须另立新字段」维持建议不改；`runs/S11` 遗留产物处置不在本轮 | — |

### 6.1 Step 6 执行记录（2026-10-14，CPU 部分完成；GPU 部分待交接）

- **SFT 样本 v7_1f（B 层重打）**：A 层免跑——`data/sft_subsets/train_full_tokens.json`（S12 建，22,941 token）与 v7 face manifest 的 train 集合**逐项相同**（实测比对 True，train split 未变，A 层跨 face 复用成立）；`serialize --subsets train_full` 重打 → `data/sft_samples/v7_1f/train_full.jsonl`（22,941 行，逐样本 v7 face parity 断言全过）+ `samples_manifest.json`（contract_version=v7）。抽验：target_text 紧凑发射（V7-A2 separators）、critical_objects 携 bev_x/bev_y/id、risk_factors 对象化形状、prompt 含 v7 归因教学。
  - 确定性复跑验证顺带完成：首跑（workspace 根，git_commit 落 "no-git"——根目录非 git repo）与重跑（DriveAlign/ repo 内，git_commit=88ac5671，与 dataset_v7 builder_commit 同源）`file_sha256` 逐字节一致（`bafd222395b2484778925baff4591d52adab64f4f02245d43b8c747bc7c17c79`）；samples_manifest sha `27ef5f2c9be0c535fd4443a231e3777512e397dbff619702a7ccb9061cfcda5b`。
- **v7 run configs 三件**（逐值复刻 frozen 先例 + V7-A2 变更，均 provisional、GPU 开跑前转 frozen 并登记）：
  1. [base_1f_v7.yaml](../../../configs/benchmark/base_1f_v7.yaml)：v6 版改 run_name/contract_version=v7/max_new_tokens=512→700；**已于 2026-10-14 开跑前冻结**（CLI `load_config` 强制 frozen；冻结 sha256 `69cd4adda7d9b904eae19a76fad725f94231272e65ae82c9b84a8c910d1da94b`；config 内不放自引用 sha 值，先例 base_1f_v6 同款偏差）；
  2. [sft_1f_v7.yaml](../../../configs/sft/sft_1f_v7.yaml)：训练 config，另改 `target_dump.separators`→`[",", ":"]`（紧凑发射，与 targets.py DUMP 同源），LoRA/优化/训练规模逐值同 sft_1f.yaml，sha `440b88a3832d190d1a5e46a67c7f364bbb794fc72ac23c41de84f3068b10888f`；**已于 2026-10-15 训练前冻结**（冻结 sha `a9eb6c2e391ed6434d77e4beb5b61566b4211e9ac23f2665a99691e07f354b5b`，上列 provisional sha 作废；逐值核对与样本复核见决策台账 §9 SFT 冻结条目，训练命令见 §6.2）；
  3. [sft_1f_v7.yaml（benchmark）](../../../configs/benchmark/sft_1f_v7.yaml)：SFT 评测 config，adapter_path 占位 `runs/S13_sft_1f_v7/checkpoints/sft_1f_v7/best`（GPU 阶段定稿，--adapter 可覆盖），max_pixels=360000 与训练侧同值，sha `03dc8b80d0f3d6d190a4c0909d0eeb207a526309b52b5a08bc7fe1704fceb5a7`。
- **parity pin 测试**：[test_sft_config_parity.py](../../../../DriveAlign/tests/unit/test_sft_config_parity.py) 新增 `TestSftV7ConfigParity`（5 测试：provisional 状态 pin、v7 代际 + 700 预算、target_dump↔DUMP 同源、训练/优化面与 v6 冻结件逐值同、A1 两侧同分辨率）——全量单测 **497 passed**（492 + 5）。v6 冻结 config 零改动（既有 parity 测试全绿佐证）。
- **token 预算复核闭环**：V7-A2 已闭环（§1.4 复测表 + 拍板），本步以三 config 写 700 落地，无预案触发。
- **遗留（GPU 阶段交接）**：① 4F smoke 补跑（S11 遗留，base 重跑前置）；② Base v7 benchmark（S09 式 1F + counterfactual battery）→ eval v3 首个正式报告（届时 eval_v3.yaml + 三 config provisional→frozen 并登记 sha）；③ SFT v7 训练（S12 式，样本已就位）；④ BEV 反事实验证（BEV rules §6.1，SFT 后）。

### 6.2 SFT v7 训练交接（GPU，tmux；S12 §2.1/§2.2 同款流程）

前置状态（2026-10-15 已完成）：`sft_1f_v7.yaml` 冻结 sha `a9eb6c2e…`；样本 `data/sft_samples/v7_1f/train_full.jsonl` 22,941 行 sha `bafd2223…`；git HEAD `88ac5671`（与 builder_commit 同源）。

```bash
# ── pre-flight（一次性，秒级；在真实终端执行）──────────────────────
nvidia-smi                                                        # ① 卡空闲
git -C DriveAlign rev-parse HEAD                                  # ② 应为 88ac5671…
sha256sum DriveAlign/configs/sft/sft_1f_v7.yaml                   # ③ 应为 a9eb6c2e…
sha256sum data/sft_samples/v7_1f/train_full.jsonl                 # ④ 应为 bafd2223…
df -h /root/autodl-tmp                                            # ⑤ 磁盘 ≥ 5 GB（11 ckpt ≈ 3.8 GB + adapter）

# ── 正式训练（tmux 内执行；约 4–4.5 h，S12 实测 2.59 s/step × 5,735 step）──
tmux new -s s13_train
conda activate autovla_codeclean
cd /root/autodl-tmp/drivealign_workspace
set -o pipefail
mkdir -p runs/S13_sft_1f_v7/reports

PYTHONPATH=DriveAlign/src python -m drivealign.cli.sft_train \
  --config DriveAlign/configs/sft/sft_1f_v7.yaml \
  --samples data/sft_samples/v7_1f/train_full.jsonl \
  --out runs/S13_sft_1f_v7/checkpoints/sft_1f_v7 \
  --dataroot data/nuscenes/trainval \
  2>&1 | tee runs/S13_sft_1f_v7/reports/step3_train.log
```

纪律（S12 先例逐条沿用）：不传 `--epochs` / `--max-steps` / `--save-steps` / `--overfit`（规模全走冻结配置，命令行不出现第二份口径）；末点 adapter = `--out` 根目录的 `adapter_model.safetensors`（收尾 `trainer.save_model` 落盘，非 `checkpoint-XXXX`）；中断续跑 `--resume-from-checkpoint runs/S13_sft_1f_v7/checkpoints/sft_1f_v7/checkpoint-XXXX` 且 `--out` 必须与首跑逐字一致（`| tee -a` 追加）。预期：optimizer step ≈ 5,735（22,941 ÷ 4）、中间 checkpoint 11 个（checkpoint-500…5500）、峰值显存记入阶段报告。

```bash
# ── 中断续跑（可选）────────────────────────────────────────────
PYTHONPATH=DriveAlign/src python -m drivealign.cli.sft_train \
  --config DriveAlign/configs/sft/sft_1f_v7.yaml \
  --samples data/sft_samples/v7_1f/train_full.jsonl \
  --out runs/S13_sft_1f_v7/checkpoints/sft_1f_v7 \
  --dataroot data/nuscenes/trainval \
  --resume-from-checkpoint runs/S13_sft_1f_v7/checkpoints/sft_1f_v7/checkpoint-XXXX \
  2>&1 | tee -a runs/S13_sft_1f_v7/reports/step3_train.log
```

### 6.3 SFT v7 评测交接（GPU，tmux；§6.2 已执行完毕，2026-10-15）

训练收口（已核验）：step 5,735 / epoch 1.0 / train_loss 0.4499 / 2.43 s/step / 峰值 10.65 GiB；best = 末点 adapter 字节拷贝（sha `7facb03d…`，溯源 `best/best_freeze.json`）；`benchmark/sft_1f_v7.yaml` 冻结 sha `c462db1d…`。SFT 面无 `bbox_output_space`——模型输出即原图坐标，评测输入无需 rescale。

```bash
# ── pre-flight（真实终端，秒级）────────────────────────────────
nvidia-smi                                       # ① 卡空闲
sha256sum DriveAlign/configs/benchmark/sft_1f_v7.yaml   # ② 应为 c462db1d…
sha256sum runs/S13_sft_1f_v7/checkpoints/sft_1f_v7/best/adapter_model.safetensors  # ③ 应为 7facb03d…

# ── 评测（tmux new -s s13_sft_eval）──────────────────────────
conda activate autovla_codeclean
cd /root/autodl-tmp/drivealign_workspace
set -o pipefail
mkdir -p runs/S13_sft_1f_v7/reports

# 0) 8 锚 smoke（~1 min，验 adapter 装载与发射形状）
PYTHONPATH=DriveAlign/src python -m drivealign.cli.batch_benchmark \
  --config DriveAlign/configs/benchmark/sft_1f_v7.yaml \
  --anchor-manifest data/face_manifests/v7/anchor_policy_manifest.json \
  --dataset-root data/dataset_v7 \
  --nuscenes-dataroot data/nuscenes/trainval \
  --out runs/S13_sft_1f_v7/predictions_smoke8.jsonl \
  --limit 8 --batch-size 16 \
  2>&1 | tee runs/S13_sft_1f_v7/reports/step4_smoke8.log

# 1) 主跑 5,468（base 同批实测 5.2 h；batched_v1 追加写，勿覆盖重跑）
PYTHONPATH=DriveAlign/src python -m drivealign.cli.batch_benchmark \
  --config DriveAlign/configs/benchmark/sft_1f_v7.yaml \
  --anchor-manifest data/face_manifests/v7/anchor_policy_manifest.json \
  --dataset-root data/dataset_v7 \
  --nuscenes-dataroot data/nuscenes/trainval \
  --out runs/S13_sft_1f_v7/predictions_1f.jsonl \
  --batch-size 16 --resume \
  2>&1 | tee runs/S13_sft_1f_v7/reports/step4_main.log

# 2) CF blank / shuffled 各 200（与 base 同子集；blank ~4 min / shuffled ~12 min）
PYTHONPATH=DriveAlign/src python -m drivealign.cli.batch_benchmark \
  --config DriveAlign/configs/benchmark/sft_1f_v7.yaml \
  --anchor-manifest data/face_manifests/v7/anchor_policy_manifest.json \
  --dataset-root data/dataset_v7 \
  --nuscenes-dataroot data/nuscenes/trainval \
  --out runs/S13_sft_1f_v7/predictions_blank.jsonl \
  --anchors-file runs/S13_base_1f_v7_batched/counterfactual_subset.txt \
  --image-transform blank --batch-size 16 \
  2>&1 | tee runs/S13_sft_1f_v7/reports/step4_cf_blank.log

PYTHONPATH=DriveAlign/src python -m drivealign.cli.batch_benchmark \
  --config DriveAlign/configs/benchmark/sft_1f_v7.yaml \
  --anchor-manifest data/face_manifests/v7/anchor_policy_manifest.json \
  --dataset-root data/dataset_v7 \
  --nuscenes-dataroot data/nuscenes/trainval \
  --out runs/S13_sft_1f_v7/predictions_shuffled.jsonl \
  --anchors-file runs/S13_base_1f_v7_batched/counterfactual_subset.txt \
  --image-transform shuffled --batch-size 16 \
  2>&1 | tee runs/S13_sft_1f_v7/reports/step4_cf_shuffled.log
```

纪律：smoke / 主跑 / CF 用**不同 `--out` 文件**（batched 追加写，覆盖同名重跑产生 duplicate → evaluator fail-fast）；中断续跑 = 同命令加 `--resume`；四件产物齐后 eval v3（CPU，本侧执行）→ paired-delta（base_v7 vs sft_v7）+ BEV 反事实验证（BEV rules §6.1）。

---

## 7. 红线与不变量汇总

1. dataset_v4 与 v5/v6 face manifests byte-frozen；v7 一律新建资产，不改历史。
2. train-only 标定；test split 全程不可见；冻结后 val 仅 sanity 不回调。
3. 物理盘（`/root/autodl-tmp/datasets/`）写入由用户在 tmux 执行。
4. request hash（v7 自有 model face）与 record hash（training_targets 结构变化：BEV 字段 + expected_output 去 reasoning，V7-C1）双代际变化均为**预期**，不得作为 parity 失败信号。
5. v7 契约资产**六件套**（reasoning_templates / enum_phrase_map 退役，V7-C1）；v6 及更早目录保持八件套 byte-frozen。`category_vocab.json` / `motion_vocab.json` 逐字节不变；risk maxItems=8、critical_objects maxItems=8 不变；motion_state 四值枚举与判定参数（45° / 0.5 m/s）不变。
6. 标定参数仅 base / thr / r / Y_brake / Y_st 五个；h(θ_gap)、坐标变换、窗口统计全为零标定确定性量。
7. yield_required = firing 并集（反事实），与模型面 taxonomy 解耦；STOP 帧 92.2% yield=false 等既有语义佐证不受本轮影响。

## 8. 待确认项（开工前逐条拍板）

1. Step 1 与 Step 2 的先后可否合并推进（配置草稿数值仅占位，代码开发依赖字段名而非终值）——建议合并为同批提交；
2. BEV 字段名 `bev_x` / `bev_y` 与 prompt 措辞定稿（随八件套复制落定）；
3. token 预算实测超限时的预案选择权（§1.4 顺序）；
4. eval 版本归属（v2 增补 vs v3 立项，§5.3）；
5. v6 legacy 函数删除时点（本方案 = risk.py 直接单路径 + extract.py 自包含 v6 公式；若倾向保留 v6 双路径过渡，需改 §2.3 处置条款）；
6. calibration plan §10 的四项待审查确认（网格取值 / 分裂判据 / 残余口径 / 终选流程）。
