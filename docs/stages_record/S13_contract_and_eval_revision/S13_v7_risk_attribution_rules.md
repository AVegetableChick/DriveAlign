# S13 Contract v7 — 风险因子归因设计（risk_factors ↔ critical_objects 关联）（拍板稿 v1）

- 状态：**拍板稿 v1**（2026-10-11 三项设计点全部拍板；Step 2 通过 G2 后的追加变更，随 Step 3 开工前落地）
- 决策依据：2026-10-11 用户三拍板（D-A1-1 ~ D-A1-3，见 §1）；决策台账 V7-A1 条目随执行登记（执行清单步骤 0）
- 机制参照：[S13_v7_risk_factor_rules.md](./S13_v7_risk_factor_rules.md)（v2.4）——**归因完全复用其因子判据，不改归因机制本身**（2026-10-12 crossing 判据改到达时间口径 §0.7，随 rules v2.4 同步，见 §3.2 表注）；BEV 字段口径见 [S13_v7_bev_gt_rules.md](./S13_v7_bev_gt_rules.md)，与本变更正交
- 代码基线：[risk.py](../../../src/drivealign/gt/risk.py)（G2 已通过版本）、[backfill.py](../../../src/drivealign/gt/backfill.py)、[output.py](../../../src/drivealign/contracts/output.py)、[output_schema.json](../../../../configs/contracts/v7/output_schema.json)（v7 未冻结）
- 单位：距离 m；id 为无量纲整数

---

## 1. 拍板总览（D-A1-1 ~ D-A1-3）

| 编号 | 决策点 | 拍板结论 | 拍板日期 |
|---|---|---|---|
| D-A1-1 | 归因落点 | **模型面**：`risk_factors` 对象化为 `[{factor, objects}]`，模型输出归因 | 2026-10-11 |
| D-A1-2 | 引用机制 | **显式 id 字段**：`critical_object` 新增 `id`（0–7 整数），`objects` 引用该 id | 2026-10-11 |
| D-A1-3 | GT 归因口径 | **全量合格对象**：因子命中逻辑与阈值逐字不动，仅"记录哪些对象"从首中即止改为全收集 | 2026-10-11 |

三项合起来 = 归因的完整定义：**GT 算什么**（§3）、**模型面长什么样**（§4）、**哪些东西不许动**（§7）。

---

## 2. 背景与时机

### 2.1 现状缺口

- v7 模型面：`risk_factors` 是纯字符串数组（maxItems 8、去重、6 枚举），`critical_object` 无 id 字段——模型面**没有归因锚点**，因子与对象在输出层是两张皮。
- GT 引擎：`derive_risk_factors`（[risk.py](../../../src/drivealign/gt/risk.py) L301–402）内部天然知道每个因子由哪些对象触发，只是最后坍缩为因子名集合——归因信息已存在，只差带出来。

### 2.2 时机：代内变更零代价

v7 尚未冻结、Step 3 未开工、dataset_v7 未建。本变更全部落在 v7 代内：

- request hash 变化被 v7 face 吸收（v7 face 本就未冻结，无历史可比性负担）；
- record hash 随 dataset_v7 首次生成，不存在重生成；
- **冻结后再改 = v8 全代际重建**，故现在是最廉价时点。

### 2.3 收益

- 监督信号闭环：对象级 grounding（bbox/BEV/motion）↔ 因子级语义在同一输出内互证，SFT 学到"哪个对象构成哪类风险"；
- Step 5 可评归因准确率（factor-level attribution precision/recall，定义细节留 eval 增补稿，与 BEV 指标族同批登记 eval 台账）。

---

## 3. GT 归因定义（精确口径，D-A1-3）

### 3.1 总原则

**因子命中判定逻辑逐字不动**：firing 引擎（K=13 采样、d_k−h_k < base 严格）、统一横向 gate、类别集合分组、两段式 stationary（含无 track 回退仅 t0）、全部严格不等号与含等号边界——均保持 rules v2.2 口径（注：D-A1-3 拍板时点的"命中判据不动"限于归因变更自身；2026-10-12 crossing 判据独立修订为到达时间口径，见 rules v2.4 §0.7，归因机制不受影响）。归因唯一改动 = **首中即止三因子去掉 `break`，改为全量收集**（不改变因子存在性，只改变归因记录范围）。

### 3.2 六因子归因口径表

| 因子 | 命中判据（不动） | 归因对象集合（D-A1-3：全量合格） | rules 条款 |
|---|---|---|---|
| `vru_crossing` | 到达时间判据（rules v2.4 §0.7：`v_toward > 0.8 ∧ t_arrive ≤ 5.0`）∧ VRU 类别 | 池内全部满足两条件的对象 | §1/§2 |
| `vehicle_crossing` | 同上（vehicle 五类） | 同上（vehicle 族） | §1/§2 |
| `lead_vehicle_braking` | 族集合 ∧ in_path(Y_brake) ∧ t0 速 > 0.5 ∧ 有 track ∧ 窗 min ≤ v0 − 0.5 | **全部**满足判据的前车（原首中即止 → 全收集；无 track / 空窗者本就不合格，不归因） | §3 |
| `stationary_obstacle` | t0 停 ∧（窗 max ≤ 0.5 或无窗回退仅 t0）∧ in_path(Y_st) | **全部**满足两段式的在径静止对象（原首中即止 → 全收集） | §4 |
| `object_near` | distance_m ≤ r（含等号） | **全部** ≤ r 的池内对象 | §5 |
| `oncoming_traffic` | 族集合 ∧ motion = oncoming ∧ distance < 20（严格） | **全部**满足的对向对象（原首中即止 → 全收集） | §6 |

三条要点：

1. **因子存在性 ⟺ 归因集合非空**（全收集口径下按构造成立），故 `objects` 恒非空、因子列表与 names 逐一对应；
2. `any_firing`（yield 并集，§7）语义逐字节不变——static firing 对象仍只喂 yield、不进任何因子归因；
3. **确定性顺序**：braking 归因按距离升序（沿用原循环排序键 `(round(distance_m, 3), ann_token)`）；其余因子按池序（距离升序 top-8）；GT 输出 `objects` 按 id 升序。

### 3.3 id 分配（D-A1-2 GT 侧）

- GT `critical_objects` 沿用池序（距离升序 top-8，既有确定性顺序不动），**id = 列表下标 0..7**；
- `risk_factors[i].objects` = 该因子归因 ann_token 集合 → id 映射，升序输出；
- id 唯一性由构造保证（同一下标只出现一次）。

### 3.4 GT 输出示例（数值为示意，用草稿阈值 base=2.0 / r=5.0 / Y_st=2.5）

```json
{
  "critical_objects": [
    {"id": 0, "category": "pedestrian", "bbox_2d": [512, 401, 601, 688],
     "motion_state": "crossing", "bev_x": 3.9, "bev_y": 0.5},
    {"id": 1, "category": "car", "bbox_2d": [743, 380, 921, 522],
     "motion_state": "stationary", "bev_x": 12.8, "bev_y": 0.0}
  ],
  "risk_factors": [
    {"factor": "vru_crossing", "objects": [0]},
    {"factor": "stationary_obstacle", "objects": [1]},
    {"factor": "object_near", "objects": [0]}
  ],
  "yield_required": true,
  "speed_action": "DECELERATE"
}
```

---

## 4. 模型面（D-A1-1 + D-A1-2 落地）

### 4.1 schema 变更（output_schema.json）

```jsonc
// critical_object：required 增加 "id"；properties 增加：
"id": { "type": "integer", "minimum": 0, "maximum": 7 }

// risk_factors：string 数组 → 对象数组
"risk_factors": {
  "type": "array",
  "maxItems": 6,                    // 8 → 6（taxonomy 尺寸，因子数上限本就 ≤ 6）
  "items": {
    "type": "object",
    "additionalProperties": false,
    "required": ["factor", "objects"],
    "properties": {
      "factor": { "enum": [/* 6 项 canonical，顺序 = RISK_TAXONOMY_ORDER */] },
      "objects": {
        "type": "array",
        "minItems": 1,              // 因子存在 ⟺ 至少一个触发对象
        "maxItems": 8,
        "uniqueItems": true,
        "items": { "type": "integer", "minimum": 0, "maximum": 7 }
      }
    }
  }
}
```

口径说明：

- `uniqueItems`（外层数组）只约束整对象重复，**同 factor 两条目 schema 拦不住**——因子去重由 parser 语义校验拒绝（§4.3）；
- `objects` 引用合法性（id 存在于 critical_objects）schema 无法表达，同样由 parser 语义校验（§4.3）；
- `id` 设硬界 0–7（与 maxItems=8 自洽）；与 bev 字段"不设硬界"的倾向不冲突——id 是离散标识符且上界由结构保证，bev 是连续物理量。

### 4.2 prompt 措辞要点（prompt.txt）

- `critical_objects` 教学：每条目增加 `"id"`（"a unique integer from 0 to 7 identifying the object within this response"）；
- `risk_factors` 教学对象化："a list of at most 6 items, each an object with exactly these keys: `factor`（6 项枚举及既有语义句不变）、`objects`（the ids of the critical objects that trigger this factor, at least one id）"；空列表语义不变；
- Restated output rules 段不动。

### 4.3 parser 语义校验清单（contracts/output.py）

沿用"从不静默修复"纪律，以下违规 = `SEMANTIC_RANGE_FAILURE`：

1. 同一 `factor` 重复条目 → failure（schema 拦不住的按成员去重）；
2. `objects` 引用的 id 不存在于 `critical_objects` → failure；
3. `critical_objects` 内 id 重复 → failure；
4. `objects` 内重复 id → failure（schema `uniqueItems` 双保险）；
5. 同义词归一化移至 `factor` 键内（risk_taxonomy 表继续消费，`normalized_terms` / `unmapped_terms` 追溯不变）。

### 4.4 四字段契约不变（V7-C1 不动摇）

顶层字段集仍为 `{critical_objects, risk_factors, yield_required, speed_action}`；变更只在 `critical_object` 条目（+id）与 `risk_factors` 条目形状（str → {factor, objects}）。`EXPECTED_OUTPUT_KEYS`（parity check）不动。

---

## 5. 代码结构影响面

### 5.1 引擎返回（gt/risk.py）

`derive_risk_factors` 返回从 `(factors, any_firing)` 扩为 **`(factors, attribution, any_firing)`**：

- `attribution: Dict[str, Tuple[str, ...]]` —— 因子名 → 归因 ann_token 元组（§3.2 确定性顺序）；
- 因子命中分支逻辑不动，仅三处 `break` 改收集。

### 5.2 GT 落字段（gt/backfill.py）

- `critical_objects` 每项加 `"id": 下标`；
- `risk_factors` = `[{"factor": f, "objects": [ids...]}]`（ann_token 归因经 id 映射）；
- `_validate_schema` 随新 schema 自动收紧，无需改逻辑。

### 5.3 解析面（contracts/output.py）

- `CriticalObject` 加 `id: int | None = None`（None 不进 `to_dict()`——v1–v6 dict 形状不变，与 bev_x/bev_y 同款模式）；
- 新增 `RiskFactorAttribution(factor: str, objects: tuple[int, ...])`；
- `StructuredDrivingOutput` 保留 `risk_factors: tuple[str, ...]`（因子名，全部版本可用，evaluator / gt_distribution 计数逻辑零改动），**新增** `risk_attribution: tuple[RiskFactorAttribution, ...] = ()`（仅 v7+ 面填充）；`to_dict()` 按 version 形状输出：有 attribution → `[{"factor", "objects"}]`，否则 → 名字列表；
- v7 面下 names 与 attribution 由同一 payload 同步构造，天然一致。

### 5.4 SFT / eval 面

- SFT targets 渲染若为整 dict JSON-dump 则形状自动跟随；若逐字段拼装则执行时核对（执行清单步骤 4 顺带确认）；
- Step 5 归因指标族（在 2D IoU 匹配对上计算 factor-level attribution P/R）只登记方向，定义细节留 eval 增补稿——本稿只定 GT/模型面口径。

---

## 6. token 预算影响（实施规划稿 §1.4 复测项）——已实测，回炉拍板 V7-A2

最坏 case（8 目标全满 + 6 因子全选，A 拍板后紧凑 JSON 基线 454 tokens）。归因落地后按 §1.4 同款 Qwen2.5-VL-3B 实测（2026-10-11，三口径 × 三发射形态）：

| 口径（8 目标全满 + 6 因子全选） | 紧凑 | 带空格 | fence |
|---|---|---|---|
| A：每因子引用 1 id（原 §6 估算口径） | 536 | 678 | 682 |
| B：GT 可达最坏（3 因子 × 全 8 id） | 545 | 699 | 703 |
| C：schema 对抗最坏（6 因子 × 全 8 id，GT 不可达） | 620 | 804 | 808 |

结论：归因增量（8 × id 字段 + 对象化包装 + id 引用列表）把**带空格/fence** 形态推过 640（SFT 监督文本是带空格形态，即模型实际发射形态），且**紧凑**形态全程 ≤ 640。实测超限触发本稿预案的"回炉拍板"。

**回炉拍板（V7-A2，2026-10-11）：SFT 目标发射改紧凑 JSON + 生成预算 700。**

- **紧凑发射**：[targets.py](../../../src/drivealign/sft/targets.py) `DUMP` separators 由 `(", ", ": ")` 改 `(",", ":")`——模型发射形态跟随 SFT 监督文本，紧凑化后三口径最坏 620 ≤ 预算全口径安全。v6 冻结样本 artifact 的 target_text 已落盘不受影响；v6 冻结 SFT config（sft_1f.yaml / sft_smoke_1f.yaml）属 v6 代际不动，v7 SFT config（GPU 阶段创建）`target_dump.separators` 写 `[",", ":"]`。
- **生成预算 640 → 700**（拍板 B 上限再评估）：700 在紧凑发射下余 80（对抗口径 C）；v7 run configs（GPU 阶段）统一写 700。
- 代价：输出文本可读性下降（无空格）；SFT 监督面 v6（带空格）→ v7（紧凑）风格代际变化，已登记决策台账 §6 V7-A2。

---

## 7. 明确不变项（红线清单）

| 项 | 口径 |
|---|---|
| 六因子命中判据 | firing / gate / 类别集合 / 两段式 / 严格不等号 / 含等号——逐字不动（rules v2.2 §0–§8） |
| 五个标定槽位 | base / thr / r / Y_brake / Y_st 零新增零改动——归因与 Step 3 标定完全正交 |
| yield_required | = firing 并集（§7）不动，static firing 仍不进归因 |
| 四字段顶层契约 | V7-C1 不动 |
| motion_state / action / pool / BEV | 全部不动 |
| v1–v6 契约面与冻结资产 | 不动（parser 兼容模式：id / attribution 仅 v7+ 面生效） |
| 因子列表顺序 | RISK_TAXONOMY_ORDER 不动 |

---

## 8. 执行清单（G2 delta，Step 3 开工前完成）

| 步骤 | 内容 | 验证 |
|---|---|---|
| 0 | 决策台账登记 V7-A1（D-A1-1~3）+ 规则稿升 v2.3（§1–§6 各因子节补归因句 + 新增归因章） | 文档交叉引用一致 |
| 1 | schema + prompt 更新（六件套中两件，§4.1/§4.2） | pin 测试更新 |
| 2 | parser：`id` 字段 + `RiskFactorAttribution` + §4.3 五条语义校验 | test_contracts_output |
| 3 | risk.py 归因返回三元组 + 三处 break 改全收集 | test_gt_risk（新增归因断言 + 全收集边界例） |
| 4 | backfill.py id 分配 + 对象化落字段（顺带核对 SFT targets 渲染路径） | test_gt_backfill（形状断言更新） |
| 5 | gt_distribution 计数适配（factor 名提取）+ parity fixtures | test_dataset_build_v4 |
| 6 | 契约对齐测试更新（enum 路径移位 / 新 pin：id required、risk_factors item 结构、maxItems 6） | test_contract_v7_alignment |
| 7 | token 预算实测，§1.4 表格刷新 | ≤ 640（B 拍板） |
| 8 | 全量单测 + G2 delta 判定 | pytest 全绿（417 基数上增改） |
| 9 | 实施规划稿新增 §2.9 归因 delta 执行记录 + project_memory 回填 | — |

## 9. 复核项与待办

| 项 | 内容 | 时点 |
|---|---|---|
| token 实测 | §6 三口径 × 三形态复测完成，超 640 → **已回炉拍板 V7-A2**（紧凑发射 + 700） | 步骤 7 ✓ |
| SFT 渲染核对 | targets.py 是整 dump 还是逐字段拼装 | 步骤 4 顺带 |
| 归因指标族 | factor-level P/R 定义（匹配对上）登记 eval 台账 | Step 5 / eval 增补稿 |
| G2 delta 判定记录 | 判定结论回填实施规划稿 | 步骤 9 |
