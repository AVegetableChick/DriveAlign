# S03 输出契约 v3 规格(Contract v3 = prompt v3 + schema v2 + 三词表)

> 依据:`configs/contracts/v3/` 文件夹原文(`output_schema.json`、`category_vocab.json`、`risk_taxonomy.json`、`motion_vocab.json`、`prompt.txt`),逐条对照整理。版本编号已统一(见主记录 §11):**contract v3 是当前冻结的默认配对**,文件夹内全部文件由 `src/drivealign/contracts/versions.py` 注册表解析。
> 姊妹文档:[S03_output_schema_v1_spec.md](S03_output_schema_v1_spec.md)(v1 契约)、[S03_open_questions_for_m09_and_s07.md](S03_open_questions_for_m09_and_s07.md)(遗留问题台账)。

## 1. 版本与定位

| 项 | 值 |
|---|---|
| contract_version | **v3**(当前默认,贯穿 parser / prompt / runner / records) |
| schema 迭代史 | v1(自由文本)→ **v2 内容**(本文件,装在 v3 文件夹内)→ v4 候选(coarse_distance 等,见 §9) |
| JSON Schema | Draft 2020-12,`$id: …/contracts/v3/output_schema.json`,title = contract v3 |
| 配套 prompt | `prompt.txt`(PROMPT_VERSION v3):v2 格式约束 + **canonical 词表全文枚举**(10 类别、4 运动状态带释义、8 风险词带释义、上限 8 对象) |
| 生成侧哲学 | **生成开放、评测封闭、映射显式**:prompt 列举词表降低开放词率(实测见 §8),归一层保证每个输出都有确定性行为,Schema 枚举闭包 |

## 2. 顶层结构(五字段,封闭对象,全部必填)

| 字段 | 类型 | v3 约束(v1 基础上的变化加粗) |
|---|---|---|
| `critical_objects` | array | **`maxItems: 8`**(token 确定性 + 防刷分;空列表合法) |
| `risk_factors` | array[string] | **封闭 8 项 taxonomy 枚举**;**`uniqueItems: true`**;**`maxItems: 8`**;空列表合法 |
| `reasoning` | string | `minLength: 1`;**保持自由文本——刻意设计,不判内容分**(M05 Gate:冗长 reasoning 不加分) |
| `yield_required` | boolean | 与 `speed_action` 语义分离(让行义务 ≠ 速度决策),M05 分开评分 |
| `speed_action` | enum | `ACCELERATE` / `KEEP_SPEED` / `DECELERATE` / `STOP`(不变) |

顶层 `additionalProperties: false`:v1 遗留字段 `coarse_position`、`behavior` 出现在生成输出中即 schema failure(已实测验证)。

## 3. critical_object 子结构(3 字段,v1 的 4 字段收缩)

| 字段 | 约束 | 监督信号 |
|---|---|---|
| `category` | **封闭 nuScenes 检测十类枚举**:`car, truck, construction_vehicle, bus, trailer, barrier, motorcycle, bicycle, pedestrian, traffic_cone` | Hard GT(投影框)+ M09 P/R/F1 |
| `bbox_2d` | `[x_min, y_min, x_max, y_max]` 绝对像素坐标,原点左上;`prefixItems` 4 个非负数 + `items: false` + `min/maxItems: 4`(与 v1 相同;corner 格式经论证保留,理由:Qwen2.5-VL 原生先验、退化框检查直接、DriveLM 中心点 `<c,CAM,x,y>` 换算在 join 层做与格式选择无关) | Hard GT + M09 matched IoU |
| `motion_state` | **4 值枚举:`stationary` / `same_direction` / `oncoming` / `crossing`**(替代 v1 自由文本 `behavior`) | GT 可从 nuScenes box velocity 相对 ego 派生(Stage 08 兑现) |

**移出生成契约的字段**:`coarse_position`(与 bbox 冗余且可互相矛盾)→ parser 派生;`behavior`(无 GT 无指标)→ 被 `motion_state` 取代,意图细节归 `reasoning`。

## 4. 词表与归一层(开放结果 → 封闭词表的映射)

三个版本化词表文件(每项含 `canonical` + `synonyms[]`,taxonomy/motion 另含 `gt_derivation` 派生规则,category 另含 `gt_source`):

| 文件 | canonical 数 | 设计要点 |
|---|---|---|
| `category_vocab.json` | 10 | 对齐 nuScenes GT 类集,匹配变精确比较;`vehicle`/`object`/`pedestrians` 等泛词/复数列入 `intentionally_unmapped`(映射即猜测,违反零静默修复) |
| `risk_taxonomy.json` | 8 | **每项必须能从未来物理证据确定性派生**(future ego corridor / occupancy / 最小距离 / 地图几何)——补掉 v1 "risk_factors 有 GT 无直接指标"的空档,M09 可做集合 F1、M05 risk 分量可做集合匹配 |
| `motion_vocab.json` | 4 | GT 从 box velocity + 相对自车运动派生;`moving` 列入 `intentionally_unmapped`(S03 实测出现过的泛词) |

Parser 归一顺序:直接 JSON 解析(失败且恰为单个 fenced 块时提取重试一次)→ 词表归一(strip + lower + **逐词单复数折叠**,如 `Pedestrians crossing` → `pedestrian_crossing`)→ Schema 校验 → semantic 检查。归一结果记 `normalized_terms`;**未映射词进 `unmapped_terms`、由 Schema 拒绝为 semantic_range_failure,从不静默修复或丢弃**。

## 5. 派生字段(在 parse result,不在生成契约)

`coarse_position`:仅 v3 且提供 `image_size` 时,由 bbox 中心按九宫格(front_left / front / front_right / mid_left / center / mid_right / rear_left / rear / rear_right)确定性派生,存 `derived_positions`。`to_dict()` 往返对 v3 schema 依然合法(派生值不回写生成侧)。

## 6. 校验层次与错误分类(与 v1 兼容,不变式)

1. **JSON 解析层**:fenced 信封规则(全文恰为单个围栏块时提取重试一次,`extracted_from_fences` 全链路记录);截断检测——raw 不以 `}` 结尾时错误消息附加 truncation hint(指向生成 token 预算,第 4 轮实测驱动,commit `33fff03`)
2. **词表归一层**(v3 新增,见 §4)
3. **Schema 层**:jsonschema Draft 2020-12 全量校验,错误带 `field` 定位(`$.critical_objects[0].motion_state`)
4. **Semantic 层**:bbox 角点乱序、传入 `image_size` 时的越界检查

错误四分类不变:generation / json_parse / schema / semantic_range。单样本任何失败降级为记录,批处理永不中断。

## 7. 字段监督信号对照(分析结论,详见遗留问题文档)

| 字段 | GT | 判分指标 |
|---|---|---|
| `category` / `bbox_2d` | Hard GT(observable 门控) | M09 P/R/F1 + IoU |
| `speed_action` / `yield_required` | Derived GT(未来物理) | M09 action/yield F1 |
| `risk_factors` | Derived GT(风险标签) | **v3 后可升级为直接集合 F1**(待 Stage 08 实现;v1 无直接指标) |
| `motion_state` | Derived GT(velocity 派生) | 待 Stage 08 定义(新增可评测轴) |
| `reasoning` | 语言参考(DriveLM,SFT 用) | 无(刻意) |

## 8. 实测备注(第 4/5 轮 sanity,2026-09-18)

- **词表枚举有效**:第 5 轮(512 预算)三变体 `unmapped_terms` 均为空——canonical 词表进 prompt 后,Base 模型在正常输入下未产出词表外词汇;归一层未被触发,作为确定性兜底保留
- **第 4 轮截断教训**:词表引导模型枚举更多对象(6+ 个,每个 +motion_state ≈ 30 tok),350+ tok 输出被旧 256 预算截断 → CLI 默认 512(schema 最坏 ≈ 400 tok)
- **对象数触顶**:original/shuffled 恰好输出 8 个对象(满上限)——Base 可能把 8 当目标数量而非上限,对 M09 precision 的影响与 GT 侧规则相关(Q1/Q5,见遗留问题文档)
- **blank 围栏连续 5 轮复现**(`extracted_from_fences: true`),标记跟踪价值持续成立

## 9. 交接

- 冻结:contract v3 全部五文件 + parser 归一层 + `coarse_position` 派生规则
- Stage 06:契约接入 `DriveAlignRecord`;补带 `available_speed` 的 sanity(路径未实测)
- Stage 07:DriveLM join 时处理**信号灯词表缺口**(DriveLM key objects 含信号灯,category 词表无此类,Q4);bbox 坐标系核对
- Stage 08:GT 侧确定性几何规则(top-K ≤ 8 + 阈值标定,Q1/Q2);`motion_state` / `risk_factors` 直接指标落地
- v4 候选(不进 v3):`coarse_distance` 深度层枚举(3D bbox 的廉价中间态,GT 从 nuScenes 3D 距离分箱派生)、信号灯类
