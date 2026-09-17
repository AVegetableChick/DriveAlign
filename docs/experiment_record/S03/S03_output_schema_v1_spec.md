# S03 输出契约 v1 结构规格（output_schema_v1）

本文档是 `configs/contracts/v1/output_schema.json` 的人工可读规格说明，两者内容一致；机器校验以 JSON 文件为单一事实源。契约于 Stage 03 冻结，后续仅允许新增版本（v2+），不得原位修改。（注：Schema 与词表文件位于 `configs/contracts/`，原 `configs/model/` 路径已在实现期迁移，`configs/model/` 仅保留模型加载配置。）

## 1. 版本与定位

- Schema 版本：`schema_version = "v1"`（`$id: https://drivealign.local/schemas/output_schema_v1.json`）
- 规范：JSON Schema Draft 2020-12
- 用途：单图驾驶场景分析的结构化输出契约，服务 Stage 03 严格解析、Stage 06 `DriveAlignRecord`、Stage 09 正式评测与 Stage 14 训练 verifier
- 配套 prompt 版本：`PROMPT_VERSION = "v2"`（`src/drivealign/contracts/prompt.py`）

## 2. 顶层结构（封闭对象，五字段全部必填）

顶层 `type: object`，`additionalProperties: false`——契约之外的任何字段都是 schema failure。

| 字段 | 类型 | 约束 | 含义 |
|---|---|---|---|
| `critical_objects` | array | 元素见 §3；允许 `[]` | 与当前驾驶决策相关的关键对象 |
| `risk_factors` | array[string] | 每项 `minLength: 1`；允许 `[]` | 可见或可推断的风险因素 |
| `reasoning` | string | `minLength: 1` | 基于当前图像与可得状态的简洁解释 |
| `yield_required` | boolean | — | 是否需要向其他道路使用者或情形让行 |
| `speed_action` | string enum | 仅 `ACCELERATE` / `KEEP_SPEED` / `DECELERATE` / `STOP` | 请求的速度控制动作 |

## 3. critical_object 子结构（`$defs/critical_object`）

数组每个元素同样是封闭对象（`additionalProperties: false`），四字段全部必填：

| 字段 | 类型 | 约束 | 含义 |
|---|---|---|---|
| `category` | string | `minLength: 1` | 简短对象类型 |
| `bbox_2d` | array[number] | 恰好 4 项（`prefixItems` 四个 `minimum: 0` 的 number + `items: false` + `minItems/maxItems: 4`），顺序 `[x_min, y_min, x_max, y_max]`，图像像素坐标 | 2D 边界框 |
| `coarse_position` | string | `minLength: 1` | 对象在图中的粗略位置 |
| `behavior` | string | `minLength: 1` | 对象的观测或推测运动 |

## 4. 校验层次（parser 实际执行顺序）

解析入口：`drivealign.contracts.output.parse_structured_output`。内容零修复——不裁剪尾部文本、不删除未知字段、不重映射非法枚举。

1. **JSON 解析层**：`json.loads` 直接解析；失败且全文恰为单个 markdown fenced 块（```` ```json ... ``` ````）时，提取内部内容重试一次，`ParseResult.extracted_from_fences` 记录该事件。截断、未闭合、多块围栏不在此列，仍为解析失败。
2. **Schema 层**：以 `output_schema_v1.json` 为单一事实源（jsonschema Draft 2020-12），覆盖缺 key、错类型、非法枚举、多余字段、bbox 数目与非负约束、空字符串等。
3. **Semantic 层**（JSON Schema 表达不了的几何约束）：
   - bbox 角点顺序：`x_min <= x_max` 且 `y_min <= y_max`
   - 图像边界：调用方传入 `image_size` 时，bbox 不得越界

## 5. 错误分类（`OutputErrorCategory`）

| 类别 | 触发条件 |
|---|---|
| `generation_failure` | 模型生成阶段异常（如图像缺失、推理报错） |
| `json_parse_failure` | raw text 不是合法 JSON（含截断、非法围栏；信封提取后仍失败时消息注明 "after fence extraction"） |
| `schema_failure` | 违反 §2/§3 任一 JSON Schema 约束，`field` 定位形如 `$.critical_objects[0].bbox_2d` |
| `semantic_range_failure` | 通过 schema 但违反 §4 第 3 层几何约束 |

每次解析收集全部错误（非首错即停）；`ParseResult` 携带 `raw_text` 回传，同一 sample id 可追溯 raw text、parse result 与 errors。

## 6. 与数据模型的衔接

- `CriticalObject` / `StructuredDrivingOutput`（frozen dataclass）与 Schema 一一对应，`to_dict()` 可往返
- 数值归一：bbox 整数在契约层归一为 float（`849` → `849.0`），不影响 schema `number` 判定
- Stage 06 将本契约接入 `DriveAlignRecord` 的模型生成侧；DriveLM QA 数据仅作训练 target/Language Reference（按 sample token join），不是输出格式

## 7. 实测行为备注（第 3 轮 sanity，见主记录 §6/§9）

- v1–v3 轮证据链证明：围栏信封是 3B 模型输入相关的概率行为，最终规则（prompt v2 + 信封提取）下 3/3 ok，`extracted_from_fences` 仅在 blank 样本为 True
- 后续正式数据需跟踪 `extracted_from_fences` 比率（交接项，见主记录 §7）
