# S03 结构化输出契约实验记录

## 1. 实验概览

- 实验阶段：Stage 03 驾驶结构化输出契约
- 实验日期：2026-09-17
- 运行环境：`autovla_codeclean`（Python 3.9.23，pytest 8.4.2，jsonschema 4.25.1）
- GPU：NVIDIA GeForce RTX 4080 SUPER（sanity 运行峰值显存 8454813184 字节 ≈ 8.45 GiB）
- 模型：Qwen2.5-VL-3B-Instruct（bfloat16，flash_attention_2，local_files_only）
- 代码产物：`src/drivealign/contracts/{output,prompt}.py`、`src/drivealign/inference/structured_runner.py`、`src/drivealign/cli/structured_sanity.py`、`tests/unit/*`
- 运行产物：`runs/structured_infer_sanity_check/`（original/blank/shuffled.jpg、records.jsonl、summary.json）

## 2. Git 与版本基线

### DriveAlign

- Repository HEAD：`e9755305e69b6f4f449dd3aeae3759b9f1e4e7b4`
- 说明：Stage 03 全部改动（schema、contracts、runner、CLI、tests）在记录时仍未提交（untracked），不应解释为已包含在该 commit 中。

### 上游仓库

- AutoVLA：`ba34eed74ce6729e7986592d0e66cbaca397b4fa`
- nuScenes-devkit：`b40adc467b919192899405d9b77871afee8efa07`
- TRL：`670458a5b804ad8beb85bfb1c2f2658271f8717d`

## 3. 目标与边界

按 `docs/project_stages/03_structured_output_contract.md`：冻结第一阶段输出 Schema，打通 Base 模型"生成原始文本 → 严格解析 → 逐项校验"闭环。使用人工选定受控图片，不构建正式数据集，不做 M09 指标。完成标准：解析失败可结构化记录且不中断批处理，成功结果满足版本化 Schema。

## 4. 输入与产物

- 输入：Stage 02 通用单图 Runner；CAM_FRONT 测试图（1600×900）；`available_speed=null`（本轮未注入因果速度）
- 产物 1：`configs/contracts/output_schema_v1.json` —— 机器可读 Schema（五顶层字段，`speed_action` 四值枚举，`additionalProperties: false`），冻结（原定 `configs/model/`，后迁至 `configs/contracts/` 以与 `src/drivealign/contracts/` 消费方对齐，`configs/model/` 保留给模型加载配置；内容与版本不变，见 §10 位置变更说明）
- 产物 2：固定 prompt 模板 `PROMPT_VERSION = "v2"`（v1 首版，因围栏问题升级，见 §6 证据链）
- 产物 3：严格 parser/validator（`contracts/output.py`）—— 错误四分类 `OutputErrorCategory`（generation / json_parse / schema / semantic_range），以 JSON Schema 文件为单一事实源（jsonschema Draft 2020-12），附加 bbox 角点顺序与图像边界 semantic 检查
- 产物 4：结构化 Runner（`inference/structured_runner.py`）—— `SampleRequest/SampleRecord`、`run_structured_inference/run_structured_batch/record_to_dict`，生成异常降级为 generation_failure 记录，批处理永不中断
- 产物 5：Sanity 入口（`cli/structured_sanity.py`）—— 原图 / blank（中灰 128,128,128）/ shuffled（4×4 patch，seed=0）三变体
- 产物 6：单元测试 40 例（`tests/unit/test_contracts_output.py`、`test_structured_runner.py`）

## 5. 执行流程

1. 实现契约层与严格 parser，`autovla_codeclean` 下自验证 11 项；
2. 实现结构化 Runner 与批处理（stub + 真实失败路径验证）；
3. Gate 用例固化为 `tests/unit` 单元测试；
4. 三轮 GPU sanity（同图三变体，greedy，max_new_tokens=256，记录见 §6）；
5. 依据 sanity 证据修订 prompt（v1→v2）并落地 parser 围栏提取规则。

## 6. 验证与 Gate

### 6.1 证据链：三轮 sanity 结果（同图、greedy 确定性）

| 运行轮次 | PROMPT_VERSION | parser 行为 | original | blank | shuffled |
|---|---|---|---|---|---|
| 第 1 轮 | v1 | 严格（围栏 = parse failure） | json_parse_failure | json_parse_failure | json_parse_failure |
| 第 2 轮 | v2 | 严格 | ok（189 tok） | json_parse_failure（128 tok，围栏） | ok（202 tok） |
| 第 3 轮（最终） | v2 | 严格 + 文档化围栏提取 | ok，extracted=False | ok，**extracted=True** | ok，extracted=False |

第 3 轮最终记录（`runs/structured_infer_sanity_check/records.jsonl`，input 2143 tok，do_sample=false，max_new_tokens=256）：

| sample_id | status | extracted_from_fences | output_tokens | generation_seconds | speed_action | yield_required | 对象数 |
|---|---|---|---|---|---|---|---|
| original | ok | False | 189 | 12.13 | KEEP_SPEED | false | 2 |
| blank | ok | True | 128 | 6.64 | KEEP_SPEED | false | 1 |
| shuffled | ok | False | 202 | 10.69 | DECELERATE | true | 2 |

### 6.2 证据链关键事实

1. **第 1 轮 3/3 失败的唯一原因是 markdown 围栏**：三条 raw_text 均以 ` ```json ` 开头；剥除围栏后内容 100% 通过 schema 校验。排除截断（output_tokens 184/145/218 < 256，尾部 `}` 完整）与内容违规。
2. **prompt v2 有效但不彻底**：v2 将围栏禁令三重显式化（开头约束 + 结尾复述 + "start with { and end with }"），第 2 轮 original/shuffled 转为 ok；blank（纯灰图，退化输入）仍复发围栏——模型在无视觉信息可依托时更依赖训练分布先验，prompt 工程无法对 3B 模型确定性根除该行为。
3. **围栏提取规则的决策依据**：S03 文档禁止的是"为提高通过率静默修复未知字段或非法枚举"；围栏提取属**信封解析约定**而非字段修复。实现为：直接解析失败且全文恰为单个 fenced 块时提取重试一次，`extracted_from_fences` 全链路记录（ParseResult → SampleRecord → records.jsonl），内容仍零修复；截断、未闭合、多块围栏仍为 json_parse_failure 且消息注明 "after fence extraction"。
4. **管线敏感性与确定性**：blank 编造 road 内容并输出 KEEP_SPEED/1 对象，shuffled 输出 DECELERATE + yield=true，original 输出 KEEP_SPEED/2 对象——三变体输出互不相同（敏感性成立）；第 3 轮 original/shuffled 输出与第 2 轮逐 token 一致（greedy 确定性成立）。
5. **内容质量不做评价**：Base 模型 bbox 粗糙、对象少（如仅 2 车）不构成 Gate 关注点；管线可重复、错误不吞、逐样本可追溯才是本阶段对象。

### 6.3 Gate 逐条核对

- 合法样例通过；缺 key、错误类型、非法 bbox、`YIELD` 作为速度动作、多余字段、负 bbox 等样例全部失败且带 `field` 定位 —— 单测参数化矩阵覆盖
- raw text、parse result 和 validation errors 由同一 sample_id 追溯 —— `SampleRecord` 携带 raw_text/output/errors/prompt_version/schema_version/extracted_from_fences，`record_to_dict` JSONL 持久化
- 单个坏输出不终止其余样本 —— stub 混合批处理 + 全失败批处理均返回全量记录
- prompt 只含当前图像、可选当前/历史可得速度和固定任务说明 —— 无 future 字样（单测断言），速度行仅前置一行且空串视为未提供
- 单元测试：`PYTHONPATH=DriveAlign/src python -m pytest DriveAlign/tests/unit -q` → **40 passed**

### 6.4 Gate 判定

通过。Schema 管线可重复、错误分类可追溯、批处理不中断；Base 内容质量低未作为失败依据。

## 7. 交接

- 冻结：`output_schema_v1`、`PROMPT_VERSION = "v2"`、parser 错误四分类 + 围栏提取信封规则（含 `extracted_from_fences` 标记）
- Stage 06 将上述契约接入 `DriveAlignRecord`；Stage 09 才进行正式 Base 内容评测
- 后续注意：(1) 正式数据统计应跟踪 `extracted_from_fences` 比率，若围栏率异常升高需重审 prompt 或改用结构化解码；(2) `available_speed` 注入路径在本轮未实测，Stage 06 Adapter 接入时需补带速度的 sanity；(3) bbox 坐标系与 DriveLM QA 对齐细节留待 Stage 07 数据构建核对

## 8. 复现实验命令

```bash
conda activate autovla_codeclean
cd /root/autodl-tmp/drivealign_workspace

# 单元测试（无 GPU）
PYTHONPATH=DriveAlign/src python -m pytest DriveAlign/tests/unit -q

# GPU sanity（三变体，产物写入 runs/structured_infer_sanity_check/）
PYTHONPATH=DriveAlign/src python -m drivealign.cli.structured_sanity \
  --image data/nuscenes/mini/samples/CAM_FRONT/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151603512404.jpg \
  --out runs/structured_infer_sanity_check
```

## 9. 附录：第 3 轮完整记录（prompt 与逐样本输出）

以下内容逐字取自 `runs/structured_infer_sanity_check/records.jsonl`（`available_speed=null`）。

### 9.1 实际使用的 prompt（PROMPT_VERSION = v2，三变体共用同一 prompt 文本）

```text
Analyze the current front-camera image of a driving scene and answer with exactly one machine-parseable JSON object. Your entire response must start with { and end with }: no code fences, no backticks, no markdown, and no text before or after the JSON object.

The JSON object must have exactly the following fields and no others:

- "critical_objects": a list of objects relevant to the current driving decision. Each item is an object with exactly these keys: "category" (short object type), "bbox_2d" (four numbers [x_min, y_min, x_max, y_max] in image pixel coordinates), "coarse_position" (rough position of the object in the image), "behavior" (observed or likely motion of the object). Use [] when there is no such object.
- "risk_factors": a list of short strings describing visible or inferable factors that affect driving risk. Use [] when there are none.
- "reasoning": a concise explanation grounded in the current image and the available ego speed, if provided.
- "yield_required": true when the ego vehicle should yield to another road user or situation, otherwise false.
- "speed_action": exactly one of "ACCELERATE", "KEEP_SPEED", "DECELERATE", "STOP".

Restated output rules: the response begins with { and ends with }; it never contains the ``` characters; it contains the JSON object only, with no commentary and no extra fields.
```

### 9.2 original（status=ok，extracted_from_fences=False，189 output tokens）

raw_text（模型原始回复）：

```json
{
  "critical_objects": [
    {
      "category": "car",
      "bbox_2d": [849, 476, 921, 526],
      "coarse_position": "middle",
      "behavior": "moving"
    },
    {
      "category": "car",
      "bbox_2d": [993, 480, 1074, 528],
      "coarse_position": "right",
      "behavior": "moving"
    }
  ],
  "risk_factors": [
    "traffic",
    "pedestrians"
  ],
  "reasoning": "There are cars on the road, which could potentially cause a collision if not careful. Additionally, pedestrians may be crossing the street, requiring caution.",
  "yield_required": false,
  "speed_action": "KEEP_SPEED"
}
```

解析后 output：与 raw_text 语义一致，仅数值被契约层归一为 float（如 `[849.0, 476.0, 921.0, 526.0]`），`to_dict` 序列化见 records.jsonl。

### 9.3 blank（status=ok，extracted_from_fences=True，128 output tokens）

raw_text（模型原始回复，**围栏复发样本，经文档化信封规则提取**）：

````text
```json
{
  "critical_objects": [
    {
      "category": "road",
      "bbox_2d": [0, 0, 643, 512],
      "coarse_position": "left side",
      "behavior": "static"
    }
  ],
  "risk_factors": [
    "no other vehicles visible",
    "smooth road surface"
  ],
  "reasoning": "The road appears to be clear with no other vehicles in sight, indicating a safe driving environment.",
  "yield_required": false,
  "speed_action": "KEEP_SPEED"
}
```
````

解析后 output：与 raw_text 内容一致（数值归一为 float）。注意模型对纯灰空白图"编造"了 road 对象——内容真伪不构成 Gate 关注点，该样本仅用于验证管线敏感性与围栏规则。

### 9.4 shuffled（status=ok，extracted_from_fences=False，202 output tokens）

raw_text（模型原始回复）：

```json
{
  "critical_objects": [
    {
      "category": "pedestrian",
      "bbox_2d": [1394, 226, 1418, 337],
      "coarse_position": "right side of the road",
      "behavior": "walking"
    },
    {
      "category": "pedestrian",
      "bbox_2d": [1000, 482, 1032, 590],
      "coarse_position": "right side of the road",
      "behavior": "walking"
    }
  ],
  "risk_factors": [
    "pedestrians crossing the street",
    "traffic lights at intersections"
  ],
  "reasoning": "There are pedestrians crossing the street and traffic lights at intersections, which may require slowing down or stopping.",
  "yield_required": true,
  "speed_action": "DECELERATE"
}
```

解析后 output：与 raw_text 内容一致（数值归一为 float）。

### 9.5 附录说明

- 三个 raw_text 与解析输出共同证明：`extracted_from_fences` 仅在 blank 上为 True，original/shuffled 的干净输出不受信封规则影响（第 2 轮同输入输出逐 token 一致，greedy 确定性）。
- 完整逐字段记录（含 prompt、prompt_version、schema_version、telemetry、errors）见 `runs/structured_infer_sanity_check/records.jsonl`。

## 10. 位置变更说明（实现期组织调整）

- `configs/model/` 中的 Schema 与词表文件（`output_schema_v1.json`、`output_schema_v2.json`、`category_vocab_v1.json`、`risk_taxonomy_v1.json`、`motion_vocab_v1.json`）迁移至 `configs/contracts/`。
- 理由：这些文件是输出契约与词表资产，由 `src/drivealign/contracts/`、M09 评测与 M05 verifier 消费，与模型本身无关；`configs/model/` 保留给模型加载配置（如 `qwen25_vl_3b.yaml`）。蓝图 `configs/` 子目录为指示性结构（"只在阶段实现时创建目录"），新增语义化子目录不违反蓝图。
- 代码同步点：`contracts/output.py` 的 `DEFAULT_SCHEMA_PATH` 与模块 docstring、`contracts/prompt.py` docstring；单测均经 `DEFAULT_SCHEMA_PATH` 引用，无需改动。
- 内容与版本均不变，纯路径组织调整。
- Prompt 存储方式变更：模板由 Python 常量改为版本化文件 `configs/contracts/prompts/structured_output_v{1,2,3}.txt`，`contracts/prompt.py` 变为加载器（`build_structured_prompt(available_speed, version)`）。v1 从第 1 轮 sanity records.jsonl 原样恢复；v2 与原常量逐字一致；v3 为 schema v2 配套草案（封闭词表枚举）。默认版本仍为 v2（与 schema v1 解析器配对），切换到 v3 须与 parser v2 同批上线。
