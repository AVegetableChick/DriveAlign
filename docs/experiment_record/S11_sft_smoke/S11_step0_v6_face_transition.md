# S11 Step 0 实施计划（v6 模型面过渡 + 1F Base 重跑，2026-10-05）

> 状态：**Step 0.1/0.2 已完成（2026-10-05，G0.1/G0.2 PASS）**；Step 0.3（GPU 重跑，用户夜间）与 Step 0.4（冻结落档）待执行。裁定依据见 [S11_reasoning_supervision_decision.md](S11_reasoning_supervision_decision.md)（最终裁定：模型面删除 reasoning 字段，升版 v6）；执行顺序为用户修订版：**0.1 → 0.2 → SFT Step 1 探索（并行）→ 0.3（用户夜间 GPU）→ 0.4**。
> 沿用先例：v4→v5 面过渡（[contracts/versions.py](/root/autodl-tmp/drivealign_workspace/DriveAlign/src/drivealign/contracts/versions.py) docstring 配对历史）——v5 当年"改 prompt、自有面、数据集不重建、manifest 派生进 runs/"的整套机制即本计划模板。

## 0. 裁定与不变式

**v6 定义**：模型面输出 schema 由 5 字段缩为 4 字段（critical_objects、risk_factors、yield_required、speed_action），prompt 同步删除 reasoning 相关指令；`MODEL_FACE_LINEAGE["v6"] = "v6"`（v6 自有面，request hash 与 v5 不同）。

**不变式（执行中任何一步不得触碰）**：

| 不变式 | 保证方式 |
|---|---|
| dataset_v4 records 逐字节冻结 | v6 只加不改；Step 0.1 末跑 `dataset_verify` 回归作零影响证明 |
| v3/v4/v5 面历史与 S09 v5 存档有效 | `configs/contracts/v1..v5/` 目录零改动（字节钉子测试 pin 住） |
| 评测指标定义不变 | eval_v1 指标规范不动，仅面引用升级 |
| 锚集不变 | manifest 重建只重算 request hash，锚点集合逐项一致 |
| `reasoning_templates.json` 保留 | 它是记录侧 GT provenance 资产（S08 回填规则），模型面不再消费但资产不可删 |

## 1. Step 0.1：v6 面定义（纯代码 + configs，CPU，先行）

1. **`configs/contracts/v6/`**：v5 八件套逐字复制，仅改两件：
   - `prompt.txt`：字段清单 5→4，删除 reasoning 字段条目及"先推理后输出"类指令行；其余措辞（包络规则、枚举说明、风险条款）逐字节不动；
   - `output_schema.json`：删除 `reasoning` 的 property 定义与 required 项；其余字段约束不变。
2. **`contracts/versions.py`**：`AVAILABLE_CONTRACT_VERSIONS` 追加 `"v6"`；`DEFAULT_CONTRACT_VERSION = "v6"`；`MODEL_FACE_LINEAGE["v6"] = "v6"`；docstring 配对历史增补 v6 条目（删除 reasoning 字段、自有面、request hash ≠ v5、数据集不重建、v5 存档继续有效）。
3. **硬编码排查**：grep `reasoning` 于 `inference/`、`evaluation/`、`records/serializer.py`，确认无 5 字段硬编码假设（schema 校验应全部由 contract 文件驱动）。
4. **单测更新**：①4 字段 schema 校验（缺字段/多字段/类型/枚举）；②`HASH_FACE_FILES` 字节钉子（v1–v5 的 prompt/category_vocab/motion_vocab 逐字节 pin）；③v5 面回归：用 v5 面构建一个合成 request，hash 与存档值一致；④`dataset_verify` 回归：records 绑定自身 contract_version（v4）验证，不受 DEFAULT 切换影响——**这是"数据集零影响"的证明性 gate**。
5. **冻结物（已完成登记，2026-10-05）**：
   - v6 `prompt.txt` sha256 = `242b632b279e87d9cf355fe42803f7381fa983adebe82d07b5597cbbe692ec10`（字段清单 5→4，删 reasoning 条目行，其余逐字节同 v5）；
   - v6 `output_schema.json` sha256 = `22efe8f8738aadb14e654a7357f02892049085ba42ab9e17fbf86cffe205419a`（`$id`/title 升 v6，删 `reasoning` 的 required+properties，description 注明 S11 裁定）；
   - v6 `category_vocab.json` / `motion_vocab.json` 与 v5 逐字节一致（`649470dc…` / `6e8e8338…`，故 hash-face 仅 prompt 字节钉子生效于 request hash）。

## 2. Step 0.2：v6-face anchor manifest + parity（CPU，紧随 0.1）

1. **重建（已完成，2026-10-05）**：以 v6 面运行 [dataset/manifest.py](/root/autodl-tmp/drivealign_workspace/DriveAlign/src/drivealign/dataset/manifest.py)（沿 S09 先例派生，不覆盖 `data/dataset_v4/` 下任何文件）：输出至 `runs/S11_sft_smoke/face_v6/anchor_policy_manifest.json`，**已归位 T2 层 `data/face_manifests/v6/`**（2026-10-05 迁移）。前提：DEFAULT 已切 v6（manifest 按当前面 stamp）。**产物**：`cv=v6, n=30940`（train 22941 / val 2531 / test 5468），sha256 = `2abe8c4feb43067e294f6af743efcc01e72729aacb94f92764d0f349c239acd1`；`manifest_assets/` 以符号链接 staging dataset_v4 shards + dataset_v1 scene manifests（S09 先例同构）。
2. **Parity 检查（已完成，25/25 PASS）**：①逐锚点 `request_hash_1f` 重算 == manifest v6 值；②抽查 25 锚点做 v5→v6 request 文本 diff，**唯一差异 = prompt 段 reasoning 相关行**，user 面图像/速度行零漂移；③image relpath 25/25 解析（存在性 + 可读）。
3. **确定性（已完成，BYTE-IDENTICAL True）**：manifest 重建两次逐字节一致（`determinism_run2/` 与 `face_v6/` sha 均 `2abe8c4f…`；比对后临时目录已删）。

## 3. Step 0.3：1F Base v6 全量重跑（GPU，用户择时夜间执行）

**前置**：0.1/0.2 完成且单测全绿。**GPU 独占**——期间 S11 Step 1/2（CPU 侧）可并行，训练类任务必须等待。

```bash
conda activate autovla_codeclean
# 主评测（eval 全锚集）
python -m drivealign.cli.base_benchmark \
  --anchor-manifest data/face_manifests/v6/anchor_policy_manifest.json \
  --output-dir runs/S09_base_benchmark/eval_v6 | tee runs/S09_base_benchmark/eval_v6/main.log
# 反事实两组（与 v5 电池同构）
python -m drivealign.cli.base_benchmark --anchor-manifest <同上> \
  --image-transform blank   --output-dir runs/S09_base_benchmark/eval_v6/blank   | tee .../blank.log
python -m drivealign.cli.base_benchmark --anchor-manifest <同上> \
  --image-transform shuffled --output-dir runs/S09_base_benchmark/eval_v6/shuffled | tee .../shuffled.log
# 指标汇总（含 visual dependence）
python -m drivealign.cli.evaluate \
  --anchor-manifest <同上> --predictions runs/S09_base_benchmark/eval_v6/predictions.jsonl \
  --counterfactual-blank runs/S09_base_benchmark/eval_v6/blank/predictions.jsonl \
  --counterfactual-shuffled runs/S09_base_benchmark/eval_v6/shuffled/predictions.jsonl
```

（具体 flag 名以 [cli/base_benchmark.py](/root/autodl-tmp/drivealign_workspace/DriveAlign/src/drivealign/cli/base_benchmark.py) / [cli/evaluate.py](/root/autodl-tmp/drivealign_workspace/DriveAlign/src/drivealign/cli/evaluate.py) argparse 为准，执行前核对。）tmux 内执行，`| tee` 前必加 `set -o pipefail`。时长预期 ≤ v5（每帧输出少 reasoning 段 ~40–60 tok）。反事实组确认全跑，保证解耦基线（blank/shuffled flip 率）同步重建。

## 4. Step 0.4：冻结与落档（CPU，收尾）

1. sha 登记：v6 prompt.txt、v6 manifest、新 eval 配置（面引用 v5→v6，指标定义不动）→ S11 决策台账。
2. 新建 `../S09_base_benchmark/S09_v6_face_rerun.md` 附录：v5→v6 主指标对照表 + **双面标注规则**——S09 存档读数为 v5 面，S12 的 Base→SFT 对比基准 **= v6 重跑读数**；此后所有文档数字必须带面标签。
3. `../S09_base_benchmark/S09_reasoning_supervision.md` 加 closure note：路线 2 预注册被 v6 删除裁定取代，指向讨论总结文档。
4. S11 决策台账条目：v6 过渡裁定 + 本 Step 0 执行记录。

## 5. Gate 表（G0.x）

| # | Gate | 判定 |
|---|---|---|
| G0.1 | v6 面单测全绿；v1–v5 字节钉子不动；dataset_verify 回归 PASS（records 零影响证明） | **PASS**（2026-10-05）`pytest DriveAlign/tests/unit -q` → **269 passed**；v1–v5 configs 逐字节未动（git diff 空，仅新增 `configs/contracts/v6/`、改 `versions.py`/`output.py`）；records 零影响由 `test_dataset_v4_records_bind_own_face_unaffected_by_default` 证明——GT 用 record 自身显式 v4 重新 parse，`serialize()` 现盖 v6 仅为 face-stamp 变化而非记录突变 |
| G0.2 | v6 manifest parity：逐锚点 hash 一致 + v5→v6 diff 仅 prompt reasoning 行 + 重建两次逐字节一致 | **PASS**（2026-10-05）manifest sha `2abe8c4f…`（`determinism_run2` 重建字节一致）；parity 抽检 25 锚点：逐锚点 hash == manifest v6 值、v5→v6 request 文本 diff 唯一差异为 reasoning 行、image relpath 25/25 可解析 |
| G0.3 | Base v6 主评测 + blank/shuffled 反事实三跑完成，无 NaN/异常中断，日志与预测文件完整 | PASS/FAIL |
| G0.4 | 指标汇总产出且与 v5 读数量级可比（大幅异常 → 先查面 diff 再下结论）；双面标注落档 | PASS/FAIL |
| G0.5 | 冻结物 sha 齐全登记；决策台账/附录/closure note 落档 | PASS/FAIL |

## 6. 风险与回退

- **回滚**：`DEFAULT_CONTRACT_VERSION` 改回 `"v5"` 一行即回（v6 目录留存不碍事）；v6-face manifest/runs 产物直接删。
- **最大账务风险**：v5/v6 双面并存期数字混淆——G0.4 的双面标注规则是硬性要求，任何引用 S09 读数的新文档必须写明面版本。
- **若 v6 Base 重跑读数大幅异常**：先做 v5→v6 request 全量 diff 定位面变更影响面，再决定是否修订 v6 prompt（走 G0.1 重冻结）；禁止在原因不明下调参解释。

## 7. 交接

Step 0.2 完成即满足 S11 Step 1（sft 包）开工条件；Step 0.4 完成即满足 S11 Step 5 关闭条件（对比基准就位）。S12/S13 直接继承 v6 面：4F 面出生即 4 字段，无 reasoning 历史包袱。
