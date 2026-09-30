# S06 冒烟记录：Record → 1F/4F serializer → processor → batch

> 状态：**Stage 06 全部 gate 通过**（2026-09-30）。
> 代码：`src/drivealign/records/{record,adapter,serializer}.py`、`src/drivealign/cli/record_build_smoke.py`、`tests/unit/test_records_*.py`。
> 分文档：格式见 `S06_record_v1_format.md`，adapter 见 `S06_adapter.md`，serializer 见 `S06_serializer.md`。

## 1. 运行环境与输入

- 环境：`autovla_codeclean`（Python 3.9.23），PYTHONPATH=DriveAlign/src，工作区根执行
- 数据：nuScenes `v1.0-mini`（`data/nuscenes/mini`），train split = devkit `splits.mini_train` 8 scenes（与 AutoVLA preprocessing 同源）
- 窗口口径（用户确认）：**严格 disjoint 32 窗口**——每 scene 锚点 index 从 3 起步长 10（4F 历史 + 6F 未来不重叠）
- provenance.builder_commit：`git -C DriveAlign rev-parse HEAD` 自动获取

## 2. 代码产物

| 产物 | 内容 |
|---|---|
| `records/record.py` | `DriveAlignRecord v1` frozen dataclass、7 顶层键白名单、`from_dict` 结构校验、`validate_record` 语义校验、`canonical_hash()`（构造期浮点定精度） |
| `records/adapter.py` | `build_record`（唯一 Record 产出点）：复用 `read_keyframe_chain`、backward-difference ego speed、沿 `sample.next` 6 帧未来、8 类 quarantine 分类（6 链路码 + `MISSING_CALIBRATION` + `NUMERIC_ANOMALY`） |
| `records/serializer.py` | `InputPolicy(1F/4F)`、`serialize`（类型上只收 `ModelInputs`）、冻结 v3 prompt + 单行因果速度、`ProcessorInputs`/`collate` 最小契约层、`future_fingerprints`/`scan_future_fingerprints` 真实未来指纹扫描（无合成 canary） |
| `cli/record_build_smoke.py` | 冒烟入口：disjoint 窗口枚举 → 逐窗口 Record/1F/4F/指纹扫描/双构建 → 五项 gate → `summary.json` + `windows.json`（含 report_sha256） |
| `tests/unit/test_records_record.py` 等 3 个 | 91 passed / 0 skipped（含 Stage 03 既有单测；mini 缺失时数据用例自动 skip） |

## 3. 执行流程（启动命令）

```bash
# 单测
conda activate autovla_codeclean && PYTHONPATH=DriveAlign/src python -m pytest DriveAlign/tests/unit -q
# 首跑
PYTHONPATH=DriveAlign/src python -m drivealign.cli.record_build_smoke
# 确定性重跑 + diff
PYTHONPATH=DriveAlign/src python -m drivealign.cli.record_build_smoke --out runs/record_build_smoke_rerun
diff runs/record_build_smoke/windows.json runs/record_build_smoke_rerun/windows.json
diff runs/record_build_smoke/summary.json runs/record_build_smoke_rerun/summary.json
```

## 4. Gate 结果（全部通过）

| Gate | 判定内容 | 结果 |
|---|---|---|
| frames_contract | valid 窗口 `validate_record` 为空且 future=6（四帧同 scene/无重复/严格递增/anchor 对齐由 validate 覆盖） | PASS |
| policy_parity | 1F/4F 共享 anchor frame_token、当前帧图像、prompt、speed；training_targets 双 None | PASS |
| leak_free | 未来 token/时间戳指纹在 prompt_1f/prompt_4f/processor/batch 命中 = 0 | PASS |
| full_coverage | 候选数 = 期望值，且全部归为 valid 或带 reason_code 的 quarantine | PASS |
| determinism | 同进程双构建 record/request hash 一致；跨进程 rerun 两份 JSON diff 为空（含 report_sha256） | PASS |

运行结果：`windows=31 valid=31 quarantine=0 reasons={}`，两轮 PASS。

## 5. 勘误：32 → 31

阶段文档"约 32 个完整窗口"基于每 scene 40 keyframe 假设；实测 mini-train 各 scene keyframe 数为 39–41：

- scene-0061 = 39 帧 → disjoint 窗口只能放 3 个（需 anchor index ≤ 39−7 = 32，而第 4 窗锚点 33 越界）
- 其余 7 个 scene 各 4 窗 → **严格 disjoint 总数 = 7×4 + 1×3 = 31**

数据中不存在 32 个 disjoint 窗口的切法，gate 期望值按现实修正为 31（"约 32"口径内的勘误）；全量 prevs≥3 锚点实为 374（10 scenes，含 val），已在 adapter 内联验证中归类通过，供后续阶段扩量参考。

## 6. 交接（Stage 07 输入）

- 冻结接口：`DriveAlignRecord v1`（不可加字段）、`build_record(nusc, token, builder_commit)`、`serialize(ModelInputs, InputPolicy)`、8 类 quarantine taxonomy、指纹扫描函数
- `training_targets` 两字段 S06 恒为 None：S07 join DriveLM 填 `language_reference`，S08 填 `expected_output`；届时监督答案须与 1F/4F 同 anchor 对齐
- 冒烟产物 `runs/record_build_smoke/windows.json` 含 31 个完整 record dict（含 `oracle_only`），可作为 S07 扩量前的本地调试样本（runs/ 不入 Git）
- Tensor 级 processor/collator（像素块化）留待 S07 训练集成，本阶段契约层形态已冻结
