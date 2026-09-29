# S05：AutoVLA nuScenes Preprocessing Mini Smoke 记录

- 日期：2026-09-29
- 脚本：`third_party/AutoVLA/tools/preprocessing/nusc_sample_generation.py`（pinned checkout，未修改，mtime 2026-09-08）
- 命令：`python tools/preprocessing/nusc_sample_generation.py --nuscenes_path <root>/data/nuscenes/mini --output_dir artifacts/s05_autovla_prep_smoke/train --split train --version v1.0-mini`
- 输出目录：`artifacts/s05_autovla_prep_smoke/train/`（不进 git）

## 1. 单样本数据格式

每个 sample 输出一个 JSON 文件，文件名为 `{sample_token}.json`，与文件内 `token` 字段一致。字段集合（raw 路径，全部 164 个样本一致）：

| 字段 | 类型 | 含义 |
|---|---|---|
| `token` | str | nuScenes sample token |
| `dataset_name` | str | 固定 `"nuscenes"` |
| `front_camera_paths` 等 6 组 | list[str]×4 | 6 相机（front / front_left / front_right / back / back_left / back_right）各 4 个**绝对路径**，顺序严格为 **[t-1.5s, t-1.0s, t-0.5s, t]**（`insert(0)` 实现，最新帧在末尾） |
| `gt_trajectory` | float[N][3] | 未来轨迹，N=10，间隔 0.5s（5s 时域）；每点 (x, y, heading)。坐标系：LIDAR_TOP lcf → x-forward/y-left 转换（`x'=y, y'=-x`），heading 由相邻点差分计算 |
| `cot_output` | list | raw 路径恒为 `[]`；传入 DriveLM 标注时为 5 段文本（fov / 关键对象 / 运动意图 / 意图推理 / 规划动作） |
| `instruction` | str | `Go Straight` / `Turn Left` / `Turn Right`，由**未来轨迹终点横向偏移**反推（阈值 ±2m）——**future-derived，Reference only** |
| `velocity` | float | 自车速度标量，m/s，由 t 与 t-0.5s 的全局 pose 差分计算 |
| `acceleration` | float | 自车加速度标量，m/s²，`(v_t - v_{t-0.5}) / 0.5`，仅用历史 pose |

注：val split 额外多一个 `future_mask` 字段（未来帧有效性），本 smoke 只跑了 train split。

## 2. 数据条目数

- 遍历 `nusc.sample` 共 **404** 条（v1.0-mini 全量）。
- 产出 **164** 个 JSON。
- 跳过计数器累计 **78**（不含 scene 过滤的静默跳过 162 条 val 场景样本）。
- 产出样本分布：6 个 scene（mini 10 场景中与官方 `splits.train` 的交集为 6 个，非 8 个）；每个 scene 内为位置 4 起、至"剩余 future ≥10"为止。
- `instruction` 分布：Go Straight 112 / Turn Left 50 / Turn Right 2。
- 数量对账公式：242（train 场景样本）− 6（pos-1）− 6（pos-2）− 6（pos-3）− 60（future<10）= **164**，分毫不差。

## 3. Sample 跳过逻辑（按代码执行顺序）

1. **场景过滤**（静默，不计数）：`sample['scene_token']` 不在当前 split 的 scene token 集合 → `continue`。mini 下 4 个 val 场景共 162 条被静默跳过。
2. **无前一帧**（计数）：`sample['prev'] == ''` → 跳过。每个场景的位置 1 样本，共 6 条。
3. **历史帧掩码不足**（计数）：`sum(ego_his_masks[:-1]) < 3` → 跳过。注意掩码切片 `[:-1]` 只统计**前序帧槽位**：
   - pos-2（只有 1 个真实前序帧）：`[0,0,1]`，sum=1 → 跳过（6 条）
   - pos-3（只有 2 个真实前序帧）：`[0,1,1]`，sum=2 → 跳过（6 条）
   - 即**要求当前帧严格存在 3 个真实前序 keyframe（位置 ≥4）**，这保证了 loader 侧 4 帧窗口总是完整。
4. **未来帧掩码不足**（计数，仅 train split）：`sum(ego_fut_masks[1:]) < 10` → 跳过。要求 ≥10 个真实未来 keyframe，共 60 条。
5. **相机路径遍历中断**（计数）：历史帧链提前耗尽导致 4 帧相机路径不完整 → 跳过（mini 中未发生）。

### 更正说明

此前会话曾判断"pos-2 样本会触发 `get_ego_velocity` 的 `prev token is empty` 断言导致脚本崩溃"。**该结论有误**：主循环中历史掩码检查（第 3 条）先于 velocity 计算执行，历史不足的样本在到达断言前已被正常跳过，断言仅在绕过主循环直接调用函数时才会触发。脚本原样运行**正常完成**（exit 0），不存在崩溃 bug。

## 验证证据

- 内容契约：164/164 通过（4 帧路径与 devkit 重建序列逐一相符、字段齐全、`gt_trajectory` shape=(10,3)、数值全部有限）。
- 确定性：独立重跑（`/tmp/s05_probe_run`）与人工运行输出 **164/164 逐字节 md5 一致**。

Loader/Collator 检查另行成文：执行记录见 `S05_loader_smoke_record.md`，batch 格式见 `references/autovla_loader_batch_format.md`，字段映射见 `references/autovla_preprocessing.md`。
