# S04 nuScenes 连续帧 Mini Smoke 实验记录

## 1. 实验概览

- 实验阶段：Stage 04 nuScenes-devkit 连续帧 Mini Smoke
- 实验日期：2026-09-22
- 运行环境：`autovla_codeclean`（Python 3.9.23，numpy 1.23.4，nuscenes-devkit 1.2.0，Pillow 导入实测 10.3.0）
- GPU：本阶段为纯 CPU 数据管线，无 GPU 参与
- 数据：nuScenes `v1.0-mini`（`data/nuscenes/mini`，含 `v1.0-mini/`、`samples/`、`sweeps/`、`maps/`；CAM_FRONT 1600×900）
- 代码产物：`src/drivealign/data/nuscenes_io.py`、`src/drivealign/evaluation/{projection,box_render}.py`、`src/drivealign/cli/nuscenes_mini_smoke.py`（另含 `data/`、`evaluation/` 两个 `__init__.py`）
- 运行产物：`runs/nuscenes_mini_smoke/`（`four_frame_grid.jpg`、4 张逐帧叠框图、`timeline.json`、`projection_report.json`、`summary.json`）

## 2. Git 与版本基线

### DriveAlign

- Repository HEAD：`00f3e20329680801787d85a5aa0ed66d55016b89`
- 说明：Stage 04 全部新增代码（`cli/nuscenes_mini_smoke.py`、`data/`、`evaluation/`）在记录时均为 untracked，不包含在该 commit 中。另注意 `docs/stages_plan/` 整体被 `.gitignore`（第 195 行）忽略，阶段文档仅本地留存。

### 上游仓库

- nuScenes-devkit：`b40adc467b919192899405d9b77871afee8efa07`（`artifacts/environment/nuscenes-devkit-sha.txt` 审计）
- AutoVLA / TRL：本阶段未触碰，沿用 S03 记录的 SHA

## 3. 目标与边界

按 `docs/stages_plan/04_nuscenes_mini_smoke.md`：用官方 nuScenes-devkit 验证 `v1.0-mini` 的 CAM_FRONT 历史链、时间戳、图片路径和当前帧 3D box 投影。本阶段不生成 DriveAlignRecord、不划分正式数据、不派生风险标签。经用户确认增补：本阶段必须输出**连续四帧可视化图像及 GT 框投影**，且**四帧各画各自的 keyframe GT**（四帧均为 keyframe，各有标注），以同时验证历史帧链与逐帧投影正确性。完成标准：至少一个 anchor 得到同 scene、严格递增的四帧窗口，当前帧至少一个可见 GT box 合法投影，全 Gate 通过且重跑可复现。

## 4. 输入与产物

- 输入 1：固定 SHA 的 nuScenes-devkit checkout（`third_party/nuscenes-devkit`），以非 editable 方式安装
- 输入 2：`v1.0-mini` 数据根（默认 anchor 由 `scene.json` 表序确定）
- 产物 1：`data/nuscenes_io.py` —— `load_nuscenes / resolve_anchor / read_keyframe_chain`，严格沿 `sample.prev`（禁用高频 `sample_data.prev`），`FrameRecord/ChainResult` frozen dataclass，六类 reason code（`INSUFFICIENT_HISTORY / CROSS_SCENE / NON_INCREASING_TIME / MISSING_IMAGE / BAD_GAP / NO_CAM_FRONT`）
- 产物 2：`evaluation/projection.py` —— `camera_boxes_for_frame`（薄封装 `get_sample_data`，box 已在 sensor frame，不二次变换）、`project_box`（`view_points` + sensor 系 z 作真实深度）、`classify_visibility`（`visible/clipped/behind/out`，复刻 `box_in_image` 语义 + 0.5 margin）
- 产物 3：`evaluation/box_render.py` —— 纯 PIL 绘制（无 matplotlib、无系统字体、无墙钟内容，保证确定性）：12 棱线框 + 朝向线（0→5），固定 10 类配色 + md5 兜底，2×2 网格合成
- 产物 4：`cli/nuscenes_mini_smoke.py` —— 入口 CLI，链校验 + 逐帧投影 + 报告 + 可视化，`summary.json` 携带 gate 布尔与 `report_sha256`
- 产物 5：`docs/stages_plan/04_nuscenes_mini_smoke.md` 产物清单与代码路径同步更新（该目录不入 Git）

## 5. 执行流程

1. 环境安装（用户执行 editable 安装报错后由本记录修复，见 §6.1）；
2. 实现 `data/nuscenes_io.py` 四帧链只读访问（含 reason code 分类）；
3. 实现 `evaluation/projection.py` 投影与可见性分类、`evaluation/box_render.py` PIL 渲染；
4. 实现 CLI 并首跑（默认 anchor），检查 `summary.json` 全 Gate；
5. 目检网格图投影合理性；
6. 重跑至 `runs/nuscenes_mini_smoke_rerun/`，diff 两份 JSON 做确定性 Gate；
7. 更新 Stage 04 阶段文档。

## 6. 验证与 Gate

### 6.1 证据链：devkit 安装修复

- **现象**：`pip install -e third_party/nuscenes-devkit/setup/ --no-deps` 在 legacy `setup.py develop` 阶段失败，子进程报 "third_party/nuscenes-devkit does not appear to be a Python project"。
- **根因**：`setup/setup.py` 第 33 行 `os.chdir('..')` 先于 `setuptools.setup()` 切到仓库根；legacy develop 命令随后在当前目录（已是仓库根）子进程执行 `pip install -e . --use-pep517 --no-deps`，而仓库根无 setup.py/pyproject.toml。
- **修复**：改非 editable 安装 `pip install --no-deps <abs>/third_party/nuscenes-devkit/setup/`，构建 wheel 成功装入 site-packages。代码由固定 SHA checkout 审计，拷贝安装不影响可追溯性。
- **验证**：`import nuscenes` 指向 `site-packages/nuscenes/`，numpy 保持 1.23.4 未被动过（`--no-deps` 防御生效）。

### 6.2 证据链：首跑结果（默认 anchor，scene-0061）

`runs/nuscenes_mini_smoke/summary.json`：anchor sample `e0845f5322254dafadbbed75aaa07969`，scene_token `cc8c0bf57f984915a77078b10eb33198`。

| offset_tag | offset_s | sample_data_token | 标注数 | 绘制数（≤50m） |
|---|---|---|---|---|
| t-1549ms | -1.549402 | `e3d495d4ac534d54b321f50006683844` | 69 | 36 |
| t-1050ms | -1.049506 | `4b6870ae200c4b969b91c50a9737f712` | 78 | 40 |
| t-0499ms | -0.499305 | `d0d9ef23e3934ea09d55afdc24db9827` | 86 | 38 |
| t0 | 0.0 | `74e7a9260c5d45b78b831528b62daf41` | 94 | 39 |

`report_sha256 = 8d0d9f6e28d40b727c55b76b5bbc74a946ea0a84f38081fa98950ef4920d9a6a`（timeline + projection_report 拼接哈希）。

### 6.3 证据链关键事实

1. **mini 的真实帧间隔非精确 0.5 s**：实测相邻间隔 0.4999/0.5002/0.4993 s，offsets 为 [-1.549, -1.050, -0.499, 0.0]。阶段文档"目标约 [-1.5,-1.0,-0.5,0.0]"成立，但 Stage 06/07 的 `time_offset_s` 必须记录真实值而非名义值。
2. **`view_points(normalize=True)` 第三行恒为 1**：真实逐角点深度必须取 sensor 系 z 坐标（实现中已修正并注释），否则可见性判定全部失效。
3. **`get_sample_data` 返回的 box 已在 sensor frame**：直接 `view_points(corners, K, normalize=True)` 即可，二次变换必错（实现前已核实 devkit 源码 L251-311）。
4. **behind/out 框的处置**：任一角点 `z ≤ 0.1m` 判 `behind`，坐标不参与绘制标签；非有限坐标的框不绘制且 `corners_2d` 记 null；`finite_values` gate 只约束 drawn 框。
5. **目检结论**：2×2 网格图中 GT 框贴合前景车辆/路障，近处大货车 3D 框因超出视场上溢属正常透视投影；朝向线、类别+距离标签可读。
6. **确定性成立**：重跑 `runs/nuscenes_mini_smoke_rerun/` 后 `timeline.json` 与 `projection_report.json` diff 为空（逐字节一致）；JPEG 不参与哈希比对（编码器版本风险，JSON 才是 gate 事实源）。

### 6.4 Gate 逐条核对（`summary.json.gates`）

| Gate | 结果 |
|---|---|
| chain_ok（链完整、图片存在、间隔正常） | true |
| same_scene（四帧同 scene，链内强制） | true |
| strictly_increasing（offsets 严格递增） | true |
| tokens_unique（4 个 sample_data token 无重复） | true |
| finite_values（drawn 框坐标全有限） | true |
| at_least_one_visible_gt（anchor 帧 drawn ≥ 1，实测 39） | true |
| all_pass | true |

失败路径覆盖：六类 reason code 在 `read_keyframe_chain` 内分类返回；链失败时 CLI 写入 `chain_ok=false + reason_code` 的 summary 并以退出码 1 终止。

### 6.5 Gate 判定

通过。四帧 CAM_FRONT keyframe 链读取、时间戳校验、GT 投影与可视化全部符合阶段文档完成标准，且同配置重跑逐字节可复现。

## 7. 交接

- 冻结：`read_keyframe_chain` 的 FrameRecord 字段（`sample_token / sample_data_token / timestamp_us / offset_s / image_relpath / image_size`）与六类 reason code —— Stage 06 `build_record` 的 `model_inputs.frames` 直接复用该读路径与字段。
- Stage 05 将在同一 devkit 安装上验证 AutoVLA 原生四帧组织方式，并与本阶段的自建链读取对照。
- 2D AABB 派生规则已明确（8 角点投影后取 min/max 并 clip 到图像界；含 behind 角点的框弃用/标 unscorable），实现留给 Stage 08 M09 oracle；`projection_report.json` 可低成本增加 `bbox_2d` 字段（已提出，未实施）。
- 上游安装约定（新增教训）： pinned checkout 一律**非 editable 安装**（`pip install --no-deps <pkg>/setup/`），可追溯性由 SHA 文件承担；legacy editable 与带 `chdir` 的 setup.py 不兼容。

## 8. 复现实验命令

```bash
conda activate autovla_codeclean
cd /root/autodl-tmp/drivealign_workspace

# 首跑（默认 anchor = scene-0061 首个有 3 个 prev 的 sample）
PYTHONPATH=DriveAlign/src python -m drivealign.cli.nuscenes_mini_smoke

# 确定性复现 + diff（两份 JSON 应逐字节一致）
PYTHONPATH=DriveAlign/src python -m drivealign.cli.nuscenes_mini_smoke --out runs/nuscenes_mini_smoke_rerun
diff runs/nuscenes_mini_smoke/timeline.json runs/nuscenes_mini_smoke_rerun/timeline.json
diff runs/nuscenes_mini_smoke/projection_report.json runs/nuscenes_mini_smoke_rerun/projection_report.json

# 可选：指定 anchor
#   --scene-name scene-0061 或 --sample-token <token>
# 可选：绘制半径
#   --max-distance-m 50
```
