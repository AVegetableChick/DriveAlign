# S01/S02 Model Smoke Test

## 1. 实验概览

- 实验阶段：Stage 01 Base 模型加载、Stage 02 通用单图推理
- 实验日期：2026-09-13
- 运行环境：`autovla_codeclean`
- GPU：NVIDIA GeForce RTX 4080 SUPER
- 模型：Qwen2.5-VL-3B-Instruct
- 本地模型目录：`models/Qwen2.5-VL-3B-Instruct`
- 结果产物目录：`artifacts/model/qwen25_vl_3b/`

## 2. Git 与版本基线

### DriveAlign

- Repository HEAD：`e9755305e69b6f4f449dd3aeae3759b9f1e4e7b4`
- 说明：上述 HEAD 是本次记录的 DriveAlign 代码版本基线；实验结果文件和当前未提交文件不应被解释为已包含在该 commit 中。

### 上游仓库

- AutoVLA：`ba34eed74ce6729e7986592d0e66cbaca397b4fa`
- nuScenes-devkit：`b40adc467b919192899405d9b77871afee8efa07`
- TRL：`670458a5b804ad8beb85bfb1c2f2658271f8717d`

## 3. 模型加载设置

配置文件：`configs/model/qwen25_vl_3b.yaml`

```yaml
model:
  path: /root/autodl-tmp/drivealign_workspace/models/Qwen2.5-VL-3B-Instruct
  revision: null
  dtype: bfloat16
  device_map: auto
  attn_implementation: flash_attention_2
  processor:
    use_fast: true

runtime:
  local_files_only: true
```

关键说明：

- `revision: null`：使用本地 checkpoint 目录，不指定 Hugging Face 分支、tag 或 commit。
- `dtype: bfloat16`：以 BF16 加载模型。
- `device_map: auto`：由 Transformers/Accelerate 自动放置模型。
- `attn_implementation: flash_attention_2`：固定使用 FlashAttention 2。
- `local_files_only: true`：实验不从网络读取模型文件。
- Processor 类型：`Qwen2_5_VLProcessor`。
- 模型类型：`Qwen2_5_VLForConditionalGeneration`。

## 4. Stage 01 结果：Base 模型加载

结果文件：`artifacts/model/qwen25_vl_3b/smoke_test.json`

| 指标 | 结果 |
|---|---:|
| 模型类型 | `Qwen2_5_VLForConditionalGeneration` |
| Processor 类型 | `Qwen2_5_VLProcessor` |
| dtype | `torch.bfloat16` |
| 设备 | `cuda:0` |
| CUDA | `true` |
| 加载耗时 | `11.167 s` |
| 峰值显存 | `7,630,138,880 bytes`，约 `7.10 GiB` |

结论：模型与 processor 成功加载到 GPU，未观察到意外 CPU fallback；Stage 01 的加载 Gate 通过，可作为 Stage 02 的模型输入。

## 5. Stage 02 结果：通用单图推理

### 5.1 输入与生成设置

- 图像：`data/nuscenes/mini/samples/CAM_FRONT/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151603512404.jpg`
- Prompt：`请描述图像中的道路场景。`
- `do_sample`：`false`
- `max_new_tokens`：`128`
- 运行次数：`3`
- 输入 token：`1852`
- 输出 token：`23`

### 5.2 FlashAttention 2 结果

结果文件：`artifacts/model/qwen25_vl_3b/infer_smoke_test/infer_smoke_test_flashattn.json`

三次输出完全一致，`greedy_outputs_identical=true`：

> 图像展示了一条城市街道的景象。街道上车辆稀少，两侧是高楼大厦和商店。

| Run | 预处理耗时 | 生成耗时 | 峰值显存 |
|---:|---:|---:|---:|
| 1 | `0.1737 s` | `4.5202 s` | `8,343,919,104 bytes` |
| 2 | `0.1616 s` | `1.3381 s` | `8,343,919,104 bytes` |
| 3 | `0.1517 s` | `1.2996 s` | `8,343,919,104 bytes` |

- 平均预处理耗时：约 `0.162 s`
- 平均生成耗时：约 `2.386 s`
- 第 2、3 次平均生成耗时：约 `1.3195 s`
- 峰值显存：约 `7.77 GiB`
- 三次 greedy 输出逐字一致：通过

说明：第 1 次生成包含模型运行预热开销，因此使用第 2、3 次结果观察稳定生成耗时更合适。

### 5.3 SDPA 对照结果

结果文件：`artifacts/model/qwen25_vl_3b/infer_smoke_test/infer_smoke_test_sdpa.json`

SDPA 对照实验同样三次输出完全一致，输出文本与 FlashAttention 2 相同。

| Run | 预处理耗时 | 生成耗时 | 峰值显存 |
|---:|---:|---:|---:|
| 1 | `0.1402 s` | `2.6260 s` | `11,424,650,240 bytes` |
| 2 | `0.1463 s` | `1.8165 s` | `11,424,650,752 bytes` |
| 3 | `0.1781 s` | `1.7907 s` | `11,424,650,752 bytes` |

- 平均生成耗时：约 `2.078 s`
- 峰值显存：约 `10.64 GiB`
- 三次 greedy 输出逐字一致：通过

## 6. 结论与交接

- Stage 01：本地 Qwen2.5-VL-3B-Instruct 在 `cuda:0` 上以 BF16 成功加载。
- Stage 02：图片已进入 Qwen VL processor，单图生成链路正常。
- Stage 02：固定 `do_sample=false` 后，三次输出逐字一致，满足可复现性要求。
- FlashAttention 2 与 SDPA 均通过一致性检查；FlashAttention 2 峰值显存低于 SDPA，因此当前默认配置固定为 `flash_attention_2`。
- 输出仍是通用视觉描述，不代表驾驶能力评测，也未进入 Stage 03 的结构化驾驶 JSON 解析。
- 当前结果可交接至 Stage 03。

## 7. 复现实验命令

激活目标环境后执行：

```bash
conda activate autovla_codeclean

PYTHONPATH=DriveAlign/src python -m \
drivealign.cli.smoke_model \
--config DriveAlign/configs/model/qwen25_vl_3b.yaml

PYTHONPATH=DriveAlign/src python -m \
drivealign.cli.infer_image \
--image data/nuscenes/mini/samples/CAM_FRONT/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151603512404.jpg \
--prompt "请描述图像中的道路场景。"
```
