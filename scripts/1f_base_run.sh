#!/usr/bin/env bash
# ============================================================================
# S12 Step 6（batched 版）：1F SFT-1F（v6-face）test 全量评测
#   —— 主评测 + 反事实两组 + 指标汇总，走 batched_v1 批量推理 path（bs=16）
#   DriveAlign/scripts/1f_base_run.sh
#
# 变更记录：
#   2026-10-07  由 bs=1 冻结面电池切换为 batched_v1 批量 path（bs=16）。
#               速度依据：docs/stages_record/S12_single_frame_sft/
#               S12_batched_inference_speed.md（bs=16 实测 3.48×，
#               GPU 利用率已近饱和，bs=32 收益边际）。
#               parity gate 延后（2026-10-07 拍板）：主读数需标注
#               "待 parity 复核"；bs=1 参照面 eval_v6/（164 行 partial）
#               原样保留，不与本脚本产物混写。
#               原 bs=1 电池版本：把 CONFIG 换回 configs/benchmark/sft_1f_v6.yaml、
#               CLI 换回 drivealign.cli.base_benchmark、输出名换回
#               predictions*.jsonl 即可（git 历史亦有原版）。
#   2026-10-07b 批大小改由批量配置承载（sft_1f_v6_batched.yaml 内
#               batch_size: 16），本脚本不再用 --batch-size CLI 覆盖。
#
# 前置（本脚本不含，由调用者自行完成）：
#   1) conda activate autovla_codeclean
#   2) cd /root/autodl-tmp/drivealign_workspace   （脚本内全是相对路径，cwd 必须是工作区根）
#
# 建议启动方式（tmux 内，日志另存一份）：
#   tmux new-session -d -s s12_step06_batched \
#     'conda activate autovla_codeclean && \
#      cd /root/autodl-tmp/drivealign_workspace && \
#      bash DriveAlign/scripts/1f_base_run.sh 2>&1 | \
#      tee runs/S12_sft_1f/eval_v6_batched/step06_batched.log'
#
# 执行内容（严格串行，GPU 独占；期间勿在别处起训练）：
#   [0] 预检：打印 sft_1f_v6_batched 配置与 v6 manifest 的 sha256（登记用，秒级）
#   [1] 主评测：test split 全量 5,468 anchors（v6 面，挂 SFT-1F 末点 adapter，
#       batch_benchmark，bs=16 由批量配置承载）
#   [2] 反事实 blank   ：200 锚子集（与 Base v6 电池同构，同一 anchors-file）
#   [3] 反事实 shuffled：200 锚子集
#   [4] 指标汇总：eval_v1（含 visual dependence），纯 CPU
#
# 关键约定：
#   - 三步推理均用 drivealign.cli.batch_benchmark（冻结面 base_benchmark 一行不改）；
#     批大小由批量配置承载（batch_size: 16），CLI 不覆盖；
#   - 输出文件名带 _bs16 后缀：batch CLI 对 predictions.jsonl / predictions_pilot.jsonl
#     有名字硬隔离（防误写冻结面产物），本脚本的命名必须避开；
#   - 三步推理均带 --resume：幂等，中断后重跑本脚本即续跑（按输出文件内已有 token 跳过）；
#   - set -e -o pipefail + 换行顺序执行：任一步失败立即停止（face hash 对不上会立刻抛错，
#     不会白跑后续步骤）；
#   - 主预测与反事实预测分文件，互不污染；反事实的变换图落到
#     $OUT/counterfactual_images/{blank,shuffled}/（scratch dir = 输出文件父目录）。
#
# 产物（均在 $OUT = runs/S12_sft_1f/eval_v6_batched/ 下）：
#   predictions_bs16.jsonl / predictions_cf_blank_bs16.jsonl / predictions_cf_shuffled_bs16.jsonl
#   main_bs16.log / cf_blank_bs16.log / cf_shuffled_bs16.log / evaluate_bs16.log
#   run_summary_predictions_bs16.json 等
#   eval_report.json / eval_report.md / anchor_scores.jsonl
# ============================================================================
set -euo pipefail

# --- 路径常量（全部相对工作区根；git sha 由运行记录登记） ------------------------
CONFIG=DriveAlign/configs/benchmark/sft_1f_v6_batched.yaml
# SFT-1F 末点 adapter 目录（1 epoch 训完的根目录 adapter_model.safetensors；
# 批量配置内 adapter_path 亦指向同目录，--adapter 运行时显式覆盖，双保险）
ADAPTER=runs/S12_sft_1f/checkpoints/sft_1f
MANIFEST=data/face_manifests/v6/anchor_policy_manifest.json
DATASET_ROOT=data/dataset_v4
NUSC_DATAROOT=data/nuscenes/trainval
CF_SUBSET=runs/S09_base_benchmark/counterfactual_subset.txt
EVAL_CONFIG=DriveAlign/configs/evaluation/eval_v1.yaml
RUNNER=drivealign.cli.batch_benchmark
OUT=runs/S12_sft_1f/eval_v6_batched

# --- 前置守卫：确认 cwd 与关键输入就位（避免相对路径静默写错地方） ---------------
[ -f "$CONFIG" ] || { echo "FATAL: 未找到 $CONFIG —— 请在 /root/autodl-tmp/drivealign_workspace 下运行"; exit 1; }
[ -f "$MANIFEST" ] || { echo "FATAL: 未找到 $MANIFEST"; exit 1; }
[ -f "$CF_SUBSET" ] || { echo "FATAL: 未找到反事实子集 $CF_SUBSET"; exit 1; }
[ -f "$ADAPTER/adapter_model.safetensors" ] || { echo "FATAL: 未找到 SFT adapter $ADAPTER/adapter_model.safetensors"; exit 1; }
export PYTHONPATH=DriveAlign/src
mkdir -p "$OUT"

echo "=================================================================="
echo "[0/4] 预检 sha256（登记用）"
sha256sum "$CONFIG" "$MANIFEST" "$ADAPTER/adapter_model.safetensors"

echo "=================================================================="
echo "[1/4] 主评测：test 全量 5,468 anchors（v6 面，SFT-1F，bs=16 配置承载）"
python -m "$RUNNER" \
  --config "$CONFIG" \
  --anchor-manifest "$MANIFEST" \
  --dataset-root "$DATASET_ROOT" \
  --nuscenes-dataroot "$NUSC_DATAROOT" \
  --adapter "$ADAPTER" \
  --resume \
  --out "$OUT/predictions_bs16.jsonl" 2>&1 | tee "$OUT/main_bs16.log"

echo "=================================================================="
echo "[2/4] 反事实 blank：200 锚子集（与 Base v6 电池同构，bs=16 配置承载）"
python -m "$RUNNER" \
  --config "$CONFIG" \
  --anchor-manifest "$MANIFEST" \
  --dataset-root "$DATASET_ROOT" \
  --nuscenes-dataroot "$NUSC_DATAROOT" \
  --adapter "$ADAPTER" \
  --anchors-file "$CF_SUBSET" \
  --image-transform blank \
  --resume \
  --out "$OUT/predictions_cf_blank_bs16.jsonl" 2>&1 | tee "$OUT/cf_blank_bs16.log"

echo "=================================================================="
echo "[3/4] 反事实 shuffled：200 锚子集（bs=16 配置承载）"
python -m "$RUNNER" \
  --config "$CONFIG" \
  --anchor-manifest "$MANIFEST" \
  --dataset-root "$DATASET_ROOT" \
  --nuscenes-dataroot "$NUSC_DATAROOT" \
  --adapter "$ADAPTER" \
  --anchors-file "$CF_SUBSET" \
  --image-transform shuffled \
  --resume \
  --out "$OUT/predictions_cf_shuffled_bs16.jsonl" 2>&1 | tee "$OUT/cf_shuffled_bs16.log"

echo "=================================================================="
echo "[4/4] 指标汇总：eval_v1（含 visual dependence，CPU）"
python -m drivealign.cli.evaluate \
  --predictions "$OUT/predictions_bs16.jsonl" \
  --anchor-manifest "$MANIFEST" \
  --dataset-root "$DATASET_ROOT" \
  --config "$EVAL_CONFIG" \
  --out "$OUT" \
  --counterfactual-blank "$OUT/predictions_cf_blank_bs16.jsonl" \
  --counterfactual-shuffled "$OUT/predictions_cf_shuffled_bs16.jsonl" 2>&1 | tee "$OUT/evaluate_bs16.log"

echo "=================================================================="
echo "S12 Step 6（batched, bs=16 配置承载）ALL DONE"
echo "产物目录：$OUT（batched_v1 path，与冻结面 eval_v6/ 分离）"
echo "gates 核对：主跑 rows=5468 / 反事实各 200 行 / eval_report.json 已生成 /"
echo "           eval_report 内 anchor_manifest_sha256 应为 2abe8c4f…（v6 manifest）"
echo "注意：parity gate 延后未做，本电池读数按约定标注『待 parity 复核』"
