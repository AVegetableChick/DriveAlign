#!/usr/bin/env bash
# ============================================================================
# S11 Step 0.3：1F Base v6-face 全量重跑（主评测 + 反事实两组 + 指标汇总）
#   DriveAlign/scripts/s11_step03_v6_base_rerun.sh
#
# 前置（本脚本不含，由调用者自行完成）：
#   1) conda activate autovla_codeclean
#   2) cd /root/autodl-tmp/drivealign_workspace   （脚本内全是相对路径，cwd 必须是工作区根）
#
# 建议启动方式（tmux 内，日志另存一份）：
#   tmux new-session -d -s s11_step03 \
#     'conda activate autovla_codeclean && \
#      cd /root/autodl-tmp/drivealign_workspace && \
#      bash DriveAlign/scripts/s11_step03_v6_base_rerun.sh 2>&1 | \
#      tee runs/S09_base_benchmark/eval_v6/step03.log'
#
# 执行内容（严格串行，GPU 独占；期间勿在别处起训练）：
#   [0] 预检：打印 v6 配置与 v6 manifest 的 sha256（Step 0.4 登记用，秒级）
#   [1] 主评测：test split 全量 5,468 anchors（v6 面）
#   [2] 反事实 blank   ：200 锚子集（与 S09 v5 电池同构，同一 anchors-file）
#   [3] 反事实 shuffled：200 锚子集
#   [4] 指标汇总：eval_v1（含 visual dependence），纯 CPU
#
# 关键约定：
#   - 三步推理均带 --resume：幂等，中断后重跑本脚本即续跑（按输出文件内已有 token 跳过）；
#   - set -e -o pipefail + 换行顺序执行：任一步失败立即停止（face hash 对不上会立刻抛错，
#     不会白跑后续步骤）；
#   - 主预测与反事实预测分文件，互不污染；反事实的变换图落到
#     $OUT/counterfactual_images/{blank,shuffled}/（scratch dir = 输出文件父目录）。
#
# 产物（均在 $OUT 下）：
#   predictions.jsonl / predictions_cf_blank.jsonl / predictions_cf_shuffled.jsonl
#   main.log / cf_blank.log / cf_shuffled.log / evaluate.log
#   run_summary_*.json
#   eval_report.json / eval_report.md / anchor_scores.jsonl
# ============================================================================
set -euo pipefail

# --- 路径常量（全部相对工作区根；两项 git sha 由 Step 0.4 台账登记） -------------
CONFIG=DriveAlign/configs/benchmark/base_1f_v6.yaml
MANIFEST=data/face_manifests/v6/anchor_policy_manifest.json
DATASET_ROOT=data/dataset_v4
NUSC_DATAROOT=data/nuscenes/trainval
CF_SUBSET=runs/S09_base_benchmark/counterfactual_subset.txt
EVAL_CONFIG=DriveAlign/configs/evaluation/eval_v1.yaml
OUT=runs/S09_base_benchmark/eval_v6

# --- 前置守卫：确认 cwd 与关键输入就位（避免相对路径静默写错地方） ---------------
[ -f "$CONFIG" ] || { echo "FATAL: 未找到 $CONFIG —— 请在 /root/autodl-tmp/drivealign_workspace 下运行"; exit 1; }
[ -f "$MANIFEST" ] || { echo "FATAL: 未找到 $MANIFEST"; exit 1; }
[ -f "$CF_SUBSET" ] || { echo "FATAL: 未找到反事实子集 $CF_SUBSET"; exit 1; }
export PYTHONPATH=DriveAlign/src
mkdir -p "$OUT"

echo "=================================================================="
echo "[0/4] 预检 sha256（登记进 S11 决策台账）"
sha256sum "$CONFIG" "$MANIFEST"

echo "=================================================================="
echo "[1/4] 主评测：test 全量 5,468 anchors（v6 面）"
python -m drivealign.cli.base_benchmark \
  --config "$CONFIG" \
  --anchor-manifest "$MANIFEST" \
  --dataset-root "$DATASET_ROOT" \
  --nuscenes-dataroot "$NUSC_DATAROOT" \
  --resume \
  --out "$OUT/predictions.jsonl" 2>&1 | tee "$OUT/main.log"

echo "=================================================================="
echo "[2/4] 反事实 blank：200 锚子集（与 v5 电池同构）"
python -m drivealign.cli.base_benchmark \
  --config "$CONFIG" \
  --anchor-manifest "$MANIFEST" \
  --dataset-root "$DATASET_ROOT" \
  --nuscenes-dataroot "$NUSC_DATAROOT" \
  --anchors-file "$CF_SUBSET" \
  --image-transform blank \
  --resume \
  --out "$OUT/predictions_cf_blank.jsonl" 2>&1 | tee "$OUT/cf_blank.log"

echo "=================================================================="
echo "[3/4] 反事实 shuffled：200 锚子集"
python -m drivealign.cli.base_benchmark \
  --config "$CONFIG" \
  --anchor-manifest "$MANIFEST" \
  --dataset-root "$DATASET_ROOT" \
  --nuscenes-dataroot "$NUSC_DATAROOT" \
  --anchors-file "$CF_SUBSET" \
  --image-transform shuffled \
  --resume \
  --out "$OUT/predictions_cf_shuffled.jsonl" 2>&1 | tee "$OUT/cf_shuffled.log"

echo "=================================================================="
echo "[4/4] 指标汇总：eval_v1（含 visual dependence，CPU）"
python -m drivealign.cli.evaluate \
  --predictions "$OUT/predictions.jsonl" \
  --anchor-manifest "$MANIFEST" \
  --dataset-root "$DATASET_ROOT" \
  --config "$EVAL_CONFIG" \
  --out "$OUT" \
  --counterfactual-blank "$OUT/predictions_cf_blank.jsonl" \
  --counterfactual-shuffled "$OUT/predictions_cf_shuffled.jsonl" 2>&1 | tee "$OUT/evaluate.log"

echo "=================================================================="
echo "S11 Step 0.3 ALL DONE"
echo "产物目录：$OUT"
echo "gates 核对：主跑 rows=5468 / 反事实各 200 行 / eval_report.json 已生成 /"
echo "           eval_report 内 anchor_manifest_sha256 应为 2abe8c4f…（v6 manifest）"
