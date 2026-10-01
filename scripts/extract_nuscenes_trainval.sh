#!/usr/bin/env bash
# Extract nuScenes v1.0-trainval subset from the AutoDL public dataset:
# full metadata + all 6 camera keyframe samples (no sweeps/lidar/radar/map),
# then symlink the result into the workspace data/ folder.
#
# Partial extraction is safe: NuScenes() only loads the metadata JSON tables
# and never validates data files; missing files would only matter on direct
# access, and drivealign io classifies missing images as MISSING_IMAGE.
#
# Run inside tmux (stream-reads ~350GB, takes 0.5-1.5h; writes ~35GB).
# Interrupted? Just rerun: tar overwrites files, idempotent per blob.
#
# Startup command:
#   ``mkdir -p /root/autodl-tmp/datasets && tmux new-session -d -s nus_extract \
#   'bash DriveAlign/scripts/extract_nuscenes_trainval.sh 2>&1 | \
#   tee /root/autodl-tmp/datasets/extract.log'``
# (mkdir must run first: tee creates the file but not its parent directory.)
set -euo pipefail

SRC=/root/autodl-pub/nuScenes/Fulldatasetv1.0/Trainval
DST=/root/autodl-tmp/datasets/nuscenes/trainval
WORKSPACE=/root/autodl-tmp/drivealign_workspace

# Whitelist: all 6 cameras, keyframe samples only.
WILDCARDS=(
  'samples/CAM_FRONT/*'
  'samples/CAM_FRONT_LEFT/*'
  'samples/CAM_FRONT_RIGHT/*'
  'samples/CAM_BACK/*'
  'samples/CAM_BACK_LEFT/*'
  'samples/CAM_BACK_RIGHT/*'
  # Uncomment to also keep camera sweeps (~+60GB):
  # 'sweeps/CAM_*/*'
  # Uncomment to also keep radar keyframes (~+1GB):
  # 'samples/RADAR_*/*'
)

mkdir -p "$DST"
cd "$DST"

echo '== [1/3] metadata (full) =='
if [ ! -d v1.0-trainval ]; then
  tar -xzf "$SRC/v1.0-trainval_meta.tgz"
fi

echo '== [2/3] 10 blobs, whitelist extraction =='
for i in 01 02 03 04 05 06 07 08 09 10; do
  f="$SRC/v1.0-trainval${i}_blobs.tgz"
  [ -f "$f" ] || { echo "MISSING: $f"; exit 1; }
  tar -xzf "$f" --wildcards "${WILDCARDS[@]}"
  echo "blob $i done | disk: $(du -sh "$DST" | cut -f1) | free: $(df -h /root/autodl-tmp | tail -1 | awk '{print $4}')"
done

echo '== [3/3] symlink into workspace =='
mkdir -p "$WORKSPACE/data/nuscenes"
ln -sfn "$DST" "$WORKSPACE/data/nuscenes/trainval"

echo '== verify: load v1.0-trainval end-to-end =='
cd "$WORKSPACE"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate autovla_codeclean
PYTHONPATH=DriveAlign/src python - <<'PY'
from pathlib import Path
from drivealign.data.nuscenes_io import (
    load_nuscenes, resolve_anchor, read_keyframe_chain,
)
from drivealign.records.adapter import build_record

nusc = load_nuscenes(Path('data/nuscenes/trainval'), 'v1.0-trainval')
print('scenes:', len(nusc.scene), '| samples:', len(nusc.sample))
token = resolve_anchor(nusc)
chain = read_keyframe_chain(nusc, token)
print('anchor:', token, '->', chain.reason_code or f'{len(chain.frames)} frames OK')
res = build_record(nusc, token, 'verify-extract')
print('record:', res.reason_code or f"OK hash={res.record.canonical_hash()[:12]}")
PY

echo 'ALL DONE'
