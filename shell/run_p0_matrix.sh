#!/bin/bash
# Run the 4 P0 direction configs sequentially on fold0. Each waits for the
# previous to exit (single GPU). Sources AutoDL network_turbo (needed to
# download mit_b3 ImageNet weights from HuggingFace via timm).
# Usage: bash scripts/run_p0_matrix.sh [start_at]
source /etc/network_turbo > /dev/null 2>&1
cd /root/crop_segmentation
LOG=/root/logs/p0_matrix.log
START=${1:-1}
CONFIGS=(ctx256_3ch ctx768_3ch ctx768_6ch ctx768_9ch)
for i in "${!CONFIGS[@]}"; do
  n=$((i + 1))
  [ "$n" -lt "$START" ] && continue
  key="${CONFIGS[$i]}"
  echo "[$(date '+%F %T')] start $key" >> "$LOG"
  /root/miniconda3/bin/python -u scripts/finetune_v2.py \
    --config "configs/p0_joint_${key}.json" \
    > "/root/logs/p0_${key}.log" 2>&1
  echo "[$(date '+%F %T')] done $key" >> "$LOG"
done
echo "[$(date '+%F %T')] P0 matrix complete" >> "$LOG"
