#!/bin/bash
# P1 production: 5-fold isolated-row OOF training with the winning P0 config
# (joint, dual-temporal 6ch, 256 context, Unet+mit_b3). Sequential on one GPU.
# Generates configs/joint_256_6ch_f{0..4}.json from configs/joint_256_6ch.json.
# Usage: bash shell/run_5fold.sh [start_fold]
cd /root/crop_segmentation
LOG=/root/logs/5fold_matrix.log
START=${1:-0}
PY=/root/miniconda3/bin/python
for FOLD in 0 1 2 3 4; do
  [ "$FOLD" -lt "$START" ] && continue
  CFG="configs/joint_256_6ch_f${FOLD}.json"
  "$PY" - <<PYEOF
import json
c = json.load(open("configs/joint_256_6ch.json"))
c["name"] = f"joint_256_6ch_f${FOLD}"
c["data"]["fold"] = ${FOLD}
json.dump(c, open("$CFG", "w"), indent=2)
PYEOF
  echo "[$(date '+%F %T')] start fold ${FOLD}" >> "$LOG"
  "$PY" -u scripts/finetune_v2.py --config "$CFG" > "/root/logs/joint_f${FOLD}.log" 2>&1
  echo "[$(date '+%F %T')] done fold ${FOLD}" >> "$LOG"
done
echo "[$(date '+%F %T')] 5-fold matrix complete" >> "$LOG"
