#!/bin/bash
# Train rice v2: per-image loss + aux cls head, train-only.
set -e
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe

$PY scripts/train_v2.py \
  --mode rice \
  --arch deeplabv3plus \
  --encoder mit_b3 \
  --seed 42 \
  --batch_size 8 \
  --acc 3 \
  --img_size 256 \
  --epochs 130 \
  --lovasz_w 0.5 \
  --focal_gamma 1.0 \
  --pos_weight 1.2 \
  --cls_w 0.5 \
  --aux \
  --copy_paste 0.3 \
  --swa_k 5 \
  --cache \
  --tag v2r 2>&1
