#!/bin/bash
# Train wheat_rape on trainpseudo (train + testA pseudo-labels), validate on val (honest).
set -e
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe

$PY scripts/train_local.py \
  --mode wheat_rape \
  --arch deeplabv3plus \
  --encoder mit_b3 \
  --seed 42 \
  --data_root data/public/trainpseudo \
  --batch_size 8 \
  --acc 3 \
  --img_size 256 \
  --epochs 130 \
  --lovasz_w 0.5 \
  --focal_gamma 2.0 \
  --pos_weight 1.5 \
  --tag tp 2>&1
