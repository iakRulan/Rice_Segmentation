#!/bin/bash
# Train rice on trainplus (train + 85% val). Rice has large dense targets,
# so standard loss (lovasz) + mild pos_weight; val on valhold.
set -e
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe

$PY scripts/train_local.py \
  --mode rice \
  --arch unet \
  --encoder mit_b3 \
  --seed 42 \
  --data_root data/public/trainplus \
  --val_root data/public/valhold \
  --batch_size 8 \
  --acc 3 \
  --img_size 256 \
  --epochs 130 \
  --lovasz_w 0.5 \
  --pos_weight 1.3 \
  --tag tp_r 2>&1
