#!/bin/bash
# Train dedicated rape model with small-object zoom augmentation.
# Target: beat s3_rape non-empty IoU (0.687) — the biggest bottleneck.
set -e
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe

$PY scripts/train_local.py \
  --mode rape \
  --arch unet \
  --encoder mit_b3 \
  --seed 42 \
  --batch_size 8 \
  --acc 3 \
  --img_size 256 \
  --crop_zoom 0.5 \
  --canvas 512 \
  --epochs 130 \
  --lovasz_w 0.5 \
  --focal_gamma 2.0 \
  --pos_weight 1.5 \
  --tag r1 2>&1
