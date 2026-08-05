#!/bin/bash
# Train rice model: UnetPlusPlus mit_b3 seed43 (architecture+seed diversity vs the
# v4 rice ensemble which is unet/deeplab seed42). Rice targets are large dense
# fields (median fg 65%), so no crop_zoom needed.
set -e
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe

$PY scripts/train_local.py \
  --mode rice \
  --arch unetpp \
  --encoder mit_b3 \
  --seed 43 \
  --batch_size 8 \
  --acc 3 \
  --img_size 256 \
  --epochs 130 \
  --lovasz_w 0.5 \
  --tag rpp1 2>&1
