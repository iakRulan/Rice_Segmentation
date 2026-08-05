#!/bin/bash
# Train wheat_rape on trainplus (train + 85% val) with focal - s7's proven recipe.
# deeplabv3plus/mit_b3 = architecture diversity vs s7's unet/mit_b3 (mit encoders
# are not supported by UnetPlusPlus).
set -e
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe

$PY scripts/train_local.py \
  --mode wheat_rape \
  --arch deeplabv3plus \
  --encoder mit_b3 \
  --seed 42 \
  --data_root data/public/trainplus \
  --val_root data/public/valhold \
  --batch_size 8 \
  --acc 3 \
  --img_size 256 \
  --epochs 130 \
  --lovasz_w 0.5 \
  --focal_gamma 2.0 \
  --pos_weight 1.5 \
  --tag tp_wr 2>&1
