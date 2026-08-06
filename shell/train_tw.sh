#!/bin/bash
# Retrain tp_wr recipe (deeplabv3plus/mit_b3 + focal) on TRAIN ONLY.
# val = full val (honest signal, model never saw val). Tests whether the
# recipe genuinely beats the v4 ensemble on testA.
set -e
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe

$PY scripts/train_local.py \
  --mode wheat_rape \
  --arch deeplabv3plus \
  --encoder mit_b3 \
  --seed 42 \
  --batch_size 8 \
  --acc 3 \
  --img_size 256 \
  --epochs 130 \
  --lovasz_w 0.5 \
  --focal_gamma 2.0 \
  --pos_weight 1.5 \
  --tag tw 2>&1
