#!/bin/bash
set -euo pipefail
cd /root/crop_segmentation
PY=/root/miniconda3/bin/python
export HF_HUB_OFFLINE=1

$PY scripts/infer_context.py \
  --checkpoint weights/ctx1_wheat_rape_deeplabv3plus_mit_b4_42_best.pth \
  --split val --output /root/ctx1_wr_val.npz --batch_size 16 --tta --cache

$PY scripts/infer_context.py \
  --checkpoint weights/ctx1_rice_deeplabv3plus_mit_b4_42_best.pth \
  --split val --output /root/ctx1_rice_val.npz --batch_size 16 --tta --cache

$PY scripts/infer_mosaic.py \
  --config cfg_mosaic1_wheat_rape.json --task multi --split val \
  --output /root/mosaic1_wr_val.npz --batch_size 4 --tta --cache

$PY scripts/infer_mosaic.py \
  --config cfg_s7_wheat_rape.json --task multi --split val \
  --output /root/s7_mosaic_wr_val.npz --batch_size 4 --tta --cache

echo "ALL_CONTEXT_EVALS_DONE"
