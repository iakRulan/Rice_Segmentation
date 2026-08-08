#!/bin/bash
set -euo pipefail
cd /root/crop_segmentation
PY=/root/miniconda3/bin/python
export HF_HUB_OFFLINE=1

$PY scripts/infer_mosaic.py --config cfg_mosaic2_rape.json --task rape \
  --split val --output /root/mosaic2_rape_val.npz --batch_size 4 --tta --cache

$PY scripts/infer_mosaic.py --config cfg_mosaic2_rice.json --task rice \
  --split val --output /root/mosaic2_rice_val.npz --batch_size 4 --tta --cache

echo MOSAIC2_EVAL_DONE
