#!/usr/bin/env bash
set -euo pipefail

cd /root/crop_segmentation
source /etc/network_turbo || true

CONFIG=${1:-configs/finetune_satlas_rape_s2tiny.json}
python_bin=${PYTHON_BIN:-/root/miniconda3/bin/python}

"$python_bin" scripts/finetune_v2.py --config "$CONFIG"

name=$(
  "$python_bin" -c 'import json,sys; print(json.load(open(sys.argv[1]))["name"])' "$CONFIG"
)
checkpoint="weights/v2/${name}/best.pth"
output="/root/${name}_val.npz"

"$python_bin" scripts/predict_v2.py \
  --checkpoint "$checkpoint" \
  --split val --tta --cache --output "$output"

echo "checkpoint=$checkpoint"
echo "validation_probabilities=$output"
