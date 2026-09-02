#!/bin/bash
set -e

source /etc/network_turbo 2>/dev/null || true
export HF_ENDPOINT=https://hf-mirror.com

cd /root/Rice_Segmentation

mkdir -p /root/autodl-tmp/weights/opt_boundary
mkdir -p data
rm -rf data/public
ln -s /root/autodl-tmp/data/public data/public

echo "=== Starting finetune_v2 for rice with boundary consistency loss ==="
exec /root/miniconda3/bin/python -u scripts/finetune_v2.py --config configs/opt_rice_boundary_mitb3.json
