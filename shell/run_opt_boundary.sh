#!/bin/bash
set -e

# 加载 AutoDL 学术加速代理与 HuggingFace 镜像
source /etc/network_turbo 2>/dev/null || true
export HF_ENDPOINT=https://hf-mirror.com

cd /root/Rice_Segmentation

# 确保输出权重目录与软链接
mkdir -p /root/autodl-tmp/weights/opt_boundary
mkdir -p data
rm -rf data/public
ln -s /root/autodl-tmp/data/public data/public

echo "=== Starting finetune_v2 with boundary consistency loss ==="
exec /root/miniconda3/bin/python -u scripts/finetune_v2.py --config configs/opt_wr_boundary_mitb3.json
