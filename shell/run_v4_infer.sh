#!/bin/bash
cd /root
PY=/root/miniconda3/bin/python
export HF_HUB_OFFLINE=1
echo "START $(date +%H:%M)"
$PY infer_ensemble.py --task multi --configs cfg_multi_v4.json > /root/logs/ens_multi_v4.log 2>&1
mv -f /root/ens_multi.npz /root/ens_multi_v4.npz
echo "MULTI_DONE $(date +%H:%M)"
$PY infer_ensemble.py --task single --configs cfg_rice_v4.json > /root/logs/ens_single_v4.log 2>&1
mv -f /root/ens_single.npz /root/ens_single_v4.npz
echo "V4_ALL_DONE $(date +%H:%M)"
