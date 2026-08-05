#!/bin/bash
# Post-v4 evals: rice raw + wheat/rape empty-clf sweep. CPU only.
cd /root
PY=/root/miniconda3/bin/python
echo "RICE_RAW $(date +%H:%M)"
$PY eval_final_sweep.py --class_name rice --preds /root/ens_single_v4.npz --channel 0 > /root/logs/eval_rice_v4.log 2>&1
echo "CLF_WHEAT $(date +%H:%M)"
$PY clf_th_fast.py wheat /root/ens_multi_v4.npz 0 0.55 30,30 > /root/logs/clf_th_fast_wheat_v4.log 2>&1
echo "CLF_RAPE $(date +%H:%M)"
$PY clf_th_fast.py rape /root/ens_multi_v4.npz 1 0.51 200,200 > /root/logs/clf_th_fast_rape_v4.log 2>&1
echo "EVALS_DONE $(date +%H:%M)"
