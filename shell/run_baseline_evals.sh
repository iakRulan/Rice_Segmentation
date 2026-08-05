#!/bin/bash
cd /root
PY=/root/miniconda3/bin/python
$PY eval_final_sweep.py --class_name wheat --preds ens_multi.npz --channel 0 --empty_preds empty_wheat_clf_pred.npy > /root/logs/baseline_wheat.log 2>&1
$PY eval_final_sweep.py --class_name rape --preds ens_multi.npz --channel 1 --empty_preds empty_rape_clf_pred.npy > /root/logs/baseline_rape.log 2>&1
$PY eval_final_sweep.py --class_name rice --preds ens_single.npz --channel 0 --empty_preds empty_rice_clf_pred.npy > /root/logs/baseline_rice.log 2>&1
echo ALL_BASELINE_DONE