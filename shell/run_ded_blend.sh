#!/bin/bash
# Infer dedicated models on val, average seeds, blend into v4 ensemble, eval raw.
cd /root
PY=/root/miniconda3/bin/python
export HF_HUB_OFFLINE=1

echo "RAPE42 $(date +%H:%M)"
$PY infer_ensemble.py --task multi --configs cfg_ded_rape42.json > /root/logs/ens_ded_rape42.log 2>&1
mv -f /root/ens_multi.npz /root/ens_ded_rape42.npz
echo "RAPE43 $(date +%H:%M)"
$PY infer_ensemble.py --task multi --configs cfg_ded_rape43.json > /root/logs/ens_ded_rape43.log 2>&1
mv -f /root/ens_multi.npz /root/ens_ded_rape43.npz
echo "WHEAT42 $(date +%H:%M)"
$PY infer_ensemble.py --task multi --configs cfg_ded_wheat42.json > /root/logs/ens_ded_wheat42.log 2>&1
mv -f /root/ens_multi.npz /root/ens_ded_wheat42.npz
echo "WHEAT43 $(date +%H:%M)"
$PY infer_ensemble.py --task multi --configs cfg_ded_wheat43.json > /root/logs/ens_ded_wheat43.log 2>&1
mv -f /root/ens_multi.npz /root/ens_ded_wheat43.npz
echo "AVG $(date +%H:%M)"
$PY avg_ded.py rape
$PY avg_ded.py wheat
echo "BLEND $(date +%H:%M)"
$PY blend_both.py --ens ens_multi_v4.npz --dw ens_ded_wheat.npz --dr ens_ded_rape.npz --alpha 0.5 --out ens_multi_v4b.npz
echo "EVAL $(date +%H:%M)"
$PY eval_final_sweep.py --class_name wheat --preds /root/ens_multi_v4b.npz --channel 0 > /root/logs/eval_wheat_v4b.log 2>&1
$PY eval_final_sweep.py --class_name rape --preds /root/ens_multi_v4b.npz --channel 1 > /root/logs/eval_rape_v4b.log 2>&1
echo "DONE $(date +%H:%M)"
