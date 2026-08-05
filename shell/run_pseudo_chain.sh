#!/bin/bash
# Chain: generate confident testA pseudo labels -> self-training retrain on train_plus.
cd /root
echo "PSEUDO_GEN $(date +%H:%M)"
/root/miniconda3/bin/python gen_pseudo_train.py > /root/logs/gen_pseudo.log 2>&1
echo "PSEUDO_GEN_DONE $(date +%H:%M)"
# verify train_plus built
if [ -f /root/competition_data/public/train_plus/label/rape/ta_clip_00665.png ]; then
  echo "RETRAIN_START $(date +%H:%M)"
  bash /root/run_pseudo_retrain.sh > /root/logs/pseudo_retrain_wrapper.log 2>&1
  echo "RETRAIN_DONE $(date +%H:%M)"
else
  echo "PSEUDO_GEN FAILED - no ta_ labels"
fi
