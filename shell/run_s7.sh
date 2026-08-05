#!/bin/bash
# Improved recipe: train on trainplus (train+val) + focal + pos_weight + lovasz.
cd /root
mkdir -p /root/logs
PY=/root/miniconda3/bin/python
export HF_HUB_OFFLINE=1

run_one() {
  local mode=$1 enc=$2 seed=$3 tag=$4 epochs=$5 bs=$6 gamma=$7 posw=$8
  local logf=/root/logs/${tag}_${mode}_${enc}_${seed}.log
  echo ">>> START $(date +%H:%M) $mode/$enc seed$seed gamma$gamma posw$posw"
  $PY train_strong.py --mode $mode --arch unet --encoder $enc --seed $seed \
      --epochs $epochs --batch_size $bs --workers 6 --tag $tag --patience 30 \
      --lovasz_w 0.5 --focal_gamma $gamma --pos_weight $posw \
      --data_root /root/competition_data/public/trainplus \
      --val_root /root/competition_data/public/valhold > $logf 2>&1
  echo ">>> DONE $(date +%H:%M) $mode/$enc seed$seed"
}

run_one wheat_rape mit_b3 42 s7 130 24 2 1.5
echo "=== S7 DONE ==="
