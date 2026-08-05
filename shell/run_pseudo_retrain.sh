#!/bin/bash
# Self-training retrain: train on train_plus (train + confident testA pseudo) + val early-stop.
cd /root
mkdir -p /root/logs
PY=/root/miniconda3/bin/python
export HF_HUB_OFFLINE=1

run_one() {
  local mode=$1 enc=$2 seed=$3 tag=$4 epochs=$5 bs=$6
  local logf=/root/logs/${tag}_${mode}_${enc}_${seed}.log
  echo ">>> START $(date +%H:%M) $mode/$enc seed$seed (pseudo)"
  $PY train_strong.py --mode $mode --arch unet --encoder $enc --seed $seed \
      --epochs $epochs --batch_size $bs --workers 6 --tag $tag --patience 30 \
      --lovasz_w 0.5 --data_root /root/competition_data/public/train_plus > $logf 2>&1
  echo ">>> DONE $(date +%H:%M) $mode/$enc seed$seed"
}

run_one wheat_rape mit_b3 42 s6 130 24
run_one rice mit_b3 42 s6 130 24
echo "=== PSEUDO RETRAIN DONE ==="
