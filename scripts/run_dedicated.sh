#!/bin/bash
# Dedicated single-class models for wheat/rape (lovasz on, default 0.5). Sequential.
cd /root
mkdir -p /root/logs
PY=/root/miniconda3/bin/python
export HF_HUB_OFFLINE=1

run_one() {
  local mode=$1 enc=$2 seed=$3 tag=$4 epochs=$5 bs=$6
  local logf=/root/logs/${tag}_${mode}_${enc}_${seed}.log
  echo ">>> START $(date +%H:%M) $mode/$enc seed$seed"
  $PY train_strong.py --mode $mode --arch unet --encoder $enc --seed $seed \
      --epochs $epochs --batch_size $bs --workers 6 --tag $tag --patience 30 --lovasz_w 0.5 > $logf 2>&1
  echo ">>> DONE $(date +%H:%M) $mode/$enc seed$seed"
}

run_one rape mit_b3 42 s3 130 24
run_one wheat mit_b3 42 s3 130 24
run_one rape mit_b3 43 s3 130 24
run_one wheat mit_b3 43 s3 130 24
echo "=== DEDICATED DONE ==="
