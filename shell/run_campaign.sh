#!/bin/bash
# Wave 1 training campaign (sequential). Logs to /root/logs/.
cd /root
mkdir -p /root/logs
PY=/root/miniconda3/bin/python
export HF_HUB_OFFLINE=1

run_one() {
  local mode=$1 arch=$2 enc=$3 seed=$4 tag=$5 epochs=$6 bs=$7
  local logf=/root/logs/${tag}_${mode}_${enc}_${seed}.log
  echo ">>> START $(date +%H:%M) $mode/$arch/$enc seed$seed"
  $PY train_strong.py --mode $mode --arch $arch --encoder $enc --seed $seed \
      --epochs $epochs --batch_size $bs --workers 6 --tag $tag > $logf 2>&1
  echo ">>> DONE $(date +%H:%M) $mode/$arch/$enc seed$seed (see $logf)"
}

run_one wheat_rape unet mit_b2 42 s2 130 32
run_one rice unet mit_b2 42 s2 130 32
run_one wheat_rape unet mit_b3 42 s2 130 24
run_one rice unet mit_b3 42 s2 130 24
run_one wheat_rape unet efficientnet-b3 43 s2 130 32
run_one rice unet efficientnet-b3 43 s2 130 32
echo "=== WAVE1 DONE ==="
