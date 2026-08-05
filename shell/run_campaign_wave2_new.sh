#!/bin/bash
# Wave 2: diverse models for ensemble (run after wave 1)
cd /root
mkdir -p /root/logs
PY=/root/miniconda3/bin/python
export HF_HUB_OFFLINE=1

run_one() {
  local mode=$1 arch=$2 enc=$3 seed=$4 tag=$5 epochs=$6 bs=$7
  local logf=/root/logs/${tag}_${mode}_${arch}_${enc}_${seed}.log
  echo ">>> START $(date +%H:%M) $mode/$arch/$enc seed$seed"
  $PY train_strong.py --mode $mode --arch $arch --encoder $enc --seed $seed \
      --epochs $epochs --batch_size $bs --workers 6 --tag $tag --patience 30 > $logf 2>&1
  echo ">>> DONE $(date +%H:%M) $mode/$arch/$enc seed$seed"
}

run_one wheat_rape deeplabv3plus mit_b2 42 s2 100 24
run_one rice deeplabv3plus mit_b2 42 s2 100 24
run_one wheat_rape unetpp efficientnet-b3 42 s2 100 16
run_one rice unetpp efficientnet-b3 42 s2 100 16
run_one wheat_rape unet mit_b5 42 s2 100 12
run_one rice unet mit_b5 42 s2 100 12
run_one wheat_rape unet mit_b2 43 s2 100 32
run_one rice unet mit_b2 43 s2 100 32
echo "=== WAVE2 DONE ==="
