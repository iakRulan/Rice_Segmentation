#!/bin/bash
# Wait for any running finetune_v2 to exit, probe GPU memory for bigger-model
# configs, then launch the requested model. Appends progress to launch_chain.log.
# AutoDL public-network accelerator (needed to download Satlas weights from HF).
source /etc/network_turbo > /dev/null 2>&1
cd /root/crop_segmentation
LOG=/root/logs/launch_chain.log
KEY=$1
case $KEY in
  s2base)   CFG=configs/finetune_satlas_rape_s2base.json;   CFGLOG=satlas_s2base_rape.log;;
  s2res152) CFG=configs/finetune_satlas_rape_s2res152.json; CFGLOG=satlas_s2res152_rape.log;;
  s2res50)  CFG=configs/finetune_satlas_rape_s2res50.json;  CFGLOG=satlas_s2res50_rape.log;;
  *) echo "unknown key: $KEY" >> "$LOG"; exit 1;;
esac

echo "[$(date '+%F %T')] $KEY: waiting for running finetune_v2 ..." >> "$LOG"
while pgrep -f 'scripts/finetune_v2.py' > /dev/null; do sleep 30; done

echo "[$(date '+%F %T')] $KEY: gpu free, probing memory ..." >> "$LOG"
/root/miniconda3/bin/python scripts/probe_models.py >> /root/logs/probe_models.log 2>&1

echo "[$(date '+%F %T')] $KEY: launching" >> "$LOG"
nohup /root/miniconda3/bin/python -u scripts/finetune_v2.py --config "$CFG" > "/root/logs/$CFGLOG" 2>&1 &
echo "[$(date '+%F %T')] $KEY launched pid=$!" >> "$LOG"
