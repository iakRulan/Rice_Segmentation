#!/bin/bash
cd /root
nohup bash /root/run_campaign.sh > /root/logs/campaign.log 2>&1 &
echo "launched $!"
