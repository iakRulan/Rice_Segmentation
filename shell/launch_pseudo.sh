#!/bin/bash
# Launcher: clean any stale gen_pseudo, then run the chain detached.
pkill -9 -f gen_pseudo_train 2>/dev/null
sleep 2
cd /root
nohup setsid bash /root/run_pseudo_chain.sh > /root/logs/pseudo_chain.log 2>&1 </dev/null &
echo "CHAIN_PID $!"
