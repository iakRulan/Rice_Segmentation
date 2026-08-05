#!/bin/bash
for p in $(pgrep -f "train_st[r]ong.py"); do kill -9 $p 2>/dev/null; done
for p in $(pgrep -f "run_campaig[n].sh"); do kill -9 $p 2>/dev/null; done
echo cleaned
