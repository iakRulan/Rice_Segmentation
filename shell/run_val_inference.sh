#!/bin/bash
# Reproduce the v4 baseline locally: run all ensemble model preds on val,
# then evaluate with blend + empty-clf + sweep.
set -e
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
SC="256,288,320"

echo "=== [1/7] multi v4 (wheat/rape) ==="
$PY scripts/local_ensemble.py --task multi --configs cfg_multi_v4.json --split val --scales $SC --out ens_multi_val.npz

echo "=== [2/7] s7 wheat/rape (pseudo-label+focal) ==="
$PY scripts/local_ensemble.py --task multi --configs cfg_s7_wheat_rape.json --split val --scales $SC --out ens_s7_val.npz

echo "=== [3/7] rice v4 ==="
$PY scripts/local_ensemble.py --task single --configs cfg_rice_v4.json --split val --scales $SC --out ens_single_val.npz

echo "=== [4/7] dedicated wheat seed42 ==="
$PY scripts/local_ensemble.py --task single --subdir wheat_rape --configs cfg_ded_wheat.json --split val --scales $SC --out ens_ded_wheat_val.npz

echo "=== [5/7] dedicated wheat seed43 ==="
$PY scripts/local_ensemble.py --task single --subdir wheat_rape --configs cfg_ded_wheat43.json --split val --scales $SC --out ens_ded_wheat43_val.npz

echo "=== [6/7] dedicated rape seed42 ==="
$PY scripts/local_ensemble.py --task single --subdir wheat_rape --configs cfg_ded_rape.json --split val --scales $SC --out ens_ded_rape_val.npz

echo "=== [7/7] dedicated rape seed43 ==="
$PY scripts/local_ensemble.py --task single --subdir wheat_rape --configs cfg_ded_rape43.json --split val --scales $SC --out ens_ded_rape43_val.npz

echo "=== ALL INFERENCE DONE ==="
