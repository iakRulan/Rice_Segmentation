#!/bin/bash
# Run all ensemble model preds on testA (for final submission).
set -e
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
SC="256,288,320"

echo "=== [1/7] multi v4 (wheat/rape) ==="
$PY scripts/local_ensemble.py --task multi --configs cfg_multi_v4.json --split testA --scales $SC --out ens_multi_testA.npz

echo "=== [2/7] s7 wheat/rape ==="
$PY scripts/local_ensemble.py --task multi --configs cfg_s7_wheat_rape.json --split testA --scales $SC --out ens_s7_testA.npz

echo "=== [3/7] rice v4 ==="
$PY scripts/local_ensemble.py --task single --configs cfg_rice_v4.json --split testA --scales $SC --out ens_single_testA.npz

echo "=== [4/7] dedicated wheat seed42 ==="
$PY scripts/local_ensemble.py --task single --subdir wheat_rape --configs cfg_ded_wheat.json --split testA --scales $SC --out ens_ded_wheat_testA.npz

echo "=== [5/7] dedicated wheat seed43 ==="
$PY scripts/local_ensemble.py --task single --subdir wheat_rape --configs cfg_ded_wheat43.json --split testA --scales $SC --out ens_ded_wheat43_testA.npz

echo "=== [6/7] dedicated rape seed42 ==="
$PY scripts/local_ensemble.py --task single --subdir wheat_rape --configs cfg_ded_rape.json --split testA --scales $SC --out ens_ded_rape_testA.npz

echo "=== [7/7] dedicated rape seed43 ==="
$PY scripts/local_ensemble.py --task single --subdir wheat_rape --configs cfg_ded_rape43.json --split testA --scales $SC --out ens_ded_rape43_testA.npz

echo "=== TESTA INFERENCE DONE ==="
