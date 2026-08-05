#!/bin/bash
# After r1 rape training: infer r1 on val, evaluate rape blends (v3 vs v3+r1),
# then launch rice training.
set -e
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
SC="256,288,320"

echo "=== r1 rape val inference ==="
$PY scripts/local_ensemble.py --task single --subdir wheat_rape --configs cfg_r1_rape.json --split val --scales $SC --out ens_r1_rape_val.npz

echo "=== eval: rape blend v3 (no r1) ==="
$PY scripts/local_blend_eval.py --spec configs/blend_v3.json --fixed --no_empty 2>&1 | grep -E "MEAN"

echo "=== eval: rape blend v3 + r1 ==="
$PY scripts/local_blend_eval.py --spec configs/blend_v3r1.json --fixed --no_empty 2>&1 | grep -E "MEAN"

echo "=== DONE (launch rice next) ==="
