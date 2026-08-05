#!/bin/bash
# Re-run dedicated single-class models on the correct wheat_rape image dir.
set -e
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
SC="256,288,320"

echo "=== dedicated wheat seed42 ==="
$PY scripts/local_ensemble.py --task single --subdir wheat_rape --configs cfg_ded_wheat.json --split val --scales $SC --out ens_ded_wheat_val.npz

echo "=== dedicated wheat seed43 ==="
$PY scripts/local_ensemble.py --task single --subdir wheat_rape --configs cfg_ded_wheat43.json --split val --scales $SC --out ens_ded_wheat43_val.npz

echo "=== dedicated rape seed42 ==="
$PY scripts/local_ensemble.py --task single --subdir wheat_rape --configs cfg_ded_rape.json --split val --scales $SC --out ens_ded_rape_val.npz

echo "=== dedicated rape seed43 ==="
$PY scripts/local_ensemble.py --task single --subdir wheat_rape --configs cfg_ded_rape43.json --split val --scales $SC --out ens_ded_rape43_val.npz

echo "=== DONE ==="
