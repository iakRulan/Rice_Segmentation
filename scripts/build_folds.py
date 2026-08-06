"""OOF (out-of-fold) split builder for honest validation on train+val.

Splits train+val (5478 per class) into K folds. Fold models train on K-1 folds
and validate on the held-out fold; OOF predictions are honest. All thresholds /
blend weights / gates are fit on OOF only.

Usage:
    python scripts/build_folds.py --k 3 --mode wheat_rape [--seed 42]
    -> saves data/folds_{mode}_k{k}.json: {fold: [img_names]}
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import DATA, TRAIN_IMG, VAL_IMG

FOLD_DIR = DATA / 'folds'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--k', type=int, default=3)
    ap.add_argument('--mode', choices=['wheat_rape', 'rice'], required=True)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    FOLD_DIR.mkdir(parents=True, exist_ok=True)
    sub = 'wheat_rape' if args.mode == 'wheat_rape' else 'rice'
    train_imgs = sorted(f for f in (TRAIN_IMG / sub).iterdir() if f.suffix == '.png')
    val_imgs = sorted(f for f in (VAL_IMG / sub).iterdir() if f.suffix == '.png')
    all_imgs = [f.name for f in train_imgs] + [f.name for f in val_imgs]
    print(f'{sub}: train={len(train_imgs)} val={len(val_imgs)} total={len(all_imgs)}')

    rng = random.Random(args.seed)
    rng.shuffle(all_imgs)
    folds = {}
    n = len(all_imgs)
    for i in range(args.k):
        # deterministic-ish stratified by index: fold i = indices i, i+K, i+2K...
        fold = all_imgs[i::args.k]
        folds[i] = fold
        print(f'  fold {i}: {len(fold)} imgs')

    out = FOLD_DIR / f'folds_{args.mode}_k{args.k}.json'
    json.dump(folds, open(out, 'w'), indent=1)
    print(f'saved {out}')


if __name__ == '__main__':
    main()
