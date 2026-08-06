"""Honest evaluation with opt_patch.postproc.triple_threshold / search_triple.

Replaces the old single-threshold + empty-classifier flow. The empty-vs-nonempty
decision is folded into (t_hi, min_size) gating, and mask drawing uses t_lo.
Thresholds are searched on a FIT split and reported on an EVAL split (OOF / half
of val), so the reported number is honest.

Usage:
    python scripts/triple_eval.py --spec configs/blend_fix.json [--holdout] [--out_json ...]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import VAL_LBL, PREDS
from opt_patch.postproc import search_triple, report

_npz_cache = {}


def load_npz(path):
    if str(path) not in _npz_cache:
        _npz_cache[str(path)] = np.load(path)
    return _npz_cache[str(path)]


def blend(spec, split):
    acc, first_imgs = None, None
    for path, ch, w in spec:
        npz = path.replace('val', split) if split != 'val' else path
        d = load_npz(PREDS / npz)
        imgs = sorted(d.files)
        if first_imgs is None:
            first_imgs = imgs
        for f in imgs:
            p = d[f].astype(np.float32)
            if p.ndim == 2:
                p = p[None]
            if acc is None:
                acc = {}
            acc[f] = acc.get(f, np.zeros_like(p[ch])) + p[ch] * w
    return first_imgs, acc


def valhold_imgs(cls):
    imgs = sorted(f.name for f in (VAL_LBL / cls).iterdir() if f.suffix == '.png')
    return {f for i, f in enumerate(imgs) if i % 7 == 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--spec', required=True)
    ap.add_argument('--subset', choices=['val', 'valhold'], default='val')
    ap.add_argument('--holdout', action='store_true',
                    help='search on even half, report on odd half (honest)')
    ap.add_argument('--out_json', default=None)
    args = ap.parse_args()

    spec = json.load(open(args.spec))
    results = {}
    for cls, entries in spec.items():
        print(f'[{cls}]')
        imgs, probs = blend(entries, 'val')
        if args.subset == 'valhold':
            keep = valhold_imgs(cls)
            imgs = [f for f in imgs if f in keep]
            probs = {f: probs[f] for f in imgs}
            print(f'  (subset=valhold n={len(imgs)})')
        P = np.stack([probs[f] for f in imgs])
        T = np.stack([(np.array(Image.open(VAL_LBL / cls / f)) > 0).astype(np.uint8) for f in imgs])

        if args.holdout:
            fit_idx = np.arange(len(imgs))[0::2]
            ev_idx = np.arange(len(imgs))[1::2]
            best = search_triple(P[fit_idx], T[fit_idx], verbose=False)
            r = report(P[ev_idx], T[ev_idx], best)
            print(f'  [honest] IoU={r["iou"]:.4f} empty={r["empty"]:.4f} nonempty={r["nonempty"]:.4f} '
                  f'(search on {len(fit_idx)}, eval on {len(ev_idx)}) settings={best}')
            results[cls] = dict(method='holdout', iou=r['iou'], settings=best)
        else:
            best = search_triple(P, T, verbose=False)
            r = report(P, T, best)
            print(f'  [fit-eval] IoU={r["iou"]:.4f} empty={r["empty"]:.4f} nonempty={r["nonempty"]:.4f}')
            results[cls] = dict(method='fiteval', iou=r['iou'], settings=best)

    mean = float(np.mean([results[c]['iou'] for c in results]))
    print(f'\n[MEAN IoU] {mean:.4f}  ' + '  '.join(f'{c}={results[c]["iou"]:.4f}' for c in results))
    if args.out_json:
        json.dump({'mean': mean, 'per_class': results}, open(args.out_json, 'w'), indent=2)


if __name__ == '__main__':
    main()
