"""Automated blend weight search per class (CPU, fast).

Given candidate npz sources per class, find weights that maximize val IoU
(with or without empty classifier). Uses fixed threshold settings for speed;
refine thresholds separately with local_blend_eval --sweep.

Usage:
    python scripts/optimize_blend.py --classes-config configs/candidates.json \
        [--no_empty] [--clf logistic|gbt] [--iters 40]
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import VAL_LBL, PREDS

FIXED_T = {'wheat': 0.45, 'rape': 0.55, 'rice': 0.53}
_npz_cache = {}


def load_npz(path):
    if str(path) not in _npz_cache:
        _npz_cache[str(path)] = np.load(path)
    return _npz_cache[str(path)]


def empty_features(pred):
    flat = pred.reshape(-1)
    feats = [float(flat.max()), float(np.percentile(flat, 99)), float(np.percentile(flat, 95)),
             float(np.percentile(flat, 90)), float(flat.mean()), float((flat > 0.1).sum()),
             float((flat > 0.3).sum()), float((flat > 0.5).sum()), float((flat > 0.7).sum())]
    bin5 = (flat > 0.5).astype(np.uint8).reshape(pred.shape[-2:])
    labeled, n = ndimage.label(bin5)
    if n > 0:
        areas = ndimage.sum(bin5, labeled, range(1, n + 1))
        feats += [float(n), float(areas.max()), float(areas.sum())]
    else:
        feats += [0.0, 0.0, 0.0]
    return np.array(feats, np.float32)


def logistic(X, y, iters=2000, lr=0.3, l2=1e-3):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xs = (X - mu) / sd
    Xb = np.hstack([np.ones((len(X), 1)), Xs])
    w = np.zeros(Xb.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(Xb @ w, -30, 30)))
        w -= lr * (Xb.T @ (p - y) / len(y) + l2 * w)
    return w, mu, sd


def proba(X, w, mu, sd):
    Xs = (X - mu) / sd
    Xb = np.hstack([np.ones((len(X), 1)), Xs])
    return 1 / (1 + np.exp(-np.clip(Xb @ w, -30, 30)))


def load_class_arrays(sources):
    """sources: list of [npz, channel]. Returns (imgs, P, y_empty).
    P: (N, nsrc, 256, 256) probs. y_empty: (N,) 1=empty."""
    first_imgs = None
    P = []
    for path, ch in sources:
        d = load_npz(PREDS / path)
        imgs = sorted(d.files)
        if first_imgs is None:
            first_imgs = imgs
        else:
            assert imgs == first_imgs, f'{path} mismatch'
        P.append(np.stack([(d[f].astype(np.float32) if d[f].ndim == 2 else d[f][ch].astype(np.float32)) for f in imgs]))
    P = np.stack(P, 1)  # (N, nsrc, 256, 256)
    return first_imgs, P


def iou_mean(blend, targets):
    m = (blend > 0.5).astype(np.uint8)
    inter = np.logical_and(m, targets).sum(axis=(1, 2))
    union = np.logical_or(m, targets).sum(axis=(1, 2))
    ious = np.where(union == 0, np.where(inter == 0, 1.0, 0.0), inter / np.maximum(union, 1))
    return ious.mean()


def score_weights(w, P, targets, y_empty, t, use_empty, clf):
    blend = np.einsum('n,Nnij->Nij', w, P)  # (N,256,256)
    if use_empty:
        X = np.array([empty_features(blend[i]) for i in range(len(blend))])
        if clf == 'gbt':
            from sklearn.ensemble import GradientBoostingClassifier
            m = GradientBoostingClassifier(n_estimators=150, max_depth=3).fit(X, y_empty)
            p = m.predict_proba(X)[:, 1]
        else:
            ww, mu, sd = logistic(X, y_empty)
            p = proba(X, ww, mu, sd)
        th = 0.5
        best = 0.0
        for tt in np.arange(0.3, 0.7, 0.05):
            acc = ((p > tt).astype(int) == y_empty).mean()
            if acc > best:
                best, th = acc, tt
        empty_mask = (p > th).astype(int)
    else:
        empty_mask = None
    m = (blend > t).astype(np.uint8)
    if empty_mask is not None:
        m[empty_mask == 1] = 0
    inter = np.logical_and(m, targets).sum(axis=(1, 2))
    union = np.logical_or(m, targets).sum(axis=(1, 2))
    ious = np.where(union == 0, np.where(inter == 0, 1.0, 0.0), inter / np.maximum(union, 1))
    return ious.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--classes-config', required=True)
    ap.add_argument('--no_empty', action='store_true')
    ap.add_argument('--clf', choices=['logistic', 'gbt'], default='logistic')
    ap.add_argument('--iters', type=int, default=60)
    ap.add_argument('--out-json', default=None)
    args = ap.parse_args()

    cc = json.load(open(args.classes_config))
    results = {}
    for cls, spec in cc.items():
        sources = [(s['npz'], s['ch']) for s in spec['sources']]
        nsrc = len(sources)
        imgs, P = load_class_arrays(sources)
        targets = np.stack([(np.array(Image.open(VAL_LBL / cls / f)) > 0).astype(np.uint8) for f in imgs])
        y_empty = np.array([int(targets[i].sum() == 0) for i in range(len(imgs))])
        t = FIXED_T[cls]
        print(f'[{cls}] {len(imgs)} imgs, {nsrc} sources', flush=True)

        rng = random.Random(42)
        best_w, best_s = None, -1
        for it in range(args.iters):
            w = np.array([rng.random() for _ in range(nsrc)])
            w = w / w.sum()
            s = score_weights(w, P, targets, y_empty, t, not args.no_empty, args.clf)
            if s > best_s:
                best_s, best_w = s, w
        # coordinate refinement
        for _ in range(5):
            improved = False
            for j in range(nsrc):
                for k in range(nsrc):
                    if j == k:
                        continue
                    for delta in [0.05, 0.1, 0.2]:
                        w2 = best_w.copy()
                        w2[j] += delta
                        w2[k] -= delta
                        if w2[j] < 0 or w2[k] < 0:
                            continue
                        w2 = w2 / w2.sum()
                        s = score_weights(w2, P, targets, y_empty, t, not args.no_empty, args.clf)
                        if s > best_s:
                            best_s, best_w, improved = s, w2, True
            if not improved:
                break
        weights = {s['name']: round(float(w), 3) for s, w in zip(spec['sources'], best_w)}
        print(f'  -> IoU={best_s:.4f} weights={weights}', flush=True)
        results[cls] = {'iou': best_s, 'weights': weights}

    print('\n=== SUMMARY ===')
    mean = float(np.mean([results[c]['iou'] for c in results]))
    for c in results:
        print(f'{c}: IoU={results[c]["iou"]:.4f}  {results[c]["weights"]}')
    print(f'MEAN = {mean:.4f}')
    if args.out_json:
        json.dump({'mean': mean, 'per_class': results}, open(args.out_json, 'w'), indent=2)
        print(f'saved {args.out_json}')


if __name__ == '__main__':
    main()
