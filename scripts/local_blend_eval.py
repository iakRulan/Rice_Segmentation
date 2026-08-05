"""Flexible blend + eval on val (CPU). For ensemble-weight experiments.

Each class's prob map is a weighted sum of (npz, channel, weight) entries.
Applies the val-fit empty classifier, then evaluates with threshold/postprocess.

Blend spec JSON:
{
  "wheat": [["ens_multi_val.npz", 0, 0.5], ["ens_ded_wheat_val.npz", 0, 0.5]],
  "rape":  [["ens_multi_val.npz", 1, 0.5], ["ens_ded_rape_val.npz", 0, 0.5]],
  "rice":  [["ens_single_val.npz", 0, 1.0]]
}

Usage:
    python scripts/local_blend_eval.py --spec blends/repro.json [--sweep] [--fixed] [--out_json ...]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import VAL_LBL, PREDS

SETTINGS = {
    'wheat': dict(t=0.45, min_area=0, max_hole=60),
    'rape': dict(t=0.55, min_area=100, max_hole=100),
    'rice': dict(t=0.53, min_area=200, max_hole=200),
}
PP_GRID = [(0, 0), (30, 30), (60, 60), (100, 100), (60, 0), (0, 60),
           (120, 60), (60, 120), (200, 200)]

_npz_cache = {}


def load_npz(path):
    p = PREDS / path
    if str(p) not in _npz_cache:
        _npz_cache[str(p)] = np.load(p)
    return _npz_cache[str(p)]


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


def logistic(X, y, iters=3000, lr=0.3, l2=1e-3):
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


def pp(mask, min_area, max_hole):
    if min_area > 0:
        labeled, n = ndimage.label(mask)
        if n:
            areas = ndimage.sum(mask, labeled, range(1, n + 1))
            for i in range(1, n + 1):
                if areas[i - 1] < min_area:
                    mask[labeled == i] = 0
    if max_hole > 0:
        inv = 1 - mask
        labeled, n = ndimage.label(inv)
        if n:
            areas = ndimage.sum(inv, labeled, range(1, n + 1))
            for i in range(1, n + 1):
                if areas[i - 1] < max_hole:
                    mask[labeled == i] = 1
    return mask


def iou(pred_bin, tgt_bin):
    inter = np.logical_and(pred_bin, tgt_bin).sum()
    union = np.logical_or(pred_bin, tgt_bin).sum()
    return 1.0 if union == 0 and inter == 0 else (0.0 if union == 0 else inter / union)


def load_targets(imgs, cls):
    return {f: (np.array(Image.open(VAL_LBL / cls / f)) > 0).astype(np.uint8) for f in imgs}


def blend_class(spec):
    """Return (imgs, probs) for a class given blend spec entries."""
    first_imgs = None
    acc = {}
    for path, ch, w in spec:
        d = load_npz(path)
        imgs = sorted(d.files)
        if first_imgs is None:
            first_imgs = imgs
        else:
            assert imgs == first_imgs, f'{path} image mismatch'
        for f in imgs:
            p = d[f].astype(np.float32)
            if p.ndim == 2:
                p = p[None]
            contrib = p[ch] * w
            acc[f] = acc.get(f, np.zeros_like(contrib)) + contrib
    return first_imgs, acc


def fit_empty_model(imgs, probs, cls, clf='logistic'):
    """Fit empty classifier on `imgs`, return (predict_fn, threshold).
    predict_fn(prob_map) -> 0/1 (1 = empty)."""
    X = np.array([empty_features(probs[f]) for f in imgs])
    y = np.array([int((np.array(Image.open(VAL_LBL / cls / f)) > 0).sum() == 0) for f in imgs])
    if clf == 'gbt':
        from sklearn.ensemble import GradientBoostingClassifier
        m = GradientBoostingClassifier(n_estimators=200, max_depth=3).fit(X, y)
        p = m.predict_proba(X)[:, 1]
        pred = lambda x: m.predict_proba(empty_features(x)[None])[:, 1][0]
    else:
        w, mu, sd = logistic(X, y)
        p = proba(X, w, mu, sd)
        pred = lambda x: proba(empty_features(x)[None], w, mu, sd)[0]
    best, bt = 0.0, 0.5
    for th in np.arange(0.1, 0.9, 0.05):
        acc = ((p > th).astype(int) == y).mean()
        if acc > best:
            best, bt = acc, th
    ec = (((p > bt).astype(int) == 1) & (y == 1)).sum() / max(1, (y == 1).sum())
    ne = (((p > bt).astype(int) == 1) & (y == 0)).sum() / max(1, (y == 0).sum())
    print(f'    [empty {cls}] {clf} fit acc={best:.4f} th={bt:.2f} empty_correct={ec:.3f} nonempty_err={ne:.3f}')
    return (pred, float(bt))


def fit_empty(imgs, probs, cls, clf='logistic'):
    pred, bt = fit_empty_model(imgs, probs, cls, clf)
    return {f: int(pred(probs[f]) > bt) for f in imgs}


def valhold_imgs(cls):
    """Every-7th sorted val image (the server's valhold split), as a name set."""
    imgs = sorted(f.name for f in (VAL_LBL / cls).iterdir() if f.suffix == '.png')
    return {f for i, f in enumerate(imgs) if i % 7 == 0}


def evaluate(imgs, probs, cls, targets, t, min_area, max_hole, empty_map):
    ious = []
    for f in imgs:
        m = (probs[f] > t).astype(np.uint8)
        if empty_map is not None and empty_map[f] == 1:
            m = np.zeros_like(m)
        m = pp(m, min_area, max_hole)
        ious.append(iou(m, targets[f]))
    return np.array(ious)


def summarize(ious, targets):
    ne = np.array([1 if targets[f].sum() > 0 else 0 for f in targets], bool)
    return (float(ious.mean()),
            float(ious[~ne].mean()) if (~ne).sum() else 1.0,
            float(ious[ne].mean()) if ne.sum() else 1.0)


def evaluate_fast(probs_arr, targets_arr, empty_arr, t, min_area, max_hole):
    """probs_arr (N,256,256), targets_arr (N,256,256) uint8, empty_arr (N,) int.
    Vectorized IoU; postprocess only when min_area/max_hole > 0."""
    N = probs_arr.shape[0]
    m = (probs_arr > t).astype(np.uint8)
    if empty_arr is not None:
        m[empty_arr == 1] = 0
    if min_area > 0 or max_hole > 0:
        for i in range(N):
            m[i] = pp(m[i], min_area, max_hole)
    inter = np.logical_and(m, targets_arr).sum(axis=(1, 2))
    union = np.logical_or(m, targets_arr).sum(axis=(1, 2))
    ious = np.where(union == 0, np.where(inter == 0, 1.0, 0.0), inter / np.maximum(union, 1))
    return ious


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--spec', required=True)
    ap.add_argument('--no_empty', action='store_true')
    ap.add_argument('--sweep', action='store_true')
    ap.add_argument('--fixed', action='store_true')
    ap.add_argument('--out_json', default=None)
    ap.add_argument('--subset', choices=['val', 'valhold'], default='val',
                    help='val = full val (v4 proxy); valhold = every-7th (fair for s7)')
    ap.add_argument('--t_step', type=float, default=0.02, help='threshold grid step')
    ap.add_argument('--clf', choices=['logistic', 'gbt'], default='logistic',
                    help='empty-classifier model (gbt is more robust for submission)')
    ap.add_argument('--holdout', action='store_true',
                    help='honest eval: fit empty-clf+threshold on even half of val, eval on odd half')
    args = ap.parse_args()

    spec = json.load(open(args.spec))
    results = {}
    for cls, entries in spec.items():
        print(f'[{cls}] blend={entries}')
        imgs, probs = blend_class(entries)
        if args.subset == 'valhold':
            keep = valhold_imgs(cls)
            imgs = [f for f in imgs if f in keep]
            probs = {f: probs[f] for f in imgs}
            print(f'    (subset=valhold n={len(imgs)})')
        targets = load_targets(imgs, cls)
        if args.holdout:
            # honest holdout: fit empty-clf + threshold on even half, eval on odd half
            fit_imgs = imgs[0::2]
            ev_imgs = imgs[1::2]
            pred, bt = None, None
            if not args.no_empty:
                pred, bt = fit_empty_model(fit_imgs, probs, cls, args.clf)
            fit_probs = np.stack([probs[f] for f in fit_imgs])
            fit_tgts = np.stack([targets[f] for f in fit_imgs])
            fit_empty_arr = None if pred is None else np.array([int(pred(probs[f]) > bt) for f in fit_imgs])
            ts = list(np.arange(0.15, 0.85, 0.02))
            best_t = max(((evaluate_fast(fit_probs, fit_tgts, fit_empty_arr, t, 0, 0).mean(), t)
                          for t in ts), key=lambda x: x[0])[1]
            ev_probs = np.stack([probs[f] for f in ev_imgs])
            ev_tgts = np.stack([targets[f] for f in ev_imgs])
            ev_map = None if pred is None else np.array([int(pred(probs[f]) > bt) for f in ev_imgs])
            ious = evaluate_fast(ev_probs, ev_tgts, ev_map, best_t, 0, 0)
            tot, emp, ne = summarize(ious, {f: targets[f] for f in ev_imgs})
            print(f'    [holdout] IoU={tot:.4f} empty={emp:.4f} nonempty={ne:.4f} @t={best_t:.2f} (fit {len(fit_imgs)}, eval {len(ev_imgs)})')
            results[cls] = dict(method='holdout', settings=dict(t=best_t, min_area=0, max_hole=0),
                                iou=tot, empty=emp, nonempty=ne)
            continue
        empty_map = None if args.no_empty else fit_empty(imgs, probs, cls, args.clf)
        # prebuild arrays
        probs_arr = np.stack([probs[f] for f in imgs])                 # (N,256,256)
        targets_arr = np.stack([targets[f] for f in imgs])             # (N,256,256)
        empty_arr = None if empty_map is None else np.array([empty_map[f] for f in imgs])
        if args.fixed or not args.sweep:
            s = SETTINGS[cls]
            ious = evaluate_fast(probs_arr, targets_arr, empty_arr, s['t'], s['min_area'], s['max_hole'])
            tot, emp, ne = summarize(ious, targets)
            print(f'    [fixed] IoU={tot:.4f} empty={emp:.4f} nonempty={ne:.4f} @t={s["t"]} pp=({s["min_area"]},{s["max_hole"]})')
            results[cls] = dict(method='fixed', settings=s, iou=tot, empty=emp, nonempty=ne)
        if args.sweep:
            ts = list(np.arange(0.15, 0.85, args.t_step))
            # phase 1: PP=(0,0) full grid (fast, no labeling)
            best0 = max(((evaluate_fast(probs_arr, targets_arr, empty_arr, t, 0, 0).mean(), t)
                         for t in ts), key=lambda x: x[0])
            # phase 2: local threshold grid around best0 for each PP config
            t0 = best0[1]
            local_ts = [t for t in ts if abs(t - t0) <= 0.12] or [t0]
            best = (-1, None)
            for min_area, max_hole in PP_GRID:
                for t in local_ts:
                    ious = evaluate_fast(probs_arr, targets_arr, empty_arr, t, min_area, max_hole)
                    if ious.mean() > best[0]:
                        best = (float(ious.mean()), (float(t), min_area, max_hole))
            tot, emp, ne = summarize(evaluate_fast(probs_arr, targets_arr, empty_arr,
                                                   best[1][0], best[1][1], best[1][2]), targets)
            print(f'    [sweep ] IoU={tot:.4f} empty={emp:.4f} nonempty={ne:.4f} @t={best[1][0]:.2f} pp=({best[1][1]},{best[1][2]})')
            results[cls] = dict(method='sweep', settings=dict(t=best[1][0], min_area=best[1][1], max_hole=best[1][2]),
                                iou=tot, empty=emp, nonempty=ne)

    mean = float(np.mean([results[c]['iou'] for c in results]))
    print(f'\n[MEAN IoU] {mean:.4f}  ' + '  '.join(f'{c}={results[c]["iou"]:.4f}' for c in results))
    if args.out_json:
        json.dump({'mean': mean, 'per_class': {c: results[c] for c in results}},
                  open(args.out_json, 'w'), indent=2)
        print(f'saved {args.out_json}')


if __name__ == '__main__':
    main()
