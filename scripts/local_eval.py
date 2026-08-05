"""Local val evaluation: blend + empty-clf + threshold/postprocess sweep -> per-image IoU.

CPU only, fast.  Mirrors the server pipeline (v4 multi + dedicated blend, val-fit
empty classifier, per-class threshold/postprocess) and adds diagnostics:
  - per-image IoU mean (the competition metric)
  - split into empty-image IoU vs non-empty-image IoU (for bottleneck analysis)

Usage:
    python scripts/local_eval.py \
        --multi outputs/preds/ens_multi_val.npz \
        --single outputs/preds/ens_single_val.npz \
        [--ded_wheat ...] [--ded_rape ...] [--blend_w 0.5] \
        [--sweep] [--fixed] [--out_json outputs/logs/eval_val.json]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import VAL_LBL

# submission settings (from make_testA_submission.py)
SETTINGS = {
    'wheat': dict(t=0.45, min_area=0, max_hole=60),
    'rape': dict(t=0.55, min_area=100, max_hole=100),
    'rice': dict(t=0.53, min_area=200, max_hole=200),
}
PP_GRID = [(0, 0), (30, 30), (60, 60), (100, 100), (60, 0), (0, 60),
           (120, 60), (60, 120), (200, 200)]


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


def load_probs(path, channel):
    d = np.load(path)
    imgs = sorted(d.files)
    if channel is None:
        return imgs, {f: d[f].astype(np.float32) for f in imgs}
    return imgs, {f: d[f][channel].astype(np.float32) for f in imgs}


def load_targets(imgs, cls):
    d = VAL_LBL / cls
    return {f: (np.array(Image.open(d / f)) > 0).astype(np.uint8) for f in imgs}


def fit_empty(imgs, probs, cls):
    """Fit empty clf on val features (same as submission pipeline), return threshold."""
    X = np.array([empty_features(probs[f]) for f in imgs])
    y = np.array([int((np.array(Image.open(VAL_LBL / cls / f)) > 0).sum() == 0) for f in imgs])
    w, mu, sd = logistic(X, y)
    p = proba(X, w, mu, sd)
    best, bt = 0.0, 0.5
    for th in np.arange(0.1, 0.9, 0.05):
        acc = ((p > th).astype(int) == y).mean()
        if acc > best:
            best, bt = acc, th
    pred = (p > bt).astype(int)
    n_empty = (y == 1).sum()
    ec = ((y == 1) & (pred == 1)).sum() / max(1, n_empty)
    ne = ((y == 0) & (pred == 1)).sum() / max(1, (y == 0).sum())
    print(f'    [empty {cls}] acc={best:.4f} th={bt:.2f} empty_correct={ec:.3f} nonempty_err={ne:.3f}')
    return {f: int((proba(empty_features(probs[f])[None], w, mu, sd)[0] > bt)) for f in imgs}


def evaluate(imgs, probs, cls, targets, t, min_area, max_hole, empty_map):
    ious = []
    for f in imgs:
        m = (probs[f] > t).astype(np.uint8)
        if empty_map is not None and empty_map[f] == 1:
            m = np.zeros_like(m)
        m = pp(m, min_area, max_hole)
        ious.append(iou(m, targets[f]))
    ious = np.array(ious)
    return ious


def summarize(ious, targets):
    nonempty = [1 if targets[f].sum() > 0 else 0 for f in targets]
    nonempty = np.array(nonempty, bool)
    return (float(ious.mean()), float(ious[~nonempty].mean()) if (~nonempty).sum() else 1.0,
            float(ious[nonempty].mean()) if nonempty.sum() else 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--multi', required=True, help='multi ensemble probs on val (wheat/rape)')
    ap.add_argument('--single', required=True, help='rice ensemble probs on val')
    ap.add_argument('--ded_wheat', default=None)
    ap.add_argument('--ded_rape', default=None)
    ap.add_argument('--blend_w', type=float, default=0.5)
    ap.add_argument('--no_empty', action='store_true', help='disable empty classifier')
    ap.add_argument('--sweep', action='store_true', help='grid sweep t x postprocess')
    ap.add_argument('--fixed', action='store_true', help='use submission SETTINGS')
    ap.add_argument('--out_json', default=None)
    args = ap.parse_args()

    mw_imgs, mw = load_probs(args.multi, 0)      # wheat channel
    mr_imgs, mr = load_probs(args.multi, 1)      # rape channel
    ri_imgs, ri = load_probs(args.single, 0)     # rice

    if args.ded_wheat:
        _, dw = load_probs(args.ded_wheat, 0)
        mw = {f: args.blend_w * mw[f] + (1 - args.blend_w) * dw[f] for f in mw}
    if args.ded_rape:
        _, dr = load_probs(args.ded_rape, 0)
        mr = {f: args.blend_w * mr[f] + (1 - args.blend_w) * dr[f] for f in mr}

    classes = {'wheat': (mw_imgs, mw), 'rape': (mr_imgs, mr), 'rice': (ri_imgs, ri)}
    results = {}
    for cls, (imgs, probs) in classes.items():
        targets = load_targets(imgs, cls)
        empty_map = None if args.no_empty else fit_empty(imgs, probs, cls)
        print(f'  [{cls}] n={len(imgs)}')

        if args.fixed or not args.sweep:
            s = SETTINGS[cls]
            ious = evaluate(imgs, probs, cls, targets, s['t'], s['min_area'], s['max_hole'], empty_map)
            tot, emp, ne = summarize(ious, targets)
            print(f'    [fixed] IoU={tot:.4f}  empty={emp:.4f}  nonempty={ne:.4f}  @t={s["t"]} pp=({s["min_area"]},{s["max_hole"]})')
            results[cls] = dict(method='fixed', settings=s, iou=tot, empty=emp, nonempty=ne)

        if args.sweep:
            best = (-1, None)
            for min_area, max_hole in PP_GRID:
                for t in np.arange(0.15, 0.85, 0.02):
                    ious = evaluate(imgs, probs, cls, targets, float(t), min_area, max_hole, empty_map)
                    if ious.mean() > best[0]:
                        best = (float(ious.mean()), (float(t), min_area, max_hole))
            tot, emp, ne = summarize(evaluate(imgs, probs, cls, targets,
                                              best[1][0], best[1][1], best[1][2], empty_map), targets)
            print(f'    [sweep ] IoU={tot:.4f}  empty={emp:.4f}  nonempty={ne:.4f}  @t={best[1][0]:.2f} pp=({best[1][1]},{best[1][2]})')
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
