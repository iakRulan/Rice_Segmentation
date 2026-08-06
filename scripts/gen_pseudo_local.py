"""Generate confident testA pseudo-labels for self-training.

Uses the final blend probs (testA) + val-fit empty classifier. Keeps images
that are confidently non-empty with max prob above a threshold. Saves to
data/pseudo/{image,label}/{wheat_rape,rice} with ta_ prefix.

Training on train + these pseudo-labels (NOT val) keeps the val signal honest.

Usage:
    python scripts/gen_pseudo_local.py --spec configs/blend_fix.json --conf 0.8
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import VAL_LBL, TESTA_IMG, PREDS, DATA, TRAIN_IMG, TRAIN_LBL

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


def fit_empty_model(imgs, probs, cls, clf='logistic'):
    X = np.array([empty_features(probs[f]) for f in imgs])
    y = np.array([int((np.array(Image.open(VAL_LBL / cls / f)) > 0).sum() == 0) for f in imgs])
    if clf == 'gbt':
        from sklearn.ensemble import GradientBoostingClassifier
        m = GradientBoostingClassifier(n_estimators=150, max_depth=3).fit(X, y)
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
    print(f'  [empty {cls}] {clf} acc={best:.3f} th={bt:.2f}')
    return (pred, float(bt))


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


SETTINGS = {
    'wheat': dict(t=0.55, min_area=30, max_hole=30),
    'rape': dict(t=0.55, min_area=0, max_hole=60),
    'rice': dict(t=0.53, min_area=200, max_hole=200),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--spec', required=True)
    ap.add_argument('--conf', type=float, default=0.8, help='min max-prob to keep')
    args = ap.parse_args()

    spec = json.load(open(args.spec))
    pseudo = DATA / 'pseudo'
    for c in ['wheat', 'rape', 'rice']:
        (pseudo / 'label' / c).mkdir(parents=True, exist_ok=True)
    for c in ['wheat_rape', 'rice']:
        (pseudo / 'image' / c).mkdir(parents=True, exist_ok=True)

    # fit empty clf on val
    clfs = {}
    for cls, entries in spec.items():
        v_imgs, v_probs = blend(entries, 'val')
        clf = 'logistic' if cls == 'rice' else 'gbt'
        clfs[cls] = fit_empty_model(v_imgs, v_probs, cls, clf)

    # testA probs
    ta = {}
    for cls, entries in spec.items():
        imgs, probs = blend(entries, 'testA')
        ta[cls] = (imgs, probs)

    all_imgs = ta['wheat'][0]
    kept = {c: 0 for c in ['wheat', 'rape', 'rice']}
    for f in all_imgs:
        # wheat/rape share image
        for cls, (imgs, probs) in ta.items():
            if cls == 'rice':
                continue
            s = SETTINGS[cls]
            p = probs[f]
            em = clfs[cls][0](p) > clfs[cls][1]
            if em or p.max() < args.conf:
                continue
            m = (p > s['t']).astype(np.uint8)
            m = pp(m, s['min_area'], s['max_hole'])
            if m.sum() == 0:
                continue
            Image.fromarray(m * 255).save(pseudo / 'label' / cls / f'ta_{f}')
            shutil.copy2(TESTA_IMG / 'wheat_rape' / f, pseudo / 'image' / 'wheat_rape' / f'ta_{f}')
            kept[cls] += 1
        # rice
        cls = 'rice'
        s = SETTINGS[cls]
        p = ta['rice'][1][f]
        em = clfs['rice'][0](p) > clfs['rice'][1]
        if em or p.max() < args.conf:
            continue
        m = (p > s['t']).astype(np.uint8)
        m = pp(m, s['min_area'], s['max_hole'])
        if m.sum() == 0:
            continue
        Image.fromarray(m * 255).save(pseudo / 'label' / 'rice' / f'ta_{f}')
        shutil.copy2(TESTA_IMG / 'rice' / f, pseudo / 'image' / 'rice' / f'ta_{f}')
        kept['rice'] += 1

    print(f'kept pseudo-labels: {kept}')
    for c in ['wheat', 'rape', 'rice']:
        print(f'  {c}: {len(list((pseudo/"label"/c).iterdir()))} labels')


if __name__ == '__main__':
    main()
