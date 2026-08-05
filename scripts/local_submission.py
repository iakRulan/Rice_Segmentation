"""Generate testA submission masks from blended prob maps (CPU).

1. Blend each class prob = weighted sum of (npz, channel, weight) entries.
   npz names in the spec refer to the val run; the testA npz is found by
   swapping 'val' -> 'testA' in the filename (e.g. ens_multi_val.npz ->
   ens_multi_testA.npz).
2. Fit the empty-image classifier on VAL features (same blend), apply to testA.
3. Threshold + postprocess (min_area / max_hole), write masks.

Usage:
    python scripts/local_submission.py --spec configs/blend_repro.json --out_dir outputs/submission \
        [--settings '{"wheat": {"t":0.45,"min_area":0,"max_hole":60}, ...}']
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import VAL_LBL, TESTA_IMG, PREDS, MASKS

# swept on val with blend_v3 (see outputs/logs/v3_sweep.json)
SETTINGS = {
    'wheat': dict(t=0.51, min_area=0, max_hole=60),
    'rape': dict(t=0.47, min_area=0, max_hole=60),
    'rice': dict(t=0.37, min_area=60, max_hole=0),
}
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


def fit_gbt(imgs, probs, cls):
    """GBT empty classifier (beat logistic on wheat/rape in holdout tests).
    Returns (model, proba_fn, th)."""
    from sklearn.ensemble import GradientBoostingClassifier
    X = np.array([empty_features(probs[f]) for f in imgs])
    y = np.array([int((np.array(Image.open(VAL_LBL / cls / f)) > 0).sum() == 0) for f in imgs])
    m = GradientBoostingClassifier(n_estimators=200, max_depth=3).fit(X, y)
    p = m.predict_proba(X)[:, 1]
    best, bt = 0.0, 0.5
    for th in np.arange(0.2, 0.8, 0.05):
        acc = ((p > th).astype(int) == y).mean()
        if acc > best:
            best, bt = acc, th
    ec = (((p > bt).astype(int) == 1) & (y == 1)).sum() / max(1, (y == 1).sum())
    ne = (((p > bt).astype(int) == 1) & (y == 0)).sum() / max(1, (y == 0).sum())
    print(f'    [empty {cls}] GBT val acc={best:.4f} th={bt:.2f} empty_correct={ec:.3f} nonempty_err={ne:.3f}')
    return (m, lambda x: m.predict_proba(x)[:, 1], bt)


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


def blend(spec, split):
    """Return (imgs, probs) blending entries; imgs from first npz."""
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


def fit_empty(imgs, probs, cls):
    X = np.array([empty_features(probs[f]) for f in imgs])
    y = np.array([int((np.array(Image.open(VAL_LBL / cls / f)) > 0).sum() == 0) for f in imgs])
    w, mu, sd = logistic(X, y)
    p = proba(X, w, mu, sd)
    best, bt = 0.0, 0.5
    for th in np.arange(0.1, 0.9, 0.05):
        if ((p > th).astype(int) == y).mean() > best:
            best, bt = ((p > th).astype(int) == y).mean(), th
    print(f'    [empty {cls}] val acc={best:.4f} th={bt:.2f}')
    return (w, mu, sd, bt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--spec', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--settings', default=None, help='JSON dict overriding SETTINGS')
    args = ap.parse_args()

    spec = json.load(open(args.spec))
    settings = SETTINGS
    if args.settings:
        settings = json.loads(args.settings)

    out = Path(args.out_dir)
    for c in settings:
        (out / c).mkdir(parents=True, exist_ok=True)

    # fit empty clf on val (per class); GBT for wheat/rape, logistic for rice
    clfs = {}
    for cls, entries in spec.items():
        v_imgs, v_probs = blend(entries, 'val')
        clf = 'logistic' if cls == 'rice' else 'gbt'
        clfs[cls] = fit_gbt(v_imgs, v_probs, cls) if clf == 'gbt' else fit_logistic(v_imgs, v_probs, cls)

    # testA inference via blend
    all_imgs = None
    ta_probs = {}
    for cls, entries in spec.items():
        imgs, probs = blend(entries, 'testA')
        ta_probs[cls] = probs
        if all_imgs is None:
            all_imgs = imgs
        else:
            assert imgs == all_imgs

    # determine image subdirs: rice uses rice/, others use wheat_rape/
    wheat_rape_dir = TESTA_IMG / 'wheat_rape'
    rice_dir = TESTA_IMG / 'rice'
    n = 0
    for f in all_imgs:
        for cls, probs in ta_probs.items():
            s = settings[cls]
            p = probs[f]
            em = clfs[cls][1](empty_features(p)[None])[0] > clfs[cls][2]
            m = (p > s['t']).astype(np.uint8)
            if em:
                m = np.zeros_like(m)
            m = pp(m, s['min_area'], s['max_hole'])
            Image.fromarray(m * 255).save(out / cls / f)
        n += 1
        if n % 200 == 0:
            print(f'  {n}/{len(all_imgs)}', flush=True)
    print(f'saved submission to {out} ({len(all_imgs)} images)', flush=True)


if __name__ == '__main__':
    main()
