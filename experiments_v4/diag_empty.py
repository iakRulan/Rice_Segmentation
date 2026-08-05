"""Diagnose empty-classifier misclassifications on the v4 blend probs.
For each class, list empty images predicted non-empty (and their IoU), and vice versa.
"""
import sys
import numpy as np
from PIL import Image
from scipy import ndimage

VAL = '/root/competition_data/public/val'


def features(pred):
    flat = pred.reshape(-1)
    feats = [flat.max(), np.percentile(flat, 99), np.percentile(flat, 95),
             np.percentile(flat, 90), flat.mean(), (flat > 0.1).sum(), (flat > 0.3).sum(),
             (flat > 0.5).sum(), (flat > 0.7).sum()]
    bin5 = (flat > 0.5).astype(np.uint8).reshape(pred.shape[-2:])
    labeled, n = ndimage.label(bin5)
    if n > 0:
        areas = ndimage.sum(bin5, labeled, range(1, n + 1))
        feats += [n, float(areas.max()), float(areas.sum())]
    else:
        feats += [0, 0.0, 0.0]
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


def iou(pred, tgt):
    inter = np.logical_and(pred, tgt).sum(); union = np.logical_or(pred, tgt).sum()
    return 1.0 if union == 0 and inter == 0 else (0.0 if union == 0 else inter / union)


def run(npz, ch, cls, t, clf_th):
    d = np.load(npz)
    imgs = sorted(d.files)
    X = []; y = []; pms = {}
    for f in imgs:
        p = d[f].astype(np.float32)
        p = p[ch] if p.ndim == 3 else p
        pms[f] = p
        lab = np.array(Image.open(f'{VAL}/label/{cls}/{f}'))
        y.append(int((lab > 0).sum() == 0))
        X.append(features(p))
    X = np.array(X); y = np.array(y)
    w, mu, sd = logistic(X, y)
    p = proba(X, w, mu, sd)
    pred = (p > clf_th).astype(int)
    print(f'[{cls}] n={len(imgs)} empty_true={y.sum()} pred_empty={(pred==1).sum()}')
    miss_empty = []  # true empty, predicted non-empty
    miss_ne = []     # true non-empty, predicted empty
    for i, f in enumerate(imgs):
        if y[i] == 1 and pred[i] == 0:
            lab = np.array(Image.open(f'{VAL}/label/{cls}/{f}'))
            m = (pms[f] > t).astype(np.uint8)
            iou_ = iou(m, lab)
            miss_empty.append((f, iou_, pms[f].max()))
        elif y[i] == 0 and pred[i] == 1:
            lab = np.array(Image.open(f'{VAL}/label/{cls}/{f}'))
            miss_ne.append((f, (lab > 0).sum()))
    print(f'  true-empty but predicted non-empty: {len(miss_empty)} (their IoU if not zeroed)')
    for f, iou_, mx in sorted(miss_empty, key=lambda x: -x[1])[:8]:
        print(f'    {f} iou_if_predicted={iou_:.3f} maxprob={mx:.3f}')
    print(f'  true-nonempty but predicted empty (killed): {len(miss_ne)}')
    for f, a in miss_ne[:8]:
        print(f'    {f} gt_area={a}')


run('/root/ens_multi_v4b.npz', 0, 'wheat', 0.45, 0.29)
run('/root/ens_multi_v4b.npz', 1, 'rape', 0.55, 0.40)
run('/root/ens_single_v4.npz', 0, 'rice', 0.53, 0.50)
