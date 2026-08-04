"""Retrain empty-image logistic clf per channel, eval on val.
Usage: clf2.py --class_name wheat --channel 0 --train_preds ens_multi_train.npz --val_preds ens_multi_v2.npz [--out preds.npy]
If --train_preds omitted, reuse /root/empty_{cls}_train_X.npy / _y.npy if present.
"""
import os, sys, argparse
import numpy as np
from PIL import Image
from scipy import ndimage

LABEL = '/root/competition_data/public'

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
    return feats

def extract(npz, label_dir, ch):
    d = np.load(npz)
    imgs = sorted(d.files)
    X, y = [], []
    for f in imgs:
        lab = np.array(Image.open(os.path.join(label_dir, f)))
        y.append(int((lab > 0).sum() == 0))
        p = d[f].astype(np.float32)
        p = p[ch] if p.ndim == 3 else p
        X.append(features(p))
    return np.array(X), np.array(y)

def logistic(X, y, lr=0.5, iters=3000, l2=1e-3):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xs = (X - mu) / sd
    Xb = np.hstack([np.ones((len(X), 1)), Xs])
    w = np.zeros(Xb.shape[1])
    for _ in range(iters):
        z = Xb @ w
        p = 1 / (1 + np.exp(-np.clip(z, -30, 30)))
        g = Xb.T @ (p - y) / len(y) + l2 * w
        w -= lr * g
    return w, mu, sd

def predict_proba(X, w, mu, sd):
    Xs = (X - mu) / sd
    Xb = np.hstack([np.ones((len(X), 1)), Xs])
    return 1 / (1 + np.exp(-np.clip(Xb @ w, -30, 30)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--class_name', required=True)
    ap.add_argument('--channel', type=int, default=0)
    ap.add_argument('--train_preds', default=None)
    ap.add_argument('--val_preds', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    cls = args.class_name

    if args.train_preds and os.path.exists(args.train_preds):
        Xtr, ytr = extract(args.train_preds, os.path.join(LABEL, 'train/label', cls), args.channel)
    else:
        Xtr = np.load(f'/root/empty_{cls}_train_X.npy'); ytr = np.load(f'/root/empty_{cls}_train_y.npy')
        # old features were extracted per-channel already; trust them
    Xva, yva = extract(args.val_preds, os.path.join(LABEL, 'val/label', cls), args.channel)
    w, mu, sd = logistic(Xtr, ytr)
    pva = predict_proba(Xva, w, mu, sd)
    pred = (pva > 0.5).astype(int)
    acc = (pred == yva).mean()
    order = np.argsort(-pva)
    rank = np.empty_like(order); rank[order] = np.arange(len(pva))
    n1 = (yva == 1).sum(); n0 = (yva == 0).sum()
    auc = (rank[yva == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else float('nan')
    ne_tot = (yva == 1).sum(); ne_ok = ((yva == 1) & (pred == 1)).sum()
    print(f'[{cls}] ch{args.channel} clf acc={acc:.4f} auc={auc:.4f} empty ok={ne_ok}/{ne_tot} ({100*ne_ok/max(1,ne_tot):.1f}%)', flush=True)
    np.save(args.out, pred)

if __name__ == '__main__':
    main()
