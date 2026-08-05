"""Train/eval an image-level 'contains-crop' classifier from segmentation prob-map features.
Features are computed by empty_features() in infer_ensemble.py. Fits on train, evals on val.
"""
import os, sys, json
import numpy as np
from PIL import Image
from scipy import ndimage
sys.path.insert(0, '/root')

LABEL = '/root/competition_data/public'


def empty_features(pred):
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


def extract(preds_npz, label_dir, out_npy):
    d = np.load(preds_npz)
    imgs = sorted(d.files)
    X, y = [], []
    for f in imgs:
        lab = np.array(Image.open(os.path.join(label_dir, f)))
        y.append(int((lab > 0).sum() == 0))
        X.append(empty_features(d[f].astype(np.float32)))
    X = np.array(X)
    y = np.array(y)
    np.save(out_npy, X)
    np.save(out_npy.replace('_X', '_y'), y)
    return X, y


def logistic(X, y, lr=0.5, iters=2000, l2=1e-3):
    """Numpy logistic regression with standardization."""
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
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--class_name', required=True)
    ap.add_argument('--train_preds')
    ap.add_argument('--val_preds')
    args = ap.parse_args()
    cls = args.class_name

    if args.train_preds:
        Xtr, ytr = extract(args.train_preds, os.path.join(LABEL, 'train/label', cls), f'/root/empty_{cls}_train_X.npy')
    else:
        Xtr = np.load(f'/root/empty_{cls}_train_X.npy'); ytr = np.load(f'/root/empty_{cls}_train_y.npy')
    Xva, yva = extract(args.val_preds, os.path.join(LABEL, 'val/label', cls), f'/root/empty_{cls}_val_X.npy')

    w, mu, sd = logistic(Xtr, ytr)
    pva = predict_proba(Xva, w, mu, sd)
    pred = (pva > 0.5).astype(int)
    acc = (pred == yva).mean()
    # AUC
    order = np.argsort(-pva)
    rank = np.empty_like(order)
    rank[order] = np.arange(len(pva))
    auc = (rank[yva == 1].sum() - len(np.where(yva == 1)[0]) * (len(np.where(yva == 1)[0]) + 1) / 2) / (len(np.where(yva == 1)[0]) * len(np.where(yva == 0)[0]))
    n_empty_total = (yva == 1).sum()
    n_empty_correct = ((yva == 1) & (pred == 1)).sum()
    print(f'[{cls}] clf acc={acc:.4f} auc={auc:.4f} empty correct={n_empty_correct}/{n_empty_total} ({n_empty_correct/n_empty_total*100:.1f}%)')
    np.save(f'/root/empty_{cls}_clf_pred.npy', pred)


if __name__ == '__main__':
    main()
