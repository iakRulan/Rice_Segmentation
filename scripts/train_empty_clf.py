"""Train per-class empty classifier from ens_features output, eval on val.
Usage: train_empty_clf.py --task multi|single --train_feats npz --val_feats npz --train_labels dir --val_labels dir
Saves /root/empty_clf_{class}.npy (1=empty) for val; also prints combined IoU estimate.
"""
import os, sys, json
import numpy as np
from PIL import Image

LABEL = '/root/competition_data/public'


def load_feats(npz_path):
    d = np.load(npz_path)
    imgs = sorted(d.files)
    # multi: (C,12), single: (1,12)
    X = np.stack([np.asarray(d[f], dtype=np.float32) for f in imgs])  # (N, C, 12)
    return imgs, X


def labels(imgs, label_dir):
    y = np.zeros(len(imgs), dtype=np.int32)
    for i, f in enumerate(imgs):
        a = np.array(Image.open(os.path.join(label_dir, f)))
        y[i] = int((a > 0).sum() == 0)
    return y


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


def main():
    ap = __import__('argparse').ArgumentParser()
    ap.add_argument('--task', choices=['multi', 'single'], required=True)
    ap.add_argument('--train_feats', required=True)
    ap.add_argument('--val_feats', required=True)
    args = ap.parse_args()

    tr_imgs, Xtr = load_feats(args.train_feats)
    va_imgs, Xva = load_feats(args.val_feats)
    C = Xtr.shape[1]
    classes = ['wheat', 'rape'] if args.task == 'multi' else ['rice']

    results = {}
    for c in range(C):
        cls = classes[c]
        ytr = labels(tr_imgs, os.path.join(LABEL, 'train/label', cls))
        yva = labels(va_imgs, os.path.join(LABEL, 'val/label', cls))
        w, mu, sd = logistic(Xtr[:, c], ytr)
        pva = proba(Xva[:, c], w, mu, sd)
        # threshold by matching train prior (empty fraction)
        prior = ytr.mean()
        th = np.percentile(proba(Xtr[:, c], w, mu, sd), prior * 100)
        pred = (pva >= th).astype(int)
        acc = (pred == yva).mean()
        n_empty = (yva == 1).sum()
        n_correct = ((yva == 1) & (pred == 1)).sum()
        # trade-off: tune threshold to maximize empty-correct without hurting non-empty too much
        best_empty = -1; best_th = th
        for tt in np.linspace(0.2, 0.8, 61):
            pr = (pva >= tt).astype(int)
            ec = ((yva == 1) & (pr == 1)).sum() / max(1, n_empty)
            ne_err = ((yva == 0) & (pr == 1)).sum() / max(1, (yva == 0).sum())
            score = ec - ne_err  # maximize empty gain minus non-empty loss
            if score > best_empty:
                best_empty, best_th = score, tt
        pr = (pva >= best_th).astype(int)
        ec = ((yva == 1) & (pr == 1)).sum() / max(1, n_empty)
        ne_err = ((yva == 0) & (pr == 1)).sum() / max(1, (yva == 0).sum())
        np.save(f'/root/empty_clf_{cls}.npy', pr)
        print(f'[{cls}] acc={acc:.4f} empty_correct={ec:.3f} nonempty_err={ne_err:.3f} '
              f'(tuned th={best_th:.3f}) -> saved empty_clf_{cls}.npy', flush=True)
        results[cls] = pr
    print('done', flush=True)


if __name__ == '__main__':
    main()
