
"""Sweep empty-clf threshold (pva>th) combined with main threshold/pp grid. CPU."""
import sys, os
import numpy as np
from PIL import Image
from scipy import ndimage
sys.path.insert(0, '/root')
from clf2 import extract, logistic, predict_proba

VAL = '/root/competition_data/public/val'

def iou(pred, tgt):
    inter = np.logical_and(pred, tgt).sum(); union = np.logical_or(pred, tgt).sum()
    return 1.0 if union == 0 and inter == 0 else (0.0 if union == 0 else inter/union)

def pp(mask, min_area, max_hole):
    if min_area > 0:
        labeled, n = ndimage.label(mask)
        if n:
            areas = ndimage.sum(mask, labeled, range(1, n+1))
            for i in range(1, n+1):
                if areas[i-1] < min_area: mask[labeled == i] = 0
    if max_hole > 0:
        inv = 1 - mask; labeled, n = ndimage.label(inv)
        if n:
            areas = ndimage.sum(inv, labeled, range(1, n+1))
            for i in range(1, n+1):
                if areas[i-1] < max_hole: mask[labeled == i] = 1
    return mask

def main():
    cls = sys.argv[1]; npz = sys.argv[2]; ch = int(sys.argv[3])
    Xtr = np.load(f'/root/empty_{cls}_train_X.npy'); ytr = np.load(f'/root/empty_{cls}_train_y.npy')
    d = np.load(npz); imgs = sorted(d.files)
    Xva, yva = extract(npz, f'{VAL}/label/{cls}', ch)
    w, mu, sd = logistic(Xtr, ytr)
    pva = predict_proba(Xva, w, mu, sd)
    targets = {}
    for f in imgs:
        lab = np.array(Image.open(f'{VAL}/label/{cls}/{f}'))
        targets[f] = (lab > 0).astype(np.uint8)
    preds = {f: (d[f][ch] if d[f].ndim == 3 else d[f]).astype(np.float32) for f in imgs}
    print(f'[{cls}] n={len(imgs)} empty_true={yva.sum()} nonempty={len(imgs)-yva.sum()}')
    configs = [(0,0),(30,30),(60,60),(100,100),(120,60),(200,200),(60,120)]
    best = (-1, None, None, None)
    for th in np.arange(0.30, 0.96, 0.05):
        empty_map = (pva > th).astype(int)
        ne_ok = ((yva==1)&(empty_map==1)).sum(); ne_bad = ((yva==0)&(empty_map==1)).sum()
        for t in np.arange(0.15, 0.85, 0.02):
            for ma, mh in configs:
                ious = []
                for f in imgs:
                    m = (preds[f] > t).astype(np.uint8)
                    if empty_map[imgs.index(f)] == 1: m = np.zeros_like(m)
                    m = pp(m, ma, mh)
                    ious.append(iou(m, targets[f]))
                sc = np.mean(ious)
                if sc > best[0]:
                    best = (sc, th, t, (ma, mh))
        if th in (0.30, 0.50, 0.70, 0.90):
            print(f'  th={th:.2f}: empty_ok={ne_ok}/{yva.sum()} nonempty_killed={ne_bad}', flush=True)
    print(f'BEST IoU={best[0]:.4f} @ clf_th={best[1]:.2f} t={best[2]:.2f} pp={best[3]}', flush=True)

if __name__ == '__main__':
    main()
