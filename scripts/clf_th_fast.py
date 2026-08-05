
"""Fast empty-clf threshold sweep at fixed t/pp, then local joint refine. CPU."""
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

def eval_at(imgs, preds, targets, empty_map, t, ppv):
    ious = []
    for f in imgs:
        m = (preds[f] > t).astype(np.uint8)
        if empty_map[imgs.index(f)] == 1: m = np.zeros_like(m)
        m = pp(m, *ppv)
        ious.append(iou(m, targets[f]))
    return np.mean(ious)

def main():
    cls = sys.argv[1]; npz = sys.argv[2]; ch = int(sys.argv[3]); base_t = float(sys.argv[4]); base_pp = sys.argv[5]
    ma, mh = (int(x) for x in base_pp.split(','))
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
    print(f'[{cls}] n={len(imgs)} empty={yva.sum()} nonempty={len(imgs)-yva.sum()}', flush=True)
    # stage 1: sweep clf_th at fixed t/pp
    best = (-1, None)
    for th in np.arange(0.05, 0.96, 0.05):
        em = (pva > th).astype(int)
        sc = eval_at(imgs, preds, targets, em, base_t, (ma, mh))
        if sc > best[0]: best = (sc, th)
        if abs(th - round(th*2)/2) < 1e-9 or th in (0.1,0.25,0.5,0.75,0.9):
            print(f'  th={th:.2f} IoU={sc:.4f}', flush=True)
    print(f'[stage1] best IoU={best[0]:.4f} @ clf_th={best[1]:.2f}', flush=True)
    # stage 2: joint refine around best clf_th
    bt = best[1]
    best2 = best
    for th in np.arange(max(0.05, bt-0.1), min(0.95, bt+0.11), 0.02):
        em = (pva > th).astype(int)
        for t in np.arange(max(0.15, base_t-0.08), min(0.84, base_t+0.09), 0.02):
            for dm, dh in [(0,0),(30,30),(60,60),(100,100),(120,60),(200,200),(60,120)]:
                sc = eval_at(imgs, preds, targets, em, t, (dm, dh))
                if sc > best2[0]: best2 = (sc, th, t, (dm, dh))
    print(f'[stage2] best IoU={best2[0]:.4f} @ clf_th={best2[1]:.2f} t={best2[2]:.2f} pp={best2[3]}', flush=True)

if __name__ == '__main__':
    main()
