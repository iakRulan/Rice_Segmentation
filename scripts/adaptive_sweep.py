
import numpy as np, os
from PIL import Image
from scipy import ndimage
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

def load(cls, npz, ch):
    d = np.load(npz); imgs = sorted(d.files)
    preds = {f: (d[f][ch] if d[f].ndim == 3 else d[f]).astype(np.float32) for f in imgs}
    targets = {}
    for f in imgs:
        lab = np.array(Image.open(f'{VAL}/label/{cls}/{f}'))
        targets[f] = (lab > 0).astype(np.uint8)
    return imgs, preds, targets

def sweep(imgs, preds, targets, empty_map, t_min=0.15, t_max=0.85, step=0.02):
    best = (-1, None, None)
    for t in np.arange(t_min, t_max, step):
        for ma, mh in [(0,0),(60,60),(100,100),(200,200),(120,60)]:
            ious = []
            for f in imgs:
                m = (preds[f] > t).astype(np.uint8)
                if empty_map is not None and empty_map[f] == 1: m = np.zeros_like(m)
                m = pp(m, ma, mh)
                ious.append(iou(m, targets[f]))
            sc = np.mean(ious)
            if sc > best[0]: best = (sc, t, (ma, mh))
    return best

def adaptive(imgs, preds, targets, empty_map, base_t, alpha):
    # per-image threshold: if pred max < 0.5, use max(base_t, alpha * pred_max)
    ious = []
    for f in imgs:
        p = preds[f]; pm = p.max()
        t = base_t
        if pm < 0.5: t = max(base_t, alpha * pm)
        m = (p > t).astype(np.uint8)
        if empty_map is not None and empty_map[f] == 1: m = np.zeros_like(m)
        m = pp(m, 100, 100)
        ious.append(iou(m, targets[f]))
    return np.mean(ious)

for cls, npz, ch, ep in [('wheat','/root/ens_multi_v2.npz',0,'/root/empty_wheat_v2clf.npy'),
                          ('rape','/root/ens_multi_v2.npz',1,'/root/empty_rape_v2clf.npy'),
                          ('rice','/root/ens_single_v2.npz',0,None)]:
    imgs, preds, targets = load(cls, npz, ch)
    empty_map = None
    if ep and os.path.exists(ep):
        ev = np.load(ep); empty_map = dict(zip(imgs, ev))
    b = sweep(imgs, preds, targets, empty_map)
    print(f'[{cls}] global best IoU={b[0]:.4f} t={b[1]:.2f} pp={b[2]}')
    for alpha in [0.5, 0.6, 0.7, 0.8]:
        a = adaptive(imgs, preds, targets, empty_map, b[1], alpha)
        print(f'  adaptive alpha={alpha}: IoU={a:.4f}  (delta={a-b[0]:+.4f})')
