
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
def analyze(cls, npz, ch, ep, t, ppv):
    d = np.load(npz); imgs = sorted(d.files)
    ev = np.load(ep) if ep else None
    res = []
    for i, f in enumerate(imgs):
        lab = np.array(Image.open(f'{VAL}/label/{cls}/{f}'))
        tgt = (lab > 0).astype(np.uint8)
        p = d[f].astype(np.float32); p = p[ch] if p.ndim == 3 else p
        m = (p > t).astype(np.uint8)
        if ev is not None and ev[i] == 1: m = np.zeros_like(m)
        m = pp(m, *ppv)
        sc = iou(m, tgt)
        res.append((sc, int(tgt.sum()), float(p.max()), int(ev[i]) if ev is not None else 0, f))
    res.sort()
    zeros = [r for r in res if r[0] == 0 and r[1] > 0]
    low = [r for r in res if 0 < r[0] < 0.5 and r[1] > 0]
    print(f'[{cls}] mean={np.mean([r[0] for r in res]):.4f} nz=0:{len(zeros)} low(0,0.5):{len(low)}')
    # zeros breakdown: empty-clf killed vs genuinely missed
    killed = [r for r in zeros if r[3] == 1]
    print(f'  zero nonempty: total={len(zeros)}, killed_by_clf={len(killed)}, genuine_miss={len(zeros)-len(killed)}')
    # GT area of zeros
    if zeros:
        areas = np.array([r[1] for r in zeros])
        print(f'  zero GT area: min={areas.min()} med={np.median(areas):.0f} max={areas.max()}')
    if low:
        areas = np.array([r[1] for r in low])
        print(f'  low(0-0.5) GT area: min={areas.min()} med={np.median(areas):.0f} max={areas.max()}')
analyze('wheat', '/root/ens_multi_v3.npz', 0, '/root/empty_wheat_v3clf.npy', 0.49, (120,60))
analyze('rape', '/root/ens_multi_v3.npz', 1, '/root/empty_rape_v3clf.npy', 0.47, (200,200))
analyze('rice', '/root/ens_single_v2.npz', 0, None, 0.41, (120,60))
