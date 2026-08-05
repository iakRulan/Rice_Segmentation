"""Diagnose probs npz: threshold sweep + empty/non-empty split + worst images."""
import sys, numpy as np
from PIL import Image
from scipy import ndimage
VAL = '/root/competition_data/public/val'

def iou(pred, tgt):
    inter = np.logical_and(pred, tgt).sum()
    union = np.logical_or(pred, tgt).sum()
    return 1.0 if union == 0 and inter == 0 else (0.0 if union == 0 else inter / union)

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

cls = sys.argv[1]
npz = sys.argv[2]
ch = int(sys.argv[3])
d = np.load(npz)
imgs = sorted(d.files)
preds = {f: (d[f][ch] if d[f].ndim == 3 else d[f]).astype(np.float32) for f in imgs}
targets = {}
empties = []
for f in imgs:
    lab = np.array(Image.open(f'{VAL}/label/{cls}/{f}'))
    targets[f] = (lab > 0).astype(np.uint8)
    empties.append((lab > 0).sum() == 0)
empties = np.array(empties)
print(f'[{cls}] n={len(imgs)} empty={empties.sum()} nonempty={(~empties).sum()}')

best = (-1, None, None)
for t in np.arange(0.15, 0.85, 0.02):
    for ma, mh in [(0,0),(30,30),(60,60),(100,100),(120,60),(200,200)]:
        ious = []
        for f in imgs:
            m = (preds[f] > t).astype(np.uint8)
            m = pp(m, ma, mh)
            ious.append(iou(m, targets[f]))
        m = np.mean(ious)
        if m > best[0]:
            best = (m, t, (ma, mh))
print(f'best IoU {best[0]:.4f} @ t={best[1]:.2f} pp={best[2]}')
t, (ma, mh) = best[1], best[2]
ious_e, ious_ne = [], []
per_img = {}
for f in imgs:
    m = (preds[f] > t).astype(np.uint8)
    m = pp(m, ma, mh)
    i = iou(m, targets[f])
    per_img[f] = i
    (ious_e if empties[imgs.index(f)] else ious_ne).append(i)
print(f'  empty IoU={np.mean(ious_e):.4f} (n={len(ious_e)}), nonempty IoU={np.mean(ious_ne):.4f} (n={len(ious_ne)})')
# worst non-empty images
worst = sorted([(v, k) for k, v in per_img.items() if not empties[imgs.index(k)]])[:10]
print('worst non-empty:', [(k, round(v, 3)) for v, k in worst])
