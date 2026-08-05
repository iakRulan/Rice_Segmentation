"""Bucket val failure modes for the current best (v4 blend) at submission thresholds.
Categories:
  A. true-empty, predicted non-empty -> garbage mask (IoU low, should be zeroed)
  B. true-nonempty, predicted empty -> killed (IoU 0)
  C. predicted non-empty but low IoU (0 < IoU < 0.5) -> wrong/partial
  D. predicted non-empty, high IoU -> ok
Also reports the maxprob distribution of bucket A images.
"""
import numpy as np
from PIL import Image
from scipy import ndimage

VAL = '/root/competition_data/public/val'

def iou(pred, tgt):
    inter = np.logical_and(pred, tgt).sum(); union = np.logical_or(pred, tgt).sum()
    return 1.0 if union == 0 and inter == 0 else (0.0 if union == 0 else inter / union)

def analyze(cls, npz, ch, t, min_area, max_hole):
    d = np.load(npz)
    imgs = sorted(d.files)
    A, B, C, D = [], [], [], []
    for f in imgs:
        p = d[f].astype(np.float32)
        p = p[ch] if p.ndim == 3 else p
        lab = np.array(Image.open(f'{VAL}/label/{cls}/{f}'))
        gt_empty = (lab > 0).sum() == 0
        m = (p > t).astype(np.uint8)
        if min_area > 0 or max_hole > 0:
            inv = 1 - m
            if max_hole > 0:
                lb, n = ndimage.label(inv)
                if n:
                    areas = ndimage.sum(inv, lb, range(1, n + 1))
                    for i in range(1, n + 1):
                        if areas[i - 1] < max_hole: m[lb == i] = 1
            if min_area > 0:
                lb, n = ndimage.label(m)
                if n:
                    areas = ndimage.sum(m, lb, range(1, n + 1))
                    for i in range(1, n + 1):
                        if areas[i - 1] < min_area: m[lb == i] = 0
        pred_empty = (m > 0).sum() == 0
        if gt_empty and not pred_empty:
            A.append((f, p.max()))
        elif (not gt_empty) and pred_empty:
            B.append((f, (lab > 0).sum()))
        elif not gt_empty:
            i = iou(m, (lab > 0).astype(np.uint8))
            (C if i < 0.5 else D).append((f, i))
    print(f'[{cls}] n={len(imgs)}')
    print(f'  A(empty->pred-nonempty): {len(A)}  maxprob: ' +
          (f'med={np.median([x[1] for x in A]):.3f} min={min(x[1] for x in A):.3f} max={max(x[1] for x in A):.3f}' if A else '-'))
    print(f'  B(nonempty->pred-empty): {len(B)}  gt_areas: ' +
          (f'med={np.median([x[1] for x in B]):.0f} min={min(x[1] for x in B)} max={max(x[1] for x in B)}' if B else '-'))
    print(f'  C(0<IoU<0.5): {len(C)}  med_iou={np.median([x[1] for x in C]):.3f}' if C else '  C: 0')
    print(f'  D(IoU>=0.5): {len(D)}')
    return A, B, C, D

analyze('wheat', '/root/ens_multi_v4b.npz', 0, 0.45, 0, 60)
analyze('rape', '/root/ens_multi_v4b.npz', 1, 0.55, 100, 100)
analyze('rice', '/root/ens_single_v4.npz', 0, 0.53, 200, 200)
