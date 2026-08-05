"""Compare shape features of predicted components between:
A) true-empty images where model hallucinates (pred non-empty, gt empty)
B) true-nonempty images correctly detected
If separable, add shape features to the empty classifier.
"""
import numpy as np
from PIL import Image
from scipy import ndimage

VAL = '/root/competition_data/public/val'
d = np.load('/root/ens_multi_v4b.npz')
imgs = sorted(d.files)
ch = 1  # rape
t = 0.55


def comp_stats(mask):
    # returns list of (area, solidity, eccentricity, border_frac) per component
    lab, n = ndimage.label(mask)
    out = []
    h, w = mask.shape
    for i in range(1, n + 1):
        comp = (lab == i)
        area = comp.sum()
        # border fraction
        border = comp[0, :].sum() + comp[-1, :].sum() + comp[:, 0].sum() + comp[:, -1].sum()
        bfrac = border / max(area, 1)
        # bounding box
        ys, xs = np.where(comp)
        hh = ys.max() - ys.min() + 1
        ww = xs.max() - xs.min() + 1
        fill = area / max(hh * ww, 1)
        out.append((area, bfrac, fill, hh / max(ww, 1)))
    return out


halluc = []   # true empty, pred non-empty (model hallucination)
correct = []  # true nonempty, correctly detected (IoU>0)
for f in imgs:
    p = d[f].astype(np.float32)
    p = p[ch] if p.ndim == 3 else p
    lab = np.array(Image.open(f'{VAL}/label/rape/{f}'))
    gt_empty = (lab > 0).sum() == 0
    m = (p > t).astype(np.uint8)
    det = (m > 0).sum() > 0
    if gt_empty and det:
        halluc.append(comp_stats(m))
    elif (not gt_empty) and det:
        inter = np.logical_and(m, lab > 0).sum()
        if inter > 0:
            correct.append(comp_stats(m))

print(f'hallucination images: {len(halluc)}, correctly-detected nonempty: {len(correct)}')
for name, arr in [('halluc', halluc), ('correct', correct)]:
    areas = [s[0] for c in arr for s in c]
    bfrac = [s[1] for c in arr for s in c]
    fill = [s[2] for c in arr for s in c]
    aspect = [s[3] for c in arr for s in c]
    print(f'{name}: n_comp_med={np.median([len(c) for c in arr]):.1f} '
          f'area_med={np.median(areas):.0f} bfrac_med={np.median(bfrac):.3f} '
          f'fill_med={np.median(fill):.3f} aspect_med={np.median(aspect):.3f}')
# distribution of maxprob for halluc vs correct
print('\nper-image maxprob:')
for name, arr in [('halluc', halluc), ('correct', correct)]:
    pass
