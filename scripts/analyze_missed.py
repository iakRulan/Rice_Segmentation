
import numpy as np, os
from PIL import Image
VAL = '/root/competition_data/public/val'
def analyze(cls, npz, ch):
    d = np.load(npz)
    imgs = sorted(d.files)
    rows = []
    for f in imgs:
        lab = np.array(Image.open(f'{VAL}/label/{cls}/{f}'))
        gt = (lab > 0)
        if gt.sum() == 0: continue
        p = d[f].astype(np.float32)
        p = p[ch] if p.ndim == 3 else p
        if (p > 0.5).sum() == 0:
            rows.append((f, int(gt.sum()), float(p.max()), float(p.mean()), int((p>0.3).sum())))
    rows.sort(key=lambda r: r[1])
    print(f'[{cls}] non-empty total={len(imgs)} fully-missed={len(rows)}')
    if rows:
        areas = np.array([r[1] for r in rows])
        print('  missed GT area: min=%d med=%d p75=%d p90=%d max=%d' % (areas.min(), np.median(areas), np.percentile(areas,75), np.percentile(areas,90), areas.max()))
        print('  missed pred max: min=%.4f med=%.4f p90=%.4f' % (min(r[2] for r in rows), np.median([r[2] for r in rows]), np.percentile([r[2] for r in rows],90)))
        print('  smallest 12:', rows[:12])
analyze('wheat', '/root/ens_multi_v2.npz', 0)
analyze('rape', '/root/ens_multi_v2.npz', 1)
analyze('rice', '/root/ens_single_v2.npz', 0)
