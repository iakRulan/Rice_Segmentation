"""Verify two premises the P0 pipeline silently assumed:
1) Mosaic continuity: are adjacent tile IDs spatially adjacent (right i->i+1,
   below i->i+grid_width)? Compare edge discontinuity against inner-column
   continuity and random pairs.
2) Dual-temporal: do rice/ and wheat_rape/ with the same tile ID image the
   same ground patch? Compare NCC of edge magnitudes same-ID cross-domain vs
   random-ID.
Plus: rice empty-image rate (validates the all-empty IoU floor ~0.217).
"""
import os
import numpy as np
from PIL import Image

R = 'data/public'
W = 83          # candidate grid width (from repo defaults - UNVERIFIED)
MAXID = 6557

def load(i, dom='wheat_rape'):
    for sp in ('train', 'val', 'testA'):
        p = f'{R}/{sp}/image/{dom}/clip_{i:05d}.png'
        if os.path.exists(p):
            return np.asarray(Image.open(p).convert('RGB'), np.int16)
    return None

# ---- 1) mosaic continuity ----
rng = np.random.default_rng(0)
inner, right, below, rand = [], [], [], []
while len(inner) < 200:
    i = int(rng.integers(1, MAXID))
    r, c = divmod(i - 1, W)
    a = load(i)
    if a is None:
        continue
    inner.append(np.abs(a[:, -1] - a[:, -2]).mean())
    if c < W - 1:
        b = load(i + 1)
        if b is not None:
            right.append(np.abs(a[:, -1] - b[:, 0]).mean())
    d = load(i + W)
    if d is not None:
        below.append(np.abs(a[-1, :] - d[0, :]).mean())
    z = load(int(rng.integers(1, MAXID)))
    if z is not None:
        rand.append(np.abs(a[:, -1] - z[:, 0]).mean())

print('=== 1) MOSAIC CONTINUITY (wheat_rape) ===')
print(f'n_inner={len(inner)} n_right={len(right)} n_below={len(below)} n_rand={len(rand)}')
for k, v in [('inner (same-tile adjacent cols)', inner),
             ('right neighbor i,i+1', right),
             ('below neighbor i,i+83', below),
             ('random pair (not-adjacent ref)', rand)]:
    print(f'  {k:32s} {np.mean(v):6.2f}')

# ---- 2) dual-temporal alignment ----
def g(x):
    y = x.mean(2).astype(np.float32)
    gy, gx = np.gradient(y)
    return np.hypot(gy, gx)

def ncc(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float((a * b).sum() / d) if d > 1e-6 else 0.0

same, diff = [], []
while len(same) < 120:
    i = int(rng.integers(1, MAXID)); j = int(rng.integers(1, MAXID))
    a, b, z = load(i, 'wheat_rape'), load(i, 'rice'), load(j, 'rice')
    if a is None or b is None or z is None:
        continue
    same.append(ncc(g(a), g(b)))
    diff.append(ncc(g(a), g(z)))

print('=== 2) DUAL-TEMPORAL ALIGNMENT ===')
print(f'  same-ID wr-vs-rice edge NCC {np.mean(same):.3f} | random-ID {np.mean(diff):.3f}')

# ---- 3) rice empty-image rate ----
def empty_rate(dom):
    e = t = 0
    for sp in ('train', 'val'):
        d = f'{R}/{sp}/label/{dom}'
        if not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            if not fn.endswith('.png'):
                continue
            a = np.asarray(Image.open(f'{d}/{fn}'), np.int16)
            e += int((a > 0).sum() == 0); t += 1
    return e / t

print('=== 3) EMPTY-IMAGE RATES (labeled tiles) ===')
for dom in ('wheat', 'rape', 'rice'):
    r = empty_rate(dom)
    print(f'  {dom:6s} empty {r*100:5.1f}%  -> all-empty IoU floor {r:.3f}')
print('DONE')
