"""Average two single-class inference npz (per-image (1,H,W)), save mean. Usage: avg_ded.py cls"""
import numpy as np
import sys

cls = sys.argv[1]  # rape or wheat
a = np.load(f'/root/ens_ded_{cls}42.npz')
b = np.load(f'/root/ens_ded_{cls}43.npz')
imgs = sorted(a.files)
out = {}
for f in imgs:
    x = a[f].astype(np.float32); y = b[f].astype(np.float32)
    if x.ndim == 2: x = x[None]
    if y.ndim == 2: y = y[None]
    out[f] = ((x + y) / 2).astype(np.float16)
np.savez(f'/root/ens_ded_{cls}.npz', **out)
print(f'saved /root/ens_ded_{cls}.npz ({len(imgs)})', flush=True)
