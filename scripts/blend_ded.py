"""Blend dedicated single-class model probs into the v4 multi ensemble, save blended npz.
Usage: blend_ded.py --ens ens_multi_v4.npz --ded ens_ded_rape.npz --ch 1 --alpha 0.5 --out ens_multi_v4b.npz
"""
import os, sys, argparse
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ens', required=True)
    ap.add_argument('--ded', required=True)
    ap.add_argument('--ch', type=int, required=True, help='channel to replace in ens')
    ap.add_argument('--alpha', type=float, default=0.5)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    de = np.load(args.ens)
    dd = np.load(args.ded)
    imgs = sorted(de.files)
    assert sorted(dd.files) == imgs, 'image mismatch'
    out = {}
    for f in imgs:
        e = de[f].astype(np.float32)   # (C,H,W) or (H,W)
        d = dd[f].astype(np.float32)   # (1,H,W) or (H,W)
        if d.ndim == 2:
            d = d[None]
        if e.ndim == 2:
            e = e[None]
        b = e.copy()
        b[args.ch] = args.alpha * e[args.ch] + (1 - args.alpha) * d[0]
        out[f] = b.astype(np.float16)
    np.savez(args.out, **out)
    print(f'saved {args.out} ({len(imgs)}) alpha={args.alpha}', flush=True)

if __name__ == '__main__':
    main()
