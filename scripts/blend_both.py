"""Blend dedicated wheat+rape probs into v4 multi ensemble (both channels).
Usage: blend_both.py --ens ens_multi_v4.npz --dw ens_ded_wheat.npz --dr ens_ded_rape.npz --alpha 0.5 --out ens_multi_v4b.npz
"""
import argparse
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ens', required=True)
    ap.add_argument('--dw', required=True)
    ap.add_argument('--dr', required=True)
    ap.add_argument('--alpha', type=float, default=0.5)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    de = np.load(args.ens)
    dw = np.load(args.dw)
    dr = np.load(args.dr)
    imgs = sorted(de.files)
    assert sorted(dw.files) == imgs and sorted(dr.files) == imgs, 'image mismatch'
    out = {}
    for f in imgs:
        e = de[f].astype(np.float32)   # (2,H,W)
        w = dw[f].astype(np.float32)
        r = dr[f].astype(np.float32)
        if w.ndim == 2: w = w[None]
        if r.ndim == 2: r = r[None]
        if e.ndim == 2: e = e[None]
        b = e.copy()
        b[0] = args.alpha * e[0] + (1 - args.alpha) * w[0]
        b[1] = args.alpha * e[1] + (1 - args.alpha) * r[0]
        out[f] = b.astype(np.float16)
    np.savez(args.out, **out)
    print(f'saved {args.out} alpha={args.alpha} ({len(imgs)})', flush=True)

if __name__ == '__main__':
    main()
