"""Submit testA with triple-threshold postprocessing (opt_patch.postproc).

For each class: search triple-threshold settings on val (search_triple), apply
to testA via triple_threshold. No separate empty classifier — the empty-vs-
nonempty decision is the (t_hi, min_size) gate.

Usage:
    python scripts/submit_triple.py --spec configs/blend_v2.json --out_dir outputs/submission_v3
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import VAL_LBL, TESTA_IMG, PREDS
from opt_patch.postproc import search_triple, triple_threshold

_npz_cache = {}


def load_npz(path):
    if str(path) not in _npz_cache:
        _npz_cache[str(path)] = np.load(path)
    return _npz_cache[str(path)]


def blend(spec, split):
    acc, first_imgs = None, None
    for path, ch, w in spec:
        npz = path.replace('val', split) if split != 'val' else path
        d = load_npz(PREDS / npz)
        imgs = sorted(d.files)
        if first_imgs is None:
            first_imgs = imgs
        for f in imgs:
            p = d[f].astype(np.float32)
            if p.ndim == 2:
                p = p[None]
            if acc is None:
                acc = {}
            acc[f] = acc.get(f, np.zeros_like(p[ch])) + p[ch] * w
    return first_imgs, acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--spec', required=True)
    ap.add_argument('--out_dir', required=True)
    args = ap.parse_args()

    spec = json.load(open(args.spec))
    out = Path(args.out_dir)
    for c in spec:
        (out / c).mkdir(parents=True, exist_ok=True)

    # search settings on val, then apply to testA
    for cls, entries in spec.items():
        v_imgs, v_probs = blend(entries, 'val')
        P = np.stack([v_probs[f] for f in v_imgs])
        T = np.stack([(np.array(Image.open(VAL_LBL / cls / f)) > 0).astype(np.uint8) for f in v_imgs])
        best = search_triple(P, T)
        print(f'[{cls}] settings={best}', flush=True)

        t_imgs, t_probs = blend(entries, 'testA')
        for f in t_imgs:
            m = triple_threshold(t_probs[f], best['t_hi'], best['min_size'], best['t_lo'],
                                 best['min_area'], best['max_hole'])
            Image.fromarray(m * 255).save(out / cls / f)
        print(f'  wrote {cls}/ ({len(t_imgs)} imgs)', flush=True)
    print(f'saved submission to {out}')


if __name__ == '__main__':
    main()
