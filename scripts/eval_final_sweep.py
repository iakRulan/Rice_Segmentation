"""Final eval: grid over threshold x postprocess params, per class. CPU only."""
import os, sys, argparse
import numpy as np
from PIL import Image
from scipy import ndimage

VAL = '/root/competition_data/public/val'


def iou(pred_bin, tgt_bin):
    inter = np.logical_and(pred_bin, tgt_bin).sum()
    union = np.logical_or(pred_bin, tgt_bin).sum()
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--class_name', required=True)
    ap.add_argument('--preds', required=True)
    ap.add_argument('--channel', type=int, default=None)
    ap.add_argument('--empty_preds', default=None)
    args = ap.parse_args()

    d = np.load(args.preds)
    imgs = sorted(d.files)
    preds = {f: (d[f].astype(np.float32) if args.channel is None else d[f][args.channel].astype(np.float32))
             for f in imgs}
    targets = {f: (np.array(Image.open(os.path.join(VAL, 'label', args.class_name, f))) > 0).astype(np.uint8)
               for f in imgs}

    empty_map = None
    if args.empty_preds:
        ep = np.load(args.empty_preds)
        empty_map = dict(zip(imgs, ep))

    best = (-1, None, None)
    configs = [(0, 0), (30, 30), (60, 60), (100, 100), (60, 0), (0, 60), (120, 60), (60, 120), (200, 200)]
    for min_area, max_hole in configs:
        best_t_local = (-1, None)
        for t in np.arange(0.15, 0.85, 0.02):
            ious = []
            for f in imgs:
                m = (preds[f] > t).astype(np.uint8)
                if empty_map is not None and empty_map[f] == 1:
                    m = np.zeros_like(m)
                m = pp(m, min_area, max_hole)
                ious.append(iou(m, targets[f]))
            sc = np.mean(ious)
            if sc > best_t_local[0]:
                best_t_local = (sc, t)
            if best_t_local[0] > best[0]:
                best = (best_t_local[0], best_t_local[1], (min_area, max_hole))
    print(f'[{args.class_name}] best IoU {best[0]:.4f} @ t={best[1]:.2f} pp={best[2]}')


if __name__ == '__main__':
    main()
