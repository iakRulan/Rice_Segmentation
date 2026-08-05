"""Evaluate saved ensemble predictions (npz) on the val set.
Computes per-class IoU under threshold sweep + optional empty-classifier zero-out + postprocess.
"""
import os, sys, argparse
import numpy as np
from PIL import Image
from scipy import ndimage

VAL = '/root/competition_data/public/val'


def iou(pred_bin, tgt_bin):
    inter = np.logical_and(pred_bin, tgt_bin).sum()
    union = np.logical_or(pred_bin, tgt_bin).sum()
    return 1.0 if union == 0 and inter == 0 else (0.0 if union == 0 else inter / union)


def postprocess(mask, min_area=0, max_hole=0):
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


def load(preds_npz, channel, label_dir):
    d = np.load(preds_npz)
    imgs = sorted(d.files)
    labs = {}
    for f in imgs:
        a = np.array(Image.open(os.path.join(label_dir, f)))
        labs[f] = (a > 0).astype(np.uint8)
    return imgs, {f: d[f].astype(np.float32) if channel is None else d[f][channel].astype(np.float32) for f in imgs}, labs


def sweep(imgs, preds, labs, t_min=0.15, t_max=0.85, step=0.01, min_area=0, max_hole=0):
    best = (-1, None)
    for t in np.arange(t_min, t_max, step):
        ious = []
        for f in imgs:
            pb = postprocess((preds[f] > t).astype(np.uint8), min_area, max_hole)
            ious.append(iou(pb, labs[f]))
        m = np.mean(ious)
        if m > best[0]:
            best = (m, float(t))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--class_name', required=True)
    ap.add_argument('--preds', required=True)
    ap.add_argument('--channel', type=int, default=None)
    ap.add_argument('--min_area', type=int, default=0)
    ap.add_argument('--max_hole', type=int, default=0)
    ap.add_argument('--empty_preds', help='npy of empty-classifier predictions (1=empty)')
    args = ap.parse_args()

    imgs, preds, labs = load(args.preds, args.channel, os.path.join(VAL, 'label', args.class_name))

    # sweep with postprocessing
    best_m, best_t = sweep(imgs, preds, labs, min_area=args.min_area, max_hole=args.max_hole)
    print(f'[{args.class_name}] best IoU {best_m:.4f} @ t={best_t:.2f} (min_area={args.min_area}, max_hole={args.max_hole})')

    if args.empty_preds:
        clf_empty = np.load(args.empty_preds)  # 1 = empty
        # apply zero-out on classifier-empty images at best threshold
        empty_map = dict(zip(sorted(imgs), clf_empty))
        ious = []
        for f in imgs:
            pb = postprocess((preds[f] > best_t).astype(np.uint8), args.min_area, args.max_hole)
            if empty_map[f] == 1:
                pb = np.zeros_like(pb)
            ious.append(iou(pb, labs[f]))
        m_clf = np.mean(ious)
        # empty-only breakdown
        e_ious, ne_ious = [], []
        for f in imgs:
            pb = postprocess((preds[f] > best_t).astype(np.uint8), args.min_area, args.max_hole)
            if empty_map[f] == 1:
                pb = np.zeros_like(pb)
            i = iou(pb, labs[f])
            (e_ious if (labs[f] > 0).sum() == 0 else ne_ious).append(i)
        print(f'  +empty-clf: IoU {m_clf:.4f}  (empty {np.mean(e_ious):.4f}, non-empty {np.mean(ne_ious):.4f})')


if __name__ == '__main__':
    main()
