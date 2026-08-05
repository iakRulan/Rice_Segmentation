"""Analyze empty-image mispredictions from saved val preds + measure object size distribution."""
import os, sys
import numpy as np
from PIL import Image

VAL = '/root/competition_data/public/val'
lb = os.path.join(VAL, 'label')


def load_preds(tag):
    d = np.load(f'/root/diag_{tag}_preds.npz')
    imgs = sorted(d.files)
    return imgs, {f: d[f].astype(np.float32) for f in imgs}


def analyze(tag, lbl_cls):
    imgs, preds = load_preds(tag)
    print(f'\n===== {tag} ({len(imgs)} imgs) =====')
    false_pos_empty = []   # max prob on empty images that got fp
    true_neg_empty = []    # max prob on empty images correctly empty
    for f in imgs:
        lab = np.array(Image.open(os.path.join(lb, lbl_cls, f)))
        p = preds[f]
        if (lab > 0).sum() == 0:
            mx = p.max()
            if (p > 0.5).any():
                false_pos_empty.append(mx)
            else:
                true_neg_empty.append(mx)
    fp = np.array(false_pos_empty)
    tn = np.array(true_neg_empty)
    print(f'  empty imgs with FP @0.5: {len(fp)} (maxprob mean {fp.mean():.3f} if any)')
    if len(fp):
        print(f'    FP maxprob: min {fp.min():.3f}, median {np.median(fp):.3f}, mean {fp.mean():.3f}, max {fp.max():.3f}')
    if len(tn):
        print(f'    TN maxprob: median {np.median(tn):.3f}, mean {tn.mean():.3f}, p90 {np.percentile(tn,90):.3f}')

    # how much does a per-image zero-out rule help?
    # rule: if maxprob < t -> zero out. sweep t on empty images only first
    best_gain = 0
    best_t = None
    for t in np.arange(0.1, 0.9, 0.02):
        # recovered = empty images with fp whose maxprob < t
        n_recover = 0
        for f in imgs:
            lab = np.array(Image.open(os.path.join(lb, lbl_cls, f)))
            p = preds[f]
            if (lab > 0).sum() == 0 and (p > 0.5).any() and p.max() < t:
                n_recover += 1
        gain = n_recover / len(imgs)
        if gain > best_gain:
            best_gain, best_t = gain, t
    print(f'  max recoverable via maxprob<t zero-out: {best_gain*len(imgs):.0f} imgs (+{best_gain:.4f} IoU) @ t={best_t:.2f}')


def object_size(tag, lbl_cls):
    imgs, _ = load_preds(tag)
    from scipy import ndimage
    sizes = []
    for f in imgs:
        lab = np.array(Image.open(os.path.join(lb, lbl_cls, f)))
        if (lab > 0).sum() == 0:
            continue
        labeled, n = ndimage.label(lab > 0)
        for i in range(1, n + 1):
            sizes.append((labeled == i).sum())
    sizes = np.array(sizes)
    print(f'\n===== {tag} object size (non-empty only) =====')
    print(f'  n objects: {len(sizes)}, med {np.median(sizes):.0f}, mean {sizes.mean():.0f}, p25 {np.percentile(sizes,25):.0f}, p75 {np.percentile(sizes,75):.0f}')
    print(f'  objects < 500px: {(sizes<500).sum()} ({(sizes<500).sum()/len(sizes)*100:.1f}%)')


if __name__ == '__main__':
    analyze('wheat', 'wheat')
    analyze('rape', 'rape')
    analyze('rice', 'rice')
    object_size('wheat', 'wheat')
    object_size('rape', 'rape')
    object_size('rice', 'rice')
