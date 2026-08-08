"""Optimize new/legacy probability blends with spatial-row cross-validation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


SPECS = {
    'wheat': [
        ('/root/mosaic1_wr_val.npz', 0, 'mosaic'),
        ('/root/ens_multi_v4b.npz', 0, 'v4b'),
        ('/root/ens_ded_wheat.npz', None, 'ded'),
        ('/root/ctx1_wr_val.npz', 0, 'context'),
    ],
    'rape': [
        ('/root/mosaic1_wr_val.npz', 1, 'mosaic'),
        ('/root/ctx1_wr_val.npz', 1, 'context'),
        ('/root/ens_ded_rape.npz', None, 'ded'),
        ('/root/ens_multi_v4b.npz', 1, 'v4b'),
    ],
    'rice': [
        ('/root/ctx1_rice_val.npz', None, 'context'),
        ('/root/ens_single_v2.npz', None, 'v2'),
        ('/root/ens_single_v3.npz', None, 'v3'),
        ('/root/ens_single_v4.npz', None, 'v4'),
    ],
}


def load_class(cls, label_root):
    arrays = []
    names = None
    labels = []
    for path, channel, label in SPECS[cls]:
        z = np.load(path)
        current = sorted(z.files)
        if names is None:
            names = current
        if current != names:
            raise RuntimeError(f'name mismatch: {path}')
        values = []
        for name in names:
            a = z[name]
            if channel is not None:
                a = a[channel]
            elif a.ndim == 3 and a.shape[0] == 1:
                a = a[0]
            values.append(a)
        arrays.append(np.asarray(values, dtype=np.float32))
        labels.append(label)
    target = np.asarray([
        np.asarray(Image.open(Path(label_root) / cls / name)) > 0 for name in names
    ])
    groups = np.asarray([(int(Path(n).stem.rsplit('_', 1)[1]) - 333) // 830
                         for n in names])
    return names, np.stack(arrays), target, groups, labels


def iou_score(pred, target, threshold, indices=None):
    if indices is not None:
        pred, target = pred[indices], target[indices]
    binary = pred > threshold
    inter = np.logical_and(binary, target).sum(axis=(1, 2))
    union = np.logical_or(binary, target).sum(axis=(1, 2))
    return float(np.divide(inter, union, out=np.ones_like(inter, dtype=float),
                           where=union > 0).mean())


def optimize(stack, target, indices, seed, iterations=None, popsize=None):
    """Fast robust search: coarse weights on 4x spatial subsample, exact t refine."""
    n = stack.shape[0]
    small = stack[:, indices, ::4, ::4]
    small_target = target[indices, ::4, ::4]
    individual_t = []
    for i in range(n):
        choices = [(iou_score(small[i], small_target, t), t)
                   for t in np.arange(.18, .831, .02)]
        individual_t.append(max(choices)[1])
    candidates = []
    for i in range(n):
        w = np.zeros(n); w[i] = 1; candidates.append(w)
    for i in range(n):
        for j in range(i + 1, n):
            for alpha in (.2, .4, .6, .8):
                w = np.zeros(n); w[i] = alpha; w[j] = 1-alpha
                candidates.append(w)
    rng = np.random.default_rng(seed)
    candidates.extend(rng.dirichlet(np.ones(n) * .7, size=40))
    ranked = []
    for w in candidates:
        blend = np.einsum('i,ijkl->jkl', w, small, optimize=True)
        t0 = float(np.dot(w, individual_t))
        best = max((iou_score(blend, small_target, t), t)
                   for t in np.clip(t0 + np.asarray([-.06, -.03, 0, .03, .06]),
                                    .12, .88))
        ranked.append((best[0], w.copy(), best[1]))
    ranked.sort(key=lambda x: -x[0])
    # Refine the best coarse candidates at native resolution.
    exact = []
    for _, w, t0 in ranked[:6]:
        blend = np.einsum('i,ijkl->jkl', w, stack[:, indices], optimize=True)
        best = max((iou_score(blend, target[indices], t), t)
                   for t in np.clip(t0 + np.arange(-.05, .051, .01), .1, .9))
        exact.append((best[0], w, best[1]))
    score, weights, threshold = max(exact, key=lambda x: x[0])
    return weights, float(threshold), float(score)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--label_root', default='/root/competition_data/public/val/label')
    ap.add_argument('--output', default='/root/candidate_blend_results.json')
    args = ap.parse_args()
    report = {}
    # Each fold holds out two non-adjacent raster rows.
    folds = [(0, 4), (1, 5), (2, 6), (3, 7)]
    for ci, cls in enumerate(('wheat', 'rape', 'rice')):
        names, stack, target, groups, labels = load_class(cls, args.label_root)
        all_idx = np.arange(len(names))
        weights, threshold, full_score = optimize(
            stack, target, all_idx, 100 + ci, iterations=10, popsize=6)
        full_blend = np.tensordot(weights, stack, axes=(0, 0))
        print(cls, 'FULL', full_score, threshold, dict(zip(labels, weights)), flush=True)
        fold_rows = []
        heldout_ious = np.zeros(len(names), dtype=np.float64)
        for fi, held_groups in enumerate(folds):
            test_idx = np.flatnonzero(np.isin(groups, held_groups))
            train_idx = np.flatnonzero(~np.isin(groups, held_groups))
            w, t, fit_score = optimize(
                stack, target, train_idx, 1000 + ci * 10 + fi,
                iterations=5, popsize=4)
            blend = np.tensordot(w, stack, axes=(0, 0))
            binary = blend[test_idx] > t
            inter = np.logical_and(binary, target[test_idx]).sum((1, 2))
            union = np.logical_or(binary, target[test_idx]).sum((1, 2))
            values = np.divide(inter, union, out=np.ones_like(inter, dtype=float),
                               where=union > 0)
            heldout_ious[test_idx] = values
            row = dict(held_groups=list(held_groups), fit_score=fit_score,
                       test_score=float(values.mean()), threshold=t,
                       weights=dict(zip(labels, map(float, w))))
            fold_rows.append(row)
            print(cls, 'FOLD', fi, row, flush=True)
        cv_score = float(heldout_ious.mean())
        report[cls] = dict(full_score=full_score, cv_score=cv_score,
                           threshold=threshold,
                           weights=dict(zip(labels, map(float, weights))),
                           folds=fold_rows)
        print(cls, 'ROW_CV', cv_score, flush=True)
        del stack, target, full_blend
    report['mean_full'] = float(np.mean([report[c]['full_score'] for c in report]))
    report['mean_row_cv'] = float(np.mean([report[c]['cv_score'] for c in
                                           ('wheat', 'rape', 'rice')]))
    with open(args.output, 'w') as f:
        json.dump(report, f, indent=2)
    print('FINAL', report['mean_full'], report['mean_row_cv'], flush=True)


if __name__ == '__main__':
    main()
