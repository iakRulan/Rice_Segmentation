"""Row-group CV for component filtering and learned empty-image gating."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from sklearn.ensemble import GradientBoostingClassifier


SPECS = {
    'wheat': [('/root/mosaic1_wr_val.npz', 0, .6),
              ('/root/ens_multi_v4b.npz', 0, .4)],
    'rape': [('/root/mosaic1_wr_val.npz', 1, .30),
             ('/root/ctx1_wr_val.npz', 1, .47),
             ('/root/ens_ded_rape.npz', None, .06),
             ('/root/ens_multi_v4b.npz', 1, .17)],
    'rice': [('/root/ctx1_rice_val.npz', None, .6),
             ('/root/ens_single_v4.npz', None, .4)],
}
CENTRES = {'wheat': .56, 'rape': .56, 'rice': .53}
FOLDS = [(0, 4), (1, 5), (2, 6), (3, 7)]


def read_prob(z, name, channel):
    a = z[name]
    if channel is not None:
        return a[channel]
    if a.ndim == 3 and a.shape[0] == 1:
        return a[0]
    return a


def load(cls):
    parts, names = [], None
    for path, channel, weight in SPECS[cls]:
        z = np.load(path)
        current = sorted(z.files)
        names = current if names is None else names
        if current != names:
            raise RuntimeError(path)
        parts.append(np.asarray([read_prob(z, n, channel) for n in names],
                                dtype=np.float32) * weight)
    probs = np.sum(parts, axis=0) / sum(x[2] for x in SPECS[cls])
    root = Path('/root/competition_data/public/val/label') / cls
    target = np.asarray([np.asarray(Image.open(root / n)) > 0 for n in names])
    groups = np.asarray([(int(Path(n).stem.rsplit('_', 1)[1]) - 333) // 830
                         for n in names])
    return names, probs, target, groups


def features(probs, centre):
    rows = []
    for p in probs:
        flat = p.reshape(-1)
        qs = np.quantile(flat, [.90, .95, .98, .99, .995, .999])
        areas = [(flat > t).sum() / flat.size for t in
                 (.1, .2, .3, .4, .5, .6, .7)]
        b = (p > centre).astype(np.uint8)
        count, _, stats, _ = cv2.connectedComponentsWithStats(b, 8)
        component_areas = stats[1:, cv2.CC_STAT_AREA] if count > 1 else np.array([])
        largest = float(component_areas.max() / flat.size) if len(component_areas) else 0.
        rows.append([flat.mean(), flat.std(), flat.max(), *qs, *areas,
                     largest, float(max(0, count-1)), float((p*(1-p)).mean())])
    x = np.asarray(rows, dtype=np.float32)
    return np.nan_to_num(x)


def precompute(probs, target, centre):
    thresholds = np.round(np.arange(centre-.08, centre+.081, .02), 3)
    min_components = (0, 20, 50, 100, 200)
    image_floors = (0, 20, 50, 100, 200, 400)
    table = {}
    for threshold in thresholds:
        by_component = {m: [] for m in min_components}
        for p in probs:
            binary = (p > threshold).astype(np.uint8)
            count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
            sizes = stats[:, cv2.CC_STAT_AREA]
            for minimum in min_components:
                if minimum == 0:
                    out = binary.astype(bool)
                else:
                    keep = sizes >= minimum; keep[0] = False
                    out = keep[labels]
                by_component[minimum].append(out)
        for minimum, masks in by_component.items():
            masks = np.asarray(masks)
            areas = masks.sum((1, 2))
            for floor in image_floors:
                gated = masks.copy()
                gated[areas < floor] = False
                inter = (gated & target).sum((1, 2))
                union = (gated | target).sum((1, 2))
                iou = np.divide(inter, union, out=np.ones_like(inter, dtype=float),
                                where=union > 0)
                table[(float(threshold), minimum, floor)] = iou
    return table


def best_setting(table, idx):
    setting, values = max(table.items(), key=lambda kv: float(kv[1][idx].mean()))
    return setting, values, float(values[idx].mean())


def train_gate(x, has_target, base_iou, train_idx, test_idx):
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=.04,
                                     max_depth=2, min_samples_leaf=12,
                                     random_state=42)
    clf.fit(x[train_idx], has_target[train_idx])
    train_p = clf.predict_proba(x[train_idx])[:, 1]
    thresholds = np.arange(.10, .91, .02)
    best = (-1., .5)
    for threshold in thresholds:
        values = base_iou[train_idx].copy()
        clear = train_p < threshold
        values[clear] = (~has_target[train_idx][clear]).astype(float)
        score = float(values.mean())
        if score > best[0]: best = (score, float(threshold))
    test_p = clf.predict_proba(x[test_idx])[:, 1]
    values = base_iou[test_idx].copy()
    clear = test_p < best[1]
    values[clear] = (~has_target[test_idx][clear]).astype(float)
    return best, values, clf


def main():
    report = {}
    for cls in ('wheat', 'rape', 'rice'):
        names, probs, target, groups = load(cls)
        has_target = target.any((1, 2))
        print(cls, 'loading/precompute', flush=True)
        x = features(probs, CENTRES[cls])
        table = precompute(probs, target, CENTRES[cls])
        all_idx = np.arange(len(names))
        setting, full_iou, base_full = best_setting(table, all_idx)
        full_gate, full_gated, _ = train_gate(x, has_target, full_iou,
                                              all_idx, all_idx)
        fold_rows, held = [], np.zeros(len(names), dtype=float)
        for fold, held_groups in enumerate(FOLDS):
            test_idx = np.flatnonzero(np.isin(groups, held_groups))
            train_idx = np.flatnonzero(~np.isin(groups, held_groups))
            fold_setting, fold_iou, fit = best_setting(table, train_idx)
            gate_fit, values, _ = train_gate(x, has_target, fold_iou,
                                             train_idx, test_idx)
            held[test_idx] = values
            row = dict(groups=list(held_groups), setting=list(fold_setting),
                       fit_base=fit, fit_gated=gate_fit[0],
                       gate_threshold=gate_fit[1], test=float(values.mean()))
            fold_rows.append(row); print(cls, 'FOLD', fold, row, flush=True)
        report[cls] = dict(full_setting=list(setting), full_base=base_full,
                           full_gated=float(full_gated.mean()),
                           full_gate_threshold=full_gate[1],
                           row_cv=float(held.mean()), folds=fold_rows)
        print(cls, report[cls], flush=True)
        del probs, target, table, x
    report['mean_full_base'] = float(np.mean([report[c]['full_base'] for c in SPECS]))
    report['mean_full_gated'] = float(np.mean([report[c]['full_gated'] for c in SPECS]))
    report['mean_row_cv'] = float(np.mean([report[c]['row_cv'] for c in SPECS]))
    json.dump(report, open('/root/postproc_cv_results.json', 'w'), indent=2)
    print('FINAL', report['mean_full_base'], report['mean_full_gated'],
          report['mean_row_cv'], flush=True)


if __name__ == '__main__': main()
