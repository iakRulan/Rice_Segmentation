"""Test richer empty-clf features + GBM vs baseline logistic, on a 50/50 val split.
Uses existing val blend npz (ens_multi_v4b.npz) + single (ens_single_v4.npz).
"""
import numpy as np
from PIL import Image
from scipy import ndimage
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

VAL = '/root/competition_data/public/val'


def rich_features(pred):
    flat = pred.reshape(-1)
    feats = [flat.max(), np.percentile(flat, 99), np.percentile(flat, 95),
             np.percentile(flat, 90), flat.mean(), (flat > 0.1).sum(), (flat > 0.3).sum(),
             (flat > 0.5).sum(), (flat > 0.7).sum()]
    for thr in (0.3, 0.5, 0.7):
        m = (flat > thr).astype(np.uint8).reshape(pred.shape[-2:])
        lab, n = ndimage.label(m)
        if n > 0:
            areas = ndimage.sum(m, lab, range(1, n + 1))
            feats += [n, float(areas.max()), float(areas.mean())]
            # solidity of largest
            ys, xs = np.where(lab == int(np.argmax(areas) + 1))
            hh = ys.max() - ys.min() + 1; ww = xs.max() - xs.min() + 1
            feats += [areas.max() / max(hh * ww, 1), max(hh, ww) / max(min(hh, ww), 1)]
        else:
            feats += [0, 0.0, 0.0, 0.0, 1.0]
    return np.array(feats, np.float32)


def run(cls, npz, ch):
    d = np.load(npz)
    imgs = sorted(d.files)
    X = []; y = []
    for f in imgs:
        p = d[f].astype(np.float32)
        p = p[ch] if p.ndim == 3 else p
        lab = np.array(Image.open(f'{VAL}/label/{cls}/{f}'))
        y.append(int((lab > 0).sum() == 0))
        X.append(rich_features(p))
    X = np.array(X); y = np.array(y)
    n = len(y)
    rng = np.random.RandomState(42)
    idx = rng.permutation(n)
    tr, te = idx[:n // 2], idx[n // 2:]
    # baseline logistic on basic 12 feats
    bl = []
    for m_name, model, feats_sel in [
        ('logit12', LogisticRegression(max_iter=2000), slice(0, 12)),
        ('logit_rich', LogisticRegression(max_iter=2000), slice(0, None)),
        ('gbm_rich', HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_depth=3, random_state=0), slice(0, None)),
    ]:
        Xt = X[tr][:, feats_sel]; Xe = X[te][:, feats_sel]
        sc = StandardScaler().fit(Xt)
        model.fit(sc.transform(Xt), y[tr])
        acc = (model.predict(sc.transform(Xe)) == y[te]).mean()
        bl.append((m_name, acc))
    return bl


for cls, npz, ch in [('wheat', '/root/ens_multi_v4b.npz', 0),
                     ('rape', '/root/ens_multi_v4b.npz', 1),
                     ('rice', '/root/ens_single_v4.npz', 0)]:
    res = run(cls, npz, ch)
    print(cls, {k: round(v, 4) for k, v in res})
