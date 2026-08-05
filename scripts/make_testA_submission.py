"""Generate testA submission with final ensemble (v4 multi + dedicated blend) + empty-clf + per-class threshold.
Usage: make_testA_submission.py --out_dir /root/submission_final
Configs (hardcoded paths); empty clf: logistic fit on val features, applied to testA.
"""
import os, sys, json, numpy as np, torch
from PIL import Image
from scipy import ndimage
sys.path.insert(0, '/root')
from infer_ensemble import build_model, get_weights_from_ckpt, tta_transforms, infer_one

NORM = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
TESTA = '/root/competition_data/public/testA'
VAL = '/root/competition_data/public/val'

CFG_MULTI = json.load(open('/root/cfg_multi_v4.json'))
CFG_RICE = json.load(open('/root/cfg_rice_v4.json'))
CFG_DED_W = json.load(open('/root/cfg_ded_wheat.json'))
CFG_DED_R = json.load(open('/root/cfg_ded_rape.json'))

# per-class threshold/postprocess from v4-blend val sweeps (t, min_area, max_hole)
SETTINGS = {
    'wheat': dict(t=0.45, min_area=0, max_hole=60),
    'rape':  dict(t=0.55, min_area=100, max_hole=100),
    'rice':  dict(t=0.53, min_area=200, max_hole=200),
}


def empty_features(pred):
    flat = pred.reshape(-1)
    feats = [flat.max(), np.percentile(flat, 99), np.percentile(flat, 95),
             np.percentile(flat, 90), flat.mean(), (flat > 0.1).sum(), (flat > 0.3).sum(),
             (flat > 0.5).sum(), (flat > 0.7).sum()]
    bin5 = (flat > 0.5).astype(np.uint8).reshape(pred.shape[-2:])
    labeled, n = ndimage.label(bin5)
    if n > 0:
        areas = ndimage.sum(bin5, labeled, range(1, n + 1))
        feats += [n, float(areas.max()), float(areas.sum())]
    else:
        feats += [0, 0.0, 0.0]
    return np.array(feats, np.float32)


def logistic(X, y, iters=3000, lr=0.3, l2=1e-3):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xs = (X - mu) / sd
    Xb = np.hstack([np.ones((len(X), 1)), Xs])
    w = np.zeros(Xb.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(Xb @ w, -30, 30)))
        w -= lr * (Xb.T @ (p - y) / len(y) + l2 * w)
    return w, mu, sd


def proba(X, w, mu, sd):
    Xs = (X - mu) / sd
    Xb = np.hstack([np.ones((len(X), 1)), Xs])
    return 1 / (1 + np.exp(-np.clip(Xb @ w, -30, 30)))


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


class Ens:
    def __init__(self, configs, device, scales=(256, 288, 320)):
        self.models = []
        for c in configs:
            m = build_model(c['arch'], c['encoder'], c['classes']).to(device).eval()
            sd = get_weights_from_ckpt(torch.load(c['weight'], map_location=device, weights_only=False))
            m.load_state_dict(sd)
            self.models.append(m)
        self.device = device
        self.tfs = tta_transforms(list(scales))

    def predict(self, image):
        preds = [infer_one(m, image, self.tfs, self.device) for m in self.models]
        return np.mean(preds, axis=0)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_dir', required=True)
    args = ap.parse_args()
    device = torch.device('cuda')

    os.makedirs(args.out_dir, exist_ok=True)
    for c in ['wheat', 'rape', 'rice']:
        os.makedirs(os.path.join(args.out_dir, c), exist_ok=True)

    # ---- fit empty clf on VAL features (from existing val blend npz, CPU) ----
    vb = np.load('/root/ens_multi_v4b.npz')
    vs = np.load('/root/ens_single_v4.npz')
    val_feats = {c: [] for c in ['wheat', 'rape', 'rice']}
    val_y = {c: [] for c in ['wheat', 'rape', 'rice']}
    for f in sorted(vb.files):
        b = vb[f].astype(np.float32)
        for i, c in enumerate(['wheat', 'rape']):
            lab = np.array(Image.open(os.path.join(VAL, 'label', c, f)))
            val_y[c].append(int((lab > 0).sum() == 0))
            val_feats[c].append(empty_features(b[i]))
        rlab = np.array(Image.open(os.path.join(VAL, 'label', 'rice', f)))
        val_y['rice'].append(int((rlab > 0).sum() == 0))
        val_feats['rice'].append(empty_features(vs[f].astype(np.float32)))
    clfs = {}
    for c in ['wheat', 'rape', 'rice']:
        X = np.array(val_feats[c]); y = np.array(val_y[c])
        w, mu, sd = logistic(X, y)
        p = proba(X, w, mu, sd)
        # threshold at 0.5-optimal on val
        best = 0; best_th = 0.5
        for th in np.arange(0.1, 0.9, 0.05):
            pred = (p > th).astype(int)
            acc = (pred == y).mean()
            if acc > best:
                best, best_th = acc, th
        clfs[c] = (w, mu, sd, best_th)
        print(f'[clf {c}] val acc={best:.4f} th={best_th:.2f}', flush=True)

    # ---- testA inference + masks ----
    ens_m = Ens(CFG_MULTI, device)
    ens_dw = Ens(CFG_DED_W, device)
    ens_dr = Ens(CFG_DED_R, device)
    ens_r = Ens(CFG_RICE, device)
    ta_wr = os.path.join(TESTA, 'image', 'wheat_rape')
    ta_rice = os.path.join(TESTA, 'image', 'rice')
    ta_wr_imgs = sorted(f for f in os.listdir(ta_wr) if f.endswith('.png'))
    ta_rice_imgs = sorted(f for f in os.listdir(ta_rice) if f.endswith('.png'))
    assert ta_wr_imgs == ta_rice_imgs, 'testA img mismatch'

    for j, f in enumerate(ta_wr_imgs):
        image = np.array(Image.open(os.path.join(ta_wr, f)).convert('RGB'))
        pm = ens_m.predict(image)
        pw = ens_dw.predict(image)
        pr = ens_dr.predict(image)
        b = pm.copy()
        b[0] = 0.5 * b[0] + 0.5 * pw[0]
        b[1] = 0.5 * b[1] + 0.5 * pr[0]
        for i, c in enumerate(['wheat', 'rape']):
            s = SETTINGS[c]
            em = int(proba(empty_features(b[i])[None], *clfs[c][:3])[0] > clfs[c][3])
            m = (b[i] > s['t']).astype(np.uint8)
            if em:
                m = np.zeros_like(m)
            m = pp(m, s['min_area'], s['max_hole'])
            Image.fromarray(m * 255).save(os.path.join(args.out_dir, c, f))
        # rice on same clip file
        rimage = np.array(Image.open(os.path.join(ta_rice, f)).convert('RGB'))
        pr_ = ens_r.predict(rimage)[0]
        s = SETTINGS['rice']
        em = int(proba(empty_features(pr_)[None], *clfs['rice'][:3])[0] > clfs['rice'][3])
        m = (pr_ > s['t']).astype(np.uint8)
        if em:
            m = np.zeros_like(m)
        m = pp(m, s['min_area'], s['max_hole'])
        Image.fromarray(m * 255).save(os.path.join(args.out_dir, 'rice', f))
        if (j + 1) % 100 == 0:
            print(f'  {j+1}/{len(ta_wr_imgs)}', flush=True)
    print(f'saved submission to {args.out_dir} ({len(ta_wr_imgs)})', flush=True)


if __name__ == '__main__':
    main()
