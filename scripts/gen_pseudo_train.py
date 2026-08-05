"""Generate confident testA pseudo-labels into a train_plus structure for self-training.
Saves: /root/competition_data/public/train_plus/image/{wheat_rape,rice}/ta_<clip>.png
       /root/competition_data/public/train_plus/label/{wheat,rape,rice}/ta_<clip>.png
Only images with empty-clf=non-empty AND maxprob>=conf_min are kept (confident).
Also copies real train images as tr_<clip>.png so the retrain set = train + confident pseudo testA.
"""
import os, sys, json, numpy as np, torch, shutil
from PIL import Image
from scipy import ndimage
sys.path.insert(0, '/root')
from infer_ensemble import build_model, get_weights_from_ckpt, tta_transforms, infer_one

NORM = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
DATA = '/root/competition_data/public'
TESTA = f'{DATA}/testA'
VAL = f'{DATA}/val'
TP = f'{DATA}/train_plus'

CFG_MULTI = json.load(open('/root/cfg_multi_v4.json'))
CFG_RICE = json.load(open('/root/cfg_rice_v4.json'))
CFG_DED_W = json.load(open('/root/cfg_ded_wheat.json'))
CFG_DED_R = json.load(open('/root/cfg_ded_rape.json'))

CONF_MIN = 0.60
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
    def __init__(self, configs, device):
        self.models = []
        for c in configs:
            m = build_model(c['arch'], c['encoder'], c['classes']).to(device).eval()
            sd = get_weights_from_ckpt(torch.load(c['weight'], map_location=device, weights_only=False))
            m.load_state_dict(sd)
            self.models.append(m)
        self.device = device
        self.tfs = tta_transforms([256, 288, 320])

    def predict(self, image):
        preds = [infer_one(m, image, self.tfs, self.device) for m in self.models]
        return np.mean(preds, axis=0)


def main():
    device = torch.device('cuda')
    # dirs
    for c in ['wheat', 'rape', 'rice']:
        os.makedirs(f'{TP}/label/{c}', exist_ok=True)
    for c in ['wheat_rape', 'rice']:
        os.makedirs(f'{TP}/image/{c}', exist_ok=True)

    # copy real train (renamed tr_)
    for split in ['wheat_rape', 'rice']:
        src = f'{DATA}/train/image/{split}'
        dst = f'{TP}/image/{split}'
        for f in os.listdir(src):
            if f.endswith('.png') and not os.path.exists(f'{dst}/tr_{f}'):
                shutil.copy2(f'{src}/{f}', f'{dst}/tr_{f}')
    for c in ['wheat', 'rape', 'rice']:
        src = f'{DATA}/train/label/{c}'
        dst = f'{TP}/label/{c}'
        for f in os.listdir(src):
            if f.endswith('.png') and not os.path.exists(f'{dst}/tr_{f}'):
                shutil.copy2(f'{src}/{f}', f'{dst}/tr_{f}')

    ens_m = Ens(CFG_MULTI, device)
    ens_dw = Ens(CFG_DED_W, device)
    ens_dr = Ens(CFG_DED_R, device)
    ens_r = Ens(CFG_RICE, device)

    # fit empty clf on val (from existing val blend npz, CPU)
    vb = np.load('/root/ens_multi_v4b.npz')
    vfeats = {c: [] for c in ['wheat', 'rape']}
    vy = {c: [] for c in ['wheat', 'rape']}
    for f in sorted(vb.files):
        b = vb[f].astype(np.float32)
        for i, c in enumerate(['wheat', 'rape']):
            lab = np.array(Image.open(f'{VAL}/label/{c}/{f}'))
            vy[c].append(int((lab > 0).sum() == 0)); vfeats[c].append(empty_features(b[i]))
    clfs = {}
    for c in ['wheat', 'rape']:
        X = np.array(vfeats[c]); y = np.array(vy[c])
        w, mu, sd = logistic(X, y); p = proba(X, w, mu, sd)
        best, bt = 0, 0.5
        for th in np.arange(0.1, 0.9, 0.05):
            acc = ((p > th).astype(int) == y).mean()
            if acc > best: best, bt = acc, th
        clfs[c] = (w, mu, sd, bt)

    # testA pseudo labels
    ta_wr = sorted(f for f in os.listdir(f'{TESTA}/image/wheat_rape') if f.endswith('.png'))
    kept = {c: 0 for c in ['wheat', 'rape', 'rice']}
    for j, f in enumerate(ta_wr):
        image = np.array(Image.open(f'{TESTA}/image/wheat_rape/{f}').convert('RGB'))
        pm = ens_m.predict(image); pw = ens_dw.predict(image); pr = ens_dr.predict(image)
        b = pm.copy(); b[0] = 0.5 * b[0] + 0.5 * pw[0]; b[1] = 0.5 * b[1] + 0.5 * pr[0]
        for i, c in enumerate(['wheat', 'rape']):
            s = SETTINGS[c]
            em = proba(empty_features(b[i])[None], *clfs[c][:3])[0] > clfs[c][3]
            confident = (not em) and b[i].max() >= CONF_MIN
            if confident:
                m = (b[i] > s['t']).astype(np.uint8)
                m = pp(m, s['min_area'], s['max_hole'])
                Image.fromarray(m * 255).save(f'{TP}/label/{c}/ta_{f}')
                shutil.copy2(f'{TESTA}/image/wheat_rape/{f}', f'{TP}/image/wheat_rape/ta_{f}')
                kept[c] += 1
        # rice
        rimage = np.array(Image.open(f'{TESTA}/image/rice/{f}').convert('RGB'))
        pr_ = ens_r.predict(rimage)[0]
        s = SETTINGS['rice']
        em = (pr_ > 0.3).sum() == 0
        if (not em) and pr_.max() >= CONF_MIN:
            m = (pr_ > s['t']).astype(np.uint8)
            m = pp(m, s['min_area'], s['max_hole'])
            Image.fromarray(m * 255).save(f'{TP}/label/rice/ta_{f}')
            shutil.copy2(f'{TESTA}/image/rice/{f}', f'{TP}/image/rice/ta_{f}')
            kept['rice'] += 1
        if (j + 1) % 100 == 0:
            print(f'  {j+1}/{len(ta_wr)}', flush=True)
    print(f'kept confident pseudo: {kept}', flush=True)
    for c in ['wheat', 'rape', 'rice']:
        print(f'  {c}: train {len(os.listdir(f"{TP}/label/{c}"))} images total', flush=True)


if __name__ == '__main__':
    main()
