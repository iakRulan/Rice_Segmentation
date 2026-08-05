
"""Hard-image second pass: re-infer low-confidence images at higher scales and blend.
Usage: ens_hard.py --task multi --configs cfg_multi_v3.json --base ens_multi_v3.npz --out ens_multi_v3h.npz [--scales 320,384,448] [--max_prob 0.5] [--bs 2]
"""
import os, sys, json, argparse
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
sys.path.insert(0, '/root')
from infer_ensemble import build_model, get_weights_from_ckpt, tta_transforms

VAL = '/root/competition_data/public/val'
TESTA = '/root/competition_data/public/testA'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', choices=['multi','single'], required=True)
    ap.add_argument('--configs', required=True)
    ap.add_argument('--base', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--scales', default='320,384,448')
    ap.add_argument('--max_prob', type=float, default=0.5)
    ap.add_argument('--bs', type=int, default=2)
    ap.add_argument('--data', choices=['val','testA'], default='val')
    ap.add_argument('--blend', type=float, default=0.5)  # weight of hard pass
    args = ap.parse_args()

    device = torch.device('cuda')
    configs = json.load(open(args.configs))
    models = []
    for c in configs:
        m = build_model(c['arch'], c['encoder'], c['classes']).to(device).eval()
        sd = torch.load(c['weight'], map_location=device, weights_only=False)
        m.load_state_dict(get_weights_from_ckpt(sd))
        models.append(m)

    scales = [int(x) for x in args.scales.split(',')]
    tfs = tta_transforms(scales)
    base = VAL if args.data == 'val' else TESTA
    img_dir = os.path.join(base, 'image', 'wheat_rape' if args.task == 'multi' else 'rice')
    imgs = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
    d = np.load(args.base)
    files = sorted(d.files)
    assert files == imgs, 'npz file order mismatch'

    # find low-confidence images (per-channel max)
    hard = []
    for i, f in enumerate(imgs):
        p = d[f].astype(np.float32)
        pmax = p.max()
        if pmax < args.max_prob:
            hard.append(f)
    print(f'hard images: {len(hard)}/{len(imgs)} (max_prob<{args.max_prob})', flush=True)
    if not hard:
        print('nothing to do'); return

    out = {f: d[f].astype(np.float32) for f in files}
    for f in hard:
        image = np.array(Image.open(os.path.join(img_dir, f)).convert('RGB'))
        preds = []
        for m in models:
            ps = []
            for name, (pp, tf) in tfs.items():
                aug = tf(image=pp(image))
                t = aug['image'].unsqueeze(0).to(device)
                with torch.no_grad(), torch.amp.autocast('cuda'):
                    o = torch.sigmoid(m(t)).float()
                if '_hflip' in name: o = torch.flip(o, dims=[-1])
                elif '_vflip' in name: o = torch.flip(o, dims=[-2])
                elif '_rot90' in name: o = torch.rot90(o, k=-1, dims=[-2,-1])
                o = F.interpolate(o, size=(256,256), mode='bilinear', align_corners=False)
                ps.append(o.squeeze(0).cpu().numpy())
            preds.append(np.mean(ps, axis=0))
        hard_pred = np.mean(preds, axis=0)
        old = out[f]
        out[f] = args.blend * hard_pred + (1 - args.blend) * old
        if f == hard[0]:
            print(f'  sample {f}: old_max={old.max():.4f} new_max={out[f].max():.4f}', flush=True)
    np.savez(args.out, **{f: out[f].astype(np.float16) for f in files})
    print(f'saved {args.out}', flush=True)

if __name__ == '__main__':
    main()
