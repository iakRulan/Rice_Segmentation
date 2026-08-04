"""Extract empty-classifier features from ensemble (single-pass, consistent) for a split.
Usage: ens_features.py --task multi|single --configs json --data_dir ... --out npz
Saves {img_name: feature_vector[12]} as float16.
"""
import os, sys, json
import numpy as np
from PIL import Image
from scipy import ndimage
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
sys.path.insert(0, '/root')
from infer_ensemble import build_model

NORM = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


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
    return feats


def main():
    ap = __import__('argparse').ArgumentParser()
    ap.add_argument('--task', choices=['multi', 'single'], required=True)
    ap.add_argument('--configs', required=True)
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    configs = json.load(open(args.configs))
    device = torch.device('cuda')
    models = [build_model(c['arch'], c['encoder'], c['classes']).to(device).eval()
              for c in configs]
    for m, c in zip(models, configs):
        sd = torch.load(c['weight'], map_location=device, weights_only=False)
        state = sd.get('model_state_dict', sd)
        if any(k.startswith('backbone.') for k in state):
            state = {k[9:]: v for k, v in state.items()}
        m.load_state_dict(state)

    tf = A.Compose([A.Normalize(**NORM), ToTensorV2()])
    img_dir = os.path.join(args.data_dir, 'image', 'wheat_rape' if args.task == 'multi' else 'rice')
    imgs = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])

    feats = {}
    for i, f in enumerate(imgs):
        image = np.array(Image.open(os.path.join(img_dir, f)).convert('RGB'))
        x = tf(image=image)['image'].unsqueeze(0).to(device)
        with torch.no_grad(), torch.amp.autocast('cuda'):
            preds = [torch.sigmoid(m(x)).float() for m in models]
        p = torch.mean(torch.stack(preds), dim=0)[0].cpu().numpy()  # (C,H,W)
        if p.ndim == 2:
            p = p[None]
        feats[f] = np.array([empty_features(p[c]) for c in range(p.shape[0])], dtype=np.float16)
        if (i + 1) % 500 == 0:
            print(f'  {i+1}/{len(imgs)}', flush=True)
    np.savez(args.out, **feats)
    print(f'saved {args.out} ({len(feats)})', flush=True)


if __name__ == '__main__':
    main()
