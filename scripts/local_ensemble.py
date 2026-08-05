"""Local GPU ensemble inference -> per-image per-class prob maps (float16 npz).

Memory-safe for the 6GB RTX 3060 Laptop: models are loaded one group at a time
and their predictions accumulated, so only ~2-3 models are ever resident.

Usage:
    python scripts/local_ensemble.py --task multi|single --configs cfg_multi_v4.json \
        --split val [--scales 256,288,320] [--out preds_multi_val.npz]

Output: npz {fname: float16 (C,256,256)}
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import load_config, PREDS, VAL_IMG, TESTA_IMG

NORM = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def build_model(arch, encoder, classes):
    import segmentation_models_pytorch as smp
    kw = dict(encoder_name=encoder, encoder_weights=None, in_channels=3,
              classes=classes, activation=None)
    table = {'unet': smp.Unet, 'unetpp': smp.UnetPlusPlus,
             'deeplabv3plus': smp.DeepLabV3Plus, 'fpn': smp.FPN,
             'manet': smp.MAnet, 'pan': smp.PAN}
    return table[arch](**kw)


def get_state(ckpt):
    sd = ckpt.get('model_state_dict', ckpt)
    if any(k.startswith('backbone.') for k in sd):
        sd = {k[len('backbone.'):]: v for k, v in sd.items()}
    return sd


def load_model(c, device):
    m = build_model(c['arch'], c['encoder'], c['classes']).to(device).eval()
    sd = torch.load(c['weight'], map_location=device, weights_only=False)
    m.load_state_dict(get_state(sd))
    return m


def make_tta(scales):
    """Return list of (name, preproc) operating on np uint8 (H,W,3) -> transformed np."""
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    preprocs = {
        'orig': lambda x: x,
        'hflip': lambda x: np.flip(x, axis=1),
        'vflip': lambda x: np.flip(x, axis=0),
        'rot90': lambda x: np.rot90(x, k=1, axes=(0, 1)),
    }
    tfs = {}
    for s in scales:
        tf = A.Compose([A.Resize(s, s), A.Normalize(**NORM), ToTensorV2()])
        for name, pp in preprocs.items():
            tfs[f'{s}_{name}'] = (pp, tf)
    return tfs


def infer_one(model, image, tfs, device, target_size=(256, 256)):
    """image: uint8 (H,W,3). Returns mean (C,256,256) prob over TTA variants.

    Batches the 4 flips of each scale into one forward (fast); falls back to
    sequential on CUDA OOM (safe for the 6GB laptop GPU).
    """
    import torch.nn.functional as F
    scales = sorted({int(k.split('_')[0]) for k in tfs})
    preds = []
    batch_mode = os.environ.get('ENS_BATCH_TTA', '1') == '1'
    for s in scales:
        group = [(k, pp, tf) for k, (pp, tf) in tfs.items() if k.startswith(f'{s}_')]
        names = [g[0] for g in group]
        # forward (batched)
        try:
            xs = [g[2](image=g[1](image))['image'] for g in group]
            t = torch.stack(xs).to(device)                       # (4,3,s,s)
            with torch.no_grad(), torch.amp.autocast('cuda'):
                out = torch.sigmoid(model(t)).float()            # (4,C,s,s)
            ok = True
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            ok = False
        if not ok or not batch_mode:
            out = []
            for g in group:
                aug = g[2](image=g[1](image))
                t = aug['image'].unsqueeze(0).to(device)
                with torch.no_grad(), torch.amp.autocast('cuda'):
                    o = torch.sigmoid(model(t)).float()          # (1,C,s,s)
                out.append(o)
            out = torch.cat(out, 0)
        # un-flip
        for i, name in enumerate(names):
            o = out[i]
            if '_hflip' in name:
                o = torch.flip(o, dims=[-1])
            elif '_vflip' in name:
                o = torch.flip(o, dims=[-2])
            elif '_rot90' in name:
                o = torch.rot90(o, k=-1, dims=[-2, -1])
            preds.append(F.interpolate(o.unsqueeze(0), size=target_size,
                                       mode='bilinear', align_corners=False).squeeze(0).cpu().numpy())
    return np.mean(preds, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', choices=['multi', 'single'], required=True)
    ap.add_argument('--configs', required=True, help='config json in configs/')
    ap.add_argument('--split', choices=['val', 'testA'], default='val')
    ap.add_argument('--scales', default='256,288,320')
    ap.add_argument('--out', default=None)
    ap.add_argument('--tag', default='')          # optional tag appended to out name
    ap.add_argument('--limit', type=int, default=0, help='smoke test: only first N images')
    ap.add_argument('--subdir', choices=['wheat_rape', 'rice'], default=None,
                    help='image subdir (default: wheat_rape for multi, rice for single)')
    args = ap.parse_args()

    configs = load_config(args.configs)
    scales = [int(s) for s in args.scales.split(',')]
    tfs = make_tta(scales)

    img_dir = VAL_IMG if args.split == 'val' else TESTA_IMG
    subdir = args.subdir or ('wheat_rape' if args.task == 'multi' else 'rice')
    img_dir = img_dir / subdir
    imgs = sorted(f for f in img_dir.iterdir() if f.suffix == '.png')
    if args.limit:
        imgs = imgs[:args.limit]
    names = [f.name for f in imgs]
    print(f'[ens] task={args.task} split={args.split} n_models={len(configs)} '
          f'scales={scales} n_img={len(imgs)}')

    device = torch.device('cuda')
    C = configs[0]['classes']
    acc = {n: np.zeros((C, 256, 256), np.float32) for n in names}  # running sum

    for i, c in enumerate(configs):
        m = load_model(c, device)
        t0 = __import__('time').time()
        for j, f in enumerate(imgs):
            image = np.array(Image.open(f).convert('RGB'))
            acc[f.name] += infer_one(m, image, tfs, device).astype(np.float32)
            if (j + 1) % 300 == 0:
                print(f'  model{i} {j+1}/{len(imgs)} {__import__("time").time()-t0:.0f}s', flush=True)
        del m
        torch.cuda.empty_cache()
        print(f'[ens] model{i} done ({c["arch"]}/{c["encoder"]})', flush=True)

    out = args.out or f'ens_{args.task}_{args.split}' + (f'_{args.tag}' if args.tag else '') + '.npz'
    out = PREDS / out
    avg = {n: (acc[n] / len(configs)).astype(np.float16) for n in names}
    np.savez(out, **avg)
    print(f'[ens] saved {out} ({len(avg)} imgs, C={C})', flush=True)


if __name__ == '__main__':
    main()
