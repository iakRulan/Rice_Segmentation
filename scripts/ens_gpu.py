"""GPU batched ensemble TTA inference (val/testA) -> npz.
Usage: ens_gpu.py --task multi|single --configs json --out npz [--scales 256,288,320] [--data_dir val|testA]
"""
import os, sys, json, argparse
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
sys.path.insert(0, '/root')
from infer_ensemble import build_model, get_weights_from_ckpt

NORM = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
VAL = '/root/competition_data/public/val'
TESTA = '/root/competition_data/public/testA'


def tta_transforms(scales):
    preprocs = {'orig': lambda x: x,
                'hflip': lambda x: np.flip(x, axis=1),
                'vflip': lambda x: np.flip(x, axis=0),
                'rot90': lambda x: np.rot90(x, k=1, axes=(0, 1))}
    tfs = {}
    for s in scales:
        tf = A.Compose([A.Resize(s, s), A.Normalize(**NORM), ToTensorV2()])
        for name, pp in preprocs.items():
            tfs[f'{s}_{name}'] = (pp, tf)
    return tfs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', choices=['multi', 'single'], required=True)
    ap.add_argument('--configs', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--scales', default='256,288,320')
    ap.add_argument('--data', choices=['val', 'testA'], default='val')
    ap.add_argument('--bs', type=int, default=0)
    args = ap.parse_args()

    configs = json.load(open(args.configs))
    base = VAL if args.data == 'val' else TESTA
    img_dir = os.path.join(base, 'image', 'wheat_rape' if args.task == 'multi' else 'rice')
    files = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
    scales = [int(x) for x in args.scales.split(',')]
    tfs = tta_transforms(scales)
    print(f'[{args.task}] {len(files)} imgs, {len(configs)} models, scales={scales}', flush=True)

    device = torch.device('cuda')
    models = []
    for c in configs:
        m = build_model(c['arch'], c['encoder'], c['classes']).to(device).eval()
        sd = torch.load(c['weight'], map_location=device, weights_only=False)
        m.load_state_dict(get_weights_from_ckpt(sd))
        models.append(m)
        print(f'  loaded {c["arch"]}/{c["encoder"]} {c["classes"]}ch', flush=True)

    images = {f: np.array(Image.open(os.path.join(img_dir, f)).convert('RGB')) for f in files}
    C = configs[0]['classes']
    preds = {}
    with torch.no_grad():
        for mi, model in enumerate(models):
            acc = {f: np.zeros((C, 256, 256), np.float32) for f in files}
            for name, (pp, tf) in tfs.items():
                bs = args.bs or max(1, min(16, 4_000_000 // (scales[0] * scales[0])))
                for i in range(0, len(files), bs):
                    batch = files[i:i + bs]
                    ts = []
                    for f in batch:
                        aug = tf(image=pp(images[f]))
                        ts.append(aug['image'])
                    t = torch.stack(ts).to(device)
                    with torch.amp.autocast('cuda'):
                        out = torch.sigmoid(model(t)).float()
                    if '_hflip' in name:
                        out = torch.flip(out, dims=[-1])
                    elif '_vflip' in name:
                        out = torch.flip(out, dims=[-2])
                    elif '_rot90' in name:
                        out = torch.rot90(out, k=-1, dims=[-2, -1])
                    out = F.interpolate(out, size=(256, 256), mode='bilinear', align_corners=False)
                    out = out.cpu().numpy()
                    for j, f in enumerate(batch):
                        acc[f] += out[j]
                print(f'  model{mi} {name} done', flush=True)
                torch.cuda.empty_cache()
            for f in files:
                acc[f] /= len(tfs)
                preds.setdefault(f, []).append(acc[f])
            del acc
            torch.cuda.empty_cache()
    out = {f: np.mean(preds[f], axis=0).astype(np.float16) for f in files}
    np.savez(args.out, **out)
    print(f'saved {args.out}', flush=True)


if __name__ == '__main__':
    main()
