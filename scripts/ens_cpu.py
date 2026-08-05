"""CPU-parallel ensemble inference (val) -> ens_{task}_v2.npz.
Loads one model per worker process; each worker handles a subset of images.
TTA: scales [256,288,320] x {orig,hflip,vflip,rot90} -> mean prob at 256x256.
Usage: ens_cpu.py --task multi|single --configs json --out npz [--workers N]
"""
import os, sys, json, argparse, math
import numpy as np
from PIL import Image
from multiprocessing import Pool
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
sys.path.insert(0, '/root')
from infer_ensemble import build_model

NORM = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
VAL = '/root/competition_data/public/val'
G = None

def _init(cfg):
    global G
    G = cfg

def _load_models(configs, device):
    models = []
    for c in configs:
        m = build_model(c['arch'], c['encoder'], c['classes']).to(device).eval()
        sd = torch.load(c['weight'], map_location=device, weights_only=False)
        state = sd.get('model_state_dict', sd)
        if any(k.startswith('backbone.') for k in state):
            state = {k[len('backbone.'):]: v for k, v in state.items()}
        m.load_state_dict(state)
        models.append(m)
    return models

def _infer_one(model, image, tfs, device):
    import torch.nn.functional as F
    preds = []
    for name, (pp, tf) in tfs.items():
        img_t = pp(image)
        aug = tf(image=img_t)
        t = aug['image'].unsqueeze(0).to(device)
        with torch.no_grad():
            out = torch.sigmoid(model(t)).float()
        if '_hflip' in name:
            out = torch.flip(out, dims=[-1])
        elif '_vflip' in name:
            out = torch.flip(out, dims=[-2])
        elif '_rot90' in name:
            out = torch.rot90(out, k=-1, dims=[-2, -1])
        out = F.interpolate(out, size=(256, 256), mode='bilinear', align_corners=False)
        preds.append(out.squeeze(0).cpu().numpy())
    return np.mean(preds, axis=0)

def _worker(files):
    cfg = G
    device = torch.device('cpu')
    torch.set_num_threads(G['threads'])
    configs = cfg['configs']
    img_dir = cfg['img_dir']
    models = _load_models(configs, device)
    tfs = cfg['tfs']
    out = {}
    for f in files:
        image = np.array(Image.open(os.path.join(img_dir, f)).convert('RGB'))
        preds = [_infer_one(m, image, tfs, device) for m in models]
        out[f] = np.mean(preds, axis=0).astype(np.float16)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', choices=['multi', 'single'], required=True)
    ap.add_argument('--configs', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--scales', default='256,288,320')
    args = ap.parse_args()

    configs = json.load(open(args.configs))
    img_dir = os.path.join(VAL, 'image', 'wheat_rape' if args.task == 'multi' else 'rice')
    files = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])

    preprocs = {'orig': lambda x: x,
                'hflip': lambda x: np.flip(x, axis=1),
                'vflip': lambda x: np.flip(x, axis=0),
                'rot90': lambda x: np.rot90(x, k=1, axes=(0, 1))}
    tfs = {}
    scales = [int(x) for x in args.scales.split(',')]
    for s in scales:
        tf = A.Compose([A.Resize(s, s), A.Normalize(**NORM), ToTensorV2()])
        for name, pp in preprocs.items():
            tfs[f'{s}_{name}'] = (pp, tf)

    chunks = [files[i::args.workers] for i in range(args.workers)]
    chunks = [ch for ch in chunks if ch]
    cfg = {'configs': configs, 'img_dir': img_dir, 'tfs': tfs, 'threads': args.threads}
    print(f'[{args.task}] {len(files)} imgs, {args.workers} workers, {len(configs)} models', flush=True)
    results = {}
    with Pool(args.workers, initializer=_init, initargs=(cfg,)) as pool:
        for i, r in enumerate(pool.imap_unordered(_worker, chunks)):
            results.update(r)
            print(f'  done {len(results)}/{len(files)}', flush=True)
    np.savez(args.out, **results)
    print(f'saved {args.out}', flush=True)

if __name__ == '__main__':
    main()
