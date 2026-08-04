"""Generate testA pseudo-labels from best available ensemble (probs + empty clf + postprocess).
Usage: gen_pseudo.py --task multi|single --configs json --out_dir dir
"""
import os, sys, json, argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from scipy import ndimage
sys.path.insert(0, '/root')
from infer_ensemble import Ensemble, postprocess

TESTA = '/root/competition_data/public/testA'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', choices=['multi', 'single'], required=True)
    ap.add_argument('--configs', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--th', type=float, default=0.35)
    ap.add_argument('--min_area', type=int, default=120)
    ap.add_argument('--max_hole', type=int, default=60)
    ap.add_argument('--empty_preds', default=None)
    args = ap.parse_args()

    device = torch.device('cuda')
    configs = json.load(open(args.configs))
    ens = Ensemble(configs, device)
    img_dir = os.path.join(TESTA, 'image', 'wheat_rape' if args.task == 'multi' else 'rice')
    imgs = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])

    empty_map = None
    if args.empty_preds and os.path.exists(args.empty_preds):
        empty_map = np.load(args.empty_preds)

    os.makedirs(args.out_dir, exist_ok=True)
    for i, f in enumerate(tqdm(imgs)):
        image = np.array(Image.open(os.path.join(img_dir, f)).convert('RGB'))
        p = ens.predict(image)  # (C,256,256)
        if p.ndim == 2:
            p = p[None]
        if empty_map is not None and empty_map[i] == 1:
            p[:] = 0.0
        # binarize per channel and postprocess
        masks = []
        for c in range(p.shape[0]):
            m = (p[c] > args.th).astype(np.uint8)
            m = postprocess(m, args.min_area, args.max_hole)
            masks.append(Image.fromarray(m * 255))
        if len(masks) == 1:
            masks[0].save(os.path.join(args.out_dir, f))
        else:
            # save as 2-class composite (wheat=1, rape=2 in one image) for yolo-style
            comp = masks[0].convert('L')
            arr = np.array(comp)
            arr2 = np.array(masks[1].convert('L'))
            arr[arr2 > 0] = 255
            Image.fromarray(arr).save(os.path.join(args.out_dir, f))
    print(f'saved pseudo labels to {args.out_dir} ({len(imgs)})', flush=True)

if __name__ == '__main__':
    main()
