
"""Crop-zoom second pass for hard images: 2x2 overlapping tiles upscaled to 256 -> stitch.
Usage: ens_crop.py --task multi --configs cfg.json --base npz --out npz [--max_prob 0.5] [--blend 0.5]
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

def make_tf():
    return A.Compose([A.Resize(256, 256), A.Normalize(**NORM), ToTensorV2()])

def tiles_2x2(img):
    H, W = img.shape[:2]
    t, step = 160, 128
    out = []
    for y in (0, step):
        for x in (0, step):
            x2 = min(x + t, W); y2 = min(y + t, H)
            x1 = x2 - t; y1 = y2 - t
            out.append((x1, y1, x2, y2, img[y1:y2, x1:x2]))
    return out

def infer_one(model, img, tf, device):
    t = tf(image=img)['image'].unsqueeze(0).to(device)
    with torch.no_grad(), torch.amp.autocast('cuda'):
        o = torch.sigmoid(model(t)).float()
    return o.squeeze(0).cpu().numpy()  # (C,256,256)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', choices=['multi','single'], required=True)
    ap.add_argument('--configs', required=True)
    ap.add_argument('--base', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--max_prob', type=float, default=0.5)
    ap.add_argument('--blend', type=float, default=0.5)
    ap.add_argument('--data', choices=['val','testA'], default='val')
    args = ap.parse_args()
    device = torch.device('cuda')
    configs = json.load(open(args.configs))
    models = []
    for c in configs:
        m = build_model(c['arch'], c['encoder'], c['classes']).to(device).eval()
        sd = torch.load(c['weight'], map_location=device, weights_only=False)
        m.load_state_dict(get_weights_from_ckpt(sd))
        models.append(m)
    ncls = configs[0]['classes']
    tf = make_tf()
    base = VAL if args.data == 'val' else TESTA
    img_dir = os.path.join(base, 'image', 'wheat_rape' if args.task == 'multi' else 'rice')
    imgs = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
    d = np.load(args.base)
    files = sorted(d.files)
    assert files == imgs
    hard = [f for f in imgs if d[f].max() < args.max_prob]
    print(f'crop-pass hard images: {len(hard)}', flush=True)
    if not hard:
        print('nothing to do'); return
    out = {f: d[f].astype(np.float32) for f in files}
    for f in hard:
        image = np.array(Image.open(os.path.join(img_dir, f)).convert('RGB'))
        H, W = image.shape[:2]
        acc = np.zeros((ncls, H, W), np.float32)
        cnt = np.zeros((H, W), np.float32)
        for x1, y1, x2, y2, tile in tiles_2x2(image):
            ps = []
            for m in models:
                ps.append(infer_one(m, tile, tf, device))
            pm = np.mean(ps, axis=0)  # (C,256,256)
            pm = F.interpolate(torch.from_numpy(pm)[None], size=(y2-y1, x2-x1), mode='bilinear', align_corners=False).squeeze(0).numpy()
            acc[:, y1:y2, x1:x2] += pm
            cnt[y1:y2, x1:x2] += 1.0
        crop_pred = acc / np.maximum(cnt[None], 1.0)
        old = out[f]
        new = args.blend * crop_pred + (1 - args.blend) * old
        out[f] = new
        if f == hard[0]:
            print(f'  sample {f}: old_max={old.max():.4f} new_max={new.max():.4f}', flush=True)
    np.savez(args.out, **{f: out[f].astype(np.float16) for f in files})
    print(f'saved {args.out}', flush=True)

if __name__ == '__main__':
    main()
