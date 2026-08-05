import os, sys, argparse, time
os.environ['PATH'] = '/root/miniconda3/bin:' + os.environ.get('PATH','')
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pathlib import Path

def iou_bin(pred, tgt):
    inter = np.logical_and(pred, tgt).sum()
    union = np.logical_or(pred, tgt).sum()
    return 1.0 if union == 0 and inter == 0 else (0.0 if union == 0 else inter / union)

def load_model(weights):
    from ultralytics import YOLO
    m = YOLO(weights)
    mm = m.model.to('cuda').eval()
    return mm

def predict_probs(mm, files, bs=32, imgsz=256):
    probs = {}
    with torch.no_grad():
        for i in range(0, len(files), bs):
            batch = files[i:i+bs]
            xs = []
            for f in batch:
                im = np.asarray(Image.open(f).convert('RGB')).astype(np.float32) / 255.0
                x = torch.from_numpy(im).permute(2, 0, 1).unsqueeze(0)
                xs.append(x)
            x = torch.cat(xs, 0).cuda()
            out = mm(x)  # B,C,H/8,W/8 logits
            p = torch.softmax(out.float(), dim=1)
            p = F.interpolate(p, size=(imgsz, imgsz), mode='bilinear', align_corners=False)
            p = p.cpu().numpy()
            for f, pi in zip(batch, p):
                probs[f.stem] = pi
    return probs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True)
    ap.add_argument('--mode', required=True, choices=['wheat_rape', 'rice'])
    ap.add_argument('--out', required=True)
    ap.add_argument('--imgsz', type=int, default=256)
    args = ap.parse_args()

    img_dir = Path('/root/yolo_data') / args.mode / 'val' / 'images'
    files = sorted(img_dir.glob('*.png'))
    print(f'loading {args.weights} ...', flush=True)
    mm = load_model(args.weights)
    print(f'inferring {len(files)} images ...', flush=True)
    t0 = time.time()
    probs = predict_probs(mm, files, bs=32, imgsz=args.imgsz)
    print(f'inference done in {time.time()-t0:.0f}s', flush=True)
    np.savez_compressed(args.out, **probs)

    if args.mode == 'wheat_rape':
        classes = [('wheat', 1), ('rape', 2)]
        labels = {'wheat': Path('/root/competition_data/public/val/label/wheat'),
                  'rape': Path('/root/competition_data/public/val/label/rape')}
    else:
        classes = [('rice', 1)]
        labels = {'rice': Path('/root/competition_data/public/val/label/rice')}

    targets = {}
    for cname in labels:
        targets[cname] = {}
        for f in files:
            lp = labels[cname] / f.name
            tgt = (np.asarray(Image.open(lp)) > 0).astype(np.uint8) if lp.exists() else np.zeros((args.imgsz, args.imgsz), np.uint8)
            targets[cname][f.stem] = tgt

    print('\n===== competition-style IoU =====', flush=True)
    means = {}
    for cname, cid in classes:
        best = (-1, None)
        for t in np.arange(0.20, 0.81, 0.05):
            ious = []
            for f in files:
                pred = probs[f.stem][cid] > t
                ious.append(iou_bin(pred, targets[cname][f.stem] > 0))
            m = float(np.mean(ious))
            if m > best[0]:
                best = (m, round(float(t), 2))
        # argmax baseline
        ious_am = []
        for f in files:
            pred = probs[f.stem].argmax(0) == cid
            ious_am.append(iou_bin(pred, targets[cname][f.stem] > 0))
        am = float(np.mean(ious_am))
        means[cname] = best[0]
        print(f'{cname}: best IoU {best[0]:.4f} @ t={best[1]} | argmax {am:.4f}', flush=True)
    print(f'MEAN(best): {np.mean(list(means.values())):.4f}', flush=True)

if __name__ == '__main__':
    main()