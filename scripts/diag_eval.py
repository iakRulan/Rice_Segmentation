"""Diagnostic eval on val set: per-class IoU, empty/non-empty split, threshold sweep."""
import os, sys
import numpy as np
from PIL import Image
import torch
from torch.cuda.amp import autocast
import albumentations as A
from albumentations.pytorch import ToTensorV2

sys.path.insert(0, '/root/crop_segmentation')
from src.models import MultiLabelModel, SingleLabelModel

VAL = '/root/competition_data/public/val'
WEIGHTS = '/root/crop_segmentation/weights'
NORM = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def make_tta():
    return {
        'original': A.Compose([A.Normalize(**NORM), ToTensorV2()]),
        'hflip': A.Compose([A.HorizontalFlip(p=1.0), A.Normalize(**NORM), ToTensorV2()]),
        'vflip': A.Compose([A.VerticalFlip(p=1.0), A.Normalize(**NORM), ToTensorV2()]),
        'rot90': A.Compose([A.RandomRotate90(p=1.0), A.Normalize(**NORM), ToTensorV2()]),
    }


def infer(model, img_path, tta, device):
    image = np.array(Image.open(img_path).convert('RGB'))
    preds = []
    for name, tf in tta.items():
        aug = tf(image=image)
        t = aug['image'].unsqueeze(0).to(device)
        with torch.no_grad(), autocast():
            out = torch.sigmoid(model(t)).cpu().numpy()[0]
        if name == 'hflip':
            out = np.flip(out, axis=-1)
        elif name == 'vflip':
            out = np.flip(out, axis=-2)
        elif name == 'rot90':
            out = np.rot90(out, k=-1, axes=(-2, -1))
        preds.append(out)
    return np.mean(preds, axis=0)


def iou(pred_bin, tgt_bin):
    inter = np.logical_and(pred_bin, tgt_bin).sum()
    union = np.logical_or(pred_bin, tgt_bin).sum()
    return 1.0 if union == 0 and inter == 0 else (0.0 if union == 0 else inter / union)


def eval_class(model, img_dir, label_path, ch, tta, device, name):
    imgs = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
    empty_gt = []
    preds = {}
    for i, f in enumerate(imgs):
        lab = np.array(Image.open(os.path.join(label_path, f)))
        empty_gt.append(int((lab > 0).sum() == 0))
        preds[f] = infer(model, os.path.join(img_dir, f), tta, device)[ch] if isinstance(ch, int) else infer(model, os.path.join(img_dir, f), tta, device).squeeze()
        if (i + 1) % 200 == 0:
            print(f'  {name}: {i+1}/{len(imgs)}', flush=True)
    np.save(f'/root/diag_{name}.npy', np.array(empty_gt))
    np.savez(f'/root/diag_{name}_preds.npz', **{f: preds[f].astype(np.float16) for f in imgs})
    return empty_gt, preds


def main():
    device = torch.device('cuda')
    tta = make_tta()

    print('Loading models...', flush=True)
    wr = MultiLabelModel()
    wr.load_state_dict(torch.load(os.path.join(WEIGHTS, 'best_wheat_rape.pth'), map_location=device, weights_only=False)['model_state_dict'])
    wr.to(device).eval()
    rice = SingleLabelModel()
    rice.load_state_dict(torch.load(os.path.join(WEIGHTS, 'best_rice.pth'), map_location=device, weights_only=False)['model_state_dict'])
    rice.to(device).eval()

    wr_dir = os.path.join(VAL, 'image/wheat_rape')
    rice_dir = os.path.join(VAL, 'image/rice')
    lb = os.path.join(VAL, 'label')

    print('Inferring wheat...', flush=True)
    we, wp = eval_class(wr, wr_dir, os.path.join(lb, 'wheat'), 0, tta, device, 'wheat')
    print('Inferring rape...', flush=True)
    re_, rp = eval_class(wr, wr_dir, os.path.join(lb, 'rape'), 1, tta, device, 'rape')
    print('Inferring rice...', flush=True)
    rice_e, ricp = eval_class(rice, rice_dir, os.path.join(lb, 'rice'), None, tta, device, 'rice')

    for name, preds, gt_label, lbl_dir in [('wheat', wp, we, 'wheat'), ('rape', rp, re_, 'rape'), ('rice', ricp, rice_e, 'rice')]:
        best = -1; best_t = None
        print(f'\n===== {name} =====')
        for t in np.arange(0.20, 0.85, 0.02):
            ious = []
            for f in sorted(preds.keys()):
                lab = np.array(Image.open(os.path.join(lb, lbl_dir, f)))
                tb = (lab > 0).astype(np.uint8)
                pb = (preds[f] > t).astype(np.uint8)
                ious.append(iou(pb, tb))
            m = np.mean(ious)
            if m > best:
                best, best_t = m, t
        # empty vs non-empty at best threshold
        ious_e, ious_ne = [], []
        for f in sorted(preds.keys()):
            lab = np.array(Image.open(os.path.join(lb, lbl_dir, f)))
            tb = (lab > 0).astype(np.uint8)
            pb = (preds[f] > best_t).astype(np.uint8)
            i = iou(pb, tb)
            (ious_e if (lab > 0).sum() == 0 else ious_ne).append(i)
        pred_frac = [1 if (p > best_t).any() else 0 for p in preds.values()]
        print(f'best IoU {best:.4f} @ t={best_t:.2f}')
        print(f'  empty: n={len(ious_e)} iou={np.mean(ious_e):.4f}, non-empty: n={len(ious_ne)} iou={np.mean(ious_ne):.4f}')
        print(f'  pred non-empty fraction: {np.mean(pred_frac):.3f}')


if __name__ == '__main__':
    main()
