"""Ensemble inference: multi-scale TTA -> model averaging -> empty-zero-out -> threshold -> postprocess.
Used for BOTH val evaluation and testA submission generation.
"""
import os, sys, argparse, json
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from scipy import ndimage

sys.path.insert(0, '/root/crop_segmentation')
import segmentation_models_pytorch as smp

NORM = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
VAL = '/root/competition_data/public/val'
TESTA = '/root/competition_data/public/testA'
WEIGHTS = '/root/crop_segmentation/weights'


def build_model(arch, encoder, classes):
    kwargs = dict(encoder_name=encoder, encoder_weights=None, in_channels=3, classes=classes, activation=None)
    if arch == 'unet':
        return smp.Unet(**kwargs)
    elif arch == 'unetpp':
        return smp.UnetPlusPlus(**kwargs)
    elif arch == 'deeplabv3plus':
        return smp.DeepLabV3Plus(**kwargs)
    elif arch == 'fpn':
        return smp.FPN(**kwargs)
    elif arch == 'manet':
        return smp.MAnet(**kwargs)
    elif arch == 'pan':
        return smp.PAN(**kwargs)
    else:
        raise ValueError(arch)


def load_ckpt(path, device):
    return torch.load(path, map_location=device, weights_only=False)


def get_weights_from_ckpt(ckpt):
    sd = ckpt.get('model_state_dict', ckpt)
    if any(k.startswith('backbone.') for k in sd):
        sd = {k[len('backbone.'):]: v for k, v in sd.items()}
    return sd


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


def infer_one(model, image, tfs, device, target_size=(256, 256)):
    import torch.nn.functional as F
    preds = []
    for name, (pp, tf) in tfs.items():
        img_t = pp(image)
        aug = tf(image=img_t)
        t = aug['image'].unsqueeze(0).to(device)
        with torch.no_grad(), torch.amp.autocast('cuda'):
            out = torch.sigmoid(model(t)).float()  # (1,C,s,s)
        if '_hflip' in name:
            out = torch.flip(out, dims=[-1])
        elif '_vflip' in name:
            out = torch.flip(out, dims=[-2])
        elif '_rot90' in name:
            out = torch.rot90(out, k=-1, dims=[-2, -1])
        out = F.interpolate(out, size=target_size, mode='bilinear', align_corners=False)
        preds.append(out.squeeze(0).cpu().numpy())
    return np.mean(preds, axis=0)


def empty_features(pred):
    """Statistics of the ensemble prob map, used by the empty-image classifier."""
    flat = pred.reshape(-1)
    feats = {
        'max': flat.max(),
        'p99': np.percentile(flat, 99),
        'p95': np.percentile(flat, 95),
        'p90': np.percentile(flat, 90),
        'mean': flat.mean(),
        'sum01': (flat > 0.1).sum(),
        'sum03': (flat > 0.3).sum(),
        'sum05': (flat > 0.5).sum(),
        'sum07': (flat > 0.7).sum(),
    }
    # largest component at 0.5
    bin5 = (flat > 0.5).astype(np.uint8).reshape(pred.shape[-2:])
    labeled, n = ndimage.label(bin5)
    if n > 0:
        areas = ndimage.sum(bin5, labeled, range(1, n + 1))
        feats['n_comp'] = n
        feats['max_comp'] = float(areas.max())
        feats['sum_comp'] = float(areas.sum())
    else:
        feats['n_comp'] = 0
        feats['max_comp'] = 0.0
        feats['sum_comp'] = 0.0
    return feats


def postprocess(mask, min_area, max_hole):
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


class Ensemble:
    def __init__(self, configs, device):
        self.models = []
        for c in configs:
            m = build_model(c['arch'], c['encoder'], c['classes']).to(device).eval()
            sd = torch.load(c['weight'], map_location=device, weights_only=False)
            state = get_weights_from_ckpt(sd)
            m.load_state_dict(state)
            self.models.append(m)
        self.device = device
        self.tfs = tta_transforms([256, 288, 320])

    def predict(self, image):
        preds = [infer_one(m, image, self.tfs, self.device) for m in self.models]
        return np.mean(preds, axis=0)


def run(configs, data_dir, out_dir, task, empty_clf=None, thresholds=None,
        min_area=40, max_hole=40, max_empty_mode='none'):
    device = torch.device('cuda')
    ens = Ensemble(configs, device)
    img_dir = os.path.join(data_dir, 'image', 'wheat_rape' if task == 'multi' else 'rice')
    imgs = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
    preds = {}
    for f in tqdm(imgs, desc=task):
        image = np.array(Image.open(os.path.join(img_dir, f)).convert('RGB'))
        preds[f] = ens.predict(image)
    np.savez(f'/root/ens_{task}.npz', **{f: preds[f].astype(np.float16) for f in imgs})
    return preds


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', choices=['multi', 'single'])
    ap.add_argument('--data_dir', default=VAL)
    ap.add_argument('--out_dir', default=None)
    ap.add_argument('--configs', required=True, help='json list of model configs')
    ap.add_argument('--save_only', action='store_true')
    args = ap.parse_args()
    configs = json.load(open(args.configs))
    preds = run(configs, args.data_dir, args.out_dir, args.task)
    if not args.save_only:
        print('saved ensemble probs to /root/ens_%s.npz' % args.task)
