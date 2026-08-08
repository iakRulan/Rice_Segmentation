"""Native-resolution mosaic inference for legacy 3-channel checkpoints.

Each prediction is produced from a 512x512 window containing the centre tile
and 128 pixels of real neighbouring tiles on every side.  The centre 256x256
prediction is retained.  Unlike resize TTA, this preserves the training pixel
scale and only removes artificial tile-border context loss.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
import segmentation_models_pytorch as smp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_context import GRID_W, tile_id

MEAN = np.asarray([0.485, 0.456, 0.406], np.float32)
STD = np.asarray([0.229, 0.224, 0.225], np.float32)


class MosaicStore:
    def __init__(self, data: Path, domain: str, cache: bool):
        self.paths = {}
        for split in ('train', 'val', 'testA'):
            for p in (data / split / 'image' / domain).glob('*.png'):
                self.paths[tile_id(p.name)] = p
        self.cache = {} if cache else None
        if cache:
            for i, p in self.paths.items():
                self.cache[i] = np.asarray(Image.open(p).convert('RGB'), np.uint8)

    def get(self, i, fallback):
        p = self.paths.get(i)
        if p is None:
            return fallback
        return self.cache[i] if self.cache is not None else np.asarray(
            Image.open(p).convert('RGB'), np.uint8)

    def window(self, i):
        centre = self.get(i, np.zeros((256, 256, 3), np.uint8))
        row, col = divmod(i - 1, GRID_W)
        rows = []
        for dy in (-1, 0, 1):
            cells = []
            for dx in (-1, 0, 1):
                rr, cc = row + dy, col + dx
                j = rr * GRID_W + cc + 1
                cells.append(centre if rr < 0 or cc < 0 or cc >= GRID_W
                             else self.get(j, centre))
            rows.append(np.concatenate(cells, 1))
        mosaic = np.concatenate(rows, 0)
        return mosaic[128:640, 128:640]


class MosaicDataset(Dataset):
    def __init__(self, data, split, domain, store):
        self.names = sorted(p.name for p in (data / split / 'image' / domain).glob('*.png'))
        self.store = store

    def __len__(self): return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        x = self.store.window(tile_id(name)).astype(np.float32) / 255
        x = (x - MEAN) / STD
        return torch.from_numpy(x.transpose(2, 0, 1)).float(), name


def build(c):
    table = {'unet': smp.Unet, 'unetpp': smp.UnetPlusPlus,
             'deeplabv3plus': smp.DeepLabV3Plus, 'fpn': smp.FPN,
             'pspnet': smp.PSPNet, 'manet': smp.MAnet}
    return table[c['arch']](encoder_name=c['encoder'], encoder_weights=None,
                            in_channels=3, classes=c['classes'], activation=None)


def state(ck):
    for key in ('model_state_dict', 'state_dict', 'model'):
        if isinstance(ck, dict) and key in ck:
            return ck[key]
    return ck


def transform(x, k, flip):
    x = torch.rot90(x, k, (-2, -1))
    return torch.flip(x, (-1,)) if flip else x


def inverse(x, k, flip):
    if flip: x = torch.flip(x, (-1,))
    return torch.rot90(x, -k, (-2, -1))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--task', choices=['multi', 'wheat', 'rape', 'rice'], required=True)
    ap.add_argument('--split', choices=['val', 'testA'], default='val')
    ap.add_argument('--data_root', default='/root/competition_data/public')
    ap.add_argument('--output', required=True)
    ap.add_argument('--batch_size', type=int, default=4)
    ap.add_argument('--tta', action='store_true')
    ap.add_argument('--cache', action='store_true')
    args = ap.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute(): config_path = ROOT / 'configs' / config_path
    cfg = json.load(open(config_path))
    models = []
    for c in cfg:
        m = build(c)
        p = Path(c['weight'])
        if not p.is_absolute(): p = ROOT / p
        m.load_state_dict(state(torch.load(p, map_location='cpu', weights_only=False)))
        models.append(m.cuda().eval())
    data = Path(args.data_root)
    domain = 'rice' if args.task == 'rice' else 'wheat_rape'
    store = MosaicStore(data, domain, args.cache)
    ds = MosaicDataset(data, args.split, domain, store)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0,
                    pin_memory=True)
    tfs = [(0, False)] if not args.tta else [(k, f) for k in range(4) for f in (0, 1)]
    result = {}
    for images, names in dl:
        images = images.cuda(non_blocking=True); total = None
        for m in models:
            mt = None
            for k, flip in tfs:
                with torch.amp.autocast('cuda'):
                    p = torch.sigmoid(m(transform(images, k, flip)))
                p = inverse(p, k, flip)
                mt = p if mt is None else mt + p
            mt /= len(tfs)
            total = mt if total is None else total + mt
        total = (total / len(models))[:, :, 128:384, 128:384].float().cpu().numpy()
        for j, name in enumerate(names):
            p = total[j].astype(np.float16)
            result[name] = p if args.task == 'multi' else p[0]
        print(f'[{len(result)}/{len(ds)}]', flush=True)
    np.savez(args.output, **result)
    print('[saved]', args.output, flush=True)

    if args.split == 'val':
        classes = ['wheat', 'rape'] if args.task == 'multi' else [args.task]
        for c, cls in enumerate(classes):
            best = (-1, None)
            for th in np.arange(.25, .71, .01):
                values = []
                for name in ds.names:
                    p = result[name] if args.task != 'multi' else result[name][c]
                    y = np.asarray(Image.open(data/'val/label'/cls/name)) > 0
                    b = p > th; u = (b | y).sum()
                    values.append((b & y).sum() / u if u else 1.)
                score = float(np.mean(values))
                if score > best[0]: best = (score, float(th))
            print(cls, 'best_raw', best, flush=True)


if __name__ == '__main__': main()
