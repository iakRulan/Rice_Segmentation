"""TTA inference and honest validation for train_context.py checkpoints."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_context import (MEAN, STD, RasterImageStore, build_model,
                                   tile_id, unpack)


class InferenceDataset(Dataset):
    def __init__(self, data: Path, split: str, mode: str,
                 store: RasterImageStore):
        self.mode = mode
        self.domain = 'wheat_rape' if mode == 'wheat_rape' else 'rice'
        self.names = sorted(p.name for p in
                            (data / split / 'image' / self.domain).glob('*.png'))
        self.store = store

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        i = tile_id(name)
        centre = self.store.get(i, np.zeros((256, 256, 3), np.uint8))
        context = self.store.context(i, centre)
        image = np.concatenate([centre, context], axis=2).astype(np.float32) / 255
        image = (image - MEAN) / STD
        return torch.from_numpy(image.transpose(2, 0, 1)).float(), name


def aug(x, k, flip):
    x = torch.rot90(x, k, (-2, -1))
    return torch.flip(x, (-1,)) if flip else x


def deaug(x, k, flip):
    if flip:
        x = torch.flip(x, (-1,))
    return torch.rot90(x, -k, (-2, -1))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--split', choices=['val', 'testA'], default='val')
    ap.add_argument('--data_root', default='/root/competition_data/public')
    ap.add_argument('--output', required=True)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--tta', action='store_true')
    ap.add_argument('--cache', action='store_true')
    args = ap.parse_args()

    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg = ck['config']
    mode = cfg['mode']
    classes = 2 if mode == 'wheat_rape' else 1
    model = build_model(cfg['arch'], cfg['encoder'], classes, cfg.get('aux', False))
    model.load_state_dict(ck['model_state_dict'])
    model.cuda().eval()

    data = Path(args.data_root)
    domain = 'wheat_rape' if mode == 'wheat_rape' else 'rice'
    store = RasterImageStore(data, domain, args.cache)
    ds = InferenceDataset(data, args.split, mode, store)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, pin_memory=True)
    transforms = [(0, False)] if not args.tta else [(k, f) for k in range(4)
                                                    for f in (False, True)]
    predictions = {}
    for images, names in loader:
        images = images.cuda(non_blocking=True)
        total = None
        for k, flip in transforms:
            with torch.amp.autocast('cuda'):
                logits = unpack(model(aug(images, k, flip)))[0]
            probs = deaug(torch.sigmoid(logits), k, flip)
            total = probs if total is None else total + probs
        total = (total / len(transforms)).float().cpu().numpy()
        for i, name in enumerate(names):
            value = total[i].astype(np.float16)
            predictions[name] = value if classes > 1 else value[0]
        print(f'[{len(predictions)}/{len(ds)}]', flush=True)
    np.savez(args.output, **predictions)
    print(f'[saved] {args.output}', flush=True)

    if args.split == 'val':
        class_names = ['wheat', 'rape'] if mode == 'wheat_rape' else ['rice']
        for c, cls in enumerate(class_names):
            scores = []
            for threshold in np.arange(0.25, 0.71, 0.01):
                ious = []
                for name in ds.names:
                    p = predictions[name]
                    if classes > 1:
                        p = p[c]
                    y = np.asarray(Image.open(data / 'val' / 'label' / cls / name)) > 0
                    b = p > threshold
                    union = np.logical_or(b, y).sum()
                    iou = np.logical_and(b, y).sum() / union if union else 1.0
                    ious.append(iou)
                scores.append((float(np.mean(ious)), float(threshold)))
            print(cls, 'best_raw', max(scores), flush=True)


if __name__ == '__main__':
    main()
