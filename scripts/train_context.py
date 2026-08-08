"""Train a crop segmenter with both tile detail and 3x3 mosaic context.

The dataset filenames form an 82x83 raster.  The historical pipeline treats
every 256x256 tile independently.  This trainer supplies the original tile in
channels 0:3 and a downsampled 3x3 neighbourhood in channels 3:6, while the
target remains the centre tile.  It can therefore use field-scale context
without sacrificing the native-resolution centre image.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
import segmentation_models_pytorch as smp

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from opt_patch.losses_v2 import MultiTaskLoss


GRID_W = 83
MEAN = np.asarray([0.485, 0.456, 0.406] * 2, np.float32)
STD = np.asarray([0.229, 0.224, 0.225] * 2, np.float32)


def tile_id(name: str) -> int:
    return int(Path(name).stem.rsplit('_', 1)[1])


class RasterImageStore:
    """Read all public split images for one domain once into host RAM."""

    def __init__(self, data: Path, domain: str, cache: bool):
        self.paths: dict[int, Path] = {}
        for split in ('train', 'val', 'testA'):
            folder = data / split / 'image' / domain
            if folder.exists():
                for p in folder.glob('*.png'):
                    self.paths[tile_id(p.name)] = p
        self.cache = {} if cache else None
        if cache:
            print(f'[cache] loading {len(self.paths)} {domain} images', flush=True)
            for i, p in self.paths.items():
                self.cache[i] = np.asarray(Image.open(p).convert('RGB'), dtype=np.uint8)

    def get(self, i: int, fallback: np.ndarray) -> np.ndarray:
        p = self.paths.get(i)
        if p is None:
            return fallback
        if self.cache is not None:
            return self.cache[i]
        return np.asarray(Image.open(p).convert('RGB'), dtype=np.uint8)

    def context(self, i: int, centre: np.ndarray) -> np.ndarray:
        z = i - 1
        row, col = divmod(z, GRID_W)
        rows = []
        for dy in (-1, 0, 1):
            cells = []
            for dx in (-1, 0, 1):
                rr, cc = row + dy, col + dx
                if rr < 0 or cc < 0 or cc >= GRID_W:
                    cells.append(centre)
                else:
                    cells.append(self.get(rr * GRID_W + cc + 1, centre))
            rows.append(np.concatenate(cells, axis=1))
        mosaic = np.concatenate(rows, axis=0)
        return cv2.resize(mosaic, (centre.shape[1], centre.shape[0]),
                          interpolation=cv2.INTER_AREA)


class ContextDataset(Dataset):
    def __init__(self, data: Path, split: str, mode: str, store: RasterImageStore,
                 augment: bool, cache_masks: bool):
        self.mode = mode
        self.domain = 'wheat_rape' if mode == 'wheat_rape' else 'rice'
        self.image_dir = data / split / 'image' / self.domain
        self.names = sorted(p.name for p in self.image_dir.glob('*.png'))
        classes = ['wheat', 'rape'] if mode == 'wheat_rape' else [mode]
        self.label_dirs = [data / split / 'label' / c for c in classes]
        self.store, self.augment = store, augment
        self.mask_cache = None
        if cache_masks:
            self.mask_cache = {}
            for name in self.names:
                self.mask_cache[name] = self._read_mask(name)

    def __len__(self):
        return len(self.names)

    def _read_mask(self, name: str) -> np.ndarray:
        masks = [(np.asarray(Image.open(d / name)) > 0).astype(np.float32)
                 for d in self.label_dirs]
        return np.stack(masks, axis=-1)

    def __getitem__(self, idx):
        name = self.names[idx]
        i = tile_id(name)
        centre = self.store.get(i, np.zeros((256, 256, 3), np.uint8)).copy()
        context = self.store.context(i, centre)
        mask = (self.mask_cache[name].copy() if self.mask_cache is not None
                else self._read_mask(name))
        image = np.concatenate([centre, context], axis=2)

        if self.augment:
            k = random.randrange(4)
            image, mask = np.rot90(image, k).copy(), np.rot90(mask, k).copy()
            if random.random() < 0.5:
                image, mask = image[:, ::-1].copy(), mask[:, ::-1].copy()
            if random.random() < 0.5:
                image, mask = image[::-1].copy(), mask[::-1].copy()
            if random.random() < 0.6:
                gain = random.uniform(0.85, 1.15)
                bias = random.uniform(-15, 15)
                image = np.clip(image.astype(np.float32) * gain + bias, 0, 255)
            if random.random() < 0.25:
                image = np.clip(image.astype(np.float32) +
                                np.random.normal(0, random.uniform(2, 8), image.shape),
                                0, 255)

        image = image.astype(np.float32) / 255.0
        image = (image - MEAN) / STD
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        mask = torch.from_numpy(mask.transpose(2, 0, 1)).float()
        return image, mask, name


def build_model(arch: str, encoder: str, classes: int, aux: bool):
    table = {'unet': smp.Unet, 'deeplabv3plus': smp.DeepLabV3Plus,
             'fpn': smp.FPN, 'manet': smp.MAnet}
    kw = dict(encoder_name=encoder, encoder_weights='imagenet', in_channels=6,
              classes=classes, activation=None)
    if aux:
        kw['aux_params'] = dict(classes=classes, pooling='avg', dropout=0.3)
    return table[arch](**kw)


def unpack(out):
    return (out[0], out[1]) if isinstance(out, (tuple, list)) else (out, None)


class EMA:
    def __init__(self, model, decay: float):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}
        self.backup = None

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if self.shadow[k].is_floating_point():
                self.shadow[k].lerp_(v.detach(), 1.0 - self.decay)
            else:
                self.shadow[k].copy_(v)

    @torch.no_grad()
    def apply(self, model):
        self.backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow)

    @torch.no_grad()
    def restore(self, model):
        model.load_state_dict(self.backup)
        self.backup = None


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    class_scores = None
    for images, masks, _ in loader:
        images = images.to(device, non_blocking=True)
        probs = torch.sigmoid(unpack(model(images))[0]).cpu().numpy()
        target = masks.numpy() > 0.5
        pred = probs > 0.5
        inter = (pred & target).sum(axis=(2, 3))
        union = (pred | target).sum(axis=(2, 3))
        iou = np.divide(inter, union, out=np.ones_like(inter, dtype=np.float64),
                        where=union > 0)
        if class_scores is None:
            class_scores = [[] for _ in range(iou.shape[1])]
        for c in range(iou.shape[1]):
            class_scores[c].extend(iou[:, c].tolist())
    per_class = [float(np.mean(x)) for x in class_scores]
    return float(np.mean(per_class)), per_class


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['wheat_rape', 'rice'], required=True)
    ap.add_argument('--data_root', default='/root/competition_data/public')
    ap.add_argument('--arch', choices=['unet', 'deeplabv3plus', 'fpn', 'manet'],
                    default='deeplabv3plus')
    ap.add_argument('--encoder', default='mit_b4')
    ap.add_argument('--epochs', type=int, default=90)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--acc', type=int, default=1)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--patience', type=int, default=25)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--workers', type=int, default=0)
    ap.add_argument('--cache', action='store_true')
    ap.add_argument('--aux', action='store_true')
    ap.add_argument('--cls_w', type=float, default=0.3)
    ap.add_argument('--lovasz_w', type=float, default=0.5)
    ap.add_argument('--focal_gamma', type=float, default=1.0)
    ap.add_argument('--pos_weight', type=float, default=1.2)
    ap.add_argument('--ema_decay', type=float, default=0.999)
    ap.add_argument('--tag', default='ctx')
    ap.add_argument('--output_dir', default='/root/crop_segmentation/weights')
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda')
    data = Path(args.data_root)
    domain = 'wheat_rape' if args.mode == 'wheat_rape' else 'rice'
    store = RasterImageStore(data, domain, args.cache)
    train_ds = ContextDataset(data, 'train', args.mode, store, True, args.cache)
    val_ds = ContextDataset(data, 'val', args.mode, store, False, args.cache)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    classes = 2 if args.mode == 'wheat_rape' else 1
    model = build_model(args.arch, args.encoder, classes, args.aux).to(device)
    criterion = MultiTaskLoss(bce=1, dice=1, lovasz=args.lovasz_w,
                              cls=args.cls_w, focal_gamma=args.focal_gamma,
                              pos_weight=args.pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    opt_per_epoch = math.ceil(len(train_loader) / args.acc)
    total_steps = opt_per_epoch * args.epochs
    warmup = max(1, int(total_steps * 0.05))

    def lr_scale(step):
        if step < warmup:
            return max(0.05, step / warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    scaler = torch.amp.GradScaler('cuda')
    ema = EMA(model, args.ema_decay)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    tag = f'{args.tag}_{args.mode}_{args.arch}_{args.encoder}_{args.seed}'
    best, stale, history, global_step = -1.0, 0, [], 0
    print(f'[config] {tag} train={len(train_ds)} val={len(val_ds)} '
          f'params={sum(p.numel() for p in model.parameters())/1e6:.1f}M', flush=True)

    for epoch in range(args.epochs):
        model.train(); optimizer.zero_grad(set_to_none=True); total = 0.0; t0 = time.time()
        for j, (images, masks, _) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            with torch.amp.autocast('cuda'):
                seg, cls = unpack(model(images))
                loss = criterion(seg, masks, cls) / args.acc
            scaler.scale(loss).backward()
            final_batch = j + 1 == len(train_loader)
            if (j + 1) % args.acc == 0 or final_batch:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step(); ema.update(model); global_step += 1
            total += loss.item() * args.acc

        ema.apply(model)
        val, per_class = evaluate(model, val_loader, device)
        row = dict(epoch=epoch + 1, train_loss=total / len(train_loader),
                   val_iou=val, class_iou=per_class,
                   lr=optimizer.param_groups[0]['lr'])
        history.append(row)
        print(f"[ep {epoch+1:03d}] loss={row['train_loss']:.4f} val={val:.4f} "
              f"classes={','.join(f'{x:.4f}' for x in per_class)} "
              f"lr={row['lr']:.2e} time={time.time()-t0:.0f}s", flush=True)
        if val > best:
            best, stale = val, 0
            torch.save({'model_state_dict': model.state_dict(), 'val_iou': val,
                        'class_iou': per_class, 'epoch': epoch + 1,
                        'config': vars(args)}, out / f'{tag}_best.pth')
            print(f'[best] {best:.4f}', flush=True)
        else:
            stale += 1
        ema.restore(model)
        with open(out / f'{tag}_history.json', 'w') as f:
            json.dump(history, f, indent=2)
        if stale >= args.patience:
            print(f'[early-stop] best={best:.4f}', flush=True)
            break


if __name__ == '__main__':
    main()
