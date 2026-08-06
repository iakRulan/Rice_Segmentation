"""Training v2: per-image loss + aux classification head + Copy-Paste + SWA.

Integrates the opt_patch fixes (user's P0/P1 roadmap):
  - MultiTaskLoss: per-image dice/iou/tversky + lovasz + BCE/focal (empty images
    keep their own gradient instead of being flattened away).
  - build_model(aux=True): image-level "has object" classification head, shared
    encoder. cls_logits are a free empty-image discriminator.
  - CopyPasteMixer: paste foreground patches from another image (never onto
    empty images by default) — the main lever for rape's small scattered targets.
  - SWA / top-k checkpoint averaging at the end.

Validation during training uses triple_threshold (t_hi/min_size/t_lo) so the
selected best epoch matches the final submission postprocessing.

Usage:
    .venv/Scripts/python.exe scripts/train_v2.py --mode wheat_rape --arch deeplabv3plus \
        --encoder mit_b3 --aux --copy_paste 0.4 --focal_gamma 2.0 --pos_weight 1.3 \
        --cls_w 0.5 --tag v2
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import DATA, CKPT
import opt_patch.model_multitask as mm
from opt_patch.losses_v2 import MultiTaskLoss, lovasz_hinge
from opt_patch.copy_paste import CopyPasteMixer
from opt_patch.postproc import triple_threshold

import albumentations as A
from albumentations.pytorch import ToTensorV2


# ---------------- Augmentation ----------------
def get_train_aug(img_size):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Transpose(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.5, border_mode=0),
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.5),
        A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=15, p=0.3),
        A.GaussNoise(std_range=(0.01, 0.06), per_channel=True, p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], is_check_shapes=False)


def get_val_aug(img_size):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], is_check_shapes=False)


def foreground_crop(image, mask, size, canvas):
    H = W = canvas
    fg = (mask > 0)
    if fg.sum() == 0:
        y = np.random.randint(0, H - size + 1)
        x = np.random.randint(0, W - size + 1)
        return image[y:y + size, x:x + size], mask[y:y + size, x:x + size]
    ys, xs = np.where(fg)
    cy, cx = int(np.mean(ys)), int(np.mean(xs))
    j = size // 3
    cy = int(np.clip(cy + np.random.randint(-j, j + 1), size // 2, H - size // 2))
    cx = int(np.clip(cx + np.random.randint(-j, j + 1), size // 2, W - size // 2))
    y0, x0 = max(0, cy - size // 2), max(0, cx - size // 2)
    if y0 + size > H: y0 = H - size
    if x0 + size > W: x0 = W - size
    return image[y0:y0 + size, x0:x0 + size], mask[y0:y0 + size, x0:x0 + size]


class CropDataset:
    def __init__(self, image_dir, label_dirs, mode, img_size, transform,
                 crop_zoom=0.0, canvas=None, copy_paste=0.0):
        self.image_dir = image_dir
        self.label_dirs = label_dirs
        self.mode = mode
        self.img_size = img_size
        self.transform = transform
        self.crop_zoom = crop_zoom
        self.canvas = canvas if (canvas and crop_zoom > 0) else None
        self.images = sorted(f for f in os.listdir(image_dir) if f.endswith('.png'))
        self.cp = CopyPasteMixer(p=copy_paste, max_objs=3) if copy_paste > 0 else None

    def __len__(self):
        return len(self.images)

    def _raw(self, idx):
        """Load image + masks as float arrays (for Copy-Paste source)."""
        img_name = self.images[idx]
        image = np.array(Image.open(os.path.join(self.image_dir, img_name)).convert('RGB'))
        masks = []
        for ld in self.label_dirs:
            p = os.path.join(ld, img_name)
            if os.path.exists(p):
                masks.append((np.array(Image.open(p)) > 0).astype(np.float32))
            else:
                masks.append(np.zeros(image.shape[:2], np.float32))
        if self.mode == 'multi':
            masks = np.stack(masks, axis=0)
            masks = np.transpose(masks, (1, 2, 0))
        else:
            masks = masks[0]
        return image, masks

    def __getitem__(self, idx):
        image, masks = self._raw(idx)

        # ---- Copy-Paste (paste foreground from another image, never onto empty) ----
        if self.cp is not None:
            src_i = np.random.randint(len(self))
            s_img, s_msk = self._raw(src_i)
            image, masks = self.cp(image, masks, s_img, s_msk)

        # ---- small-object zoom crop ----
        if self.canvas and self.canvas > self.img_size and np.random.rand() < self.crop_zoom:
            image = np.array(Image.fromarray(image).resize((self.canvas, self.canvas), Image.BILINEAR))
            if self.mode == 'multi':
                masks_rz = np.stack([np.array(Image.fromarray((masks[:, :, c] * 255).astype(np.uint8))
                                              .resize((self.canvas, self.canvas), Image.NEAREST)) / 255.0
                                     for c in range(masks.shape[-1])], axis=-1)
            else:
                masks_rz = np.array(Image.fromarray((masks * 255).astype(np.uint8))
                                    .resize((self.canvas, self.canvas), Image.NEAREST)) / 255.0
            image, masks = foreground_crop(image, masks_rz, self.img_size, self.canvas)

        aug = self.transform(image=image, mask=masks)
        image, masks = aug['image'], aug['mask']
        if isinstance(masks, torch.Tensor):
            masks = masks.permute(2, 0, 1) if self.mode == 'multi' else masks.unsqueeze(0)
        else:
            masks = torch.from_numpy(np.transpose(masks, (2, 0, 1)) if self.mode == 'multi' else masks[None])
        return image, masks.float(), self.images[idx]


# ---------------- EMA ----------------
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = p.data.clone()

    def update(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.shadow[n].mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def apply_shadow(self, model):
        self.backup = {}
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.backup[n] = p.data.clone()
                p.data.copy_(self.shadow[n])

    def restore(self, model):
        for n, p in model.named_parameters():
            if n in self.backup:
                p.data.copy_(self.backup[n])
        self.backup = {}


def compute_iou(pred, target):
    inter = (pred & target).sum()
    union = (pred | target).sum()
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return inter / union


def evaluate(model, loader, device, img_size):
    """Per-image IoU with triple-threshold settings tuned once (fixed t_hi/t_lo)."""
    from opt_patch.postproc import iou
    model.eval()
    class_ious = []
    with torch.no_grad():
        for images, masks, _ in loader:
            images = images.to(device)
            seg, cls = mm.unpack(model(images))
            out = torch.sigmoid(seg).cpu().numpy()
            m = masks.cpu().numpy()
            for c in range(out.shape[1]):
                per = []
                for j in range(len(out)):
                    pr = triple_threshold(out[j, c], 0.50, 0, 0.45)
                    per.append(iou(pr, (m[j, c] > 0).astype(np.uint8)))
                if len(class_ious) <= c:
                    class_ious.append([])
                class_ious[c].extend(per)
    means = [np.mean(x) for x in class_ious]
    return np.mean(means), means


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['wheat_rape', 'rice', 'wheat', 'rape'], required=True)
    ap.add_argument('--arch', default='unet')
    ap.add_argument('--encoder', default='mit_b3')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--epochs', type=int, default=130)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--acc', type=int, default=3)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--img_size', type=int, default=256)
    ap.add_argument('--crop_zoom', type=float, default=0.0)
    ap.add_argument('--canvas', type=int, default=512)
    ap.add_argument('--patience', type=int, default=40)
    ap.add_argument('--ema_decay', type=float, default=0.999)
    ap.add_argument('--lovasz_w', type=float, default=0.5)
    ap.add_argument('--iou_w', type=float, default=0.0)
    ap.add_argument('--tv_w', type=float, default=0.0)
    ap.add_argument('--cls_w', type=float, default=0.5, help='aux cls head loss weight')
    ap.add_argument('--focal_gamma', type=float, default=0.0)
    ap.add_argument('--pos_weight', type=float, default=1.0)
    ap.add_argument('--aux', action='store_true', help='add image-level classification head')
    ap.add_argument('--copy_paste', type=float, default=0.0, help='Copy-Paste prob')
    ap.add_argument('--swa_k', type=int, default=5, help='top-k checkpoints to average at end (0=off)')
    ap.add_argument('--tag', default='v2')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--data_root', default=None)
    ap.add_argument('--val_root', default=None)
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True

    data_dir = Path(args.data_root) if args.data_root else DATA
    val_dir = Path(args.val_root) if args.val_root else data_dir
    classes = 2 if args.mode == 'wheat_rape' else 1

    model = mm.build_model(args.arch, args.encoder, classes, aux=args.aux).to(device)
    print(f'[config] arch={args.arch} enc={args.encoder} aux={args.aux} params='
          f'{sum(p.numel() for p in model.parameters())/1e6:.1f}M seed={args.seed} '
          f'bs={args.batch_size} acc={args.acc} cp={args.copy_paste}', flush=True)

    if args.mode == 'wheat_rape':
        tr_img, tr_lbl = data_dir / 'train/image/wheat_rape', [data_dir / 'train/label/wheat', data_dir / 'train/label/rape']
        va_img, va_lbl = val_dir / 'val/image/wheat_rape', [val_dir / 'val/label/wheat', val_dir / 'val/label/rape']
        mode = 'multi'
    elif args.mode in ('wheat', 'rape'):
        tr_img, tr_lbl = data_dir / 'train/image/wheat_rape', [data_dir / f'train/label/{args.mode}']
        va_img, va_lbl = val_dir / 'val/image/wheat_rape', [val_dir / f'val/label/{args.mode}']
        mode = 'single'
    else:
        tr_img, tr_lbl = data_dir / 'train/image/rice', [data_dir / 'train/label/rice']
        va_img, va_lbl = val_dir / 'val/image/rice', [val_dir / 'val/label/rice']
        mode = 'single'

    train_ds = CropDataset(tr_img, tr_lbl, mode, args.img_size, get_train_aug(args.img_size),
                           args.crop_zoom, args.canvas, args.copy_paste)
    val_ds = CropDataset(va_img, va_lbl, mode, args.img_size, get_val_aug(args.img_size))

    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)
    print(f'[data] train={len(train_ds)} val={len(val_ds)}', flush=True)

    criterion = MultiTaskLoss(bce=1.0, dice=1.0, lovasz=args.lovasz_w,
                              iou=args.iou_w, tversky=args.tv_w, cls=args.cls_w,
                              focal_gamma=args.focal_gamma, pos_weight=args.pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    warmup = int(total_steps * 0.05)

    def lr_fn(step):
        if step < warmup:
            return step / max(1, warmup)
        prog = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)
    scaler = torch.amp.GradScaler('cuda')
    ema = EMA(model, args.ema_decay)

    best = 0.0
    patience = 0
    _TOP = []  # rolling top-k checkpoints for SWA
    tag = f'{args.tag}_{args.mode}_{args.encoder.replace("-", "")}_{args.seed}'
    hist = {'train_loss': [], 'val_iou': [], 'lr': []}
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        t0 = time.time()
        for i, (images, masks, _) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)
            with torch.amp.autocast('cuda'):
                seg, cls = mm.unpack(model(images))
                loss = criterion(seg, masks, cls) / args.acc
            scaler.scale(loss).backward()
            if (i + 1) % args.acc == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                ema.update(model)
                sched.step()
                global_step += 1
            total_loss += loss.item() * args.acc
            if (i + 1) % 200 == 0:
                print(f'  ep{epoch+1} it{i+1} loss {total_loss/(i+1):.4f} lr {sched.get_last_lr()[0]:.2e}', flush=True)
        train_loss = total_loss / len(train_loader)

        ema.apply_shadow(model)
        val_iou, class_ious = evaluate(model, val_loader, device, args.img_size)
        ema.restore(model)
        print(f'[ep{epoch+1}] loss {train_loss:.4f} val_iou {val_iou:.4f} '
              f'({",".join(f"{x:.4f}" for x in class_ious)}) time {time.time()-t0:.0f}s', flush=True)
        hist['train_loss'].append(train_loss)
        hist['val_iou'].append(val_iou)
        hist['lr'].append(sched.get_last_lr()[0])

        # rolling top-k checkpoints (keep at most swa_k best), plus always the best
        ema.apply_shadow(model)
        ck = {'model_state_dict': {k: v.clone() for k, v in model.state_dict().items()},
              'val_iou': val_iou, 'epoch': epoch, 'config': vars(args)}
        improved = val_iou > best
        if improved:
            best = val_iou
            patience = 0
            torch.save(ck, CKPT / f'{tag}_best.pth')
            print(f'  -> saved best {best:.4f}', flush=True)
        else:
            patience += 1
        if args.swa_k > 0:
            p = CKPT / f'{tag}_ep{epoch:03d}.pth'
            torch.save(ck, p)
            _TOP.append((float(val_iou), str(p)))
            _TOP.sort(key=lambda q: -q[0])
            while len(_TOP) > args.swa_k:
                _, victim = _TOP.pop()
                if os.path.exists(victim):
                    os.remove(victim)
        ema.restore(model)
        if patience >= args.patience:
            print(f'[early stop] ep{epoch+1}', flush=True)
            break

    # ---- SWA / top-k checkpoint averaging ----
    if args.swa_k > 0 and _TOP:
        top_paths = [p for _, p in _TOP[:args.swa_k]]
        avg = mm.average_state_dicts(top_paths)
        model.load_state_dict(avg, strict=False)
        torch.save({'model_state_dict': avg, 'val_iou': best,
                    'epoch': epoch, 'config': vars(args), 'swa_top': top_paths},
                   CKPT / f'{tag}_swa.pth')
        print(f'[swa] averaged {len(top_paths)} ckpts -> {tag}_swa.pth', flush=True)

    torch.save({'model_state_dict': model.state_dict(), 'ema': ema.shadow,
                'val_iou': best, 'epoch': epoch, 'config': vars(args)},
               CKPT / f'{tag}_last.pth')
    with open(CKPT / f'{tag}_history.json', 'w') as f:
        json.dump(hist, f)
    print(f'[done] best {best:.4f}', flush=True)


if __name__ == '__main__':
    main()
