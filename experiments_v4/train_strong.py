"""Strong training script for crop segmentation.
- Configurable arch/encoder via smp (256x256 native)
- Loss: BCE + soft Dice + Lovasz-hinge (binary IoU optimizer)
- AdamW + cosine warmup, AMP, grad clip, EMA
- Multi-seed capable; saves best + last, writes history
"""
import os, sys, argparse, json, math, time
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, '/root/crop_segmentation')
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

DATA = '/root/competition_data/public'
OUT = '/root/crop_segmentation/weights'


# ---------------- Lovasz hinge (binary) ----------------
def lovasz_grad(gt_sorted):
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_hinge_flat(logits, labels):
    if len(labels) == 0:
        return logits.sum() * 0.0
    signs = 2.0 * labels.float() - 1.0
    errors = 1.0 - logits * signs
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    return torch.dot(F.relu(errors_sorted), grad)


def lovasz_hinge(logits, labels, per_image=True):
    if per_image:
        return _lovasz_per_image_vectorized(logits, labels)
    else:
        return lovasz_hinge_flat(logits.reshape(-1), labels.reshape(-1))


def _lovasz_per_image_vectorized(logits, labels):
    losses = []
    B = logits.size(0)
    for i in range(B):
        lg = logits[i].reshape(-1)
        lb = labels[i].reshape(-1)
        losses.append(lovasz_hinge_flat(lg, lb))
    return torch.mean(torch.stack(losses))


class CombinedLoss(nn.Module):
    def __init__(self, bce=1.0, dice=1.0, lovasz=0.5):
        super().__init__()
        self.bce_w = bce
        self.dice_w = dice
        self.lov_w = lovasz
        self.bce = nn.BCEWithLogitsLoss()
        self.smooth = 1.0

    def soft_dice(self, logits, target):
        p = torch.sigmoid(logits).reshape(-1)
        t = target.reshape(-1)
        inter = (p * t).sum()
        return 1.0 - (2.0 * inter + self.smooth) / (p.sum() + t.sum() + self.smooth)

    def forward(self, logits, target):
        bce = self.bce(logits, target) * self.bce_w
        dice = self.soft_dice(logits, target) * self.dice_w
        lov = torch.tensor(0.0, device=logits.device)
        if self.lov_w > 0:
            # average per-class lovasz (handles multi-channel)
            lovs = []
            for c in range(logits.size(1)):
                lovs.append(lovasz_hinge(logits[:, c], target[:, c]))
            lov = torch.mean(torch.stack(lovs)) * self.lov_w
        return bce + dice + lov


# ---------------- Augmentation ----------------
def get_train_aug(img_size):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Transpose(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.5, border_mode=0),
        A.OneOf([
            A.ElasticTransform(alpha=120, sigma=120*0.05, p=0.5),
            A.GridDistortion(p=0.5),
            A.OpticalDistortion(distort_limit=1.5, p=0.5),
        ], p=0.3),
        A.OneOf([
            A.CLAHE(clip_limit=4.0, p=0.5),
            A.Sharpen(p=0.5),
            A.Emboss(p=0.5),
            A.GaussianBlur(blur_limit=(3, 5), p=0.5),
        ], p=0.4),
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.5),
        A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=15, p=0.3),
        A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(16, 48),
                        hole_width_range=(16, 48), fill=0, fill_mask=0, p=0.2),
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


# ---------------- Dataset ----------------
class CropDataset(Dataset):
    def __init__(self, image_dir, label_dirs, mode, transform):
        self.image_dir = image_dir
        self.label_dirs = label_dirs
        self.mode = mode
        self.transform = transform
        self.images = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        image = np.array(Image.open(os.path.join(self.image_dir, img_name)).convert('RGB'))
        masks = []
        for ld in self.label_dirs:
            p = os.path.join(ld, img_name)
            if os.path.exists(p):
                m = (np.array(Image.open(p)) > 0).astype(np.float32)
            else:
                m = np.zeros(image.shape[:2], np.float32)
            masks.append(m)
        if self.mode == 'multi':
            masks = np.stack(masks, axis=0)          # (C,H,W)
            masks = np.transpose(masks, (1, 2, 0))   # (H,W,C)
        else:
            masks = masks[0]
        aug = self.transform(image=image, mask=masks)
        image = aug['image']
        masks = aug['mask']
        if isinstance(masks, torch.Tensor):
            if self.mode == 'multi':
                masks = masks.permute(2, 0, 1)
            else:
                masks = masks.unsqueeze(0)
        else:
            if self.mode == 'multi':
                masks = np.transpose(masks, (2, 0, 1))
            else:
                masks = masks[None]
        return image, masks.float() if isinstance(masks, torch.Tensor) else torch.from_numpy(masks), img_name


# ---------------- EMA ----------------
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = p.data.clone()

    def update(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.shadow[n].mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def apply_shadow(self, model):
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


def evaluate(model, loader, device):
    model.eval()
    class_ious = []
    with torch.no_grad():
        for images, masks, _ in loader:
            images = images.to(device)
            out = torch.sigmoid(model(images)).cpu().numpy()
            m = masks.cpu().numpy()
            for c in range(out.shape[1]):
                pb = (out[:, c] > 0.5).astype(np.uint8)
                tb = (m[:, c] > 0).astype(np.uint8)
                per = [compute_iou(pb[j], tb[j]) for j in range(len(pb))]
                if len(class_ious) <= c:
                    class_ious.append([])
                class_ious[c].extend(per)
    means = [np.mean(x) for x in class_ious]
    return np.mean(means), means


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['wheat_rape', 'rice', 'wheat', 'rape'], required=True)
    ap.add_argument('--arch', default='unet')
    ap.add_argument('--encoder', default='efficientnet-b3')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--epochs', type=int, default=120)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--img_size', type=int, default=256)
    ap.add_argument('--acc', type=int, default=1)
    ap.add_argument('--patience', type=int, default=40)
    ap.add_argument('--ema_decay', type=float, default=0.999)
    ap.add_argument('--lovasz_w', type=float, default=0.5, help='weight of lovasz-hinge loss (0 disables)')
    ap.add_argument('--tag', default='strong')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--data_root', default=DATA, help='train root (e.g. train_plus for self-training)')
    args = ap.parse_args()

    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True

    classes = 2 if args.mode == 'wheat_rape' else 1
    model = smp.Unet(encoder_name=args.encoder, encoder_weights='imagenet',
                     in_channels=3, classes=classes, activation=None)
    # if using deeplab/fpn we still create Unet unless arch set
    if args.arch == 'deeplabv3plus':
        model = smp.DeepLabV3Plus(encoder_name=args.encoder, encoder_weights='imagenet',
                                  in_channels=3, classes=classes, activation=None)
    elif args.arch == 'fpn':
        model = smp.FPN(encoder_name=args.encoder, encoder_weights='imagenet',
                        in_channels=3, classes=classes, activation=None)
    elif args.arch == 'unetpp':
        model = smp.UnetPlusPlus(encoder_name=args.encoder, encoder_weights='imagenet',
                                 in_channels=3, classes=classes, activation=None)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'[config] arch={args.arch} encoder={args.encoder} params={n_params/1e6:.1f}M seed={seed}', flush=True)

    data_dir = args.data_root
    val_dir = DATA
    if args.mode == 'wheat_rape':
        tr_img, tr_lbl = os.path.join(data_dir, 'train/image/wheat_rape'), [
            os.path.join(data_dir, 'train/label/wheat'), os.path.join(data_dir, 'train/label/rape')]
        va_img, va_lbl = os.path.join(val_dir, 'val/image/wheat_rape'), [
            os.path.join(val_dir, 'val/label/wheat'), os.path.join(val_dir, 'val/label/rape')]
        mode = 'multi'
    elif args.mode in ('wheat', 'rape'):
        tr_img, tr_lbl = os.path.join(data_dir, 'train/image/wheat_rape'), [
            os.path.join(data_dir, f'train/label/{args.mode}')]
        va_img, va_lbl = os.path.join(val_dir, 'val/image/wheat_rape'), [
            os.path.join(val_dir, f'val/label/{args.mode}')]
        mode = 'single'
    else:
        tr_img, tr_lbl = os.path.join(data_dir, 'train/image/rice'), [os.path.join(data_dir, 'train/label/rice')]
        va_img, va_lbl = os.path.join(val_dir, 'val/image/rice'), [os.path.join(val_dir, 'val/label/rice')]
        mode = 'single'

    train_ds = CropDataset(tr_img, tr_lbl, mode, get_train_aug(args.img_size))
    val_ds = CropDataset(va_img, va_lbl, mode, get_val_aug(args.img_size))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True, persistent_workers=True)
    print(f'[data] train={len(train_ds)} val={len(val_ds)}', flush=True)

    criterion = CombinedLoss(bce=1.0, dice=1.0, lovasz=args.lovasz_w)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    warmup = int(total_steps * 0.05)

    def lr_fn(step):
        if step < warmup:
            return step / max(1, warmup)  # factor 0 -> 1
        prog = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * prog))  # factor 1 -> 0

    sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)
    scaler = torch.cuda.amp.GradScaler()
    ema = EMA(model, args.ema_decay)

    best = 0.0
    patience = 0
    hist = {'train_loss': [], 'val_iou': [], 'lr': []}
    tag = f'{args.tag}_{args.mode}_{args.encoder.replace("-","")}_{seed}'
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        t0 = time.time()
        for i, (images, masks, _) in enumerate(train_loader):
            images = images.to(device)
            masks = masks.to(device)
            with torch.cuda.amp.autocast():
                out = model(images)
                loss = criterion(out, masks) / args.acc
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
            if (i + 1) % 100 == 0:
                print(f'  ep{epoch+1} it{i+1} loss {total_loss/(i+1):.4f} lr {sched.get_last_lr()[0]:.2e}', flush=True)
        train_loss = total_loss / len(train_loader)

        ema.apply_shadow(model)
        val_iou, class_ious = evaluate(model, val_loader, device)
        ema.restore(model)

        print(f'[ep{epoch+1}] loss {train_loss:.4f} val_iou {val_iou:.4f} '
              f'({",".join(f"{x:.4f}" for x in class_ious)}) time {time.time()-t0:.0f}s', flush=True)
        hist['train_loss'].append(train_loss)
        hist['val_iou'].append(val_iou)
        hist['lr'].append(sched.get_last_lr()[0])

        if val_iou > best:
            best = val_iou
            patience = 0
            ema.apply_shadow(model)
            torch.save({'model_state_dict': {k: v.clone() for k, v in model.state_dict().items()},
                        'val_iou': val_iou, 'epoch': epoch, 'config': vars(args)},
                       os.path.join(OUT, f'{tag}_best.pth'))
            ema.restore(model)
            print(f'  -> saved best {best:.4f}', flush=True)
        else:
            patience += 1
        if patience >= args.patience:
            print(f'[early stop] ep{epoch+1}', flush=True)
            break

    # save EMA weights as final (usually better than raw)
    torch.save({'model_state_dict': model.state_dict(), 'ema': ema.shadow,
                'val_iou': best, 'epoch': epoch, 'config': vars(args)},
               os.path.join(OUT, f'{tag}_last.pth'))
    with open(os.path.join(OUT, f'{tag}_history.json'), 'w') as f:
        json.dump(hist, f)
    print(f'[done] best {best:.4f}', flush=True)


if __name__ == '__main__':
    main()
