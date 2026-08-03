"""
优化训练脚本 v2
- 更强的数据增强 (ColorJitter, RandomBrightnessContrast等)
- 更大的图像尺寸 (512x512, 使用梯度累积管理显存)
- Tversky loss替代BCE+Dice
- 更长的训练周期 + 更小的最小学习率
- 多尺度训练
"""
import os
import sys
import argparse
import json
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

from src.models import MultiLabelModel, SingleLabelModel


class TverskyLoss(nn.Module):
    """Tversky Loss - 更好地处理类别不平衡"""
    def __init__(self, alpha=0.7, beta=0.3, smooth=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        
        # 展平
        pred = pred.view(-1)
        target = target.view(-1)
        
        # TP, FP, FN
        tp = (pred * target).sum()
        fp = ((1 - target) * pred).sum()
        fn = (target * (1 - pred)).sum()
        
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1 - tversky


class FocalTverskyLoss(nn.Module):
    """Focal Tversky Loss"""
    def __init__(self, alpha=0.7, beta=0.3, gamma=1.33, smooth=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred = pred.view(-1)
        target = target.view(-1)
        
        tp = (pred * target).sum()
        fp = ((1 - target) * pred).sum()
        fn = (target * (1 - pred)).sum()
        
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return (1 - tversky) ** self.gamma


class CombinedLoss(nn.Module):
    """组合损失: Focal + Tversky + BCE"""
    def __init__(self, focal_weight=0.3, tversky_weight=0.3, bce_weight=0.4):
        super().__init__()
        self.focal_weight = focal_weight
        self.tversky_weight = tversky_weight
        self.bce_weight = bce_weight
        self.focal = smp.losses.FocalLoss(mode='binary')
        self.tversky = FocalTverskyLoss(alpha=0.7, beta=0.3)
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, pred, target):
        return (self.focal_weight * self.focal(pred, target) +
                self.tversky_weight * self.tversky(pred, target) +
                self.bce_weight * self.bce(pred, target))


def get_training_augmentation_v2(img_size=512):
    """更强的数据增强"""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Transpose(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.15,
            scale_limit=0.3,
            rotate_limit=45,
            p=0.7,
            border_mode=0
        ),
        A.OneOf([
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.5),
            A.GridDistortion(p=0.5),
            A.OpticalDistortion(distort_limit=2, p=0.5),
        ], p=0.4),
        A.OneOf([
            A.CLAHE(clip_limit=4.0, p=0.5),
            A.Sharpen(p=0.5),
            A.Emboss(p=0.5),
        ], p=0.3),
        A.RandomBrightnessContrast(
            brightness_limit=0.3,
            contrast_limit=0.3,
            p=0.5
        ),
        A.HueSaturationValue(
            hue_shift_limit=20,
            sat_shift_limit=30,
            val_shift_limit=20,
            p=0.3
        ),
        A.GaussNoise(var_limit=(10, 50), p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], is_check_shapes=False)


def get_validation_augmentation(img_size=512):
    """验证集数据增强"""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], is_check_shapes=False)


class CropDataset(Dataset):
    """农作物分割数据集"""
    def __init__(self, image_dir, label_dirs, mode='multi', transform=None, val=False):
        self.image_dir = image_dir
        self.label_dirs = label_dirs
        self.mode = mode
        self.transform = transform
        self.val = val
        
        self.images = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
    
    def __len__(self):
        return len(self.images)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = np.array(Image.open(img_path).convert('RGB'))
        
        masks = []
        for label_dir in self.label_dirs:
            label_path = os.path.join(label_dir, img_name)
            if os.path.exists(label_path):
                mask = np.array(Image.open(label_path))
                mask = (mask > 0).astype(np.float32)
            else:
                mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
            masks.append(mask)
        
        if self.mode == 'multi':
            masks = np.stack(masks, axis=0)  # (C, H, W)
            masks = np.transpose(masks, (1, 2, 0))  # (H, W, C)
        else:
            masks = masks[0]  # (H, W)
        
        if self.transform:
            augmented = self.transform(image=image, mask=masks)
            image = augmented['image']
            masks = augmented['mask']
            if self.mode == 'multi':
                masks = np.transpose(masks, (2, 0, 1))  # (H, W, C) -> (C, H, W)
            else:
                masks = np.expand_dims(masks, axis=0)  # (H, W) -> (1, H, W)
        
        return image, masks, img_name


def compute_iou(pred, target, threshold=0.5):
    """计算IoU"""
    pred_binary = (pred > threshold).astype(np.uint8)
    target_binary = (target > 0).astype(np.uint8)
    
    intersection = np.logical_and(pred_binary, target_binary).sum()
    union = np.logical_or(pred_binary, target_binary).sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union


def train_one_epoch(model, dataloader, optimizer, criterion, scaler, device, accumulation_steps=1):
    """训练一个epoch，支持梯度累积"""
    model.train()
    total_loss = 0
    total_iou = 0
    num_batches = 0
    
    optimizer.zero_grad()
    
    for i, (images, masks, _) in enumerate(tqdm(dataloader, desc='Training')):
        images = images.to(device)
        masks = masks.to(device)
        
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, masks) / accumulation_steps
        
        scaler.scale(loss).backward()
        
        if (i + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps
        
        # 计算IoU
        with torch.no_grad():
            preds = torch.sigmoid(outputs).cpu().numpy()
            masks_np = masks.cpu().numpy()
            
            if preds.shape[1] > 1:
                # 多标签 - 计算每个类别的IoU取平均
                class_ious = []
                for c in range(preds.shape[1]):
                    pred_binary = (preds[:, c] > 0.5).astype(np.uint8)
                    mask_binary = (masks_np[:, c] > 0).astype(np.uint8)
                    ious = []
                    for j in range(len(pred_binary)):
                        intersection = np.logical_and(pred_binary[j], mask_binary[j]).sum()
                        union = np.logical_or(pred_binary[j], mask_binary[j]).sum()
                        if union == 0:
                            iou = 1.0 if intersection == 0 else 0.0
                        else:
                            iou = intersection / union
                        ious.append(iou)
                    class_ious.append(np.mean(ious))
                total_iou += np.mean(class_ious)
            else:
                pred_binary = (preds[:, 0] > 0.5).astype(np.uint8)
                mask_binary = (masks_np[:, 0] > 0).astype(np.uint8)
                ious = []
                for j in range(len(pred_binary)):
                    intersection = np.logical_and(pred_binary[j], mask_binary[j]).sum()
                    union = np.logical_or(pred_binary[j], mask_binary[j]).sum()
                    if union == 0:
                        iou = 1.0 if intersection == 0 else 0.0
                    else:
                        iou = intersection / union
                    ious.append(iou)
                total_iou += np.mean(ious)
            num_batches += 1
    
    return total_loss / num_batches, total_iou / num_batches


def validate(model, dataloader, criterion, device, mode='multi'):
    """验证"""
    model.eval()
    total_loss = 0
    total_iou = 0
    num_batches = 0
    
    with torch.no_grad():
        for images, masks, _ in tqdm(dataloader, desc='Validation'):
            images = images.to(device)
            masks = masks.to(device)
            
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)
            
            total_loss += loss.item()
            
            preds = torch.sigmoid(outputs).cpu().numpy()
            masks_np = masks.cpu().numpy()
            
            if mode == 'multi':
                # 计算每个类别的IoU，取平均
                class_ious = []
                for c in range(preds.shape[1]):
                    pred_binary = (preds[:, c] > 0.5).astype(np.uint8)
                    mask_binary = (masks_np[:, c] > 0).astype(np.uint8)
                    
                    ious = []
                    for j in range(len(pred_binary)):
                        intersection = np.logical_and(pred_binary[j], mask_binary[j]).sum()
                        union = np.logical_or(pred_binary[j], mask_binary[j]).sum()
                        if union == 0:
                            iou = 1.0 if intersection == 0 else 0.0
                        else:
                            iou = intersection / union
                        ious.append(iou)
                    class_ious.append(np.mean(ious))
                
                total_iou += np.mean(class_ious)
            else:
                pred_binary = (preds[:, 0] > 0.5).astype(np.uint8)
                mask_binary = (masks_np[:, 0] > 0).astype(np.uint8)
                
                ious = []
                for j in range(len(pred_binary)):
                    intersection = np.logical_and(pred_binary[j], mask_binary[j]).sum()
                    union = np.logical_or(pred_binary[j], mask_binary[j]).sum()
                    if union == 0:
                        iou = 1.0 if intersection == 0 else 0.0
                    else:
                        iou = intersection / union
                    ious.append(iou)
                
                total_iou += np.mean(ious)
            
            num_batches += 1
    
    return total_loss / num_batches, total_iou / num_batches


def main():
    parser = argparse.ArgumentParser(description='Optimized training v2')
    parser.add_argument('--mode', type=str, choices=['multi', 'single'], required=True)
    parser.add_argument('--architecture', type=str, default='unet')
    parser.add_argument('--encoder', type=str, default='efficientnet-b3')
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--accumulation_steps', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--min_lr', type=float, default=1e-7)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--output_dir', type=str, default='/root/crop_segmentation/weights')
    parser.add_argument('--loss_type', type=str, default='focal_tversky', 
                       choices=['bce_dice', 'focal', 'tversky', 'focal_tversky'])
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Image size: {args.img_size}")
    print(f"Batch size: {args.batch_size} x {args.accumulation_steps} = {args.batch_size * args.accumulation_steps} effective")
    
    # 创建模型
    if args.mode == 'multi':
        model = MultiLabelModel(architecture=args.architecture, encoder_name=args.encoder)
    else:
        model = SingleLabelModel(architecture=args.architecture, encoder_name=args.encoder)
    
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total_params:,}")
    
    # 数据
    data_dir = '/root/competition_data/public'
    
    if args.mode == 'multi':
        image_dir = os.path.join(data_dir, 'train/image/wheat_rape')
        label_dirs = [
            os.path.join(data_dir, 'train/label/wheat'),
            os.path.join(data_dir, 'train/label/rape'),
        ]
    else:
        image_dir = os.path.join(data_dir, 'train/image/rice')
        label_dirs = [os.path.join(data_dir, 'train/label/rice')]
    
    train_transform = get_training_augmentation_v2(args.img_size)
    val_transform = get_validation_augmentation(args.img_size)
    
    # 训练集
    train_dataset = CropDataset(image_dir, label_dirs, mode=args.mode, transform=train_transform, val=False)
    
    # 验证集 - 使用单独的val目录
    if args.mode == 'multi':
        val_image_dir = os.path.join(data_dir, 'val/image/wheat_rape')
        val_label_dirs = [
            os.path.join(data_dir, 'val/label/wheat'),
            os.path.join(data_dir, 'val/label/rape'),
        ]
    else:
        val_image_dir = os.path.join(data_dir, 'val/image/rice')
        val_label_dirs = [os.path.join(data_dir, 'val/label/rice')]
    
    val_dataset = CropDataset(val_image_dir, val_label_dirs, mode=args.mode, transform=val_transform, val=True)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                              num_workers=8, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                            num_workers=8, pin_memory=True, persistent_workers=True)
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # 损失函数
    if args.loss_type == 'focal_tversky':
        criterion = CombinedLoss()
    elif args.loss_type == 'focal':
        criterion = smp.losses.FocalLoss(mode='binary')
    elif args.loss_type == 'tversky':
        criterion = FocalTverskyLoss()
    else:
        criterion = smp.losses.DiceLoss(mode='binary') + nn.BCEWithLogitsLoss()
    
    # 优化器和调度器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2, eta_min=args.min_lr
    )
    
    scaler = GradScaler()
    
    # 训练循环
    best_iou = 0
    patience_counter = 0
    history = {'train_loss': [], 'train_iou': [], 'val_loss': [], 'val_iou': [], 'lr': []}
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print(f"LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        train_loss, train_iou = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device, args.accumulation_steps
        )
        val_loss, val_iou = validate(model, val_loader, criterion, device, args.mode)
        
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f} IoU: {train_iou:.4f}")
        print(f"Val Loss: {val_loss:.4f} IoU: {val_iou:.4f}")
        
        history['train_loss'].append(train_loss)
        history['train_iou'].append(train_iou)
        history['val_loss'].append(val_loss)
        history['val_iou'].append(val_iou)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # 保存最佳模型
        if val_iou > best_iou:
            best_iou = val_iou
            patience_counter = 0
            save_path = os.path.join(args.output_dir, f'v2_best_{args.mode}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_iou': val_iou,
                'train_iou': train_iou,
            }, save_path)
            print(f"  -> Saved best model (IoU: {best_iou:.4f})")
        else:
            patience_counter += 1
        
        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    # 保存历史
    history_path = os.path.join(args.output_dir, f'history_v2_{args.mode}.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nTraining completed. Best Val IoU: {best_iou:.4f}")


if __name__ == '__main__':
    main()
