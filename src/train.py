"""
训练脚本
支持小麦+油菜（多标签）和水稻（单标签）两种模式
"""
import os
import sys
import time
import json
import random
import argparse
import numpy as np
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast

import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.dataset import CropSegmentationDataset, get_training_augmentation, get_validation_augmentation
from src.models import get_model, MultiLabelModel, SingleLabelModel
from src.losses import CombinedLoss, DiceLoss, MultiLabelLoss


def set_seed(seed=42):
    """固定随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_iou(pred, target, threshold=0.5):
    """计算单张图的IoU"""
    pred = torch.sigmoid(pred)
    pred_binary = (pred > threshold).float()
    
    # 确保形状一致
    if pred_binary.dim() > target.dim():
        pred_binary = pred_binary.squeeze(1)
    if target.dim() > pred_binary.dim():
        target = target.squeeze(1)
    
    intersection = (pred_binary * target).sum()
    union = pred_binary.sum() + target.sum() - intersection
    
    if union == 0:
        # 都为空
        return 1.0
    return (intersection / union).item()


def compute_batch_iou(preds, targets, threshold=0.5):
    """计算batch的平均IoU"""
    ious = []
    for i in range(preds.shape[0]):
        iou = compute_iou(preds[i], targets[i], threshold)
        ious.append(iou)
    return np.mean(ious)


def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device, epoch, log_interval=50):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    running_iou = 0.0
    num_batches = 0
    
    for batch_idx, (images, masks, _) in enumerate(dataloader):
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(images)
            # 处理形状匹配
            if outputs.shape[1] == 1 and masks.dim() == 3:
                # 水稻: outputs [B,1,H,W], masks [B,H,W]
                loss = criterion(outputs.squeeze(1), masks)
            else:
                loss = criterion(outputs, masks)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # 计算IoU
        with torch.no_grad():
            iou = compute_batch_iou(outputs, masks)
        
        running_loss += loss.item()
        running_iou += iou
        num_batches += 1
        
        if (batch_idx + 1) % log_interval == 0:
            avg_loss = running_loss / num_batches
            avg_iou = running_iou / num_batches
            print(f"  Epoch {epoch} [{batch_idx+1}/{len(dataloader)}] "
                  f"Loss: {avg_loss:.4f} IoU: {avg_iou:.4f}")
    
    return running_loss / num_batches, running_iou / num_batches


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    """验证"""
    model.eval()
    running_loss = 0.0
    running_iou = 0.0
    num_batches = 0
    
    for images, masks, _ in dataloader:
        images = images.to(device)
        masks = masks.to(device)
        
        with autocast():
            outputs = model(images)
            # 处理形状匹配
            if outputs.shape[1] == 1 and masks.dim() == 3:
                loss = criterion(outputs.squeeze(1), masks)
            else:
                loss = criterion(outputs, masks)
        
        iou = compute_batch_iou(outputs, masks)
        
        running_loss += loss.item()
        running_iou += iou
        num_batches += 1
    
    return running_loss / num_batches, running_iou / num_batches


def get_positive_weight(dataloader, device):
    """计算正样本权重用于BCE loss"""
    total_pixels = 0
    positive_pixels = 0
    
    for _, masks, _ in dataloader:
        total_pixels += masks.numel()
        positive_pixels += masks.sum().item()
    
    if positive_pixels == 0:
        return None
    
    # pos_weight = negative / positive
    pos_weight = (total_pixels - positive_pixels) / positive_pixels
    return torch.tensor([pos_weight], device=device)


def train(args):
    """主训练函数"""
    set_seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 数据集路径
    data_root = args.data_dir
    
    if args.mode == 'wheat_rape':
        # 小麦+油菜多标签
        train_image_dir = os.path.join(data_root, 'train/image/wheat_rape')
        train_label_dirs = [
            os.path.join(data_root, 'train/label/wheat'),
            os.path.join(data_root, 'train/label/rape')
        ]
        val_image_dir = os.path.join(data_root, 'val/image/wheat_rape')
        val_label_dirs = [
            os.path.join(data_root, 'val/label/wheat'),
            os.path.join(data_root, 'val/label/rape')
        ]
        num_classes = 2
        model_class = MultiLabelModel
    else:
        # 水稻单标签
        train_image_dir = os.path.join(data_root, 'train/image/rice')
        train_label_dirs = [os.path.join(data_root, 'train/label/rice')]
        val_image_dir = os.path.join(data_root, 'val/image/rice')
        val_label_dirs = [os.path.join(data_root, 'val/label/rice')]
        num_classes = 1
        model_class = SingleLabelModel
    
    # 数据增强
    train_transform = get_training_augmentation(args.img_size)
    val_transform = get_validation_augmentation(args.img_size)
    
    # 数据集
    train_dataset = CropSegmentationDataset(
        image_dir=train_image_dir,
        label_dirs=train_label_dirs,
        transform=train_transform,
        mode='multi' if args.mode == 'wheat_rape' else 'single'
    )
    
    val_dataset = CropSegmentationDataset(
        image_dir=val_image_dir,
        label_dirs=val_label_dirs,
        transform=val_transform,
        mode='multi' if args.mode == 'wheat_rape' else 'single'
    )
    
    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # 模型
    model = model_class(
        architecture=args.architecture,
        encoder_name=args.encoder
    )
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}, Trainable: {trainable_params:,}")
    
    # 损失函数
    if args.mode == 'wheat_rape':
        criterion = MultiLabelLoss(num_classes=2, loss_type='combined')
    else:
        criterion = CombinedLoss(bce_weight=0.5, dice_weight=0.5)
    
    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # 学习率调度
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # 混合精度
    scaler = GradScaler()
    
    # 训练循环
    best_iou = 0.0
    patience_counter = 0
    history = {'train_loss': [], 'train_iou': [], 'val_loss': [], 'val_iou': []}
    
    print(f"\nStarting training for {args.epochs} epochs...")
    print("=" * 60)
    
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        
        # 训练
        train_loss, train_iou = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch
        )
        
        # 验证
        val_loss, val_iou = validate(model, val_loader, criterion, device)
        
        # 更新学习率
        scheduler.step()
        
        elapsed = time.time() - start_time
        
        print(f"Epoch {epoch}/{args.epochs} ({elapsed:.1f}s) - "
              f"Train Loss: {train_loss:.4f} IoU: {train_iou:.4f} | "
              f"Val Loss: {val_loss:.4f} IoU: {val_iou:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_iou'].append(train_iou)
        history['val_loss'].append(val_loss)
        history['val_iou'].append(val_iou)
        
        # 保存最佳模型
        if val_iou > best_iou:
            best_iou = val_iou
            patience_counter = 0
            save_path = os.path.join(args.output_dir, f'best_{args.mode}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_iou': val_iou,
                'args': vars(args)
            }, save_path)
            print(f"  -> Saved best model (IoU: {best_iou:.4f})")
        else:
            patience_counter += 1
        
        # 早停
        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break
    
    # 保存最终模型
    save_path = os.path.join(args.output_dir, f'final_{args.mode}.pth')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_iou': val_iou,
        'args': vars(args)
    }, save_path)
    
    # 保存训练历史
    with open(os.path.join(args.output_dir, f'history_{args.mode}.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nTraining completed. Best Val IoU: {best_iou:.4f}")
    return best_iou


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train crop segmentation model')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, default='/root/competition_data/public')
    parser.add_argument('--mode', type=str, choices=['wheat_rape', 'rice'], required=True)
    parser.add_argument('--img_size', type=int, default=256)
    
    # 模型参数
    parser.add_argument('--architecture', type=str, default='unet',
                       choices=['unet', 'deeplabv3plus', 'fpn', 'pspnet', 'manet'])
    parser.add_argument('--encoder', type=str, default='efficientnet-b3')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--patience', type=int, default=20)
    
    # 输出
    parser.add_argument('--output_dir', type=str, default='/root/crop_segmentation/weights')
    
    args = parser.parse_args()
    train(args)
