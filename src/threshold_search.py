"""
阈值优化脚本
在验证集上搜索最佳阈值
"""
import os
import sys
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

from src.dataset import CropSegmentationDataset, get_validation_augmentation
from src.models import MultiLabelModel, SingleLabelModel


def compute_iou(pred, target, threshold=0.5):
    """计算IoU"""
    pred_binary = (pred > threshold).astype(np.uint8)
    target_binary = (target > 0).astype(np.uint8)
    
    intersection = np.logical_and(pred_binary, target_binary).sum()
    union = np.logical_or(pred_binary, target_binary).sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return intersection / union


def search_threshold(model, dataloader, device, mode, thresholds=None):
    """搜索最佳阈值"""
    if thresholds is None:
        thresholds = np.arange(0.1, 0.9, 0.05)
    
    model.eval()
    all_preds = []
    all_masks = []
    
    with torch.no_grad():
        for images, masks, _ in tqdm(dataloader, desc='Predicting'):
            images = images.to(device)
            
            with autocast():
                outputs = model(images)
            
            preds = torch.sigmoid(outputs).cpu().numpy()
            masks_np = masks.numpy()
            
            all_preds.append(preds)
            all_masks.append(masks_np)
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_masks = np.concatenate(all_masks, axis=0)
    
    results = {}
    
    if mode == 'multi':
        # 多标签模式：分别搜索每个类别的阈值
        class_names = ['wheat', 'rape']
        for i, name in enumerate(class_names):
            best_iou = 0
            best_thresh = 0.5
            
            for thresh in thresholds:
                ious = []
                for j in range(len(all_preds)):
                    iou = compute_iou(all_preds[j, i], all_masks[j, i], thresh)
                    ious.append(iou)
                
                avg_iou = np.mean(ious)
                if avg_iou > best_iou:
                    best_iou = avg_iou
                    best_thresh = thresh
            
            results[name] = {'threshold': best_thresh, 'iou': best_iou}
            print(f"{name}: best threshold = {best_thresh:.2f}, IoU = {best_iou:.4f}")
    else:
        # 单标签模式
        best_iou = 0
        best_thresh = 0.5
        
        for thresh in thresholds:
            ious = []
            for j in range(len(all_preds)):
                pred = all_preds[j].squeeze()
                mask = all_masks[j].squeeze()
                iou = compute_iou(pred, mask, thresh)
                ious.append(iou)
            
            avg_iou = np.mean(ious)
            if avg_iou > best_iou:
                best_iou = avg_iou
                best_thresh = thresh
        
        results['rice'] = {'threshold': best_thresh, 'iou': best_iou}
        print(f"rice: best threshold = {best_thresh:.2f}, IoU = {best_iou:.4f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Search optimal threshold')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--mode', type=str, choices=['multi', 'single'], required=True)
    parser.add_argument('--data_dir', type=str, default='/root/competition_data/public')
    parser.add_argument('--output_file', type=str, default='/root/crop_segmentation/thresholds.json')
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载模型
    if args.mode == 'multi':
        model = MultiLabelModel()
    else:
        model = SingleLabelModel()
    
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    print(f"Loaded model from {args.model_path}")
    
    # 准备验证集
    transform = get_validation_augmentation(256)
    
    if args.mode == 'multi':
        val_dataset = CropSegmentationDataset(
            image_dir=os.path.join(args.data_dir, 'train/image/wheat_rape'),
            label_dirs=[
                os.path.join(args.data_dir, 'train/label/wheat'),
                os.path.join(args.data_dir, 'train/label/rape'),
            ],
            mode='multi',
            transform=transform,
        )
    else:
        val_dataset = CropSegmentationDataset(
            image_dir=os.path.join(args.data_dir, 'train/image/rice'),
            label_dirs=[
                os.path.join(args.data_dir, 'train/label/rice'),
            ],
            mode='single',
            transform=transform,
        )
    
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    print(f"Val samples: {len(val_dataset)}")
    
    # 搜索阈值
    results = search_threshold(model, val_loader, device, args.mode)
    
    # 保存结果
    import json
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved thresholds to {args.output_file}")


if __name__ == '__main__':
    main()
