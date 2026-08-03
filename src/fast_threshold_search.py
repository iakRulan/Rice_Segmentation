"""
快速阈值优化脚本 - 使用向量化numpy操作
"""
import os
import sys
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import MultiLabelModel, SingleLabelModel


def compute_iou_vectorized(preds, targets, thresholds):
    """
    向量化IoU计算
    preds: (N, H, W) 预测概率
    targets: (N, H, W) 真实标签
    thresholds: (T,) 阈值数组
    返回: (T,) 每个阈值对应的平均IoU
    """
    results = []
    for thresh in thresholds:
        pred_binary = (preds > thresh).astype(np.uint8)
        target_binary = (targets > 0).astype(np.uint8)
        
        intersection = np.logical_and(pred_binary, target_binary).sum(axis=(1, 2))
        union = np.logical_or(pred_binary, target_binary).sum(axis=(1, 2))
        
        # 处理union=0的情况
        iou = np.where(union == 0, 
                       np.where(intersection == 0, 1.0, 0.0),
                       intersection / union)
        results.append(iou.mean())
    
    return np.array(results)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    val_image_dir = '/root/competition_data/public/val/image'
    val_label_dir = '/root/competition_data/public/val/label'
    
    # Load models
    print("Loading Wheat+Rape model...")
    wheat_rape_model = MultiLabelModel()
    checkpoint = torch.load('/root/crop_segmentation/weights/best_wheat_rape.pth', map_location=device, weights_only=False)
    wheat_rape_model.load_state_dict(checkpoint['model_state_dict'])
    wheat_rape_model = wheat_rape_model.to(device)
    wheat_rape_model.eval()
    
    print("Loading Rice model...")
    rice_model = SingleLabelModel()
    checkpoint = torch.load('/root/crop_segmentation/weights/best_rice.pth', map_location=device, weights_only=False)
    rice_model.load_state_dict(checkpoint['model_state_dict'])
    rice_model = rice_model.to(device)
    rice_model.eval()
    
    # Simple inference (no TTA for speed)
    print("\nRunning inference on val...")
    
    # Wheat+Rape
    wheat_rape_images = sorted([f for f in os.listdir(os.path.join(val_image_dir, 'wheat_rape')) if f.endswith('.png')])
    wheat_rape_preds = []
    wheat_rape_targets = []
    
    for img_name in tqdm(wheat_rape_images, desc='Wheat+Rape'):
        img_path = os.path.join(val_image_dir, 'wheat_rape', img_name)
        image = np.array(Image.open(img_path).convert('RGB'))
        
        # Transform
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        transform = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
        augmented = transform(image=image)
        img_tensor = augmented['image'].unsqueeze(0).to(device)
        
        with torch.no_grad(), autocast():
            output = wheat_rape_model(img_tensor)
        
        pred = torch.sigmoid(output).cpu().numpy()[0]  # (2, H, W)
        wheat_rape_preds.append(pred)
        
        # Load targets
        wheat_label = np.array(Image.open(os.path.join(val_label_dir, 'wheat', img_name)))
        rape_label = np.array(Image.open(os.path.join(val_label_dir, 'rape', img_name)))
        wheat_rape_targets.append(np.stack([
            (wheat_label > 0).astype(np.float32),
            (rape_label > 0).astype(np.float32)
        ], axis=0))
    
    wheat_rape_preds = np.array(wheat_rape_preds)  # (N, 2, H, W)
    wheat_rape_targets = np.array(wheat_rape_targets)  # (N, 2, H, W)
    
    # Rice
    rice_images = sorted([f for f in os.listdir(os.path.join(val_image_dir, 'rice')) if f.endswith('.png')])
    rice_preds = []
    rice_targets = []
    
    for img_name in tqdm(rice_images, desc='Rice'):
        img_path = os.path.join(val_image_dir, 'rice', img_name)
        image = np.array(Image.open(img_path).convert('RGB'))
        
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        transform = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
        augmented = transform(image=image)
        img_tensor = augmented['image'].unsqueeze(0).to(device)
        
        with torch.no_grad(), autocast():
            output = rice_model(img_tensor)
        
        pred = torch.sigmoid(output).cpu().numpy()[0]  # (1, H, W)
        rice_preds.append(pred)
        
        rice_label = np.array(Image.open(os.path.join(val_label_dir, 'rice', img_name)))
        rice_targets.append((rice_label > 0).astype(np.float32)[np.newaxis, :, :])
    
    rice_preds = np.array(rice_preds)  # (N, 1, H, W)
    rice_targets = np.array(rice_targets)  # (N, 1, H, W)
    
    # Threshold search
    print("\n" + "="*60)
    print("THRESHOLD OPTIMIZATION")
    print("="*60)
    
    thresholds = np.arange(0.1, 0.9, 0.01)
    
    # Wheat
    wheat_ious = compute_iou_vectorized(
        wheat_rape_preds[:, 0],  # (N, H, W)
        wheat_rape_targets[:, 0],
        thresholds
    )
    best_wheat_idx = np.argmax(wheat_ious)
    best_wheat_thresh = thresholds[best_wheat_idx]
    best_wheat_iou = wheat_ious[best_wheat_idx]
    print(f"Wheat: best threshold = {best_wheat_thresh:.2f}, IoU = {best_wheat_iou:.6f}")
    
    # Rape
    rape_ious = compute_iou_vectorized(
        wheat_rape_preds[:, 1],
        wheat_rape_targets[:, 1],
        thresholds
    )
    best_rape_idx = np.argmax(rape_ious)
    best_rape_thresh = thresholds[best_rape_idx]
    best_rape_iou = rape_ious[best_rape_idx]
    print(f"Rape: best threshold = {best_rape_thresh:.2f}, IoU = {best_rape_iou:.6f}")
    
    # Rice
    rice_ious = compute_iou_vectorized(
        rice_preds[:, 0],
        rice_targets[:, 0],
        thresholds
    )
    best_rice_idx = np.argmax(rice_ious)
    best_rice_thresh = thresholds[best_rice_idx]
    best_rice_iou = rice_ious[best_rice_idx]
    print(f"Rice: best threshold = {best_rice_thresh:.2f}, IoU = {best_rice_iou:.6f}")
    
    avg_iou = (best_wheat_iou + best_rape_iou + best_rice_iou) / 3
    print(f"\nBest average IoU: {avg_iou:.6f}")
    
    # Save results
    import json
    results = {
        'wheat': {'threshold': float(best_wheat_thresh), 'iou': float(best_wheat_iou)},
        'rape': {'threshold': float(best_rape_thresh), 'iou': float(best_rape_iou)},
        'rice': {'threshold': float(best_rice_thresh), 'iou': float(best_rice_iou)},
        'average_iou': float(avg_iou)
    }
    
    with open('/root/crop_segmentation/thresholds_optimized.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to /root/crop_segmentation/thresholds_optimized.json")


if __name__ == '__main__':
    main()
