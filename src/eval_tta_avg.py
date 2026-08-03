"""
最终优化推理 - 使用更高阈值和后处理
"""
import os
import sys
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast
import albumentations as A
from albumentations.pytorch import ToTensorV2
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import MultiLabelModel, SingleLabelModel


def remove_small_components(mask, min_area=100):
    """去除小连通域"""
    labeled, num_features = ndimage.label(mask)
    for i in range(1, num_features + 1):
        component = labeled == i
        if component.sum() < min_area:
            mask[component] = 0
    return mask


def fill_holes(mask, max_hole_area=100):
    """填充小孔洞"""
    inverted = 1 - mask
    labeled, num_features = ndimage.label(inverted)
    for i in range(1, num_features + 1):
        hole = labeled == i
        if hole.sum() < max_hole_area:
            mask[hole] = 1
    return mask


def apply_tta_average(model, image_dir, device, img_size=256):
    """TTA with soft averaging"""
    transforms = {
        'original': A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
        'hflip': A.Compose([
            A.HorizontalFlip(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
        'vflip': A.Compose([
            A.VerticalFlip(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
        'rot90': A.Compose([
            A.RandomRotate90(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
    }
    
    images = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
    results = {}
    
    for img_name in tqdm(images, desc='TTA Average'):
        img_path = os.path.join(image_dir, img_name)
        image = np.array(Image.open(img_path).convert('RGB'))
        
        all_preds = []
        for name, transform in transforms.items():
            augmented = transform(image=image)
            img_tensor = augmented['image'].unsqueeze(0).to(device)
            
            with autocast():
                output = model(img_tensor)
            
            pred = torch.sigmoid(output).detach().cpu().numpy()[0]
            
            if name == 'hflip':
                pred = np.flip(pred, axis=-1)
            elif name == 'vflip':
                pred = np.flip(pred, axis=-2)
            elif name == 'rot90':
                pred = np.rot90(pred, k=-1, axes=(-2, -1))
            
            all_preds.append(pred)
        
        avg_pred = np.mean(all_preds, axis=0)
        results[img_name] = avg_pred
    
    return results


def compute_iou(pred_binary, target_binary):
    intersection = np.logical_and(pred_binary, target_binary).sum()
    union = np.logical_or(pred_binary, target_binary).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union


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
    
    # TTA Inference
    print("\nTTA Inference on val...")
    wheat_rape_results = apply_tta_average(wheat_rape_model, os.path.join(val_image_dir, 'wheat_rape'), device)
    rice_results = apply_tta_average(rice_model, os.path.join(val_image_dir, 'rice'), device)
    
    # Evaluate with different thresholds
    print("\n" + "="*60)
    print("THRESHOLD OPTIMIZATION")
    print("="*60)
    
    best_total = 0
    best_thresholds = (0.5, 0.5, 0.5)
    
    thresholds = np.arange(0.3, 0.8, 0.05)
    
    for wheat_thresh in thresholds:
        for rape_thresh in thresholds:
            for rice_thresh in thresholds:
                # Wheat
                w_ious = []
                for img_name in sorted(wheat_rape_results.keys()):
                    pred = wheat_rape_results[img_name][0]
                    pred_binary = (pred > wheat_thresh).astype(np.uint8)
                    label_path = os.path.join(val_label_dir, 'wheat', img_name)
                    label = np.array(Image.open(label_path))
                    label_binary = (label > 0).astype(np.uint8)
                    w_ious.append(compute_iou(pred_binary, label_binary))
                
                # Rape
                r_ious = []
                for img_name in sorted(wheat_rape_results.keys()):
                    pred = wheat_rape_results[img_name][1]
                    pred_binary = (pred > rape_thresh).astype(np.uint8)
                    label_path = os.path.join(val_label_dir, 'rape', img_name)
                    label = np.array(Image.open(label_path))
                    label_binary = (label > 0).astype(np.uint8)
                    r_ious.append(compute_iou(pred_binary, label_binary))
                
                # Rice
                rc_ious = []
                for img_name in sorted(rice_results.keys()):
                    pred = rice_results[img_name].squeeze()
                    pred_binary = (pred > rice_thresh).astype(np.uint8)
                    label_path = os.path.join(val_label_dir, 'rice', img_name)
                    label = np.array(Image.open(label_path))
                    label_binary = (label > 0).astype(np.uint8)
                    rc_ious.append(compute_iou(pred_binary, label_binary))
                
                total = (np.mean(w_ious) + np.mean(r_ious) + np.mean(rc_ious)) / 3
                
                if total > best_total:
                    best_total = total
                    best_thresholds = (wheat_thresh, rape_thresh, rice_thresh)
    
    print(f"Best thresholds: wheat={best_thresholds[0]:.2f}, rape={best_thresholds[1]:.2f}, rice={best_thresholds[2]:.2f}")
    print(f"Best average IoU: {best_total:.6f}")
    
    # Per-class best
    print("\n" + "="*60)
    print("PER-CLASS BEST THRESHOLDS")
    print("="*60)
    
    for cls, results, idx in [('wheat', wheat_rape_results, 0), ('rape', wheat_rape_results, 1), ('rice', rice_results, None)]:
        best_iou = 0
        best_t = 0.5
        for t in thresholds:
            ious = []
            for img_name in sorted(results.keys()):
                if idx is not None:
                    pred = results[img_name][idx]
                else:
                    pred = results[img_name].squeeze()
                pred_binary = (pred > t).astype(np.uint8)
                label_path = os.path.join(val_label_dir, cls, img_name)
                label = np.array(Image.open(label_path))
                label_binary = (label > 0).astype(np.uint8)
                ious.append(compute_iou(pred_binary, label_binary))
            avg = np.mean(ious)
            if avg > best_iou:
                best_iou = avg
                best_t = t
        print(f"{cls}: best threshold = {best_t:.2f}, IoU = {best_iou:.6f}")


if __name__ == '__main__':
    main()
