"""
评估优化后的推理在验证集上的表现
"""
import os
import sys
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


def remove_small_components(mask, min_area=50):
    labeled, num_features = ndimage.label(mask)
    for i in range(1, num_features + 1):
        component = labeled == i
        if component.sum() < min_area:
            mask[component] = 0
    return mask


def fill_holes(mask, max_hole_area=50):
    inverted = 1 - mask
    labeled, num_features = ndimage.label(inverted)
    for i in range(1, num_features + 1):
        hole = labeled == i
        if hole.sum() < max_hole_area:
            mask[hole] = 1
    return mask


def apply_tta_voting(model, image_dir, device, threshold=0.5, img_size=256):
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
    
    for img_name in tqdm(images, desc='TTA Voting'):
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
        votes = np.array([p > threshold for p in all_preds], dtype=np.float32)
        vote_ratio = np.mean(votes, axis=0)
        final_pred = np.where((avg_pred > threshold) & (vote_ratio > 0.5), avg_pred, 0.0)
        
        results[img_name] = final_pred
    
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
    wheat_rape_results = apply_tta_voting(wheat_rape_model, os.path.join(val_image_dir, 'wheat_rape'), device, threshold=0.6)
    rice_results = apply_tta_voting(rice_model, os.path.join(val_image_dir, 'rice'), device, threshold=0.6)
    
    # Evaluate
    print("\n" + "="*60)
    print("EVALUATION WITH TTA VOTING (threshold=0.6)")
    print("="*60)
    
    # Wheat
    wheat_ious = []
    for img_name in sorted(wheat_rape_results.keys()):
        pred = wheat_rape_results[img_name][0]
        pred_binary = (pred > 0.6).astype(np.uint8)
        
        label_path = os.path.join(val_label_dir, 'wheat', img_name)
        label = np.array(Image.open(label_path))
        label_binary = (label > 0).astype(np.uint8)
        
        iou = compute_iou(pred_binary, label_binary)
        wheat_ious.append(iou)
    
    wheat_mean = np.mean(wheat_ious)
    print(f"Wheat IoU: {wheat_mean:.6f}")
    
    # Rape
    rape_ious = []
    for img_name in sorted(wheat_rape_results.keys()):
        pred = wheat_rape_results[img_name][1]
        pred_binary = (pred > 0.6).astype(np.uint8)
        
        label_path = os.path.join(val_label_dir, 'rape', img_name)
        label = np.array(Image.open(label_path))
        label_binary = (label > 0).astype(np.uint8)
        
        iou = compute_iou(pred_binary, label_binary)
        rape_ious.append(iou)
    
    rape_mean = np.mean(rape_ious)
    print(f"Rape IoU: {rape_mean:.6f}")
    
    # Rice
    rice_ious = []
    for img_name in sorted(rice_results.keys()):
        pred = rice_results[img_name].squeeze()
        pred_binary = (pred > 0.6).astype(np.uint8)
        
        label_path = os.path.join(val_label_dir, 'rice', img_name)
        label = np.array(Image.open(label_path))
        label_binary = (label > 0).astype(np.uint8)
        
        iou = compute_iou(pred_binary, label_binary)
        rice_ious.append(iou)
    
    rice_mean = np.mean(rice_ious)
    print(f"Rice IoU: {rice_mean:.6f}")
    
    total = (wheat_mean + rape_mean + rice_mean) / 3
    print(f"\nAverage IoU: {total:.6f}")
    
    # Try different thresholds
    print("\n" + "="*60)
    print("THRESHOLD OPTIMIZATION")
    print("="*60)
    
    best_total = 0
    best_threshold = 0.6
    
    for threshold in np.arange(0.3, 0.8, 0.05):
        w_ious = []
        for img_name in sorted(wheat_rape_results.keys()):
            pred = wheat_rape_results[img_name][0]
            pred_binary = (pred > threshold).astype(np.uint8)
            label_path = os.path.join(val_label_dir, 'wheat', img_name)
            label = np.array(Image.open(label_path))
            label_binary = (label > 0).astype(np.uint8)
            w_ious.append(compute_iou(pred_binary, label_binary))
        
        r_ious = []
        for img_name in sorted(wheat_rape_results.keys()):
            pred = wheat_rape_results[img_name][1]
            pred_binary = (pred > threshold).astype(np.uint8)
            label_path = os.path.join(val_label_dir, 'rape', img_name)
            label = np.array(Image.open(label_path))
            label_binary = (label > 0).astype(np.uint8)
            r_ious.append(compute_iou(pred_binary, label_binary))
        
        rc_ious = []
        for img_name in sorted(rice_results.keys()):
            pred = rice_results[img_name].squeeze()
            pred_binary = (pred > threshold).astype(np.uint8)
            label_path = os.path.join(val_label_dir, 'rice', img_name)
            label = np.array(Image.open(label_path))
            label_binary = (label > 0).astype(np.uint8)
            rc_ious.append(compute_iou(pred_binary, label_binary))
        
        total = (np.mean(w_ious) + np.mean(r_ious) + np.mean(rc_ious)) / 3
        
        if total > best_total:
            best_total = total
            best_threshold = threshold
    
    print(f"Best threshold: {best_threshold:.2f}")
    print(f"Best average IoU: {best_total:.6f}")


if __name__ == '__main__':
    main()
