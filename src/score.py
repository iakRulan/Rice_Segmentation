"""
计算验证集得分
按照比赛规则计算每个类别的IoU，然后取平均
"""
import os
import sys
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

from src.dataset import get_validation_augmentation
from src.models import MultiLabelModel, SingleLabelModel


class InferenceDataset(torch.utils.data.Dataset):
    """推理数据集"""
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = np.array(Image.open(img_path).convert('RGB'))
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        return image, img_name


def compute_iou(pred_binary, target_binary):
    """计算IoU"""
    intersection = np.logical_and(pred_binary, target_binary).sum()
    union = np.logical_or(pred_binary, target_binary).sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return intersection / union


def inference(model, dataloader, device):
    """推理"""
    results = {}
    
    with torch.no_grad():
        for images, img_names in tqdm(dataloader, desc='Inference'):
            images = images.to(device)
            
            with autocast():
                outputs = model(images)
            
            preds = torch.sigmoid(outputs).cpu().numpy()
            
            for i, name in enumerate(img_names):
                results[name] = preds[i]
    
    return results


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 数据增强
    transform = get_validation_augmentation(256)
    
    val_image_dir = '/root/competition_data/public/val/image'
    val_label_dir = '/root/competition_data/public/val/label'
    
    # ===== Wheat + Rape 模型 =====
    print("\n" + "="*60)
    print("Loading Wheat+Rape model...")
    wheat_rape_model = MultiLabelModel()
    checkpoint = torch.load('/root/crop_segmentation/weights/best_wheat_rape.pth', map_location=device, weights_only=False)
    wheat_rape_model.load_state_dict(checkpoint['model_state_dict'])
    wheat_rape_model = wheat_rape_model.to(device)
    wheat_rape_model.eval()
    
    # 推理
    wheat_rape_image_dir = os.path.join(val_image_dir, 'wheat_rape')
    wheat_rape_dataset = InferenceDataset(wheat_rape_image_dir, transform=transform)
    wheat_rape_loader = DataLoader(wheat_rape_dataset, batch_size=16, shuffle=False, num_workers=4)
    
    wheat_rape_results = inference(wheat_rape_model, wheat_rape_loader, device)
    
    # 计算每个类别的IoU
    print("\n" + "="*60)
    print("Computing scores...")
    print("="*60)
    
    # Wheat
    wheat_ious = []
    wheat_label_dir = os.path.join(val_label_dir, 'wheat')
    for img_name in sorted(wheat_rape_results.keys()):
        pred = wheat_rape_results[img_name][0]  # channel 0: wheat
        pred_binary = (pred > 0.5).astype(np.uint8)
        
        label_path = os.path.join(wheat_label_dir, img_name)
        if os.path.exists(label_path):
            label = np.array(Image.open(label_path))
            label_binary = (label > 0).astype(np.uint8)
        else:
            label_binary = np.zeros((256, 256), dtype=np.uint8)
        
        iou = compute_iou(pred_binary, label_binary)
        wheat_ious.append(iou)
    
    wheat_mean_iou = np.mean(wheat_ious)
    print(f"\nWheat IoU: {wheat_mean_iou:.4f}")
    
    # Rape
    rape_ious = []
    rape_label_dir = os.path.join(val_label_dir, 'rape')
    for img_name in sorted(wheat_rape_results.keys()):
        pred = wheat_rape_results[img_name][1]  # channel 1: rape
        pred_binary = (pred > 0.5).astype(np.uint8)
        
        label_path = os.path.join(rape_label_dir, img_name)
        if os.path.exists(label_path):
            label = np.array(Image.open(label_path))
            label_binary = (label > 0).astype(np.uint8)
        else:
            label_binary = np.zeros((256, 256), dtype=np.uint8)
        
        iou = compute_iou(pred_binary, label_binary)
        rape_ious.append(iou)
    
    rape_mean_iou = np.mean(rape_ious)
    print(f"Rape IoU: {rape_mean_iou:.4f}")
    
    # ===== Rice 模型 =====
    print("\n" + "="*60)
    print("Loading Rice model...")
    rice_model = SingleLabelModel()
    checkpoint = torch.load('/root/crop_segmentation/weights/best_rice.pth', map_location=device, weights_only=False)
    rice_model.load_state_dict(checkpoint['model_state_dict'])
    rice_model = rice_model.to(device)
    rice_model.eval()
    
    # 推理
    rice_image_dir = os.path.join(val_image_dir, 'rice')
    rice_dataset = InferenceDataset(rice_image_dir, transform=transform)
    rice_loader = DataLoader(rice_dataset, batch_size=16, shuffle=False, num_workers=4)
    
    rice_results = inference(rice_model, rice_loader, device)
    
    # Rice
    rice_ious = []
    rice_label_dir = os.path.join(val_label_dir, 'rice')
    for img_name in sorted(rice_results.keys()):
        pred = rice_results[img_name].squeeze()
        pred_binary = (pred > 0.5).astype(np.uint8)
        
        label_path = os.path.join(rice_label_dir, img_name)
        if os.path.exists(label_path):
            label = np.array(Image.open(label_path))
            label_binary = (label > 0).astype(np.uint8)
        else:
            label_binary = np.zeros((256, 256), dtype=np.uint8)
        
        iou = compute_iou(pred_binary, label_binary)
        rice_ious.append(iou)
    
    rice_mean_iou = np.mean(rice_ious)
    print(f"Rice IoU: {rice_mean_iou:.4f}")
    
    # 总分
    total_score = (wheat_mean_iou + rape_mean_iou + rice_mean_iou) / 3
    
    print("\n" + "="*60)
    print("FINAL SCORE SUMMARY")
    print("="*60)
    print(f"  Wheat IoU:  {wheat_mean_iou:.4f}")
    print(f"  Rape IoU:   {rape_mean_iou:.4f}")
    print(f"  Rice IoU:   {rice_mean_iou:.4f}")
    print(f"  Average IoU: {total_score:.4f}")
    print("="*60)
    
    # 详细统计
    print("\nDetailed statistics:")
    print(f"  Wheat - min: {np.min(wheat_ious):.4f}, max: {np.max(wheat_ious):.4f}, std: {np.std(wheat_ious):.4f}")
    print(f"  Rape  - min: {np.min(rape_ious):.4f}, max: {np.max(rape_ious):.4f}, std: {np.std(rape_ious):.4f}")
    print(f"  Rice  - min: {np.min(rice_ious):.4f}, max: {np.max(rice_ious):.4f}, std: {np.std(rice_ious):.4f}")


if __name__ == '__main__':
    main()
