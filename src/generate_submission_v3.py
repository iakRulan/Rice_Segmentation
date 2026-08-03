"""
优化版提交生成脚本
结合 TTA + 优化阈值 + 后处理（去除小连通域、填充孔洞）
"""
import os
import sys
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
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


def apply_tta(model, image, device):
    """TTA推理：多种变换取平均"""
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
    
    all_preds = []
    for name, transform in transforms.items():
        augmented = transform(image=image)
        img_tensor = augmented['image'].unsqueeze(0).to(device)
        
        with torch.no_grad(), autocast():
            output = model(img_tensor)
        
        pred = torch.sigmoid(output).detach().cpu().numpy()[0]
        
        # 逆向变换
        if name == 'hflip':
            pred = np.flip(pred, axis=-1)
        elif name == 'vflip':
            pred = np.flip(pred, axis=-2)
        elif name == 'rot90':
            pred = np.rot90(pred, k=-1, axes=(-2, -1))
        
        all_preds.append(pred)
    
    return np.mean(all_preds, axis=0)


def main():
    parser = argparse.ArgumentParser(description='Generate optimized submission')
    parser.add_argument('--wheat_rape_model', type=str, default='/root/crop_segmentation/weights/best_wheat_rape.pth')
    parser.add_argument('--rice_model', type=str, default='/root/crop_segmentation/weights/best_rice.pth')
    parser.add_argument('--data_dir', type=str, default='/root/competition_data/public/testA')
    parser.add_argument('--output_dir', type=str, default='/root/crop_segmentation/submission_v3')
    parser.add_argument('--wheat_threshold', type=float, default=0.69)
    parser.add_argument('--rape_threshold', type=float, default=0.31)
    parser.add_argument('--rice_threshold', type=float, default=0.41)
    parser.add_argument('--min_area', type=int, default=100)
    parser.add_argument('--max_hole_area', type=int, default=100)
    parser.add_argument('--use_tta', action='store_true', default=True)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建输出目录
    os.makedirs(os.path.join(args.output_dir, 'wheat'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'rape'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'rice'), exist_ok=True)
    
    # Load models
    print("Loading Wheat+Rape model...")
    wheat_rape_model = MultiLabelModel()
    checkpoint = torch.load(args.wheat_rape_model, map_location=device, weights_only=False)
    wheat_rape_model.load_state_dict(checkpoint['model_state_dict'])
    wheat_rape_model = wheat_rape_model.to(device)
    wheat_rape_model.eval()
    
    print("Loading Rice model...")
    rice_model = SingleLabelModel()
    checkpoint = torch.load(args.rice_model, map_location=device, weights_only=False)
    rice_model.load_state_dict(checkpoint['model_state_dict'])
    rice_model = rice_model.to(device)
    rice_model.eval()
    
    # ===== Wheat + Rape =====
    print("\n=== Wheat + Rape ===")
    wheat_rape_image_dir = os.path.join(args.data_dir, 'image/wheat_rape')
    wheat_rape_images = sorted([f for f in os.listdir(wheat_rape_image_dir) if f.endswith('.png')])
    
    for img_name in tqdm(wheat_rape_images, desc='Wheat+Rape TTA'):
        img_path = os.path.join(wheat_rape_image_dir, img_name)
        image = np.array(Image.open(img_path).convert('RGB'))
        
        # TTA推理
        pred = apply_tta(wheat_rape_model, image, device)
        
        # 应用阈值
        wheat_mask = (pred[0] > args.wheat_threshold).astype(np.uint8)
        rape_mask = (pred[1] > args.rape_threshold).astype(np.uint8)
        
        # 后处理
        wheat_mask = remove_small_components(wheat_mask, args.min_area)
        wheat_mask = fill_holes(wheat_mask, args.max_hole_area)
        rape_mask = remove_small_components(rape_mask, args.min_area)
        rape_mask = fill_holes(rape_mask, args.max_hole_area)
        
        # 保存
        Image.fromarray(wheat_mask * 255).save(os.path.join(args.output_dir, 'wheat', img_name))
        Image.fromarray(rape_mask * 255).save(os.path.join(args.output_dir, 'rape', img_name))
    
    # ===== Rice =====
    print("\n=== Rice ===")
    rice_image_dir = os.path.join(args.data_dir, 'image/rice')
    rice_images = sorted([f for f in os.listdir(rice_image_dir) if f.endswith('.png')])
    
    for img_name in tqdm(rice_images, desc='Rice TTA'):
        img_path = os.path.join(rice_image_dir, img_name)
        image = np.array(Image.open(img_path).convert('RGB'))
        
        # TTA推理
        pred = apply_tta(rice_model, image, device)
        
        # 应用阈值
        rice_mask = (pred.squeeze() > args.rice_threshold).astype(np.uint8)
        
        # 后处理
        rice_mask = remove_small_components(rice_mask, args.min_area)
        rice_mask = fill_holes(rice_mask, args.max_hole_area)
        
        # 保存
        Image.fromarray(rice_mask * 255).save(os.path.join(args.output_dir, 'rice', img_name))
    
    # 统计
    print("\n=== Summary ===")
    wheat_files = os.listdir(os.path.join(args.output_dir, 'wheat'))
    rape_files = os.listdir(os.path.join(args.output_dir, 'rape'))
    rice_files = os.listdir(os.path.join(args.output_dir, 'rice'))
    print(f"Wheat: {len(wheat_files)} files")
    print(f"Rape: {len(rape_files)} files")
    print(f"Rice: {len(rice_files)} files")
    print(f"\nSubmission saved to {args.output_dir}")


if __name__ == '__main__':
    main()
