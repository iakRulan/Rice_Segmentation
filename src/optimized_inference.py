"""
优化推理脚本
1. TTA (Test Time Augmentation)
2. 阈值优化
3. 后处理（去除小连通域）
"""
import os
import sys
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
import albumentations as A
from albumentations.pytorch import ToTensorV2
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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


def apply_tta(model, image_dir, device, img_size=256):
    """测试时增强（TTA）"""
    # 定义多种变换
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
        'rot180': A.Compose([
            A.RandomRotate90(p=1.0),
            A.RandomRotate90(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
        'rot270': A.Compose([
            A.RandomRotate90(p=1.0),
            A.RandomRotate90(p=1.0),
            A.RandomRotate90(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
    }
    
    images = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
    results = {}
    
    for img_name in tqdm(images, desc='TTA Inference'):
        img_path = os.path.join(image_dir, img_name)
        image = np.array(Image.open(img_path).convert('RGB'))
        
        all_preds = []
        
        for name, transform in transforms.items():
            augmented = transform(image=image)
            img_tensor = augmented['image'].unsqueeze(0).to(device)
            
            with autocast():
                output = model(img_tensor)
            
            pred = torch.sigmoid(output).detach().cpu().numpy()[0]
            
            # 反转变换
            if name == 'hflip':
                pred = np.flip(pred, axis=-1)
            elif name == 'vflip':
                pred = np.flip(pred, axis=-2)
            elif name == 'rot90':
                pred = np.rot90(pred, k=-1, axes=(-2, -1))
            elif name == 'rot180':
                pred = np.rot90(pred, k=-2, axes=(-2, -1))
            elif name == 'rot270':
                pred = np.rot90(pred, k=-3, axes=(-2, -1))
            
            all_preds.append(pred)
        
        # 平均所有预测
        avg_pred = np.mean(all_preds, axis=0)
        results[img_name] = avg_pred
    
    return results


def predict(args):
    """主推理函数"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建输出目录
    os.makedirs(os.path.join(args.output_dir, 'wheat'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'rape'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'rice'), exist_ok=True)
    
    # ===== Wheat + Rape 模型 =====
    print("\n=== Wheat + Rape Model ===")
    wheat_rape_model = MultiLabelModel()
    checkpoint = torch.load(args.wheat_rape_model, map_location=device, weights_only=False)
    wheat_rape_model.load_state_dict(checkpoint['model_state_dict'])
    wheat_rape_model = wheat_rape_model.to(device)
    wheat_rape_model.eval()
    print(f"Loaded model from {args.wheat_rape_model}")
    
    # 推理
    wheat_rape_image_dir = os.path.join(args.data_dir, 'image/wheat_rape')
    wheat_rape_results = apply_tta(wheat_rape_model, wheat_rape_image_dir, device, args.img_size)
    
    # 保存小麦和油菜结果
    for img_name, pred in tqdm(wheat_rape_results.items(), desc='Saving wheat/rape'):
        # pred shape: (2, H, W) - channel 0: wheat, channel 1: rape
        wheat_mask = (pred[0] > args.wheat_threshold).astype(np.uint8) * 255
        rape_mask = (pred[1] > args.rape_threshold).astype(np.uint8) * 255
        
        # 后处理
        if args.post_process:
            wheat_mask = remove_small_components(wheat_mask, args.min_area)
            wheat_mask = fill_holes(wheat_mask, args.max_hole_area)
            rape_mask = remove_small_components(rape_mask, args.min_area)
            rape_mask = fill_holes(rape_mask, args.max_hole_area)
        
        wheat_img = Image.fromarray(wheat_mask)
        rape_img = Image.fromarray(rape_mask)
        
        wheat_img.save(os.path.join(args.output_dir, 'wheat', img_name))
        rape_img.save(os.path.join(args.output_dir, 'rape', img_name))
    
    # ===== Rice 模型 =====
    print("\n=== Rice Model ===")
    rice_model = SingleLabelModel()
    checkpoint = torch.load(args.rice_model, map_location=device, weights_only=False)
    rice_model.load_state_dict(checkpoint['model_state_dict'])
    rice_model = rice_model.to(device)
    rice_model.eval()
    print(f"Loaded model from {args.rice_model}")
    
    # 推理
    rice_image_dir = os.path.join(args.data_dir, 'image/rice')
    rice_results = apply_tta(rice_model, rice_image_dir, device, args.img_size)
    
    # 保存水稻结果
    for img_name, pred in tqdm(rice_results.items(), desc='Saving rice'):
        # pred shape: (1, H, W)
        rice_mask = (pred.squeeze() > args.rice_threshold).astype(np.uint8) * 255
        
        # 后处理
        if args.post_process:
            rice_mask = remove_small_components(rice_mask, args.min_area)
            rice_mask = fill_holes(rice_mask, args.max_hole_area)
        
        rice_img = Image.fromarray(rice_mask)
        rice_img.save(os.path.join(args.output_dir, 'rice', img_name))
    
    # 统计
    print("\n=== Summary ===")
    wheat_files = os.listdir(os.path.join(args.output_dir, 'wheat'))
    rape_files = os.listdir(os.path.join(args.output_dir, 'rape'))
    rice_files = os.listdir(os.path.join(args.output_dir, 'rice'))
    print(f"Wheat: {len(wheat_files)} files")
    print(f"Rape: {len(rape_files)} files")
    print(f"Rice: {len(rice_files)} files")
    print(f"\nSubmission saved to {args.output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Optimized inference with TTA and post-processing')
    parser.add_argument('--wheat_rape_model', type=str, default='/root/crop_segmentation/weights/best_wheat_rape.pth')
    parser.add_argument('--rice_model', type=str, default='/root/crop_segmentation/weights/best_rice.pth')
    parser.add_argument('--data_dir', type=str, default='/root/competition_data/public/testA')
    parser.add_argument('--output_dir', type=str, default='/root/crop_segmentation/submission_v2')
    parser.add_argument('--wheat_threshold', type=float, default=0.5)
    parser.add_argument('--rape_threshold', type=float, default=0.5)
    parser.add_argument('--rice_threshold', type=float, default=0.5)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--post_process', action='store_true', default=True)
    parser.add_argument('--min_area', type=int, default=100)
    parser.add_argument('--max_hole_area', type=int, default=100)
    args = parser.parse_args()
    
    predict(args)


if __name__ == '__main__':
    main()
