"""
高级TTA推理 + 后处理
- 多尺度TTA (256, 384, 512)
- 多种变换组合
- 连通域分析后处理
- 形态学操作
"""
import os
import sys
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.models import MultiLabelModel, SingleLabelModel


class TTADataset(Dataset):
    """TTA数据集"""
    def __init__(self, image_dir, img_size, scale_factor=1.0):
        self.image_dir = image_dir
        self.images = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
        self.img_size = img_size
        self.scale_factor = scale_factor
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = np.array(Image.open(img_path).convert('RGB'))
        
        # 多尺度
        h, w = image.shape[:2]
        new_h, new_w = int(h * self.scale_factor), int(w * self.scale_factor)
        image = cv2.resize(image, (new_w, new_h))
        
        transform = A.Compose([
            A.Resize(self.img_size, self.img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
        
        augmented = transform(image=image)
        return augmented['image'], img_name


def get_tta_transforms(img_size):
    """获取所有TTA变换"""
    base_norm = A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    transforms = {
        'original': A.Compose([
            A.Resize(img_size, img_size),
            base_norm,
            ToTensorV2(),
        ]),
        'hflip': A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=1.0),
            base_norm,
            ToTensorV2(),
        ]),
        'vflip': A.Compose([
            A.Resize(img_size, img_size),
            A.VerticalFlip(p=1.0),
            base_norm,
            ToTensorV2(),
        ]),
        'rot90': A.Compose([
            A.Resize(img_size, img_size),
            A.RandomRotate90(p=1.0),
            base_norm,
            ToTensorV2(),
        ]),
        'rot180': A.Compose([
            A.Resize(img_size, img_size),
            A.Rotate(limit=(180, 180), p=1.0, border_mode=0),
            base_norm,
            ToTensorV2(),
        ]),
        'transpose': A.Compose([
            A.Resize(img_size, img_size),
            A.Transpose(p=1.0),
            base_norm,
            ToTensorV2(),
        ]),
        'brightness': A.Compose([
            A.Resize(img_size, img_size),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
            base_norm,
            ToTensorV2(),
        ]),
    }
    
    return transforms


def apply_tta(model, image, device, img_size):
    """应用TTA并返回平均预测"""
    model.eval()
    transforms = get_tta_transforms(img_size)
    
    all_preds = []
    
    with torch.no_grad():
        for name, transform in transforms.items():
            augmented = transform(image=image)
            input_tensor = augmented['image'].unsqueeze(0).to(device)
            
            with autocast():
                output = model(input_tensor)
                pred = torch.sigmoid(output)
            
            # 逆变换
            pred_np = pred.cpu().numpy()[0]  # (C, H, W)
            
            if name == 'hflip':
                pred_np = np.flip(pred_np, axis=2)
            elif name == 'vflip':
                pred_np = np.flip(pred_np, axis=1)
            elif name == 'rot90':
                pred_np = np.rot90(pred_np, k=-1, axes=(1, 2))
            elif name == 'rot180':
                pred_np = np.rot90(pred_np, k=2, axes=(1, 2))
            elif name == 'transpose':
                pred_np = np.transpose(pred_np, (0, 2, 1))
            
            all_preds.append(pred_np)
    
    # 平均所有预测
    avg_pred = np.mean(all_preds, axis=0)
    return avg_pred


def post_process_mask(mask, min_area=100, max_hole_area=100, morph_kernel_size=3):
    """后处理：连通域分析 + 形态学操作"""
    result = np.zeros_like(mask)
    
    # 连通域分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            component = (labels == i).astype(np.uint8)
            
            # 填充小孔洞
            contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            filled = np.zeros_like(component)
            cv2.drawContours(filled, contours, -1, 1, -1)
            
            holes = filled - component
            if holes.sum() > 0 and holes.sum() <= max_hole_area:
                component = filled
            
            result = np.logical_or(result, component).astype(np.uint8)
    
    # 形态学闭运算填充小间隙
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
    
    # 形态学开运算去除小噪点
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
    
    return result


def generate_submission(model, image_dir, output_dir, device, mode='multi', 
                        img_size=512, threshold=0.5, use_tta=True):
    """生成提交文件"""
    os.makedirs(output_dir, exist_ok=True)
    
    images = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
    
    for img_name in tqdm(images, desc='Generating predictions'):
        img_path = os.path.join(image_dir, img_name)
        image = np.array(Image.open(img_path).convert('RGB'))
        
        if use_tta:
            pred = apply_tta(model, image, device, img_size)
        else:
            model.eval()
            transform = A.Compose([
                A.Resize(img_size, img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
            augmented = transform(image=image)
            input_tensor = augmented['image'].unsqueeze(0).to(device)
            
            with torch.no_grad():
                with autocast():
                    output = model(input_tensor)
                    pred = torch.sigmoid(output).cpu().numpy()[0]
        
        # 保存每个类别的预测
        if mode == 'multi':
            for c, class_name in enumerate(['wheat', 'rape']):
                class_dir = os.path.join(output_dir, class_name)
                os.makedirs(class_dir, exist_ok=True)
                
                mask = pred[c]
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]), 
                                 interpolation=cv2.INTER_LINEAR)
                binary_mask = (mask > threshold).astype(np.uint8) * 255
                
                # 后处理
                binary_mask = post_process_mask(binary_mask // 255) * 255
                
                output_path = os.path.join(class_dir, img_name)
                Image.fromarray(binary_mask.astype(np.uint8)).save(output_path)
        else:
            class_dir = os.path.join(output_dir, 'rice')
            os.makedirs(class_dir, exist_ok=True)
            
            mask = pred[0]
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]),
                             interpolation=cv2.INTER_LINEAR)
            binary_mask = (mask > threshold).astype(np.uint8) * 255
            
            binary_mask = post_process_mask(binary_mask // 255) * 255
            
            output_path = os.path.join(class_dir, img_name)
            Image.fromarray(binary_mask.astype(np.uint8)).save(output_path)


def main():
    parser = argparse.ArgumentParser(description='Advanced TTA inference')
    parser.add_argument('--mode', type=str, choices=['multi', 'single'], required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--architecture', type=str, default='unet')
    parser.add_argument('--encoder', type=str, default='efficientnet-b3')
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--use_tta', action='store_true', default=True)
    parser.add_argument('--data_dir', type=str, default='/root/competition_data/public')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载模型
    if args.mode == 'multi':
        model = MultiLabelModel(architecture=args.architecture, encoder_name=args.encoder)
    else:
        model = SingleLabelModel(architecture=args.architecture, encoder_name=args.encoder)
    
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Loaded model from {args.model_path}")
    print(f"Model val IoU: {checkpoint.get('val_iou', 'N/A')}")
    
    # 生成提交
    if args.mode == 'multi':
        image_dir = os.path.join(args.data_dir, 'test/image/wheat_rape')
    else:
        image_dir = os.path.join(args.data_dir, 'test/image/rice')
    
    generate_submission(model, image_dir, args.output_dir, device, 
                       mode=args.mode, img_size=args.img_size, 
                       threshold=args.threshold, use_tta=args.use_tta)
    
    print(f"Submission saved to {args.output_dir}")


if __name__ == '__main__':
    main()
