"""
推理脚本
支持TTA（测试时增强）和后处理
"""
import os
import sys
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

import albumentations as A
from albumentations.pytorch import ToTensorV2

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
        
        # 读取影像
        image = np.array(Image.open(img_path).convert('RGB'))
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        return image, img_name


def load_model(model_path, mode, architecture='unet', encoder='efficientnet-b3', device='cuda'):
    """加载模型"""
    if mode == 'wheat_rape':
        model = MultiLabelModel(architecture=architecture, encoder_name=encoder)
    else:
        model = SingleLabelModel(architecture=architecture, encoder_name=encoder)
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model


@torch.no_grad()
def inference(model, dataloader, device, use_tta=False):
    """推理"""
    results = {}
    
    for images, img_names in tqdm(dataloader, desc='Inference'):
        images = images.to(device)
        
        with autocast():
            outputs = model(images)
        
        # 应用sigmoid
        preds = torch.sigmoid(outputs)
        
        # 转为numpy
        preds = preds.cpu().numpy()
        
        for i, name in enumerate(img_names):
            results[name] = preds[i]
    
    return results


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
            
            pred = torch.sigmoid(output).cpu().numpy()[0]
            
            # 反转变换
            if name == 'hflip':
                pred = np.flip(pred, axis=-1)
            elif name == 'vflip':
                pred = np.flip(pred, axis=-2)
            elif name == 'rot90':
                pred = np.rot90(pred, k=-1, axes=(-2, -1))
            
            all_preds.append(pred)
        
        # 平均所有预测
        avg_pred = np.mean(all_preds, axis=0)
        results[img_name] = avg_pred
    
    return results


def save_predictions(results, output_dir, threshold=0.5, min_area=0):
    """
    保存预测结果
    
    Args:
        results: 预测结果字典 {filename: prediction_array}
        output_dir: 输出目录
        threshold: 二值化阈值
        min_area: 最小连通域面积
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for img_name, pred in results.items():
        # 二值化
        mask = (pred > threshold).astype(np.uint8) * 255
        
        # 确保尺寸正确
        if mask.shape[-2:] != (256, 256):
            mask = np.array(Image.fromarray(mask).resize((256, 256), Image.NEAREST))
        
        # 保存
        mask_img = Image.fromarray(mask)
        mask_img.save(os.path.join(output_dir, img_name))


def predict(args):
    """主推理函数"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载模型
    model = load_model(
        args.model_path, 
        args.mode, 
        args.architecture, 
        args.encoder, 
        device
    )
    print(f"Loaded model from {args.model_path}")
    
    # 推理
    if args.tta:
        print("Using TTA...")
        results = apply_tta(model, args.image_dir, device, args.img_size)
    else:
        transform = get_validation_augmentation(args.img_size)
        dataset = InferenceDataset(args.image_dir, transform=transform)
        dataloader = DataLoader(
            dataset, 
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=args.num_workers
        )
        results = inference(model, dataloader, device)
    
    # 保存结果
    if args.mode == 'wheat_rape':
        # 小麦和油菜分别保存
        wheat_results = {k: v[0:1] for k, v in results.items()}
        rape_results = {k: v[1:2] for k, v in results.items()}
        
        wheat_dir = os.path.join(args.output_dir, 'wheat')
        rape_dir = os.path.join(args.output_dir, 'rape')
        
        save_predictions(wheat_results, wheat_dir, args.threshold, args.min_area)
        save_predictions(rape_results, rape_dir, args.threshold, args.min_area)
        
        print(f"Saved wheat predictions to {wheat_dir}")
        print(f"Saved rape predictions to {rape_dir}")
    else:
        rice_dir = os.path.join(args.output_dir, 'rice')
        save_predictions(results, rice_dir, args.threshold, args.min_area)
        print(f"Saved rice predictions to {rice_dir}")
    
    print(f"Total predictions: {len(results)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference for crop segmentation')
    
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--mode', type=str, choices=['wheat_rape', 'rice'], required=True)
    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    
    parser.add_argument('--architecture', type=str, default='unet')
    parser.add_argument('--encoder', type=str, default='efficientnet-b3')
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--min_area', type=int, default=0)
    parser.add_argument('--tta', action='store_true')
    
    args = parser.parse_args()
    predict(args)
