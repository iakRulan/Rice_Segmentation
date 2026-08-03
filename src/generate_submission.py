"""
生成提交文件
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
    parser = argparse.ArgumentParser(description='Generate submission')
    parser.add_argument('--wheat_rape_model', type=str, default='/root/crop_segmentation/weights/best_wheat_rape.pth')
    parser.add_argument('--rice_model', type=str, default='/root/crop_segmentation/weights/best_rice.pth')
    parser.add_argument('--data_dir', type=str, default='/root/competition_data/public/testA')
    parser.add_argument('--output_dir', type=str, default='/root/crop_segmentation/submission')
    parser.add_argument('--wheat_threshold', type=float, default=0.5)
    parser.add_argument('--rape_threshold', type=float, default=0.5)
    parser.add_argument('--rice_threshold', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建输出目录
    os.makedirs(os.path.join(args.output_dir, 'wheat'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'rape'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'rice'), exist_ok=True)
    
    # 数据增强
    transform = get_validation_augmentation(256)
    
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
    wheat_rape_dataset = InferenceDataset(wheat_rape_image_dir, transform=transform)
    wheat_rape_loader = DataLoader(wheat_rape_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    wheat_rape_results = inference(wheat_rape_model, wheat_rape_loader, device)
    
    # 保存小麦和油菜结果
    for img_name, pred in tqdm(wheat_rape_results.items(), desc='Saving wheat/rape'):
        # pred shape: (2, H, W) - channel 0: wheat, channel 1: rape
        wheat_mask = (pred[0] > args.wheat_threshold).astype(np.uint8) * 255
        rape_mask = (pred[1] > args.rape_threshold).astype(np.uint8) * 255
        
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
    rice_dataset = InferenceDataset(rice_image_dir, transform=transform)
    rice_loader = DataLoader(rice_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    rice_results = inference(rice_model, rice_loader, device)
    
    # 保存水稻结果
    for img_name, pred in tqdm(rice_results.items(), desc='Saving rice'):
        # pred shape: (1, H, W)
        rice_mask = (pred.squeeze() > args.rice_threshold).astype(np.uint8) * 255
        
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


if __name__ == '__main__':
    main()
