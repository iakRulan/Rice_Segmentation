"""
遥感影像农作物分割数据集
支持小麦+油菜（多标签）和水稻（单标签）两种模式
"""
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2


class CropSegmentationDataset(Dataset):
    """
    农作物分割数据集
    """
    def __init__(self, image_dir, label_dirs=None, label_values=None, 
                 transform=None, mode='multi'):
        self.image_dir = image_dir
        self.label_dirs = label_dirs if label_dirs else []
        self.label_values = label_values if label_values else []
        self.transform = transform
        self.mode = mode
        
        # 获取所有影像文件名
        self.images = sorted([f for f in os.listdir(image_dir) 
                            if f.endswith('.png')])
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        # 读取影像 (RGBA -> RGB)
        image = np.array(Image.open(img_path).convert('RGB'))
        
        # 读取标签
        if self.label_dirs:
            masks = []
            for label_dir in self.label_dirs:
                label_path = os.path.join(label_dir, img_name)
                if os.path.exists(label_path):
                    mask = np.array(Image.open(label_path))
                    # 二值化：非0为1
                    mask = (mask > 0).astype(np.float32)
                else:
                    mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
                masks.append(mask)
            
            if self.mode == 'multi':
                # 多标签：堆叠为 (num_classes, H, W) -> 转置为 (H, W, num_classes) 给 albumentations
                masks = np.stack(masks, axis=0)
                masks = np.transpose(masks, (1, 2, 0))  # (C,H,W) -> (H,W,C)
            else:
                # 单标签：只取第一个
                masks = masks[0]
            
            # 数据增强
            if self.transform:
                augmented = self.transform(image=image, mask=masks)
                image = augmented['image']
                masks = augmented['mask']
                # 多标签：转回 (C, H, W)
                if self.mode == 'multi':
                    masks = np.transpose(masks, (2, 0, 1))  # (H,W,C) -> (C,H,W)
            
            return image, masks, img_name
        else:
            # 测试集无标签
            if self.transform:
                augmented = self.transform(image=image)
                image = augmented['image']
            return image, img_name


def get_training_augmentation(img_size=256):
    """训练数据增强"""
    return A.Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Transpose(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1, 
            scale_limit=0.2, 
            rotate_limit=30, 
            p=0.5,
            border_mode=0
        ),
        A.OneOf([
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.5),
            A.GridDistortion(p=0.5),
            A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=0.5),
        ], p=0.3),
        A.OneOf([
            A.CLAHE(clip_limit=4.0, p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.3, 
                contrast_limit=0.3, 
                p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=20, 
                sat_shift_limit=30, 
                val_shift_limit=20, 
                p=0.5
            ),
        ], p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2(),
    ], is_check_shapes=False)


def get_validation_augmentation(img_size=256):
    """验证/测试数据增强（仅归一化）"""
    return A.Compose([
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2(),
    ], is_check_shapes=False)
