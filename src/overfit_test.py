"""
最小闭环验证脚本
用少量数据过拟合，验证整个流程是否正确
"""
import os
import sys
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import CropSegmentationDataset, get_training_augmentation, get_validation_augmentation
from src.models import MultiLabelModel, SingleLabelModel
from src.losses import CombinedLoss, MultiLabelLoss


def overfit_test(mode='wheat_rape', num_samples=20, num_epochs=50, lr=1e-3):
    """
    过拟合测试：用少量数据训练，看是否能达到高IoU
    如果过拟合失败，说明模型/数据/代码有问题
    """
    print(f"\n{'='*60}")
    print(f"过拟合测试: {mode}, {num_samples} samples, {num_epochs} epochs")
    print(f"{'='*60}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    data_root = '/root/competition_data/public'
    
    if mode == 'wheat_rape':
        image_dir = os.path.join(data_root, 'train/image/wheat_rape')
        label_dirs = [
            os.path.join(data_root, 'train/label/wheat'),
            os.path.join(data_root, 'train/label/rape')
        ]
        model = MultiLabelModel(architecture='unet', encoder_name='efficientnet-b3')
        criterion = MultiLabelLoss(num_classes=2, loss_type='combined')
    else:
        image_dir = os.path.join(data_root, 'train/image/rice')
        label_dirs = [os.path.join(data_root, 'train/label/rice')]
        model = SingleLabelModel(architecture='unet', encoder_name='efficientnet-b3')
        criterion = CombinedLoss()
    
    # 数据集 - 过拟合测试不使用数据增强！
    dataset = CropSegmentationDataset(
        image_dir=image_dir,
        label_dirs=label_dirs,
        transform=get_validation_augmentation(),  # 仅归一化，无增强
        mode='multi' if mode == 'wheat_rape' else 'single'
    )
    
    # 只取前num_samples个
    indices = list(range(min(num_samples, len(dataset))))
    subset = Subset(dataset, indices)
    
    dataloader = DataLoader(subset, batch_size=4, shuffle=False, num_workers=0)
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    print(f"\n开始训练...")
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0
        total_iou = 0
        num_batches = 0
        
        for images, masks, _ in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            # 处理形状匹配
            if mode == 'rice':
                # outputs: [B, 1, H, W], masks: [B, H, W]
                loss = criterion(outputs.squeeze(1), masks)
            else:
                loss = criterion(outputs, masks)
            
            loss.backward()
            optimizer.step()
            
            # 计算IoU
            with torch.no_grad():
                preds = torch.sigmoid(outputs)
                if mode == 'rice':
                    preds = preds.squeeze(1)
                
                preds_binary = (preds > 0.5).float()
                
                intersection = (preds_binary * masks).sum()
                union = preds_binary.sum() + masks.sum() - intersection
                
                if union == 0:
                    iou = 1.0
                else:
                    iou = (intersection / union).item()
            
            total_loss += loss.item()
            total_iou += iou
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_iou = total_iou / num_batches
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}: Loss={avg_loss:.4f}, IoU={avg_iou:.4f}")
    
    print(f"\n最终 IoU: {avg_iou:.4f}")
    
    if avg_iou > 0.9:
        print("✓ 过拟合测试通过！模型可以正常学习。")
        return True
    elif avg_iou > 0.7:
        print("△ 过拟合测试部分通过，可能需要更多epoch")
        return True
    else:
        print("✗ 过拟合测试失败！请检查数据和模型。")
        return False


def verify_pipeline():
    """验证整个流程"""
    print("\n" + "="*60)
    print("流程验证")
    print("="*60)
    
    # 1. 检查数据加载
    print("\n1. 检查数据加载...")
    data_root = '/root/competition_data/public'
    
    # 检查小麦+油菜
    dataset = CropSegmentationDataset(
        image_dir=os.path.join(data_root, 'train/image/wheat_rape'),
        label_dirs=[
            os.path.join(data_root, 'train/label/wheat'),
            os.path.join(data_root, 'train/label/rape')
        ],
        transform=get_validation_augmentation(),
        mode='multi'
    )
    
    image, mask, name = dataset[0]
    print(f"  小麦+油菜: image shape={image.shape}, mask shape={mask.shape}, name={name}")
    
    # 检查水稻
    dataset_rice = CropSegmentationDataset(
        image_dir=os.path.join(data_root, 'train/image/rice'),
        label_dirs=[os.path.join(data_root, 'train/label/rice')],
        transform=get_validation_augmentation(),
        mode='single'
    )
    
    image_r, mask_r, name_r = dataset_rice[0]
    print(f"  水稻: image shape={image_r.shape}, mask shape={mask_r.shape}, name={name_r}")
    
    # 2. 检查模型前向传播
    print("\n2. 检查模型前向传播...")
    
    model_wr = MultiLabelModel(architecture='unet', encoder_name='efficientnet-b3')
    model_rice = SingleLabelModel(architecture='unet', encoder_name='efficientnet-b3')
    
    with torch.no_grad():
        # 小麦+油菜
        img_batch = image.unsqueeze(0)
        out_wr = model_wr(img_batch)
        print(f"  小麦+油菜输出: shape={out_wr.shape}")
        
        # 水稻
        img_r_batch = image_r.unsqueeze(0)
        out_r = model_rice(img_r_batch)
        print(f"  水稻输出: shape={out_r.shape}")
    
    # 3. 检查损失函数
    print("\n3. 检查损失函数...")
    
    criterion_wr = MultiLabelLoss(num_classes=2, loss_type='combined')
    criterion_rice = CombinedLoss()
    
    # 小麦+油菜: pred [B, C, H, W], mask [B, C, H, W]
    loss_wr = criterion_wr(out_wr, mask.unsqueeze(0))
    
    # 水稻: pred [B, 1, H, W], mask [B, H, W] -> 需要 squeeze pred 或 unsqueeze mask
    # 这里我们 squeeze pred 的通道维度
    loss_rice = criterion_rice(out_r.squeeze(1), mask_r.unsqueeze(0))
    
    print(f"  小麦+油菜损失: {loss_wr.item():.4f}")
    print(f"  水稻损失: {loss_rice.item():.4f}")
    
    print("\n✓ 流程验证完成！")
    return True


if __name__ == '__main__':
    # 先验证流程
    verify_pipeline()
    
    # 过拟合测试
    print("\n" + "="*60)
    print("开始过拟合测试")
    print("="*60)
    
    # 小麦+油菜
    overfit_test(mode='wheat_rape', num_samples=20, num_epochs=50)
    
    # 水稻
    overfit_test(mode='rice', num_samples=20, num_epochs=50)
