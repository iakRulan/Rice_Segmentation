"""
损失函数
结合多种损失函数以提升分割精度
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Dice Loss"""
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred_flat = pred.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice


class CombinedLoss(nn.Module):
    """组合损失：BCE + Dice"""
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
    
    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class MultiLabelLoss(nn.Module):
    """
    多标签损失
    对每个类别独立计算损失后求和
    """
    def __init__(self, num_classes=2, class_weights=None, loss_type='combined'):
        super().__init__()
        self.num_classes = num_classes
        self.class_weights = class_weights if class_weights else [1.0] * num_classes
        
        if loss_type == 'combined':
            self.loss_fn = CombinedLoss()
        elif loss_type == 'dice':
            self.loss_fn = DiceLoss()
        else:
            self.loss_fn = nn.BCEWithLogitsLoss()
    
    def forward(self, pred, target):
        """
        pred: (B, C, H, W)
        target: (B, C, H, W)
        """
        total_loss = 0.0
        for c in range(self.num_classes):
            class_pred = pred[:, c:c+1, :, :]
            class_target = target[:, c:c+1, :, :]
            loss = self.loss_fn(class_pred, class_target)
            total_loss += self.class_weights[c] * loss
        
        return total_loss / self.num_classes
