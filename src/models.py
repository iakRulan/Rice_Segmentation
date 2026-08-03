"""
模型定义
支持多种分割架构：U-Net, DeepLabV3+, FPN, PSPNet
使用 segmentation_models_pytorch 库
"""
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


def get_model(architecture='unet', encoder_name='efficientnet-b3', 
              in_channels=3, classes=1, activation=None):
    """
    获取分割模型
    """
    model_kwargs = dict(
        encoder_name=encoder_name,
        encoder_weights='imagenet',
        in_channels=in_channels,
        classes=classes,
        activation=activation,
    )
    
    if architecture == 'unet':
        model = smp.Unet(**model_kwargs)
    elif architecture == 'deeplabv3plus':
        model = smp.DeepLabV3Plus(**model_kwargs)
    elif architecture == 'fpn':
        model = smp.FPN(**model_kwargs)
    elif architecture == 'pspnet':
        model = smp.PSPNet(**model_kwargs)
    elif architecture == 'manet':
        model = smp.MAnet(**model_kwargs)
    elif architecture == 'pan':
        model = smp.PAN(**model_kwargs)
    elif architecture == 'linknet':
        model = smp.Linknet(**model_kwargs)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    return model


class MultiLabelModel(nn.Module):
    """
    多标签分割模型 - 用于小麦+油菜共享影像
    输出2个通道：小麦和油菜的独立二值掩膜
    """
    def __init__(self, architecture='unet', encoder_name='efficientnet-b3'):
        super().__init__()
        self.backbone = get_model(
            architecture=architecture,
            encoder_name=encoder_name,
            in_channels=3,
            classes=2,
            activation=None
        )
    
    def forward(self, x):
        return self.backbone(x)


class SingleLabelModel(nn.Module):
    """
    单标签分割模型 - 用于水稻的独立分割
    输出1个通道：水稻的二值掩膜
    """
    def __init__(self, architecture='unet', encoder_name='efficientnet-b3'):
        super().__init__()
        self.backbone = get_model(
            architecture=architecture,
            encoder_name=encoder_name,
            in_channels=3,
            classes=1,
            activation=None
        )
    
    def forward(self, x):
        return self.backbone(x)
