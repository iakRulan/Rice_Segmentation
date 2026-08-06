"""带图像级分类头的分割模型 + 权重平均工具。

为什么要分类头
--------------
现在的空图判定 (`empty_features`) 只用了概率图上的 12 个手工特征（max、若干分位数、
阈值以上像素数、连通域统计），而且是**在 val 上拟合、又在 val 上评估**的。真正
的信息在图像本身——encoder 早就编码了"这张图有没有这种作物"，只是被 decoder 压
成概率图之后丢掉了。

smp 原生支持 `aux_params`，会在 encoder 最深层特征上接 GAP→Dropout→Linear，
forward 直接返回 (mask_logits, cls_logits)，改动量极小。分类头和分割共享 encoder，
既是正则化也是免费的空图判别器。训练完直接把 sigmoid(cls_logits) 当作
"该图有目标"的概率喂给 postproc.triple_threshold 的 cls_prob。

权重平均
--------
在 664 张 val 上按 best-epoch 选 checkpoint，等于在噪声（σ≈0.005）上取 130 次
最大值，有明显的向上偏差且不迁移。改成对末段多个 epoch 做权重平均（SWA）或
top-k checkpoint 平均，通常更稳、更高。
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

ARCHS = ('unet', 'unetpp', 'deeplabv3plus', 'fpn', 'manet', 'pan')


def build_model(arch: str, encoder: str, classes: int,
                encoder_weights: str | None = 'imagenet',
                aux: bool = True, dropout: float = 0.3) -> nn.Module:
    """aux=True 时 forward 返回 (seg_logits, cls_logits)。

    encoder 可以直接写 timm 的名字，例如 'tu-convnext_small'、'tu-convnextv2_tiny'、
    'tu-swinv2_tiny_window8_256'，比 mit_b3 / efficientnet-b3 有更大的上限。
    """
    import segmentation_models_pytorch as smp
    table = {'unet': smp.Unet, 'unetpp': smp.UnetPlusPlus,
             'deeplabv3plus': smp.DeepLabV3Plus, 'fpn': smp.FPN,
             'manet': smp.MAnet, 'pan': smp.PAN}
    kw = dict(encoder_name=encoder, encoder_weights=encoder_weights,
              in_channels=3, classes=classes, activation=None)
    if aux:
        kw['aux_params'] = dict(classes=classes, dropout=dropout, pooling='avg')
    return table[arch](**kw)


def unpack(out):
    """兼容 aux / 非 aux 两种返回。"""
    if isinstance(out, (tuple, list)):
        return out[0], out[1]
    return out, None


# --------------------------------------------------------- 权重平均
@torch.no_grad()
def average_state_dicts(paths, key: str = 'model_state_dict'):
    """对多个 checkpoint 做等权重平均（SWA / top-k ensembling in weight space）。

    只对浮点张量求平均；整型 buffer（如 num_batches_tracked）取第一个。
    平均后如果模型含 BatchNorm，最好再用训练集跑一遍前向重估 BN 统计量
    （torch.optim.swa_utils.update_bn）；纯 Transformer encoder(mit_*/swin) 不需要。
    """
    paths = [Path(p) for p in paths]
    acc, n = None, 0
    for p in paths:
        ck = torch.load(p, map_location='cpu', weights_only=False)
        sd = ck.get(key, ck) if isinstance(ck, dict) else ck
        if acc is None:
            acc = {k: (v.clone().float() if v.is_floating_point() else v.clone())
                   for k, v in sd.items()}
        else:
            for k, v in sd.items():
                if acc[k].is_floating_point():
                    acc[k] += v.float()
        n += 1
    for k, v in acc.items():
        if v.is_floating_point():
            acc[k] = (v / n)
    return acc


def topk_checkpoints(ckpt_dir, pattern: str, k: int = 3, key: str = 'val_iou'):
    """按 checkpoint 里记录的 val_iou 取前 k 个路径。"""
    items = []
    for p in sorted(Path(ckpt_dir).glob(pattern)):
        ck = torch.load(p, map_location='cpu', weights_only=False)
        items.append((float(ck.get(key, 0.0)), p))
    items.sort(reverse=True)
    return [p for _, p in items[:k]]
