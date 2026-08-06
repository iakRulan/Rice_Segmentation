"""指标对齐的损失函数 —— 替换 scripts/train_local.py 里的 CombinedLoss。

核心修正
--------
原 CombinedLoss.soft_dice 把整个 batch（含 batch 维和通道维）flatten 后算一个
Dice。比赛指标是 **逐图 IoU 再平均**，两者严重不匹配：

  * batch-level Dice 下，一张空图上误检 200 个像素，会被同 batch 其它图的
    大面积正样本稀释掉 —— 而在指标里这张图直接 IoU = 0。
  * 数据里 wheat/rape 约 48-50% 是空图，rice 约 22%。也就是说近一半样本的梯度
    信号被稀释了。

本文件的 dice / iou / tversky 全部在 (B, C) 维度上逐图逐类计算再平均。对空图，
smooth 项使 loss ≈ 1 - smooth/(pred_sum + smooth)，预测越多惩罚越大，正好对应
"空图必须全零"的指标行为。

用法
----
    from opt_patch.losses_v2 import MultiTaskLoss
    criterion = MultiTaskLoss(bce=1.0, dice=1.0, lovasz=0.5, cls=0.5,
                              focal_gamma=2.0, pos_weight=1.3)
    seg_logits, cls_logits = model(images)          # aux_params 模型
    loss = criterion(seg_logits, masks, cls_logits)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- Lovasz
def lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_hinge_flat(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if labels.numel() == 0:
        return logits.sum() * 0.0
    signs = 2.0 * labels.float() - 1.0
    errors = 1.0 - logits * signs
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    grad = lovasz_grad(labels[perm])
    return torch.dot(F.relu(errors_sorted), grad)


def lovasz_hinge(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """logits/labels: (B, H, W) —— 逐图计算后平均（与指标一致）。"""
    losses = [lovasz_hinge_flat(logits[i].reshape(-1), labels[i].reshape(-1))
              for i in range(logits.size(0))]
    return torch.stack(losses).mean()


# ------------------------------------------------- 逐图 region losses
def _per_image_stats(logits: torch.Tensor, target: torch.Tensor):
    """返回逐图逐类的 (tp, fp, fn)，形状均为 (B, C)。"""
    p = torch.sigmoid(logits)
    dims = (-2, -1)
    tp = (p * target).sum(dims)
    fp = (p * (1.0 - target)).sum(dims)
    fn = ((1.0 - p) * target).sum(dims)
    return tp, fp, fn


def soft_dice_per_image(logits, target, smooth: float = 1.0):
    tp, fp, fn = _per_image_stats(logits, target)
    dice = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)
    return (1.0 - dice).mean()


def soft_iou_per_image(logits, target, smooth: float = 1.0):
    """比 Dice 更贴近评测指标（评测就是 IoU）。"""
    tp, fp, fn = _per_image_stats(logits, target)
    iou = (tp + smooth) / (tp + fp + fn + smooth)
    return (1.0 - iou).mean()


def tversky_per_image(logits, target, alpha: float = 0.3, beta: float = 0.7,
                      gamma: float = 1.0, smooth: float = 1.0):
    """beta > alpha 时更重罚漏检 —— 对油菜这种小而散的目标召回不足很有效。
    gamma > 1 即 Focal-Tversky。"""
    tp, fp, fn = _per_image_stats(logits, target)
    tv = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    loss = 1.0 - tv
    if gamma != 1.0:
        loss = loss.clamp_min(1e-6) ** gamma
    return loss.mean()


# ------------------------------------------------------------ 组合损失
class MultiTaskLoss(nn.Module):
    """分割 + 图像级"有无目标"分类的多任务损失。

    cls_logits 传 None 时退化为纯分割损失，可直接替换原 CombinedLoss。
    """

    def __init__(self, bce: float = 1.0, dice: float = 1.0, lovasz: float = 0.5,
                 iou: float = 0.0, tversky: float = 0.0, cls: float = 0.0,
                 focal_gamma: float = 0.0, pos_weight: float = 1.0,
                 tversky_alpha: float = 0.3, tversky_beta: float = 0.7,
                 tversky_gamma: float = 1.0, smooth: float = 1.0):
        super().__init__()
        self.w = dict(bce=bce, dice=dice, lovasz=lovasz, iou=iou,
                      tversky=tversky, cls=cls)
        self.focal_gamma = focal_gamma
        self.pos_weight = pos_weight
        self.tv = (tversky_alpha, tversky_beta, tversky_gamma)
        self.smooth = smooth
        self._bce = nn.BCEWithLogitsLoss(reduction='none')

    def pointwise(self, logits, target):
        """逐像素 BCE（可加 focal / pos_weight），但按图平均而非全局平均。"""
        bce = self._bce(logits, target)
        if self.pos_weight != 1.0:
            bce = bce * (target * self.pos_weight + (1.0 - target))
        if self.focal_gamma > 0:
            prob = torch.sigmoid(logits)
            pt = torch.where(target > 0.5, prob, 1.0 - prob)
            bce = bce * (1.0 - pt).clamp_min(1e-6) ** self.focal_gamma
        return bce.mean(dim=(-2, -1)).mean()

    def forward(self, seg_logits, target, cls_logits=None):
        loss = seg_logits.sum() * 0.0
        if self.w['bce']:
            loss = loss + self.pointwise(seg_logits, target) * self.w['bce']
        if self.w['dice']:
            loss = loss + soft_dice_per_image(seg_logits, target, self.smooth) * self.w['dice']
        if self.w['iou']:
            loss = loss + soft_iou_per_image(seg_logits, target, self.smooth) * self.w['iou']
        if self.w['tversky']:
            a, b, g = self.tv
            loss = loss + tversky_per_image(seg_logits, target, a, b, g, self.smooth) * self.w['tversky']
        if self.w['lovasz']:
            lovs = [lovasz_hinge(seg_logits[:, c], target[:, c])
                    for c in range(seg_logits.size(1))]
            loss = loss + torch.stack(lovs).mean() * self.w['lovasz']
        if self.w['cls'] and cls_logits is not None:
            # 图像级标签：该类在这张图上是否存在
            has = (target.sum(dim=(-2, -1)) > 0).float()          # (B, C)
            loss = loss + F.binary_cross_entropy_with_logits(cls_logits, has) * self.w['cls']
        return loss
