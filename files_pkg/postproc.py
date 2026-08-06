"""指标驱动的后处理与阈值搜索 —— 替换 local_blend_eval.py 里的 pp/fit_empty。

三个改动
--------
1. **三阈值方案**（SIIM-ACR Pneumothorax 冠军方案的标准做法）：
   用高阈值 t_hi 做"是否有目标"的判定门，面积不足 min_size 就整图置空；
   通过判定后改用低阈值 t_lo 画 mask。
   现有代码只有"单阈值 + 去小连通域"，无法把"整图置空"和"mask 精细度"解耦。
   在 per-image IoU 指标下这两件事的最优阈值本来就不同（置空要保守，画 mask 要激进）。

2. **空图判定按指标优化，而不是按分类准确率**。原 fit_empty_model 里
   `acc = ((p > th) == y).mean()` 选阈值 —— 这是最大化分类准确率，不是最大化 IoU。
   两者的最优点不同：设 q = P(空图)，非空图的期望 IoU 为 J，则
       置空的期望收益 = q
       出图的期望收益 = (1 - q) * J
   最优规则是 q > J / (1 + J)。J≈0.80 时门限是 0.444，J≈0.87 时是 0.465 ——
   都明显低于 0.5，而且随类别不同。所以应当直接对最终指标扫。

3. 阈值/权重的拟合必须在 **OOF（out-of-fold）** 预测上做。在同一份 val 上
   既拟合又评估，得到的分数是虚高的（PROGRESS 里已经踩过一次这个坑）。
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


# ------------------------------------------------------------ 基础工具
def remove_small(mask: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 0:
        return mask
    labeled, n = ndimage.label(mask)
    if n == 0:
        return mask
    areas = ndimage.sum(mask, labeled, range(1, n + 1))
    kill = np.flatnonzero(areas < min_area) + 1
    if kill.size:
        mask = mask.copy()
        mask[np.isin(labeled, kill)] = 0
    return mask


def fill_holes(mask: np.ndarray, max_hole: int) -> np.ndarray:
    if max_hole <= 0:
        return mask
    inv = 1 - mask
    labeled, n = ndimage.label(inv)
    if n == 0:
        return mask
    areas = ndimage.sum(inv, labeled, range(1, n + 1))
    fill = np.flatnonzero(areas < max_hole) + 1
    if fill.size:
        mask = mask.copy()
        mask[np.isin(labeled, fill)] = 1
    return mask


def iou(pred: np.ndarray, tgt: np.ndarray) -> float:
    inter = np.logical_and(pred, tgt).sum()
    union = np.logical_or(pred, tgt).sum()
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return float(inter) / float(union)


# ------------------------------------------------------- 三阈值后处理
def triple_threshold(prob: np.ndarray, t_hi: float, min_size: int, t_lo: float,
                     min_area: int = 0, max_hole: int = 0,
                     cls_prob: float | None = None, cls_gate: float | None = None) -> np.ndarray:
    """prob: (H, W) 概率图，返回 uint8 mask。

    cls_prob 是（可选的）分类头给出的"这张图有目标"的概率；低于 cls_gate 直接置空。
    """
    if cls_prob is not None and cls_gate is not None and cls_prob < cls_gate:
        return np.zeros(prob.shape, np.uint8)
    gate = prob > t_hi
    if gate.sum() < min_size:                      # 判定为空图
        return np.zeros(prob.shape, np.uint8)
    mask = (prob > t_lo).astype(np.uint8)
    mask = remove_small(mask, min_area)
    mask = fill_holes(mask, max_hole)
    return mask


def empty_gate_from_expected_iou(j_hat: float) -> float:
    """给定非空图的平均 IoU，返回置空决策的贝叶斯最优门限 q*。"""
    return j_hat / (1.0 + j_hat)


# ------------------------------------------------------------ 阈值搜索
def _score(probs: np.ndarray, tgts: np.ndarray, t_hi, min_size, t_lo,
           min_area, max_hole, cls_probs=None, cls_gate=None) -> float:
    """probs (N,H,W) float32, tgts (N,H,W) uint8。返回平均逐图 IoU。"""
    n = probs.shape[0]
    gate_area = (probs > t_hi).reshape(n, -1).sum(1)
    keep = gate_area >= min_size
    if cls_probs is not None and cls_gate is not None:
        keep &= (cls_probs >= cls_gate)
    masks = (probs > t_lo).astype(np.uint8)
    masks[~keep] = 0
    if min_area > 0 or max_hole > 0:
        for i in np.flatnonzero(keep):
            m = remove_small(masks[i], min_area)
            masks[i] = fill_holes(m, max_hole)
    inter = np.logical_and(masks, tgts).sum(axis=(1, 2))
    union = np.logical_or(masks, tgts).sum(axis=(1, 2))
    ious = np.where(union == 0, np.where(inter == 0, 1.0, 0.0),
                    inter / np.maximum(union, 1))
    return float(ious.mean())


def search_triple(probs: np.ndarray, tgts: np.ndarray,
                  cls_probs: np.ndarray | None = None,
                  t_hi_grid=(0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80),
                  min_size_grid=(0, 50, 100, 200, 400, 800, 1600, 3000),
                  t_lo_grid=(0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55),
                  pp_grid=((0, 0), (30, 30), (60, 60), (0, 60), (60, 0), (120, 120)),
                  cls_gate_grid=(None, 0.20, 0.30, 0.40, 0.45, 0.50, 0.60),
                  verbose=True) -> dict:
    """坐标下降式搜索（全网格太贵）：先 (t_hi, min_size)，再 t_lo，再 pp，再 cls_gate，
    循环两轮。**务必只在 OOF / 拟合半区上调用，再到另一半评估。**"""
    best = dict(t_hi=0.55, min_size=0, t_lo=0.45, min_area=0, max_hole=0, cls_gate=None)
    best_s = _score(probs, tgts, **best, cls_probs=cls_probs)
    for _ in range(2):
        for th in t_hi_grid:
            for ms in min_size_grid:
                c = dict(best, t_hi=th, min_size=ms)
                s = _score(probs, tgts, **c, cls_probs=cls_probs)
                if s > best_s:
                    best_s, best = s, c
        for tl in t_lo_grid:
            c = dict(best, t_lo=tl)
            s = _score(probs, tgts, **c, cls_probs=cls_probs)
            if s > best_s:
                best_s, best = s, c
        for ma, mh in pp_grid:
            c = dict(best, min_area=ma, max_hole=mh)
            s = _score(probs, tgts, **c, cls_probs=cls_probs)
            if s > best_s:
                best_s, best = s, c
        if cls_probs is not None:
            for cg in cls_gate_grid:
                c = dict(best, cls_gate=cg)
                s = _score(probs, tgts, **c, cls_probs=cls_probs)
                if s > best_s:
                    best_s, best = s, c
        if verbose:
            print(f'  [search] IoU={best_s:.4f} {best}')
    return dict(best, iou=best_s)


def report(probs, tgts, settings, cls_probs=None) -> dict:
    """在给定设置下汇报 总体 / 空图 / 非空图 三个 IoU，用于定位瓶颈。"""
    cfg = {k: settings[k] for k in
           ('t_hi', 'min_size', 't_lo', 'min_area', 'max_hole', 'cls_gate')}
    n = probs.shape[0]
    ious = np.empty(n)
    for i in range(n):
        cp = None if cls_probs is None else float(cls_probs[i])
        m = triple_threshold(probs[i], cls_prob=cp, **cfg)
        ious[i] = iou(m, tgts[i])
    ne = tgts.reshape(n, -1).sum(1) > 0
    return dict(iou=float(ious.mean()),
                empty=float(ious[~ne].mean()) if (~ne).any() else 1.0,
                nonempty=float(ious[ne].mean()) if ne.any() else 1.0,
                n_empty=int((~ne).sum()), n_nonempty=int(ne.sum()))
