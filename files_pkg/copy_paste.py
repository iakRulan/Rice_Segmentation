"""Copy-Paste 增强 —— 针对油菜（目标小而散、非空图 IoU 只有 0.78）的主要杠杆。

做法：从另一张有目标的图上抠出若干前景连通域，随机缩放/翻转后贴到当前图上，
同时更新 mask。相比 crop_zoom（只是把目标放大，上下文被破坏），Copy-Paste
在保持原始尺度和背景统计的前提下增加正样本密度，是小目标分割的标配增强。

注意：约一半样本是空图。**不要**给空图贴前景（会把空图样本从训练集里抹掉，而
空图判定占了指标的一半）。默认只在非空图之间做 paste；如果想让分类头见到更难的
负样本，可以用 `to_empty_p` 少量地往空图上贴（此时该图不再是空图，标签会同步更新）。

集成方式（改 scripts/train_local.py 的 CropDataset.__getitem__）：

    from opt_patch.copy_paste import CopyPasteMixer
    # __init__ 里
    self.cp = CopyPasteMixer(p=0.4, max_objs=3) if copy_paste else None
    # __getitem__ 里，读完 image/masks 之后、crop_zoom 之前
    if self.cp is not None:
        src_i = np.random.randint(len(self))
        s_img, s_msk = self._raw(src_i)
        image, masks = self.cp(image, masks, s_img, s_msk)
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def _as_hwc(mask: np.ndarray) -> tuple[np.ndarray, bool]:
    """统一成 (H, W, C)，返回是否原本是单通道。"""
    if mask.ndim == 2:
        return mask[:, :, None], True
    return mask, False


class CopyPasteMixer:
    def __init__(self, p: float = 0.4, max_objs: int = 3,
                 scale_range: tuple[float, float] = (0.7, 1.4),
                 min_src_area: int = 40, to_empty_p: float = 0.0,
                 feather: int = 1, rng: np.random.Generator | None = None):
        self.p = p
        self.max_objs = max_objs
        self.scale_range = scale_range
        self.min_src_area = min_src_area
        self.to_empty_p = to_empty_p
        self.feather = feather
        self.rng = rng or np.random.default_rng()

    # ------------------------------------------------------------------
    def __call__(self, image: np.ndarray, mask: np.ndarray,
                 src_image: np.ndarray, src_mask: np.ndarray):
        """image (H,W,3) uint8; mask (H,W) 或 (H,W,C) float{0,1}。"""
        mask_hwc, was_2d = _as_hwc(mask)
        src_hwc, _ = _as_hwc(src_mask)
        if src_hwc.shape[-1] != mask_hwc.shape[-1]:
            return image, mask

        tgt_empty = mask_hwc.sum() == 0
        if tgt_empty and self.rng.random() >= self.to_empty_p:
            return image, mask
        if not tgt_empty and self.rng.random() >= self.p:
            return image, mask
        if src_hwc.sum() == 0:
            return image, mask

        image = image.copy()
        mask_hwc = mask_hwc.copy()
        n_pasted = 0
        C = mask_hwc.shape[-1]
        order = self.rng.permutation(C)
        for c in order:
            if n_pasted >= self.max_objs:
                break
            comps = self._components(src_hwc[:, :, c])
            self.rng.shuffle(comps)
            for (sl_y, sl_x) in comps:
                if n_pasted >= self.max_objs:
                    break
                ok = self._paste_one(image, mask_hwc, src_image, src_hwc,
                                     c, sl_y, sl_x)
                n_pasted += int(ok)

        out_mask = mask_hwc[:, :, 0] if was_2d else mask_hwc
        return image, out_mask

    # ------------------------------------------------------------------
    def _components(self, m: np.ndarray):
        binm = (m > 0).astype(np.uint8)
        if binm.sum() == 0:
            return []
        labeled, n = ndimage.label(binm)
        out = []
        for sl in ndimage.find_objects(labeled):
            if sl is None:
                continue
            h = sl[0].stop - sl[0].start
            w = sl[1].stop - sl[1].start
            if h * w >= self.min_src_area:
                out.append((sl[0], sl[1]))
        return out

    def _paste_one(self, image, mask_hwc, src_image, src_hwc, c, sl_y, sl_x) -> bool:
        patch = src_image[sl_y, sl_x]                       # (h,w,3)
        pm = (src_hwc[sl_y, sl_x, c] > 0).astype(np.float32)  # (h,w)
        h, w = pm.shape
        H, W = image.shape[:2]

        s = float(self.rng.uniform(*self.scale_range))
        nh, nw = max(2, int(round(h * s))), max(2, int(round(w * s)))
        if nh >= H or nw >= W:
            return False
        patch = _resize_nn(patch, nh, nw)
        pm = _resize_nn(pm[:, :, None], nh, nw)[:, :, 0]

        if self.rng.random() < 0.5:
            patch, pm = patch[:, ::-1], pm[:, ::-1]
        if self.rng.random() < 0.5:
            patch, pm = patch[::-1], pm[::-1]

        y0 = int(self.rng.integers(0, H - nh + 1))
        x0 = int(self.rng.integers(0, W - nw + 1))
        alpha = pm
        if self.feather > 0:
            alpha = ndimage.uniform_filter(pm, size=2 * self.feather + 1)
            alpha = np.clip(alpha, 0, 1) * pm  # 只在前景内部羽化，不外扩
        a3 = alpha[:, :, None]
        region = image[y0:y0 + nh, x0:x0 + nw].astype(np.float32)
        image[y0:y0 + nh, x0:x0 + nw] = (
            region * (1 - a3) + patch.astype(np.float32) * a3
        ).astype(image.dtype)
        sub = mask_hwc[y0:y0 + nh, x0:x0 + nw, c]
        mask_hwc[y0:y0 + nh, x0:x0 + nw, c] = np.maximum(sub, (pm > 0.5).astype(sub.dtype))
        return True


def _resize_nn(arr: np.ndarray, nh: int, nw: int) -> np.ndarray:
    """最近邻缩放，避免引入 cv2/PIL 依赖，也保证 mask 不产生中间值。"""
    h, w = arr.shape[:2]
    yi = (np.arange(nh) * (h / nh)).astype(np.int64).clip(0, h - 1)
    xi = (np.arange(nw) * (w / nw)).astype(np.int64).clip(0, w - 1)
    return arr[yi][:, xi]
