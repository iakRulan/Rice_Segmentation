from __future__ import annotations

import numpy as np


def per_image_iou(probabilities: np.ndarray, targets: np.ndarray,
                  threshold: float) -> np.ndarray:
    pred = probabilities > threshold
    truth = targets > 0.5
    intersection = (pred & truth).sum(axis=(-2, -1))
    union = (pred | truth).sum(axis=(-2, -1))
    return np.divide(
        intersection, union, out=np.ones_like(intersection, dtype=np.float64),
        where=union > 0,
    )


def mean_iou(probabilities: np.ndarray, targets: np.ndarray,
             threshold: float = 0.5) -> tuple[float, list[float]]:
    values = per_image_iou(probabilities, targets, threshold)
    per_class = values.mean(axis=0).tolist()
    return float(np.mean(per_class)), [float(v) for v in per_class]


def search_threshold(probabilities: np.ndarray, targets: np.ndarray,
                     low: float = 0.30, high: float = 0.70,
                     step: float = 0.01) -> tuple[float, float]:
    best_score, best_threshold = -1.0, 0.5
    for threshold in np.arange(low, high + step / 2, step):
        score, _ = mean_iou(probabilities, targets, float(threshold))
        if score > best_score:
            best_score, best_threshold = score, float(threshold)
    return best_score, best_threshold
