"""Segmentation metrics, implemented to match the VDT-2048 protocol.

* ``iou``  intersection-over-union of the binary masks
* ``fbeta`` F-measure with beta^2 = 0.3, the saliency convention that weights
  precision more than recall
* ``mae``  mean absolute difference between the probability map and the 0/1 mask
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    p = pred.astype(bool)
    g = gt.astype(bool)
    union = np.logical_or(p, g).sum()
    return float(np.logical_and(p, g).sum() / union) if union else 1.0


def fbeta(pred: np.ndarray, gt: np.ndarray, beta2: float = 0.3) -> float:
    p = pred.astype(bool)
    g = gt.astype(bool)
    tp = float(np.logical_and(p, g).sum())
    fp = float(np.logical_and(p, ~g).sum())
    fn = float(np.logical_and(~p, g).sum())
    denom = (1 + beta2) * tp + beta2 * fn + fp
    return float((1 + beta2) * tp / denom) if denom else 1.0


def mae(prob: np.ndarray, gt: np.ndarray) -> float:
    return float(np.abs(prob.astype(np.float64) - gt.astype(np.float64)).mean())


def evaluate(prob: np.ndarray, gt: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    pred = prob >= threshold
    return {"iou": iou(pred, gt), "fbeta": fbeta(pred, gt), "mae": mae(prob, gt)}
