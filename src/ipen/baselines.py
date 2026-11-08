"""Strong classical baselines that are given extra prior knowledge.

These are deliberately *generous* to the classical side, so that the reported
ceiling cannot be blamed on missing tuning:

* ``grabcut_gtbox``  - GrabCut initialised with the ground-truth bounding box
  (the "user drew a perfect box" scenario);
* ``area_oracle``    - the threshold is chosen so that the predicted area equals
  the true object area ("we know how big the object is");
* ``vote_ensemble``  - majority vote of Otsu, adaptive and top-hat thresholds.
"""
from __future__ import annotations

import cv2
import numpy as np

from .fuse import to_gray
from .segment import adaptive, otsu, postprocess, tophat


def gt_box(gt: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(gt)
    if len(ys) == 0:
        return None
    h, w = gt.shape
    return (max(0, int(xs.min()) - 4), max(0, int(ys.min()) - 4),
            min(w - 1, int(xs.max()) + 4), min(h - 1, int(ys.max()) + 4))


def grabcut_gtbox(img: np.ndarray, gt: np.ndarray, scale: float = 0.5) -> np.ndarray:
    """GrabCut with the *ground-truth* bounding box as initialisation."""
    box = gt_box(gt)
    h, w = gt.shape
    if box is None:
        return np.zeros((h, w), bool)
    small = cv2.resize(img, (max(2, int(w * scale)), max(2, int(h * scale))),
                       interpolation=cv2.INTER_AREA)
    x0, y0, x1, y1 = [int(v * scale) for v in box]
    rect = (x0, y0, max(2, x1 - x0), max(2, y1 - y0))
    gc = np.zeros(small.shape[:2], np.uint8)
    bgm = np.zeros((1, 65), np.float64)
    fgm = np.zeros((1, 65), np.float64)
    cv2.setRNGSeed(0)
    try:
        cv2.grabCut(small, gc, rect, bgm, fgm, 3, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return np.zeros((h, w), bool)
    out = ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).astype(np.uint8)
    return cv2.resize(out, (w, h), interpolation=cv2.INTER_NEAREST) > 0


def area_oracle(img: np.ndarray, gt: np.ndarray, use_tophat: bool = True) -> np.ndarray:
    """Threshold picked so that the predicted area matches the true object area."""
    resp = tophat(img, 21) if use_tophat else to_gray(img) * 255.0
    frac = float(gt.mean())
    if frac <= 0:
        return np.zeros(gt.shape, bool)
    q = max(0.0, 1.0 - frac)
    t = np.quantile(resp, q)
    return resp > t


def vote_ensemble(img: np.ndarray) -> np.ndarray:
    """Majority vote of three textbook segmenters (no prior knowledge)."""
    votes = (otsu(img).astype(np.int8)
             + adaptive(img, 51, 10).astype(np.int8)
             + (tophat(img, 21) > np.quantile(tophat(img, 21), 0.995)).astype(np.int8))
    return votes >= 2
