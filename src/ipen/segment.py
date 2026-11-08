"""Classical segmentation operators.

Six families, all classical image processing:

* ``otsu``        - global threshold from the between-class variance
* ``triangle``    - triangle (Zack) threshold, robust for unimodal dark images
* ``adaptive``    - local mean threshold with a window and a bias
* ``kmeans``      - 2-means clustering in (L, a, b) colour space
* ``region_grow`` - seeded region growing from the brightest pixel
* ``grabcut``     - OpenCV's graph-cut refinement inside the Otsu bounding box

They all return a boolean mask of the same shape as the input.
"""
from __future__ import annotations

import cv2
import numpy as np

from .fuse import to_gray


def _u8(x: np.ndarray) -> np.ndarray:
    if x.dtype == np.uint8:
        return x
    return np.clip(x, 0, 255).astype(np.uint8)


def otsu(img: np.ndarray) -> np.ndarray:
    g = _u8(to_gray(img) * 255)
    gb = ndimage_median(g, 3)
    t, _ = cv2.threshold(gb, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return gb >= t


def ndimage_median(x: np.ndarray, k: int) -> np.ndarray:
    from scipy import ndimage
    return ndimage.median_filter(x, size=k)


def triangle(img: np.ndarray) -> np.ndarray:
    g = _u8(to_gray(img) * 255)
    gb = ndimage_median(g, 3)
    t, _ = cv2.threshold(gb, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
    return gb >= t


def adaptive(img: np.ndarray, block: int = 51, c: float = 10.0) -> np.ndarray:
    g = _u8(to_gray(img) * 255)
    block = block if block % 2 == 1 else block + 1
    out = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY, block, -c)
    return out > 0


def kmeans(img: np.ndarray, k: int = 2, seed: int = 0) -> np.ndarray:
    lab = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    data = lab.reshape(-1, 3)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    cv2.setRNGSeed(seed)
    _, labels, centers = cv2.kmeans(data, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    labels = labels.ravel()
    brightness = centers[:, 0]                      # L channel of each cluster
    bright = int(np.argmax(brightness))
    return (labels == bright).reshape(img.shape[:2])


def region_grow(img: np.ndarray, tol: float = 0.18) -> np.ndarray:
    """Seeded region growing from the brightest 0.5 % of pixels."""
    g = to_gray(img).astype(np.float32)
    g = (g - g.min()) / max(float(np.ptp(g)), 1e-6)
    thr = np.quantile(g, 0.995)
    ys, xs = np.nonzero(g >= thr)
    if len(ys) == 0:
        return np.zeros(g.shape, bool)
    mask = np.zeros(g.shape, bool)
    mask[ys, xs] = True
    for _ in range(200):
        grown = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        frontier = grown & ~mask
        if not frontier.any():
            break
        ok = frontier & (g >= thr - tol)
        if not ok.any():
            break
        mask |= ok
    return mask


def grabcut(img: np.ndarray, mask: np.ndarray | None = None,
            scale: float = 0.5) -> np.ndarray:
    """GrabCut initialised from the Otsu bounding box (user-drawn box surrogate).

    The graph cut runs on a ``scale``-sized copy and the mask is upsampled, which
    keeps the cost at a few hundred milliseconds per frame; the segmentation
    boundary of a small object is barely affected.
    """
    base = otsu(img) if mask is None else mask
    h, w = base.shape
    small = cv2.resize(img, (max(2, int(w * scale)), max(2, int(h * scale))),
                       interpolation=cv2.INTER_AREA)
    base_s = cv2.resize(base.astype(np.uint8), (small.shape[1], small.shape[0]),
                        interpolation=cv2.INTER_NEAREST) > 0
    ys, xs = np.nonzero(base_s)
    if len(ys) < 5:
        return np.zeros((h, w), bool)
    pad = 6
    x0, x1 = max(0, int(xs.min()) - pad), min(small.shape[1] - 1, int(xs.max()) + pad)
    y0, y1 = max(0, int(ys.min()) - pad), min(small.shape[0] - 1, int(ys.max()) + pad)
    gc = np.full(small.shape[:2], cv2.GC_BGD, np.uint8)
    gc[y0:y1 + 1, x0:x1 + 1] = cv2.GC_PR_BGD
    gc[base_s] = cv2.GC_FGD
    bgm = np.zeros((1, 65), np.float64)
    fgm = np.zeros((1, 65), np.float64)
    cv2.setRNGSeed(0)
    try:
        cv2.grabCut(small, gc, None, bgm, fgm, 2, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return cv2.resize(base.astype(np.uint8), (w, h),
                          interpolation=cv2.INTER_NEAREST) > 0
    out = ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).astype(np.uint8)
    return cv2.resize(out, (w, h), interpolation=cv2.INTER_NEAREST) > 0


def postprocess(mask: np.ndarray, area_min: int = 120) -> np.ndarray:
    """Morphological clean-up: open, close and drop tiny components."""
    m = mask.astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    out = np.zeros_like(m)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= area_min:
            out[labels == i] = 1
    return out > 0



def tophat(img: np.ndarray, size: int = 21) -> np.ndarray:
    """White top-hat response: small bright structures on a dark background.

    This is the textbook morphological operator for small-object detection: the
    opening removes everything smaller than the structuring element, so the
    residual image keeps exactly the small bright blobs.
    """
    g = _u8(to_gray(img) * 255).astype(np.float32)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.morphologyEx(g, cv2.MORPH_TOPHAT, ker)


SEGMENTERS = {
    "Otsu": otsu,
    "三角形阈值": triangle,
    "自适应阈值": adaptive,
    "K-means": kmeans,
    "区域生长": region_grow,
    "GrabCut": grabcut,
}
