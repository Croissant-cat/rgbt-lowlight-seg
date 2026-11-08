"""RGB / thermal fusion operators.

Thermal images carry the object silhouette even when the visible image is almost
black, so fusing the two modalities is the classical answer to low light.  Four
families are compared:

* ``weighted``   - per-pixel weighted average (a = 1 - alpha * a + alpha * b)
* ``pca``        - principal-component fusion (the 1st eigenvector weights)
* ``laplacian``  - Laplacian-pyramid fusion (Burt & Adelson), a classic of the
  course: local energy picks the sharper source at every scale
* ``max``        - take the brighter pixel, a naive upper bound
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage


def _to_float(img: np.ndarray) -> np.ndarray:
    return img.astype(np.float32) / 255.0


def _to_u8(img: np.ndarray) -> np.ndarray:
    return np.clip(img * 255.0, 0, 255).round().astype(np.uint8)


def to_gray(img: np.ndarray) -> np.ndarray:
    """Luminance of an RGB or grayscale image, always ``float32`` in [0, 1]."""
    f = _to_float(img)
    if f.ndim == 2:
        return f
    return 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]


def rggb_gray(rgb: np.ndarray) -> np.ndarray:
    return to_gray(rgb)


def weighted(a: np.ndarray, b: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    return _to_u8((1 - alpha) * _to_float(a) + alpha * _to_float(b))


def pca(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Principal-component fusion on the two luminance channels."""
    x = np.stack([to_gray(a).ravel(), to_gray(b).ravel()], axis=1)
    x = x - x.mean(axis=0, keepdims=True)
    cov = np.cov(x, rowvar=False)
    w, v = np.linalg.eigh(cov)
    weights = np.abs(v[:, int(np.argmax(w))])
    weights /= max(weights.sum(), 1e-6)
    fused = weights[0] * to_gray(a) + weights[1] * to_gray(b)
    return _to_u8((fused - fused.min()) / max(float(np.ptp(fused)), 1e-6))


def _lap_pyramid(x: np.ndarray, levels: int) -> list[np.ndarray]:
    pyr = [x]
    for _ in range(levels - 1):
        pyr.append(cv2.pyrDown(pyr[-1]))
    return pyr


def laplacian(a: np.ndarray, b: np.ndarray, levels: int = 4) -> np.ndarray:
    """Laplacian-pyramid fusion with local-energy (absolute Laplacian) selection."""
    ga, gb = to_gray(a), to_gray(b)
    la, lb = _lap_pyramid(ga, levels), _lap_pyramid(gb, levels)
    merged = []
    for level in range(levels):
        if level == levels - 1:                 # coarsest: plain average
            merged.append(0.5 * (la[level] + lb[level]))
            continue
        up_a = cv2.pyrUp(la[level + 1], dstsize=(la[level].shape[1], la[level].shape[0]))
        up_b = cv2.pyrUp(lb[level + 1], dstsize=(lb[level].shape[1], lb[level].shape[0]))
        da = la[level] - up_a
        db = lb[level] - up_b
        ea = ndimage.uniform_filter(np.abs(da), 5)
        eb = ndimage.uniform_filter(np.abs(db), 5)
        merged.append(np.where(ea >= eb, da, db))
    out = merged[-1]
    for level in range(levels - 2, -1, -1):
        out = cv2.pyrUp(out, dstsize=(merged[level].shape[1], merged[level].shape[0])) + merged[level]
    return _to_u8(np.clip(out, 0.0, 1.0))


def max_fuse(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return _to_u8(np.maximum(_to_float(a), _to_float(b)))


FUSIONS = {
    "仅RGB": None,
    "仅热红外": None,
    "加权平均": weighted,
    "PCA融合": pca,
    "拉普拉斯金字塔": laplacian,
    "取最大值": max_fuse,
}


def _as_rgb(x: np.ndarray) -> np.ndarray:
    """Replicate a grayscale result to 3 channels so every consumer sees RGB."""
    return x if x.ndim == 3 else np.repeat(x[..., None], 3, axis=2)


def fuse(name: str, rgb: np.ndarray, thermal: np.ndarray) -> np.ndarray:
    """Return a single RGB image for the requested fusion (grayscale is expanded)."""
    if name == "仅RGB":
        return _as_rgb(rgb)
    if name == "仅热红外":
        return _as_rgb(thermal)
    return _as_rgb(FUSIONS[name](rgb, thermal))
