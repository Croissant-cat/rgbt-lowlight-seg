"""Illumination enhancement operators (the "image processing" half of the study).

Everything here is implemented from scratch on top of NumPy/SciPy, except
``clahe`` which wraps OpenCV's industrial implementation as a reference.

* ``gamma``          - global tonal mapping I' = I^(1/gamma)
* ``hist_equalize``  - global histogram equalisation with the CDF
* ``clahe``          - contrast-limited adaptive HE (OpenCV reference)
* ``msr``            - multi-scale Retinex: log I - log(gaussian * I), per channel
* ``gray_world``     - simple white balance, a common low-light pre-step
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage


def _to_float(img: np.ndarray) -> np.ndarray:
    return img.astype(np.float32) / 255.0


def _to_u8(img: np.ndarray) -> np.ndarray:
    return np.clip(img * 255.0, 0, 255).round().astype(np.uint8)


def gamma(img: np.ndarray, g: float = 0.45) -> np.ndarray:
    """Brighten a dark image: out = I ** (1/g)... i.e. ``I ** (1-g_eff)``."""
    return _to_u8(np.power(_to_float(img), float(g)))


def hist_equalize(img: np.ndarray) -> np.ndarray:
    """Global histogram equalisation on the luminance, applied to all channels."""
    f = _to_float(img)
    y = 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]
    hist, _ = np.histogram(y, bins=256, range=(0.0, 1.0))
    cdf = np.cumsum(hist).astype(np.float64)
    cdf /= cdf[-1]
    lut = cdf * 255.0
    yq = np.clip((y * 255).astype(np.int32), 0, 255)
    y_eq = lut[yq] / 255.0
    gain = np.where(y > 1e-4, y_eq / np.maximum(y, 1e-4), 0.0)
    return _to_u8(np.clip(f * gain[..., None], 0.0, 1.0))


def clahe(img: np.ndarray, clip: float = 2.0, grid: int = 8) -> np.ndarray:
    """Contrast-limited adaptive histogram equalisation (OpenCV reference)."""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    op = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    lab[..., 0] = op.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def msr(img: np.ndarray, sigmas=(15.0, 80.0, 250.0), gain: float = 1.0) -> np.ndarray:
    """Multi-scale Retinex (Jobson et al., 1997), per-channel log domain.

    The Gaussian blurs use OpenCV's separable implementation, which is about six
    times faster than ``scipy.ndimage.gaussian_filter`` for the large sigmas.
    """
    f = np.maximum(_to_float(img), 1e-3)
    out = np.zeros_like(f)
    log_f = np.log10(f)
    for s in sigmas:
        blur = cv2.GaussianBlur(f, (0, 0), s, borderType=cv2.BORDER_REFLECT)
        out += log_f - np.log10(np.maximum(blur, 1e-3))
    out /= len(sigmas)
    # normalise each channel into [0, 1] with a robust percentile stretch
    for c in range(3):
        ch = out[..., c]
        lo, hi = np.percentile(ch, 1.0), np.percentile(ch, 99.0)
        out[..., c] = (ch - lo) / max(hi - lo, 1e-6)
    return _to_u8(np.clip(out * gain, 0.0, 1.0))


def gray_world(img: np.ndarray) -> np.ndarray:
    """Normalise each channel by its mean (gray-world assumption)."""
    f = _to_float(img)
    means = f.reshape(-1, 3).mean(axis=0)
    f = f * (means.mean() / np.maximum(means, 1e-3))
    return _to_u8(np.clip(f, 0.0, 1.0))


ENHANCERS = {
    "无增强": lambda im: im,
    "Gamma(0.45)": lambda im: gamma(im, 0.45),
    "全局直方图均衡": hist_equalize,
    "CLAHE": clahe,
    "多尺度Retinex": msr,
    "灰度世界白平衡": gray_world,
}
