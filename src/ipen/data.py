"""Dataset access for the low-light RGB-T segmentation experiment.

12 hand-picked frames of the public VDT-2048 benchmark are shipped with the
repository (``data/cases/<id>/{RGB.png, thermal.png, gt.png}``).  Each frame has
a visible-light image, a thermal image and the pixel-accurate ground-truth mask,
so every segmentation method can be scored exactly.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
CASES = ROOT / "data" / "cases"

RGB_NAME = "RGB.png"
THERMAL_NAME = "热红外.png"
GT_NAME = "真实掩码GT.png"


@dataclass
class Frame:
    sid: str
    rgb: np.ndarray        # (H, W, 3) uint8
    thermal: np.ndarray    # (H, W, 3) uint8
    gt: np.ndarray         # (H, W) bool

    @property
    def shape(self):
        return self.gt.shape


def _read(path: Path, mode: str = "RGB") -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert(mode))


def load_frame(sid: str) -> Frame:
    d = CASES / sid
    return Frame(sid, _read(d / RGB_NAME), _read(d / THERMAL_NAME),
                 _read(d / GT_NAME, "L") >= 128)


def load_frames(limit: int | None = None) -> list[Frame]:
    ids = sorted(p.name for p in CASES.iterdir() if p.is_dir())
    if limit:
        ids = ids[:limit]
    return [load_frame(sid) for sid in ids]


def luminance(rgb: np.ndarray) -> np.ndarray:
    f = rgb.astype(np.float32) / 255.0
    return 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]
