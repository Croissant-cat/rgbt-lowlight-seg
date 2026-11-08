"""Unit tests for the low-light RGB-T segmentation experiment.

Run with pytest (``pytest tests``) or standalone (``python tests/test_core.py``).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ipen.data import load_frames                                   # noqa: E402
from ipen.enhance import ENHANCERS, gamma                            # noqa: E402
from ipen.fuse import FUSIONS, fuse, to_gray                         # noqa: E402
from ipen.metrics import evaluate, fbeta, iou, mae                     # noqa: E402
from ipen.segment import SEGMENTERS, postprocess, tophat             # noqa: E402


def _frame():
    return load_frames(limit=1)[0]


def test_dataset_has_pairs():
    frames = load_frames()
    assert len(frames) == 12
    for fr in frames:
        assert fr.rgb.shape[:2] == fr.gt.shape
        assert fr.thermal.shape[:2] == fr.gt.shape
        assert fr.gt.dtype == bool


def test_enhancers_keep_shape_and_dtype():
    fr = _frame()
    for name, fn in ENHANCERS.items():
        out = fn(fr.rgb)
        assert out.shape == fr.rgb.shape, name
        assert out.dtype == np.uint8, name
        assert 0 <= out.min() and out.max() <= 255


def test_gamma_brightens_dark_image():
    fr = _frame()
    assert gamma(fr.rgb, 0.45).mean() > fr.rgb.mean()


def test_fusion_returns_rgb_and_is_finite():
    fr = _frame()
    for name in FUSIONS:
        out = fuse(name, fr.rgb, fr.thermal)
        assert out.ndim == 3 and out.shape[2] == 3, name
        assert np.isfinite(out).all()


def test_to_gray_accepts_2d_and_3d():
    a = np.zeros((4, 5), np.uint8)
    b = np.zeros((4, 5, 3), np.uint8)
    assert to_gray(a).shape == (4, 5)
    assert to_gray(b).shape == (4, 5)


def test_segmenters_return_masks_of_right_shape():
    fr = _frame()
    for name, fn in SEGMENTERS.items():
        m = fn(fr.rgb)
        assert m.shape == fr.gt.shape, name
        assert m.dtype == bool, name
    t = tophat(fr.rgb)
    assert t.shape == fr.gt.shape and t.dtype == np.float32


def test_metrics_identity_and_bounds():
    fr = _frame()
    g = fr.gt.astype(np.float32)
    assert iou(g >= 0.5, fr.gt) == 1.0
    assert abs(fbeta(g >= 0.5, fr.gt) - 1.0) < 1e-9
    assert mae(g, fr.gt) == 0.0
    empty = np.zeros_like(g)
    assert iou(empty >= 0.5, fr.gt) == 0.0
    sc = evaluate(g, fr.gt)
    assert all(np.isfinite(v) for v in sc.values())


def test_postprocess_drops_specks():
    m = np.zeros((60, 60), bool)
    m[10:30, 10:30] = True          # a real blob
    m[50, 50] = True                # a single speck
    out = postprocess(m, area_min=50)
    assert out[20, 20]
    assert not out[50, 50]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok", fn.__name__)
    print(f"{len(fns)} tests passed")
