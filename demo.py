"""Single-pair segmentation demo.

    python demo.py --rgb data/cases/1001/RGB.png \
                   --thermal data/cases/1001/热红外.png [--method ours|classic|both]

``best``          - Gamma(0.45) + thermal only + adaptive threshold (best priors-free)
``fusion``        - Gamma + Laplacian-pyramid fusion + adaptive threshold
``grabcut_gtbox`` - GrabCut initialised with the ground-truth box (needs --gt)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from ipen.enhance import ENHANCERS                     # noqa: E402
from ipen.fuse import fuse                             # noqa: E402
from ipen.segment import SEGMENTERS, postprocess       # noqa: E402


def read(path: str) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def overlay(img: np.ndarray, mask: np.ndarray, color=(255, 40, 40)) -> np.ndarray:
    out = img.astype(np.float32).copy()
    m = mask.astype(bool)
    out[m] = 0.45 * out[m] + 0.55 * np.array(color, np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def classic(rgb, thermal, enhance="Gamma(0.45)", fusion="仅热红外", seg="自适应阈值"):
    t0 = time.perf_counter()
    img = fuse(fusion, rgb, thermal)
    if enhance != "无增强":
        img = ENHANCERS[enhance](img)
    mask = postprocess(SEGMENTERS[seg](img), max(80, int(0.0005 * rgb.shape[0] * rgb.shape[1])))
    return mask, time.perf_counter() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rgb", required=True)
    ap.add_argument("--thermal", required=True)
    ap.add_argument("--gt", default="", help="optional ground-truth mask for scoring")
    ap.add_argument("--method", default="best",
                    choices=["best", "fusion", "grabcut_gtbox"],
                    help="best = Gamma + thermal + adaptive; fusion = pyramid fusion; "
                         "grabcut_gtbox needs --gt")
    ap.add_argument("--out", default="demo_out.png")
    args = ap.parse_args()

    rgb, thermal = read(args.rgb), read(args.thermal)
    print(f"RGB {rgb.shape}  thermal {thermal.shape}")

    gt = None
    if args.gt:
        with Image.open(args.gt) as im:
            gt = np.asarray(im.convert("L")) >= 128

    panels, labels = [rgb, thermal], ["visible", "thermal"]
    if args.method == "grabcut_gtbox":
        if gt is None:
            raise SystemExit("--method grabcut_gtbox requires --gt")
        from ipen.baselines import grabcut_gtbox
        t0 = time.perf_counter()
        mask = grabcut_gtbox(fuse("拉普拉斯金字塔", rgb, thermal), gt)
        print(f"GT-box GrabCut: {(time.perf_counter()-t0)*1000:.0f} ms")
        panels.append(overlay(rgb, mask))
        labels.append("GT box + GrabCut")
    else:
        enh, fus = (("Gamma(0.45)", "仅热红外") if args.method == "best"
                    else ("Gamma(0.45)", "拉普拉斯金字塔"))
        t0 = time.perf_counter()
        mask, _ = classic(rgb, thermal, enh, fus)
        print(f"classical ({enh} + {fus}): {(time.perf_counter()-t0)*1000:.0f} ms")
        panels.append(overlay(rgb, mask))
        labels.append("classical")

    if gt is not None:
        from ipen.metrics import fbeta, iou
        print(f"IoU {iou(mask, gt):.4f}   Fbeta {fbeta(mask, gt):.4f}")

    Image.fromarray(np.concatenate(panels, axis=1)).save(args.out)
    print("written", args.out, labels)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
