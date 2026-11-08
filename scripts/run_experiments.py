"""All experiments of the low-light RGB-T segmentation study (classical only).

Four blocks, each written to its own CSV under ``results/``:

1. ``grid.csv``    enhancement x fusion x segmenter, 12 frames each
2. ``oracle.csv``  best single-threshold IoU - the ceiling of any thresholding
                   method, so "is it tuning or information?"
3. ``strong.csv``  prior-assisted classical baselines (ground-truth box for
                   GrabCut, area-prior threshold, three-way vote ensemble)
4. ``degrade.csv`` robustness when the visible image is artificially darkened

Usage:  python scripts/run_experiments.py
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ipen.baselines import area_oracle, grabcut_gtbox, vote_ensemble   # noqa: E402
from ipen.data import load_frames                                     # noqa: E402
from ipen.enhance import ENHANCERS                                    # noqa: E402
from ipen.fuse import fuse                                            # noqa: E402
from ipen.metrics import evaluate, iou                                # noqa: E402
from ipen.segment import SEGMENTERS, postprocess, tophat              # noqa: E402

RESULTS = ROOT / "results"
ENH = ["无增强", "Gamma(0.45)", "CLAHE", "多尺度Retinex"]
FUS = ["仅RGB", "仅热红外", "加权平均", "拉普拉斯金字塔"]
SEG = ["Otsu", "三角形阈值", "自适应阈值", "K-means", "区域生长", "GrabCut"]
BEST = ("Gamma(0.45)", "仅热红外", "自适应阈值")


def _area_min(frame) -> int:
    return max(80, int(0.0005 * frame.gt.size))


def run_grid(frames):
    """enhancement x fusion x segmenter; the enhanced image is built once."""
    rows = []
    for enh, fus in itertools.product(ENH, FUS):
        for fr in frames:
            img = fuse(fus, fr.rgb, fr.thermal)
            if enh != "无增强":
                img = ENHANCERS[enh](img)
            amin = _area_min(fr)
            for seg in SEG:
                mask = postprocess(SEGMENTERS[seg](img), amin)
                rows.append({"enhance": enh, "fusion": fus, "segmenter": seg,
                             "image": fr.sid,
                             **evaluate(mask.astype(np.float32), fr.gt)})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "grid.csv", index=False)
    print(f"[grid] {len(df)} rows")
    return df


def run_oracle(frames):
    """Best single threshold on the top-hat response - the method's own ceiling."""
    rows = []
    levels = np.linspace(0.80, 0.9995, 100)
    for fus in FUS:
        for fr in frames:
            resp = tophat(fuse(fus, fr.rgb, fr.thermal), size=21)
            best, best_q = 0.0, 0.0
            for q in levels:
                m = resp > np.quantile(resp, q)
                v = iou(m, fr.gt)
                if v > best:
                    best, best_q = v, q
            rows.append({"fusion": fus, "image": fr.sid, "oracle_iou": best,
                         "quantile": best_q, "gt_area": float(fr.gt.mean())})
        sub = [r["oracle_iou"] for r in rows if r["fusion"] == fus]
        print(f"[oracle] {fus:8s} mean best-IoU {np.mean(sub):.3f}")
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "oracle.csv", index=False)
    return df


def run_strong(frames):
    """Prior-assisted baselines: how far can classical methods be pushed?"""
    rows = []
    for fr in frames:
        img_t = fuse("仅热红外", fr.rgb, fr.thermal)
        img_f = fuse("拉普拉斯金字塔", fr.rgb, fr.thermal)
        img_g = ENHANCERS["Gamma(0.45)"](img_t)
        cand = {
            "投票集成(Otsu+自适应+顶帽)": postprocess(vote_ensemble(img_f), _area_min(fr)),
            "面积先验oracle(仅热红外)": area_oracle(img_t, fr.gt),
            "面积先验oracle(金字塔)": area_oracle(img_f, fr.gt),
            "GT框+GrabCut": grabcut_gtbox(img_f, fr.gt),
            "GT框+GrabCut(Gamma增强)": grabcut_gtbox(img_g, fr.gt),
        }
        for name, mask in cand.items():
            sc = evaluate(mask.astype(np.float32), fr.gt)
            rows.append({"method": name, "image": fr.sid, "prior": True, **sc})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "strong.csv", index=False)
    for m, sub in df.groupby("method"):
        print(f"[strong] {m:26s} IoU={sub.iou.mean():.3f} Fbeta={sub.fbeta.mean():.3f}")
    return df


def run_degrade(frames):
    """Darken the visible image and see how the classical pipeline reacts."""
    rows = []
    for level, tag in [(1.0, "原始"), (0.5, "0.5x"), (0.2, "0.2x"), (0.1, "0.1x")]:
        for fr in frames:
            rgb_d = np.clip(fr.rgb.astype(np.float32) * level, 0, 255).astype(np.uint8)
            # each method keeps its own best configuration from the grid
            img1 = fuse(BEST[1], rgb_d, fr.thermal)
            m1 = postprocess(SEGMENTERS[BEST[2]](ENHANCERS[BEST[0]](img1)), _area_min(fr))
            img2 = fuse("拉普拉斯金字塔", rgb_d, fr.thermal)
            m2 = grabcut_gtbox(img2, fr.gt)
            m3 = postprocess(SEGMENTERS[BEST[2]](ENHANCERS["Gamma(0.45)"](img2)), _area_min(fr))
            rows.append({"method": "仅热红外+自适应(最优)", "level": tag, "image": fr.sid,
                         **evaluate(m1.astype(np.float32), fr.gt)})
            rows.append({"method": "金字塔融合+自适应", "level": tag, "image": fr.sid,
                         **evaluate(m3.astype(np.float32), fr.gt)})
            rows.append({"method": "GT框+GrabCut", "level": tag, "image": fr.sid,
                         **evaluate(m2.astype(np.float32), fr.gt)})
        for m in ("仅热红外+自适应(最优)", "金字塔融合+自适应", "GT框+GrabCut"):
            sub = [r["iou"] for r in rows if r["level"] == tag and r["method"] == m]
            print(f"[degrade] {tag:5s} {m:12s} IoU {np.mean(sub):.3f}")
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "degrade.csv", index=False)
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    frames = load_frames()
    print(f"{len(frames)} frames, GT area "
          f"{min(f.gt.mean() for f in frames)*100:.2f}% - "
          f"{max(f.gt.mean() for f in frames)*100:.2f}%")

    run_grid(frames)
    run_oracle(frames)
    run_strong(frames)
    run_degrade(frames)
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
