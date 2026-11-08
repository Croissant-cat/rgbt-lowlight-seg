"""Figures for the low-light RGB-T segmentation report (classical only).

    python scripts/make_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ipen.baselines import grabcut_gtbox                      # noqa: E402
from ipen.data import load_frames                             # noqa: E402
from ipen.enhance import ENHANCERS                            # noqa: E402
from ipen.fuse import fuse                                    # noqa: E402
from ipen.metrics import iou                                   # noqa: E402
from ipen.segment import SEGMENTERS, postprocess               # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130
plt.rcParams["savefig.bbox"] = "tight"

RESULTS = ROOT / "results"
FIGS = RESULTS / "figs"
FIGS.mkdir(parents=True, exist_ok=True)
BEST = ("Gamma(0.45)", "仅热红外", "自适应阈值")
GRY, ACC, RED = "#7f8c8d", "#2980b9", "#c0392b"


def _panels(images, titles, path, cols=None, figsize=None, fontsize=9):
    n = len(images)
    cols = cols or n
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=figsize or (2.6 * cols, 2.4 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, im, t in zip(axes, images, titles):
        ax.imshow(im)
        ax.set_title(t, fontsize=fontsize)
    fig.tight_layout(pad=0.4)
    fig.savefig(path)
    plt.close(fig)


def overlay(img, mask, color=(255, 40, 40)):
    out = img.astype(np.float32).copy()
    m = mask.astype(bool)
    out[m] = 0.45 * out[m] + 0.55 * np.array(color, np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def fig_cases():
    frames = load_frames()
    picks = [frames[3], frames[0], frames[6]]
    rows = []
    for fr in picks:
        img = ENHANCERS[BEST[0]](fuse(BEST[1], fr.rgb, fr.thermal))
        trad = postprocess(SEGMENTERS[BEST[2]](img), 153)
        gc = grabcut_gtbox(fuse("拉普拉斯金字塔", fr.rgb, fr.thermal), fr.gt)
        ims = [fr.rgb, fr.thermal,
               np.repeat((fr.gt * 255)[..., None], 3, 2).astype(np.uint8),
               overlay(fr.rgb, trad), overlay(fr.rgb, gc)]
        ts = [f"可见光 {fr.sid}", "热红外", f"GT ({fr.gt.mean()*100:.1f}%)",
              f"常规组合 IoU={iou(trad, fr.gt):.3f}",
              f"GT框+GrabCut IoU={iou(gc, fr.gt):.3f}"]
        rows.append((ims, ts))
    ims = [im for r, _ in rows for im in r]
    titles = [t for _, ts in rows for t in ts]
    _panels(ims, titles, FIGS / "fig_cases.png", cols=5, figsize=(14, 6.4))


def fig_enhance():
    fr = load_frames()[3]
    names = ["无增强", "Gamma(0.45)", "CLAHE", "多尺度Retinex", "灰度世界白平衡"]
    ims = [ENHANCERS[n](fr.rgb) if n != "无增强" else fr.rgb for n in names]
    stats = [f"均值 {im.mean():.0f}" for im in ims]
    _panels(ims, [f"{n}\n{s}" for n, s in zip(names, stats)],
            FIGS / "fig_enhance.png", cols=5, figsize=(14, 2.9))


def fig_fusion_demo():
    fr = load_frames()[0]
    names = ["仅RGB", "仅热红外", "加权平均", "拉普拉斯金字塔"]
    ims = [fuse(n, fr.rgb, fr.thermal) for n in names]
    _panels(ims, names, FIGS / "fig_fusion.png", cols=4, figsize=(11, 2.6))


def fig_factors():
    g = pd.read_csv(RESULTS / "grid.csv")
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    seg = g.groupby("segmenter")["iou"].mean().sort_values()
    axes[0].barh(range(len(seg)), seg.values, color=GRY, edgecolor="black", lw=0.4)
    axes[0].set_yticks(range(len(seg)))
    axes[0].set_yticklabels(seg.index, fontsize=9)
    axes[0].set_title("不同分割算子（4 增强 × 4 融合平均）", fontsize=10)

    fus = g.groupby("fusion")["iou"].mean().sort_values()
    axes[1].barh(range(len(fus)), fus.values, color=ACC, edgecolor="black", lw=0.4)
    axes[1].set_yticks(range(len(fus)))
    axes[1].set_yticklabels(fus.index, fontsize=9)
    axes[1].set_title("不同模态/融合方式", fontsize=10)

    enh = g.groupby("enhance")["iou"].mean().sort_values()
    axes[2].barh(range(len(enh)), enh.values, color="#f39c12", edgecolor="black", lw=0.4)
    axes[2].set_yticks(range(len(enh)))
    axes[2].set_yticklabels(enh.index, fontsize=9)
    axes[2].set_title("不同低照度增强方法", fontsize=10)
    for ax in axes:
        ax.set_xlabel("IoU")
        ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_factors.png")
    plt.close(fig)


def fig_prior_ladder():
    """How far the classical side can be pushed by handing it more prior."""
    g = pd.read_csv(RESULTS / "grid.csv")
    o = pd.read_csv(RESULTS / "oracle.csv")
    st = pd.read_csv(RESULTS / "strong.csv")
    gmean = st.groupby("method")["iou"].mean()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 3.9))

    labels = ["无先验\n最佳组合", "面积先验\n(仅热红外)", "单阈值\n最优上界",
              "三算子\n投票集成", "GT框+\nGrabCut", "GT框+Gamma\n+GrabCut"]
    vals = [g.groupby(["enhance", "fusion", "segmenter"])["iou"].mean().max(),
            gmean.get("面积先验oracle(仅热红外)", np.nan),
            o.groupby("fusion")["oracle_iou"].mean().max(),
            gmean.get("投票集成(Otsu+自适应+顶帽)", np.nan),
            gmean.get("GT框+GrabCut", np.nan),
            gmean.get("GT框+GrabCut(Gamma增强)", np.nan)]
    cols = [GRY, "#95a5a6", "#95a5a6", "#95a5a6", "#e67e22", RED]
    axes[0].bar(range(len(vals)), vals, color=cols, edgecolor="black", lw=0.4)
    axes[0].set_xticks(range(len(vals)))
    axes[0].set_xticklabels(labels, fontsize=8)
    axes[0].set_ylabel("IoU")
    axes[0].set_ylim(0, 0.86)
    axes[0].set_title("先验阶梯：位置先验决定成败", fontsize=10)
    for i, v in enumerate(vals):
        axes[0].text(i, v + 0.012, f"{v:.3f}", ha="center", fontsize=8)
    axes[0].grid(axis="y", alpha=0.3)

    dg = pd.read_csv(RESULTS / "degrade.csv")
    for m, c, mk in [("仅热红外+自适应(最优)", GRY, "o"),
                     ("金字塔融合+自适应", ACC, "^"),
                     ("GT框+GrabCut", RED, "s")]:
        sub = dg[dg.method == m].groupby("level")["iou"].mean()
        order = [lv for lv in ["原始", "0.5x", "0.2x", "0.1x"] if lv in sub.index]
        axes[1].plot(order, [sub[l] for l in order], marker=mk, color=c, label=m)
    axes[1].set_xlabel("可见光亮度系数（1.0 = 原始）")
    axes[1].set_ylabel("IoU")
    axes[1].set_title("可见光被进一步压暗时的鲁棒性", fontsize=10)
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_prior_ladder.png")
    plt.close(fig)


def fig_perimage():
    g = pd.read_csv(RESULTS / "grid.csv")
    st = pd.read_csv(RESULTS / "strong.csv")
    best = g[(g.enhance == BEST[0]) & (g.fusion == BEST[1]) & (g.segmenter == BEST[2])]
    best = best.set_index("image")["iou"]
    gc = st[st.method == "GT框+GrabCut(Gamma增强)"].set_index("image")["iou"]
    ids = sorted(best.index, key=lambda k: best[k])
    x = np.arange(len(ids))
    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.bar(x - 0.2, [best[i] for i in ids], width=0.4, color=GRY, edgecolor="black",
           lw=0.4, label="常规组合（无先验）")
    ax.bar(x + 0.2, [gc[i] for i in ids], width=0.4, color=RED, edgecolor="black",
           lw=0.4, label="GT框+GrabCut+Gamma（最强先验）")
    ax.set_xticks(x)
    ax.set_xticklabels(ids, fontsize=8)
    ax.set_ylabel("IoU")
    ax.set_xlabel("帧编号（按无先验 IoU 升序）")
    ax.set_title("逐帧对比：给 GT 框 + Gamma 预增强后多数帧达到 0.75 以上", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGS / "fig_perimage.png")
    plt.close(fig)


def main() -> int:
    fig_cases()
    fig_enhance()
    fig_fusion_demo()
    fig_factors()
    fig_prior_ladder()
    fig_perimage()
    print("figures ->", FIGS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
