# 低照度 RGB-T 物品分割：传统图像处理的能力边界

> 图像处理课程大作业 · 李嘉怡 · 人工智能 2402 · 202456435

在几乎无光的室内场景里，可见光图像几乎没有信息，目标只占画面的 0.3%~6%。本仓库把
"增强 → 融合 → 分割 → 形态学后处理"这条经典流水线拆开逐环量化：**6 种低照度增强 × 4 种多模态融合 ×
6 种分割算子**，共 1152 次评测，再用**先验阶梯**与**最优阈值上界**两种工具界定它的能力边界。

全部算子与指标均由本仓库自行实现（NumPy / SciPy / OpenCV），**不使用任何预训练模型或深度学习组件**。

![逐帧对比](results/figs/fig_perimage.png)

## 主要结果

| 条件 | 额外先验 | IoU |
|:--|:--|--:|
| 无先验最佳组合（Gamma + 仅热红外 + 自适应阈值） | — | 0.247 |
| 面积先验 oracle（知道目标多大） | 面积 | 0.212 |
| 单阈值最优上界（100 档阈值取最优） | 阈值 | 0.236 |
| 三算子投票集成 | — | 0.223 |
| 真值框 + GrabCut | 位置 | 0.463 |
| **真值框 + Gamma 预增强 + GrabCut** | 位置 + 预处理 | **0.744** |

**三个结论**

1. **真正稀缺的是"位置"。** 知道目标多大帮不上忙（0.212 反而更低），把阈值调到最优也只有 0.236；
   但只要给出真值框，分数立刻接近翻倍——GrabCut 可以把颜色模型限制在框内。
2. **"增强有没有用"取决于下游。** 对 Otsu 这类全局阈值，增强改变直方图形状 → 有害
   （无增强 0.064 → Retinex 0.035）；对 GrabCut 这类基于 GMM 的方法，Gamma 拉开框内灰度差 →
   极大有益（0.463 → 0.744）。同一算子，下游不同，效果相反。
3. **热红外是主要信息源，可见光在此场景下是干扰。** 仅热红外 IoU 0.074 vs 仅 RGB 0.019（4 倍）；
   把可见光压暗到 0.1 倍后，依赖 RGB 的融合方法反而从 0.162 升到 0.239——融合会自动倒向热红外。

## 方法要点

| 环节 | 实现的算子 |
|:--|:--|
| 低照度增强 | Gamma(0.45) / 全局直方图均衡 / CLAHE / 多尺度 Retinex / 灰度世界白平衡 |
| 多模态融合 | 仅RGB / 仅热红外 / 加权平均 / PCA / 拉普拉斯金字塔 / 取最大值 |
| 分割 | Otsu / 三角形阈值 / 自适应阈值 / K-means / 区域生长 / GrabCut |
| 小目标专用 | 形态学顶帽（I − opening）+ 连通域面积过滤 |
| 上界分析 | 面积先验 oracle、单阈值最优上界、真值框 GrabCut |

## 目录结构

```
rgbt-lowlight-seg/
├── demo.py                    # 单张 RGB-T 的分割演示
├── src/ipen/                  # 核心算法（全部自研）
│   ├── data.py                # 12 组 RGB-T 帧加载
│   ├── enhance.py             # Gamma / 直方图均衡 / CLAHE / MSR / 灰度世界
│   ├── fuse.py                # 加权 / PCA / 拉普拉斯金字塔 / 取最大 + 灰度化
│   ├── segment.py             # Otsu / 三角形 / 自适应 / K-means / 区域生长 / GrabCut / 顶帽
│   ├── baselines.py           # 面积先验 oracle、真值框 GrabCut、投票集成
│   └── metrics.py             # IoU / Fβ(β²=0.3) / MAE
├── scripts/
│   ├── run_experiments.py     # 主网格 + 先验阶梯 + 最优阈值上界 + 压暗实验
│   ├── make_figures.py        # 全部插图
│   └── build_report.py        # 由结果生成课程报告
├── data/cases/                # 12 组 RGB / 热红外 / GT
├── results/                   # grid/oracle/strong/degrade 四个 CSV + 插图
└── tests/test_core.py         # 8 项单元测试
```

## 快速开始

```bash
pip install -r requirements.txt

# 无先验最佳组合
python demo.py --rgb data/cases/1001/RGB.png --thermal data/cases/1001/热红外.png

# 给真值框的上界参照（需要 GT）
python demo.py --rgb data/cases/1001/RGB.png --thermal data/cases/1001/热红外.png \
               --gt data/cases/1001/真实掩码GT.png --method grabcut_gtbox

# 复现全部实验（CPU，约 4 分钟）
python scripts/run_experiments.py
python scripts/make_figures.py
python scripts/build_report.py
```

`run_experiments.py` 写出四个 CSV：`grid.csv`（1152 行）、`oracle.csv`（最优阈值上界）、
`strong.csv`（先验阶梯）、`degrade.csv`（压暗实验）。

## 测试

```bash
python tests/test_core.py     # 或 pytest tests
```

## 数据说明

`data/cases/` 的 12 组帧（可见光 / 热红外 / 真值掩码）取自公开的多模态分割基准 **VDT-2048**，
仅用于课程实验评测。仓库不包含任何预训练权重或第三方模型代码。

## License

MIT，见 [LICENSE](LICENSE)。图像版权归原基准作者所有。
