"""生成 lab04 讲义里的图。

    PYTHONPATH=src:. uv run --frozen python labs/lab04_factor_investing/figures/make.py

图中文字用英文（容器里没有中文字体），中文说明写在讲义的图注里。
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from quantlab import factors  # noqa: E402

OUT = Path(__file__).resolve().parent
COLORS = ["#2a6fb0", "#d1732b", "#3a9a5b"]


def neutralization() -> None:
    """横截面中性化：斜率 gamma、行业截距 delta_j、残差 u 在一张图上的样子。"""
    rng = np.random.default_rng(7)
    n, sectors = 60, ["Bank", "Tech", "Energy"]
    names = [f"S{i:02d}" for i in range(n)]
    size = pd.Series(rng.normal(0, 1, n), index=names)
    sector = pd.Series(rng.choice(sectors, n), index=names)
    level = {"Bank": -0.8, "Tech": 0.9, "Energy": 0.0}

    # 原始信号 = 小盘暴露（规模越小分数越高） + 行业水平 + 真正有用的部分
    raw = -0.7 * size + sector.map(level) + rng.normal(0, 0.45, n)
    resid = factors.neutralize(raw.to_frame().T, size, sector).iloc[0]

    # 与 neutralize 相同的回归，取出系数用来画线：X = [size, 行业哑变量]
    dummies = pd.get_dummies(sector).astype(float)
    X = np.column_stack([size.to_numpy(), dummies.to_numpy()])
    coef = np.linalg.lstsq(X, raw.to_numpy(), rcond=None)[0]
    gamma, deltas = coef[0], dict(zip(dummies.columns, coef[1:], strict=True))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    xs = np.linspace(size.min() - 0.2, size.max() + 0.2, 2)
    for color, sec in zip(COLORS, sectors, strict=True):
        m = sector == sec
        ax1.scatter(size[m], raw[m], s=18, color=color, alpha=0.8, label=sec)
        ax1.plot(xs, deltas[sec] + gamma * xs, color=color, lw=1.6)
        ax1.annotate(
            rf"$\delta_{{{sec}}}$",
            xy=(0, deltas[sec]),
            xytext=(-2.9, deltas[sec]),
            color=color,
            va="center",
            fontsize=11,
            arrowprops={"arrowstyle": "-", "color": color, "lw": 0.6, "ls": ":"},
        )
        ax2.scatter(size[m], resid[m], s=18, color=color, alpha=0.8, label=sec)

    # 挑一只残差较大的股票，把 u 画成竖线
    pick = resid.abs().idxmax()
    fitted = raw[pick] - resid[pick]
    ax1.vlines(size[pick], fitted, raw[pick], color="black", lw=1.4)
    ax1.annotate(
        r"$u_{i,t}$",
        xy=(size[pick], (fitted + raw[pick]) / 2),
        xytext=(8, 0),
        textcoords="offset points",
        va="center",
        fontsize=12,
    )
    ax2.vlines(size[pick], 0, resid[pick], color="black", lw=1.4)
    ax2.annotate(
        r"$u_{i,t}$",
        xy=(size[pick], resid[pick] / 2),
        xytext=(8, 0),
        textcoords="offset points",
        va="center",
        fontsize=12,
    )

    ax1.axvline(0, color="grey", lw=0.5)
    ax1.text(
        0.97,
        0.95,
        rf"slope $\gamma_t$ = {gamma:.2f}",
        transform=ax1.transAxes,
        ha="right",
        va="top",
        fontsize=11,
    )
    ax1.set_xlim(-3.2, size.max() + 0.3)
    ax1.set_title(r"Raw signal $s_{i,t}$ on one day $t$")
    ax1.set_xlabel(r"size exposure $\mathrm{size}_i$ (small $\leftarrow$ $\rightarrow$ large)")
    ax1.set_ylabel("signal")
    ax1.legend(title="industry $j$", loc="lower left", fontsize=9)

    ax2.axhline(0, color="grey", lw=0.8)
    ax2.set_title(r"Neutralized signal: residual $u_{i,t}$")
    ax2.set_xlabel(r"size exposure $\mathrm{size}_i$")
    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT / "neutralization.svg")
    if preview := os.environ.get("FIG_PREVIEW_DIR"):  # 只用于查看效果，不入库
        fig.savefig(Path(preview) / "neutralization.png", dpi=110)
    plt.close(fig)

    # 供讲义图注核对：残差与 size 的相关、各行业残差均值都应为 0
    print(f"gamma = {gamma:.3f}; deltas = { {k: round(v, 3) for k, v in deltas.items()} }")
    print(f"corr(raw, size) = {np.corrcoef(raw, size)[0, 1]:.3f}")
    print(f"corr(resid, size) = {np.corrcoef(resid, size)[0, 1]:.2e}")
    print("resid mean by industry:", resid.groupby(sector).mean().round(12).to_dict())


if __name__ == "__main__":
    neutralization()
