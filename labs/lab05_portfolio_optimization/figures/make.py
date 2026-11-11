"""生成 lab05 讲义里的图。

    PYTHONPATH=src:. uv run --frozen python labs/lab05_portfolio_optimization/figures/make.py

数据设定与讲义正文的模拟完全相同（同样的 seed），图注里的数字以本脚本的输出为准。
风格（字号、字体、SVG 文字设置）见 quantlab.plotting。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from quantlab import portfolio
from quantlab.plotting import COLORS, FULL_WIDTH, save_svg, use_lecture_style

OUT = Path(__file__).resolve().parent
DAYS = 252


def _unit_gross(x: np.ndarray) -> np.ndarray:
    return x / np.abs(x).sum()


def tangency_weights() -> None:
    """§2：mu 的估计噪声被优化器放大成极端的、带做空的权重。"""
    r = portfolio.simulate_assets(n_assets=10, n_days=5 * DAYS, seed=0)
    true_vol = r.std().to_numpy() * np.sqrt(DAYS)
    true_mu = 0.4 * true_vol
    cov = r.cov().to_numpy() * DAYS
    mu_hat = r.mean().to_numpy() * DAYS
    w_true = _unit_gross(np.linalg.solve(cov, true_mu))
    w_hat = _unit_gross(np.linalg.solve(cov, mu_hat))

    order = np.argsort(true_mu)
    x = np.arange(len(order))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL_WIDTH, 3.3))

    ax1.scatter(x, true_mu[order] * 100, color=COLORS[0], s=28, label=r"true $\mu$", zorder=3)
    ax1.scatter(
        x, mu_hat[order] * 100, color=COLORS[1], s=28, label=r"estimated $\hat\mu$", zorder=3
    )
    ax1.vlines(x, true_mu[order] * 100, mu_hat[order] * 100, color="grey", lw=0.8)
    ax1.axhline(0, color="grey", lw=0.6)
    ax1.set_xticks(x, [str(i + 1) for i in x])
    ax1.set_xlabel("asset (sorted by true $\\mu$)")
    ax1.set_ylabel("annual expected return (%)")
    ax1.set_title("Inputs: 5 years of data")
    ax1.legend(frameon=False, loc="upper left")

    width = 0.38
    ax2.bar(x - width / 2, w_true[order], width, color=COLORS[0], label="true tangency")
    ax2.bar(x + width / 2, w_hat[order], width, color=COLORS[1], label="estimated tangency")
    ax2.axhline(0, color="grey", lw=0.6)
    ax2.set_xticks(x, [str(i + 1) for i in x])
    ax2.set_xlabel("asset (sorted by true $\\mu$)")
    ax2.set_ylabel(r"weight ($\sum |w| = 1$)")
    ax2.set_title("Output: optimizer weights")
    ax2.legend(frameon=False, loc="lower right")

    fig.tight_layout()
    save_svg(fig, OUT / "tangency_weights.svg")
    plt.close(fig)
    print(
        f"[tangency] true mu range {true_mu.min():.1%}..{true_mu.max():.1%}; "
        f"mu_hat min {mu_hat.min():.1%}, negatives {(mu_hat < 0).sum()}; "
        f"true max|w| {np.abs(w_true).max():.2f}, shorts {(w_true < 0).sum()}; "
        f"est max|w| {np.abs(w_hat).max():.2f}, shorts {(w_hat < 0).sum()}"
    )


def ex_ante_ex_post() -> None:
    """§2：N/T 越大，最小方差组合报告的风险越被低估；收缩缓解这一点。"""
    r = portfolio.simulate_assets(n_assets=50, n_days=1000 + DAYS, seed=1)
    test = r.iloc[-DAYS:]
    ts = [60, 100, 150, 250, 500, 1000]
    ratios = {"sample": [], "ledoit_wolf": []}
    for t in ts:
        train = r.iloc[-DAYS - t : -DAYS]
        for est in ratios:
            pred, real = portfolio.oos_min_variance(train, test, estimator=est)
            ratios[est].append(real / pred)

    fig, ax = plt.subplots(figsize=(FULL_WIDTH * 0.62, 3.2))
    labels = {"sample": "sample covariance", "ledoit_wolf": "Ledoit-Wolf shrinkage"}
    for color, est in zip(COLORS[:2], ratios, strict=True):
        ax.plot(ts, ratios[est], marker="o", ms=4, color=color, label=labels[est])
    ax.axhline(1, color="grey", lw=0.8, ls="--")
    ax.text(1000, 0.93, "honest: realized = predicted", ha="right", va="top", color="grey")
    ax.set_ylim(0.5, None)
    ax.set_xscale("log")
    ax.set_xticks(ts, [str(t) for t in ts])
    ax.minorticks_off()
    ax.set_xlabel("estimation window T (days), N = 50 assets")
    ax.set_ylabel("realized vol / predicted vol")
    ax.set_title("Min-variance: how much the optimizer understates risk")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    save_svg(fig, OUT / "ex_ante_ex_post.svg")
    plt.close(fig)
    print(
        "[ex-ante/ex-post] "
        + "; ".join(
            f"T={t}: sample {s:.2f}, LW {lw:.2f}"
            for t, s, lw in zip(ts, ratios["sample"], ratios["ledoit_wolf"], strict=True)
        )
    )


def diversification() -> None:
    """§4：等权组合的波动随相关系数上升，分散化的下限是 sigma * sqrt(rho)。"""
    sigma = 0.20
    rho = np.linspace(0, 1, 101)
    fig, ax = plt.subplots(figsize=(FULL_WIDTH * 0.62, 3.2))
    for color, n in zip(COLORS[:3], [5, 10, 100], strict=True):
        ax.plot(rho, 100 * sigma * np.sqrt(rho + (1 - rho) / n), color=color, label=f"N = {n}")

    def ew_vol(r: float, n: int = 10) -> float:
        return sigma * np.sqrt(r + (1 - r) / n)

    lo, hi = ew_vol(0.2), ew_vol(0.8)
    ax.scatter([0.2, 0.8], [100 * lo, 100 * hi], color="black", s=22, zorder=4)
    ax.annotate(
        "",
        xy=(0.8, 100 * hi),
        xytext=(0.2, 100 * lo),
        arrowprops={"arrowstyle": "->", "color": "black", "lw": 1.0},
    )
    ax.text(0.24, 100 * lo - 0.8, f"calm: {lo:.1%}", ha="left", va="top")
    ax.text(0.77, 100 * hi + 0.9, f"crisis: {hi:.1%}", ha="right", va="center")
    ax.axhline(100 * sigma, color="grey", lw=0.6, ls=":")
    ax.text(0.01, 100 * sigma + 0.4, "single asset (20%)", color="grey", va="bottom")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 23)
    ax.set_xlabel(r"pairwise correlation $\rho$")
    ax.set_ylabel("equal-weight portfolio vol (%)")
    ax.set_title("Diversification fades as correlation rises")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    save_svg(fig, OUT / "diversification.svg")
    plt.close(fig)
    print(f"[diversification] N=10: rho 0.2 -> {lo:.1%}, rho 0.8 -> {hi:.1%}")


if __name__ == "__main__":
    use_lecture_style()
    tangency_weights()
    ex_ante_ex_post()
    diversification()
