"""生成 lab00 讲义里的图。

    PYTHONPATH=src:. uv run --frozen python labs/lab00_returns_basics/figures/make.py

风格（字号、字体、SVG 文字设置）见 quantlab.plotting。随机数种子与讲义里引用数字时的模拟相同，
所以图上的终点数值和讲义正文一致。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter

from quantlab import returns
from quantlab.plotting import COLORS, FULL_WIDTH, save_svg, use_lecture_style

OUT = Path(__file__).resolve().parent
DAYS = 252


def leveraged_etf() -> None:
    """§2：零漂移、年化波动 30% 的标的与它的 3 倍日再平衡 ETF，5 年里中位数与均值的走势。"""
    n_paths, years = 20_000, 5
    rng = np.random.default_rng(0)
    rng.normal(0.0, 0.30 / np.sqrt(DAYS), (DAYS, n_paths))  # 讲义里先模拟了 1 年，跳过这一段随机数
    n = years * DAYS
    under = np.ones(n_paths)
    lev3 = np.ones(n_paths)
    t_axis, med_u, med_3, mean_3 = [0.0], [1.0], [1.0], [1.0]
    # 按行分块生成：与一次性生成 (n, n_paths) 的随机数序列完全相同，但不用占用几百 MB 内存
    for day in range(1, n + 1):
        r = rng.normal(0.0, 0.30 / np.sqrt(DAYS), n_paths)
        under *= 1 + r
        lev3 *= 1 + 3 * r
        if day % 5 == 0:
            t_axis.append(day / DAYS)
            med_u.append(np.median(under))
            med_3.append(np.median(lev3))
            mean_3.append(lev3.mean())

    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 3.3))
    ax.plot(t_axis, med_u, color=COLORS[0], label="underlying: median")
    ax.plot(t_axis, mean_3, color=COLORS[1], ls="--", label="3x ETF: mean")
    ax.plot(t_axis, med_3, color=COLORS[1], label="3x ETF: median")
    ax.axhline(1.0, color="grey", lw=0.8)
    ax.set_yscale("log")
    ax.set_yticks([0.1, 0.2, 0.5, 1])
    ax.set_yticklabels(["0.1", "0.2", "0.5", "1"])
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(0.1, 1.4)
    ax.set_xlabel("years held")
    ax.set_ylabel("wealth (log scale)")
    ax.set_title("Zero-drift index, 30% vol: 3x daily-rebalanced ETF")
    for y, txt, c in [
        (med_u[-1], f"{med_u[-1]:.3f}", COLORS[0]),
        (mean_3[-1], f"{mean_3[-1]:.3f}", COLORS[1]),
        (med_3[-1], f"{med_3[-1]:.3f}", COLORS[1]),
    ]:
        ax.annotate(
            txt, xy=(t_axis[-1], y), xytext=(4, 0), textcoords="offset points", va="center", color=c
        )
    ax.set_xlim(0, years + 0.45)
    ax.legend(loc="lower left", frameon=False)
    fig.tight_layout()
    save_svg(fig, OUT / "leveraged_etf.svg")
    plt.close(fig)
    print(
        f"[leveraged_etf] 5y: underlying median {med_u[-1]:.3f}; "
        f"3x median {med_3[-1]:.3f}, mean {mean_3[-1]:.3f}, loss share {(lev3 < 1).mean():.0%}"
    )


def mean_vs_median() -> None:
    """§3：mu=8%、sigma=40% 的 GBM 持有 20 年，均值、中位数与 5%–95% 区间。"""
    mu, sigma, years, n_paths = 0.08, 0.40, 20, 20_000
    rng = np.random.default_rng(1)  # 与 returns.simulate_gbm(..., seed=1) 的随机数序列相同
    dt = 1.0 / DAYS
    log_w = np.zeros(n_paths)
    t_axis, mean, med, lo, hi = [0.0], [1.0], [1.0], [1.0], [1.0]
    for day in range(1, years * DAYS + 1):
        z = rng.standard_normal(n_paths)
        log_w += (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
        if day % 21 == 0:
            w = np.exp(log_w)
            t_axis.append(day / DAYS)
            mean.append(w.mean())
            med.append(np.median(w))
            q = np.quantile(w, [0.05, 0.95])
            lo.append(q[0])
            hi.append(q[1])
    w = np.exp(log_w)

    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 3.4))
    ax.fill_between(t_axis, lo, hi, color=COLORS[0], alpha=0.15, lw=0, label="5%–95% of paths")
    ax.plot(t_axis, mean, color=COLORS[1], label="mean")
    ax.plot(t_axis, med, color=COLORS[0], label="median")
    ax.plot(
        t_axis,
        np.exp(mu * np.array(t_axis)),
        color=COLORS[1],
        ls=":",
        lw=1.2,
        label=r"theory: $e^{\mu T}$",
    )
    ax.set_yscale("log")
    ax.set_yticks([0.1, 0.3, 1, 3, 10, 30])
    ax.set_yticklabels(["0.1", "0.3", "1", "3", "10", "30"])
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("years held")
    ax.set_ylabel("wealth (log scale)")
    ax.set_title(r"GBM with $\mu$ = 8%, $\sigma$ = 40%: the mean grows, the median does not")
    ax.annotate(
        f"mean {w.mean():.2f}",
        xy=(years, w.mean()),
        xytext=(4, 0),
        textcoords="offset points",
        va="center",
        color=COLORS[1],
    )
    ax.annotate(
        f"median {np.median(w):.2f}",
        xy=(years, np.median(w)),
        xytext=(4, 0),
        textcoords="offset points",
        va="center",
        color=COLORS[0],
    )
    ax.set_xlim(0, years + 3.2)
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    save_svg(fig, OUT / "mean_vs_median.svg")
    plt.close(fig)
    top = np.sort(w)[-n_paths // 100 :].sum() / w.sum()
    print(
        f"[mean_vs_median] 20y: mean {w.mean():.2f}, median {np.median(w):.2f}, "
        f"loss share {(w < 1).mean():.0%}, top-1% share {top:.0%}, "
        f"5%/95% {np.quantile(w, 0.05):.2f}/{np.quantile(w, 0.95):.2f}"
    )


def growth_vs_leverage() -> None:
    """§3：固定杠杆的长期增长率 g(L) = L mu - L^2 sigma^2 / 2，mu=8%、sigma=20%。"""
    mu, sigma = 0.08, 0.20
    levs = np.linspace(0, 5, 201)
    g = np.array([returns.growth_rate(mu, sigma, L) for L in levs])
    l_star, l_zero = mu / sigma**2, 2 * mu / sigma**2

    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 3.0))
    ax.plot(levs, 100 * g, color=COLORS[0])
    ax.axhline(0, color="grey", lw=0.8)
    ax.plot([l_star], [100 * returns.growth_rate(mu, sigma, l_star)], "o", color=COLORS[1])
    peak = 100 * returns.growth_rate(mu, sigma, l_star)
    arrow = {"arrowstyle": "-", "lw": 0.8}
    # 两个标注都放在空白区域：弧线下方、以及 x>4 时零线上方
    ax.annotate(
        rf"Kelly $L^*=\mu/\sigma^2$ = {l_star:.0f}" + f"\ngrowth {peak:.0f}%/yr",
        xy=(l_star, peak),
        xytext=(l_star, 2.5),
        ha="center",
        va="center",
        color=COLORS[1],
        arrowprops={**arrow, "color": COLORS[1]},
    )
    ax.plot([l_zero], [0], "o", color=COLORS[4])
    ax.annotate(
        rf"$L = 2\mu/\sigma^2$ = {l_zero:.0f}:" + "\ngrowth back to 0",
        xy=(l_zero, 0),
        xytext=(4.2, 4.5),
        ha="left",
        va="center",
        color=COLORS[4],
        arrowprops={**arrow, "color": COLORS[4]},
    )
    ax.set_xlabel("leverage $L$ (risky asset / wealth, rebalanced daily)")
    ax.set_ylabel("long-run growth, %/yr")
    ax.set_title(r"Growth rate vs leverage: $\mu$ = 8%, $\sigma$ = 20%")
    fig.tight_layout()
    save_svg(fig, OUT / "growth_vs_leverage.svg")
    plt.close(fig)
    print(
        f"[growth_vs_leverage] L*={l_star:.1f}, g(L*)={peak / 100:.3f}, "
        f"zero at L={l_zero:.1f}, g(5)={returns.growth_rate(mu, sigma, 5):.3f}"
    )


if __name__ == "__main__":
    use_lecture_style()
    leveraged_etf()
    mean_vs_median()
    growth_vs_leverage()
