"""生成 lab06 讲义里的图。

    PYTHONPATH=src:. uv run --frozen python labs/lab06_mean_reversion_pairs/figures/make.py

数据设定与讲义正文的模拟完全相同（同样的 seed），图注里的数字以本脚本的输出为准。
风格（字号、字体、SVG 文字设置）见 quantlab.plotting。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from quantlab import data, pairs
from quantlab.plotting import COLORS, FULL_WIDTH, save_svg, use_lecture_style

OUT = Path(__file__).resolve().parent
DAYS = 252


def _slope_t(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    """OLS y = a + b x 的斜率 t 值与 R²。"""
    X = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    s2 = resid @ resid / (len(y) - 2)
    se = np.sqrt(s2 * np.linalg.inv(X.T @ X)[1, 1])
    r2 = 1 - resid @ resid / ((y - y.mean()) ** 2).sum()
    return float(beta[1] / se), float(r2)


def spurious_regression() -> None:
    """§2：两条独立随机游走的价格回归，斜率“显著”的比例远超 5%。"""
    rng = np.random.default_rng(0)
    t_level, t_ret, paths = [], [], []
    for _ in range(1000):
        y = np.cumsum(rng.normal(0, 0.015, 756))
        x = np.cumsum(rng.normal(0, 0.015, 756))
        t_level.append(_slope_t(y, x)[0])
        t_ret.append(_slope_t(np.diff(y), np.diff(x))[0])
        paths.append((x, y))
    t_level, t_ret = np.array(t_level), np.array(t_ret)
    # 展示一个“典型”的例子：斜率为正，|t| 最接近所有 |t| 的中位数
    med_abs_t = np.median(np.abs(t_level))
    candidates = np.flatnonzero(t_level > 0)
    pick = candidates[np.argmin(np.abs(t_level[candidates] - med_abs_t))]
    example = paths[pick]
    share_level = np.mean(np.abs(t_level) > 2)
    share_ret = np.mean(np.abs(t_ret) > 2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL_WIDTH, 3.3))
    x, y = example
    days = np.arange(len(x))
    ax1.plot(days, 100 * np.exp(x), color=COLORS[0], lw=1.3, label="stock X")
    ax1.plot(days, 100 * np.exp(y), color=COLORS[1], lw=1.3, label="stock Y")
    t_ex, r2_ex = _slope_t(y, x)
    ax1.set_title(f"Independent random walks (t = {t_ex:.0f})")
    ax1.set_xlabel("trading day")
    ax1.set_ylabel("price")
    ax1.legend(frameon=False, loc="upper right")

    bins = np.linspace(-40, 40, 81)
    ax2.hist(
        np.clip(t_level, -40, 40),
        bins=bins,
        color=COLORS[1],
        alpha=0.75,
        label=f"prices: {share_level:.0%}",
    )
    ax2.hist(t_ret, bins=bins, color=COLORS[0], alpha=0.85, label=f"returns: {share_ret:.0%}")
    for v in (-2, 2):
        ax2.axvline(v, color="black", lw=0.8, ls="--")
    ax2.set_title("Slope t-stat, 1000 independent pairs")
    ax2.set_xlabel("t-stat of the slope (clipped at ±40)")
    ax2.set_ylabel("count")
    ax2.legend(frameon=False, loc="upper left", title="share with |t| > 2")
    fig.tight_layout()
    save_svg(fig, OUT / "spurious_regression.svg")
    plt.close(fig)
    print(
        f"[spurious] |t|>2 prices {share_level:.0%}, returns {share_ret:.0%}; "
        f"median |t| {med_abs_t:.1f}; example pair t={t_ex:.1f}, R2={r2_ex:.2f}"
    )


def half_life_estimates() -> None:
    """§1：真实半衰期 15 天，1 年和 5 年数据的估计分布。"""
    est = {}
    for years in [1, 5]:
        vals = []
        for s in range(500):
            spread = data.cointegrated_pair(n_days=years * DAYS, half_life=15, seed=1000 + s)[
                "spread"
            ]
            vals.append(pairs.half_life(pairs.fit_ar1(spread)[1]))
        vals = np.array(vals)
        est[years] = vals[np.isfinite(vals)]

    fig, ax = plt.subplots(figsize=(FULL_WIDTH * 0.62, 3.2))
    bins = np.linspace(0, 45, 46)
    for color, years in zip(COLORS[1::-1], [1, 5], strict=True):
        med = np.median(est[years])
        ax.hist(
            np.clip(est[years], 0, 45),
            bins=bins,
            color=color,
            alpha=0.7,
            label=f"{years} year{'s' if years > 1 else ''} of data (median {med:.1f})",
        )
    top = ax.get_ylim()[1] * 1.3
    ax.set_ylim(0, top)
    ax.axvline(15, color="black", lw=1.2)
    ax.text(15.6, top * 0.97, "true: 15 days", va="top")
    ax.set_xlabel("estimated half-life (days, clipped at 45)")
    ax.set_ylabel("count (500 samples)")
    ax.set_title("Half-life estimates: biased low, and wide")
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.0, 0.82))
    fig.tight_layout()
    save_svg(fig, OUT / "half_life_estimates.svg")
    plt.close(fig)
    for years in [1, 5]:
        lo, med, hi = np.quantile(est[years], [0.05, 0.5, 0.95])
        print(f"[half-life] {years}y: median {med:.1f}, 90% [{lo:.1f}, {hi:.1f}]")


def zscore_trades() -> None:
    """§4：价差、滚动 z-score 与进出场。"""
    pair = data.cointegrated_pair(n_days=10 * DAYS, half_life=10, seed=5)
    ly, lx = np.log(pair["y"]), np.log(pair["x"])
    eg = pairs.engle_granger(ly, lx)
    z = pairs.rolling_zscore(eg["spread"], 60)
    pos = pairs.pair_positions(z)

    window = slice(DAYS, DAYS + 252)  # 展示一年
    zz, pp = z.iloc[window], pos.iloc[window]
    days = np.arange(len(zz))

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(FULL_WIDTH, 4.2), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]}
    )
    ax1.plot(days, zz.to_numpy(), color=COLORS[0], lw=1.2)
    for lvl, style in [(2, "--"), (-2, "--"), (0.5, ":"), (-0.5, ":")]:
        ax1.axhline(lvl, color="grey", lw=0.9, ls=style)
    ax1.fill_between(
        days, -4, 4, where=pp.to_numpy() < 0, color=COLORS[1], alpha=0.15, lw=0, step="post"
    )
    ax1.fill_between(
        days, -4, 4, where=pp.to_numpy() > 0, color=COLORS[2], alpha=0.15, lw=0, step="post"
    )
    ax1.set_ylim(-3.6, 3.6)
    ax1.set_xlim(0, len(days) - 1)
    ax1.set_ylabel("z-score of spread")
    ax1.set_title("Enter at |z| > 2, exit inside ±0.5 (one year shown)")
    for lvl, label in [(2, "entry +2"), (0.5, "exit +0.5"), (-0.5, "exit -0.5"), (-2, "entry -2")]:
        ax1.text(len(days) + 4, lvl, label, ha="left", va="center", color="grey", clip_on=False)

    ax2.step(days, pp.to_numpy(), where="post", color="black", lw=1.2)
    ax2.set_yticks([-1, 0, 1], ["short", "flat", "long"])
    ax2.set_ylim(-1.4, 1.4)
    ax2.set_ylabel("position")
    ax2.set_xlabel("trading day")
    fig.tight_layout()
    save_svg(fig, OUT / "zscore_trades.svg")
    plt.close(fig)
    n_trades = int(((pp != 0) & (pp.shift(fill_value=0) == 0)).sum())
    print(f"[zscore] trades opened in the shown year: {n_trades}")


if __name__ == "__main__":
    use_lecture_style()
    half_life_estimates()
    spurious_regression()
    zscore_trades()
