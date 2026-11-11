"""生成 lab07 讲义里的图。

    PYTHONPATH=src:. uv run --frozen python labs/lab07_volatility_garch/figures/make.py

数据设定与讲义正文的模拟完全相同（同样的 seed），图注里的数字以本脚本的输出为准。
风格（字号、字体、SVG 文字设置）见 quantlab.plotting。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from quantlab import data, vol
from quantlab.plotting import COLORS, FULL_WIDTH, save_svg, use_lecture_style

OUT = Path(__file__).resolve().parent
DAYS = 252


def _acf(x: np.ndarray, lags: int) -> np.ndarray:
    x = x - x.mean()
    denom = x @ x
    return np.array([(x[:-k] @ x[k:]) / denom for k in range(1, lags + 1)])


def clustering() -> None:
    """§1：收益本身几乎没有自相关，平方收益的自相关显著为正且衰减很慢。"""
    g = data.garch_returns(5000, seed=0)
    r = g["ret"].to_numpy()
    lags = 40
    acf_r, acf_r2 = _acf(r, lags), _acf(r**2, lags)
    band = 1.96 / np.sqrt(len(r))

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(FULL_WIDTH, 3.3), gridspec_kw={"width_ratios": [1.35, 1]}
    )
    shown = slice(0, 1500)
    days = np.arange(len(r))[shown]
    ax1.plot(days, 100 * r[shown], color=COLORS[0], lw=0.6)
    sig = 100 * g["sigma"].to_numpy()[shown]
    ax1.plot(days, 2 * sig, color=COLORS[1], lw=1.2, label=r"$\pm 2\sigma_t$ (true)")
    ax1.plot(days, -2 * sig, color=COLORS[1], lw=1.2)
    ax1.set_xlabel("trading day")
    ax1.set_ylabel("daily return (%)")
    ax1.set_title("Calm and turbulent stretches")
    ax1.set_ylim(-5, 5.8)
    ax1.legend(frameon=False, loc="upper right")

    k = np.arange(1, lags + 1)
    ax2.bar(k - 0.2, acf_r, width=0.4, color=COLORS[0], label="returns $r_t$")
    ax2.bar(k + 0.2, acf_r2, width=0.4, color=COLORS[1], label="squared returns $r_t^2$")
    ax2.axhspan(-band, band, color="grey", alpha=0.2, lw=0)
    ax2.axhline(0, color="grey", lw=0.6)
    ax2.set_xlabel("lag (days)")
    ax2.set_ylabel("autocorrelation")
    ax2.set_title("Autocorrelation")
    ax2.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    save_svg(fig, OUT / "clustering.svg")
    plt.close(fig)
    print(
        f"[clustering] acf r lag1 {acf_r[0]:+.3f}, max|acf r| {np.abs(acf_r).max():.3f}; "
        f"acf r2 lag1 {acf_r2[0]:.3f}, lag20 {acf_r2[19]:.3f}, lag40 {acf_r2[39]:.3f}; "
        f"95% band ±{band:.3f}"
    )


def vol_target() -> None:
    """§3：波动率目标把组合的滚动波动压平。"""
    g = data.garch_returns(5000, seed=0)
    r = g["ret"]
    vt = vol.vol_target_returns(r, target_vol=0.10, max_leverage=2.0)

    def rolling(x: pd.Series) -> pd.Series:
        return x.rolling(63).std() * np.sqrt(DAYS)

    def vol_of_vol(x: pd.Series) -> float:
        return float(rolling(x).std())

    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 2.9))
    days = np.arange(len(r))
    ax.plot(days, 100 * rolling(r).to_numpy(), color=COLORS[0], lw=1.0, label="buy and hold")
    ax.plot(days, 100 * rolling(vt).to_numpy(), color=COLORS[1], lw=1.0, label="vol target 10%")
    ax.axhline(10, color="black", lw=0.8, ls="--")
    ax.set_xlim(0, len(r) - 1)
    ax.set_xlabel("trading day")
    ax.set_ylabel("63-day realized vol (%)")
    ax.set_title("Volatility targeting stabilizes risk")
    ax.legend(frameon=False, loc="upper right", ncol=2)
    ax.set_ylim(0, 37)
    fig.tight_layout()
    save_svg(fig, OUT / "vol_target.svg")
    plt.close(fig)
    print(f"[vol target] vol-of-vol {vol_of_vol(r):.3f} -> {vol_of_vol(vt):.3f}")


def kelly() -> None:
    """§4：增长率是杠杆的抛物线；用估计参数下注时，完全 Kelly 经常为负，半 Kelly 稳得多。"""
    mu_ex, sigma = 0.06, 0.20
    fs = np.linspace(0, 4.2, 200)

    def growth(f):
        return f * mu_ex - 0.5 * f**2 * sigma**2

    rng = np.random.default_rng(2)
    res = {1.0: [], 0.5: []}
    for _ in range(2000):
        x = rng.normal(mu_ex / DAYS, sigma / np.sqrt(DAYS), 5 * DAYS)
        f_hat = (x.mean() * DAYS) / (x.var(ddof=1) * DAYS)
        for frac in res:
            f = float(np.clip(frac * f_hat, 0, 5))
            res[frac].append(growth(f))
    res = {k: np.array(v) for k, v in res.items()}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL_WIDTH, 3.3))
    ax1.plot(fs, 100 * growth(fs), color=COLORS[0])
    ax1.axhline(0, color="grey", lw=0.6)
    kelly_f = mu_ex / sigma**2
    for f, label, offset, ha in [
        (kelly_f / 2, "half Kelly", (-4, 8), "right"),
        (kelly_f, "Kelly", (0, 8), "center"),
        (2 * kelly_f, "2× Kelly", (8, 4), "left"),
    ]:
        ax1.scatter([f], [100 * growth(f)], color="black", s=20, zorder=3)
        ax1.annotate(
            f"{label}\n{growth(f):+.1%}",
            xy=(f, 100 * growth(f)),
            xytext=offset,
            textcoords="offset points",
            ha=ha,
            va="bottom",
        )
    ax1.set_ylim(-9, 7.5)
    ax1.set_xlabel(r"leverage $f$")
    ax1.set_ylabel("excess log growth per year (%)")
    ax1.set_title(r"Known $\mu$, $\sigma$ (Sharpe 0.3)")

    bins = np.linspace(-0.25, 0.06, 63)
    for color, frac, name in [(COLORS[1], 1.0, "full Kelly"), (COLORS[2], 0.5, "half Kelly")]:
        ax2.hist(
            100 * res[frac],
            bins=100 * bins,
            color=color,
            alpha=0.7,
            label=f"{name}: mean {res[frac].mean():+.1%}",
        )
    ax2.axvline(0, color="grey", lw=0.8)
    ax2.axvline(100 * growth(kelly_f), color="black", lw=1.0, ls="--")
    ax2.text(
        100 * growth(kelly_f) + 0.5,
        ax2.get_ylim()[1] * 0.97,
        "best possible",
        rotation=90,
        ha="left",
        va="top",
    )
    ax2.set_xlabel("realized excess growth (%)")
    ax2.set_ylabel("count (2000 universes)")
    ax2.set_title("Betting on 5-year estimates")
    ax2.legend(frameon=False, loc="center left")
    fig.tight_layout()
    save_svg(fig, OUT / "kelly.svg")
    plt.close(fig)
    print(
        f"[kelly] known: half {growth(kelly_f / 2):+.4f}, kelly {growth(kelly_f):+.4f}, "
        f"2x {growth(2 * kelly_f):+.4f}; estimated full mean {res[1.0].mean():+.4f} "
        f"(neg {np.mean(res[1.0] < 0):.0%}), half mean {res[0.5].mean():+.4f} "
        f"(neg {np.mean(res[0.5] < 0):.0%})"
    )


if __name__ == "__main__":
    use_lecture_style()
    clustering()
    vol_target()
    kelly()
