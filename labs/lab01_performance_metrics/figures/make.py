"""生成 lab01 讲义里的图。

    PYTHONPATH=src:. uv run --frozen python labs/lab01_performance_metrics/figures/make.py

风格见 quantlab.plotting。数据生成方式与随机数种子与讲义里引用数字时的模拟相同。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from quantlab import data, risk
from quantlab.plotting import COLORS, FULL_WIDTH, save_svg, use_lecture_style

OUT = Path(__file__).resolve().parent
DAYS = 252


def normal_strategy(years: float, seed: int, sharpe: float = 1.0, vol: float = 0.10) -> pd.Series:
    n = int(years * DAYS)
    g = np.random.default_rng(seed)
    r = g.normal(sharpe * vol / DAYS, vol / np.sqrt(DAYS), n)
    return pd.Series(r, index=data.business_index(n))


def sell_insurance(years: float, seed: int) -> pd.Series:
    """平时每天赚一点，平均每两年一次 8%~15% 的大亏。"""
    n = int(years * DAYS)
    g = np.random.default_rng(seed)
    r = g.normal(0.0004, 0.002, n)
    crash = g.random(n) < 1 / (2 * DAYS)
    r[crash] -= g.uniform(0.08, 0.15, crash.sum())
    return pd.Series(r, index=data.business_index(n))


def same_sharpe() -> None:
    """§1：均值和波动完全相同（Sharpe 相同）的两个 5 年策略，净值路径截然不同。"""
    calm_seed = next(s for s in range(1000) if (sell_insurance(5, s) < -0.05).sum() == 1)
    ins = sell_insurance(5, calm_seed)
    z = normal_strategy(5, seed=0)
    norm = ins.mean() + ins.std() * (z - z.mean()) / z.std()  # 样本均值和波动与 ins 完全相同

    w_norm, w_ins = (1 + norm).cumprod(), (1 + ins).cumprod()
    years = np.arange(len(ins)) / DAYS
    crash_day = int(np.argmin(ins.to_numpy()))

    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 3.3))
    ax.plot(
        years,
        w_norm.to_numpy(),
        color=COLORS[0],
        lw=1.4,
        label=f"normal returns (Sharpe {risk.sharpe(norm):.2f})",
    )
    ax.plot(
        years,
        w_ins.to_numpy(),
        color=COLORS[1],
        lw=1.4,
        label=f"sell insurance (Sharpe {risk.sharpe(ins):.2f})",
    )
    ax.annotate(
        f"one day: {ins.iloc[crash_day]:.1%}",
        xy=(years[crash_day], w_ins.iloc[crash_day]),
        xytext=(years[crash_day] - 0.25, 1.04),  # 右下角的空白处
        color=COLORS[1],
        ha="right",
        va="center",
        arrowprops={"arrowstyle": "-", "color": COLORS[1], "lw": 0.8},
    )
    ax.set_xlabel("years")
    ax.set_ylabel("wealth")
    ax.set_title("Same mean, same volatility, same Sharpe")
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    save_svg(fig, OUT / "same_sharpe.svg")
    plt.close(fig)
    print(
        f"[same_sharpe] sharpe {risk.sharpe(norm):.3f} / {risk.sharpe(ins):.3f}; "
        f"crash day {ins.iloc[crash_day]:.2%} at year {years[crash_day]:.2f}; "
        f"max dd {risk.max_drawdown(norm):.1%} / {risk.max_drawdown(ins):.1%}"
    )


def smoothing() -> None:
    """§6：估值平滑。真实月收益与管理人报告的平滑月收益（前 5 年）。"""
    months = 12 * 30
    g = np.random.default_rng(2)
    true = pd.Series(g.normal(0.10 / 12, 0.15 / np.sqrt(12), months))
    reported = true.copy()
    for t in range(1, months):
        reported.iloc[t] = 0.4 * true.iloc[t] + 0.6 * reported.iloc[t - 1]

    show = 60
    x = np.arange(1, show + 1)
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 3.2))
    ax.bar(
        x,
        100 * true.iloc[:show],
        color=COLORS[0],
        alpha=0.35,
        width=0.8,
        label=f"true monthly return (vol {true.std() * np.sqrt(12):.1%}/yr)",
    )
    ax.plot(
        x,
        100 * reported.iloc[:show],
        color=COLORS[1],
        lw=1.8,
        marker="o",
        ms=3,
        label=f"reported (smoothed) return (vol {reported.std() * np.sqrt(12):.1%}/yr)",
    )
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xlabel("month")
    ax.set_ylabel("monthly return, %")
    ax.set_title("Appraisal smoothing: reported = 0.4 × true + 0.6 × last reported")
    ax.set_xlim(0, show + 1)
    lim = 1.15 * 100 * true.iloc[:show].abs().max()
    ax.set_ylim(-lim, lim * 1.35)
    ax.legend(loc="upper left", frameon=False, ncol=1)
    fig.tight_layout()
    save_svg(fig, OUT / "smoothing.svg")
    plt.close(fig)
    print(
        f"[smoothing] 30y vol true {true.std() * np.sqrt(12):.1%}, "
        f"reported {reported.std() * np.sqrt(12):.1%}; "
        f"autocorr reported {reported.autocorr(1):.2f}"
    )


if __name__ == "__main__":
    use_lecture_style()
    same_sharpe()
    smoothing()
