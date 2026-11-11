"""生成 lab03 讲义里的图。

    PYTHONPATH=src:. uv run --frozen python labs/lab03_backtest_statistics/figures/make.py

风格见 quantlab.plotting。数据生成方式与随机数种子与讲义里引用数字时的模拟相同。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from quantlab import data, perf, stats
from quantlab.plotting import COLORS, FULL_WIDTH, save_svg, use_lecture_style

OUT = Path(__file__).resolve().parent
DAYS = 252


def strategy_with_sharpe(sr_annual: float, years: float, seed: int) -> pd.Series:
    """真实年化 Sharpe 恰为 sr_annual 的 iid 正态日收益（年化波动 10%）。"""
    n = int(years * DAYS)
    vol = 0.10 / np.sqrt(DAYS)
    mu = sr_annual * vol / np.sqrt(DAYS)
    r = np.random.default_rng(seed).normal(mu, vol, n)
    return pd.Series(r, index=data.business_index(n))


def sharpe_sampling() -> None:
    """§1：真实 Sharpe 0.5 的策略，3 / 10 / 30 年回测的样本 Sharpe 分布。"""
    true_sr, n_sims = 0.5, 2000
    bins = np.linspace(-1.6, 2.6, 85)
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 3.3))
    rows = []
    for years, color in [(3, COLORS[4]), (10, COLORS[0]), (30, COLORS[2])]:
        srs = np.array(
            [perf.sharpe(strategy_with_sharpe(true_sr, years, s)) for s in range(n_sims)]
        )
        lo, hi = np.quantile(srs, [0.025, 0.975])
        neg = (srs < 0).mean()
        rows.append((years, lo, hi, neg, srs.std()))
        ax.hist(
            srs,
            bins=bins,
            density=True,
            histtype="step",
            lw=1.8,
            color=color,
            label=f"{years:>2} years: 95% in [{lo:.2f}, {hi:.2f}], {neg:.0%} negative",
        )
    ax.axvline(true_sr, color="black", ls="--", lw=1)
    ax.axvline(0, ymax=0.6, color="grey", lw=0.8)  # 只画到直方图上方，给标签留出空间
    ax.text(
        true_sr - 0.04,
        0.72,
        "true Sharpe 0.5",
        transform=ax.get_xaxis_transform(),
        ha="right",
        va="center",
    )
    ax.set_xlabel("sample Sharpe ratio (annualized)")
    ax.set_ylabel("density")
    ax.set_yticks([])
    ax.spines[["left"]].set_visible(False)
    ax.set_title("Same strategy, true Sharpe 0.5: what a backtest can show")
    ax.legend(loc="upper right", frameon=False)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.75)  # 直方图占下半部分，图例放在上方的空白
    fig.tight_layout()
    save_svg(fig, OUT / "sharpe_sampling.svg")
    plt.close(fig)
    for years, lo, hi, neg, sd in rows:
        print(
            f"[sharpe_sampling] {years}y: 95% [{lo:.2f}, {hi:.2f}], negative {neg:.1%}, sd {sd:.3f}"
        )


def max_sharpe_vs_trials() -> None:
    """§3：N 个纯噪声策略（10 年）里最好的那个，年化 Sharpe 的期望随 N 增长。"""
    sigma_sr = 1 / np.sqrt(10)  # 10 年数据，年化样本 Sharpe 的标准误（§1）
    ns = np.unique(np.round(np.logspace(0, 3, 60)).astype(int))
    exact = [stats.expected_max_sharpe(n, sigma_sr**2) if n > 1 else 0.0 for n in ns]
    asym = sigma_sr * np.sqrt(2 * np.log(ns))

    # 模拟：每个 N 重复 400 次，从 N(0, sigma_sr^2) 里抽 N 个样本 Sharpe 取最大
    rng = np.random.default_rng(0)
    sim_n = np.array([2, 5, 10, 20, 50, 100, 200, 500, 1000])
    sim = [rng.normal(0, sigma_sr, (400, n)).max(axis=1).mean() for n in sim_n]

    good = 0.5
    n_cross = next(int(n) for n, e in zip(ns, exact, strict=True) if e > good)

    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 3.4))
    ax.plot(ns, exact, color=COLORS[0], label="expected best of $N$ (Bailey & López de Prado)")
    ax.plot(
        ns,
        asym,
        color=COLORS[0],
        ls=":",
        lw=1.3,
        label=r"$\sigma_{\mathrm{SR}}\,(2\ln N)^{1/2}$ (overstates)",
    )
    ax.plot(sim_n, sim, "o", color=COLORS[1], ms=5, label="simulated average best of $N$")
    ax.axhline(good, color=COLORS[2], ls="--", lw=1.2)
    ax.text(
        1000,
        good + 0.02,
        "a genuinely good strategy: true Sharpe 0.5",
        color=COLORS[2],
        ha="right",
        va="bottom",
    )
    ax.annotate(
        f"pure noise beats it\nafter ~{n_cross} trials",
        xy=(n_cross, good),
        xytext=(n_cross * 3.5, 0.22),
        color=COLORS[2],
        ha="left",
        va="center",
        arrowprops={"arrowstyle": "-", "color": COLORS[2], "lw": 0.8},
    )
    ax.set_xscale("log")
    ax.set_xticks([1, 10, 100, 1000])
    ax.set_xticklabels(["1", "10", "100", "1000"])
    ax.set_xlabel("number of strategies tried, $N$ (all with true Sharpe 0)")
    ax.set_ylabel("best annualized Sharpe")
    ax.set_title("10 years of data: how good the best of $N$ pure-noise strategies looks")
    ax.legend(loc="upper left", frameon=False)
    ax.set_ylim(0, 1.45)
    fig.tight_layout()
    save_svg(fig, OUT / "max_sharpe.svg")
    plt.close(fig)
    pick = {n: e for n, e in zip(ns, exact, strict=True) if n in (10, 100, 1000)}
    print(
        f"[max_sharpe] expected best: { {k: round(v, 3) for k, v in pick.items()} }; "
        f"crosses {good} at N={n_cross}; simulated "
        f"{dict(zip(sim_n.tolist(), np.round(sim, 3).tolist(), strict=True))}"
    )


if __name__ == "__main__":
    use_lecture_style()
    sharpe_sampling()
    max_sharpe_vs_trials()
