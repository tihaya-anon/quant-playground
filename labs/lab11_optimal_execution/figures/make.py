"""生成 lab11 讲义里的图。

    PYTHONPATH=src:. uv run --frozen python labs/lab11_optimal_execution/figures/make.py

风格（字号、字体、SVG 文字设置）见 quantlab.plotting。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter

from quantlab import execution
from quantlab.plotting import COLORS, FULL_WIDTH, save_svg, use_lecture_style

OUT = Path(__file__).resolve().parent

# Almgren-Chriss (2000) 原文的例子：5 天内卖出 100 万股，股价 50，日波动 0.95 美元；
# 每半天一个交易区间（N=10），与讲义 §4 的表格和 tests/test_execution.py 一致
AC = {"X": 1e6, "T": 5.0, "N": 10, "sigma": 0.95, "eta": 2.5e-6, "gamma": 2.5e-7, "epsilon": 0.0625}
LAMS = [1e-9, 1e-6, 1e-5]


def almgren_chriss() -> None:
    """风险厌恶越高，执行越前置：期望成本上升、成本波动下降。"""
    X, T, N = AC["X"], AC["T"], AC["N"]
    sigma, eta, gamma, eps = AC["sigma"], AC["eta"], AC["gamma"], AC["epsilon"]
    tau = T / N
    days = np.arange(N + 1) * tau

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL_WIDTH, 3.3))
    names = ["λ ≈ 0  (TWAP)", "λ = 1e-6", "λ = 1e-5"]
    points = []
    for color, lam, name in zip(COLORS[:3], LAMS, names, strict=True):
        x = execution.ac_trajectory(X, T, N, sigma, eta, gamma, lam)
        ax1.plot(days, x / X * 100, "o-", color=color, ms=3, label=name)
        e = execution.expected_cost(x, sigma, eta, gamma, eps, tau)
        sd = np.sqrt(execution.cost_variance(x, sigma, tau))
        points.append((e, sd))
        day1 = int(round(1 / tau))
        print(f"lam {lam:.0e}: first-day sold {(1 - x[day1] / X):.0%}, E {e:,.0f}, sd {sd:,.0f}")
    ax1.set_title("Optimal trajectory $x(t)$")
    ax1.set_xlabel("day")
    ax1.set_ylabel("shares still to sell (%)")
    ax1.legend(loc="upper right", frameon=False, fontsize=9)

    lams = np.logspace(-9, -4.5, 60)
    fr = execution.efficient_frontier(X, T, N, sigma, eta, gamma, eps, lams)
    ax2.plot(np.sqrt(fr["variance"]) / 1e6, fr["expected_cost"] / 1e6, color="grey", lw=1.4)
    for color, (e, sd) in zip(COLORS[:3], points, strict=True):
        ax2.plot(sd / 1e6, e / 1e6, "o", color=color, ms=7)
    ax2.set_title("Efficient frontier")
    ax2.set_xlabel("s.d. of cost ($ millions)")
    ax2.set_ylabel("expected cost ($ millions)")

    fig.tight_layout()
    save_svg(fig, OUT / "almgren_chriss.svg")
    plt.close(fig)


def capacity() -> None:
    """三种“容量”：净 Sharpe 门槛、利润最大、净收益归零，按讲义 §2 的公式和算例参数计算。"""
    alpha, vol, tau_d, n_names, adv, half_spread, Y, sigma = (
        0.15,
        0.10,
        0.10,
        50,
        5e7,
        0.0003,
        0.8,
        0.02,
    )
    A = np.logspace(7, np.log10(4e9), 4000)
    r_net = alpha - 252 * tau_d * (half_spread + Y * sigma * np.sqrt(tau_d * A / (n_names * adv)))
    sharpe = r_net / vol
    profit = r_net * A

    a_sharpe = A[sharpe >= 1.0].max()
    i_max = int(profit.argmax())
    a_profit = A[i_max]
    a_zero = A[r_net > 0].max()
    print(
        f"Sharpe>=1 up to {a_sharpe / 1e8:.2f}e8; profit max at {a_profit / 1e8:.2f}e8 "
        f"(net Sharpe {sharpe[i_max]:.2f}, profit {profit[i_max] / 1e8:.2f}e8/yr); "
        f"zero at {a_zero / 1e8:.2f}e8"
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL_WIDTH, 3.3), sharex=True)
    ax1.semilogx(A / 1e8, sharpe, color=COLORS[0])
    ax1.axhline(1.0, color="grey", lw=0.8, ls="--")
    ax1.axhline(0.0, color="grey", lw=0.8)
    ax1.set_ylabel("net Sharpe")
    ax1.set_title("Net Sharpe falls with size")

    ax2.semilogx(A / 1e8, profit / 1e8, color=COLORS[1])
    ax2.axhline(0.0, color="grey", lw=0.8)
    ax2.set_ylabel("annual profit (100m CNY)")
    ax2.set_title("Profit peaks much later")

    marks = [
        (a_sharpe, COLORS[0], "Sharpe = 1"),
        (a_profit, COLORS[1], "max profit"),
        (a_zero, COLORS[4], "net = 0"),
    ]
    ticks = [0.1, 0.3, 1, 3, 10, 30]
    for ax in (ax1, ax2):
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_major_formatter(FixedFormatter([f"{t:g}" for t in ticks]))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("AUM (100m CNY)")
        for a, color, label in marks:
            ax.axvline(a / 1e8, color=color, lw=1.2, ls=":", label=f"{label} ({a / 1e8:.1f})")
    # 三条竖线用图例说明：右图左下角是空的（规模小时利润不为负）
    ax2.legend(loc="lower left", frameon=False, fontsize=9, handlelength=1.5)

    fig.tight_layout()
    save_svg(fig, OUT / "capacity.svg")
    plt.close(fig)


if __name__ == "__main__":
    use_lecture_style()
    almgren_chriss()
    capacity()
