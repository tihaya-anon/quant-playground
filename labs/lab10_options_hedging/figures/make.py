"""生成 lab10 讲义里的图。

    PYTHONPATH=src:. uv run --frozen python labs/lab10_options_hedging/figures/make.py

风格（字号、字体、SVG 文字设置）见 quantlab.plotting。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter

from quantlab.options import delta_hedge_pnl
from quantlab.plotting import COLORS, FULL_WIDTH, save_svg, use_lecture_style

OUT = Path(__file__).resolve().parent

S0, K, T, R, SIGMA_IMPL = 100.0, 100.0, 0.25, 0.0, 0.20


def hedging_pnl() -> None:
    """卖出平值看涨期权并 delta 对冲：P&L 取决于实现波动，误差随对冲次数按 1/sqrt(N) 缩小。"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL_WIDTH, 3.3))

    # 左：按 20% 卖出、每天对冲，实现波动 15% / 20% / 25%
    bins = np.linspace(-4, 4, 81)
    for color, real in zip(COLORS[:3], [0.15, 0.20, 0.25], strict=True):
        pnl = delta_hedge_pnl(S0, K, T, R, real, SIGMA_IMPL, 63, 20_000, seed=1) / S0 * 100
        ax1.hist(pnl, bins=bins, color=color, alpha=0.55, label=f"realized {real:.0%}")
        print(
            f"realized {real:.0%}: mean {pnl.mean():+.2f}%, sd {pnl.std():.2f}%, "
            f"loss share {(pnl < 0).mean():.0%}"
        )
    ax1.axvline(0, color="grey", lw=0.8)
    ax1.set_title("Sold at 20% implied, hedged daily")
    ax1.set_xlabel("P&L at expiry (% of spot)")
    ax1.set_yticks([])
    ax1.spines[["left"]].set_visible(False)
    ax1.legend(loc="upper left", frameon=False, fontsize=9)

    # 右：用正确的波动率对冲，P&L 标准差随对冲次数的变化
    ns = np.array([4, 8, 16, 32, 63, 126, 252, 630])
    sds = np.array(
        [
            (
                delta_hedge_pnl(S0, K, T, R, SIGMA_IMPL, SIGMA_IMPL, n, 20_000, seed=2) / S0 * 100
            ).std()
            for n in ns
        ]
    )
    # 参考线：斜率 -1/2，略往下平移，免得被模拟曲线盖住
    ref = 0.8 * sds[4] * np.sqrt(ns[4] / ns)
    ax2.loglog(ns, ref, color="grey", lw=1.0, ls="--", label="∝ 1/√N")
    ax2.loglog(ns, sds, "o-", color=COLORS[0], ms=4, label="simulated")
    for n, label in [(13, "weekly"), (63, "daily")]:
        ax2.axvline(n, color="grey", lw=0.5, ls=":")
        ax2.text(n * 1.08, sds.max() * 0.95, label, fontsize=9, va="top")
    ticks = [0.1, 0.2, 0.5, 1.0, 2.0]
    ax2.yaxis.set_major_locator(FixedLocator(ticks))
    ax2.yaxis.set_major_formatter(FixedFormatter([f"{t:g}" for t in ticks]))
    ax2.yaxis.set_minor_formatter(NullFormatter())
    ax2.set_ylim(0.1, 2.2)
    ax2.set_title("Hedging error vs. rebalances")
    ax2.set_xlabel("rebalances over 3 months, $N$")
    ax2.set_ylabel("P&L s.d. (% of spot)")
    ax2.legend(loc="lower left", frameon=False)

    fig.tight_layout()
    save_svg(fig, OUT / "hedging_pnl.svg")
    plt.close(fig)

    for n, sd in zip(ns, sds, strict=True):
        print(f"N={n:4d}: sd {sd:.3f}%")
    print(f"slope (log-log, first to last): {np.polyfit(np.log(ns), np.log(sds), 1)[0]:.2f}")


if __name__ == "__main__":
    use_lecture_style()
    hedging_pnl()
