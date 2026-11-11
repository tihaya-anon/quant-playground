"""生成 lab02 讲义里的图。

    PYTHONPATH=src:. uv run --frozen python labs/lab02_vectorized_backtest/figures/make.py

风格见 quantlab.plotting。数据生成方式与随机数种子与讲义里引用数字时的模拟相同。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter

from quantlab import backtest as bt
from quantlab import data, perf
from quantlab.plotting import COLORS, FULL_WIDTH, save_svg, use_lecture_style

OUT = Path(__file__).resolve().parent


def timeline() -> None:
    """§1：信号在 t 收盘才算出，只能吃到 t+1 的收益。"""
    # 三个交易日的日内价格路径（装饰用，来自零漂移 GBM），交易日之间留出隔夜的空隙
    px = data.gbm_prices(3 * 40, mu=0.0, sigma=0.6, seed=4).to_numpy()
    px = 100 * px / px[0]
    sessions = [(0.0, 1.0, "day t-1"), (1.35, 2.35, "day t"), (2.7, 3.7, "day t+1")]

    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 3.4))
    closes = []
    for k, (a, b, name) in enumerate(sessions):
        seg = px[k * 40 : (k + 1) * 40]
        ax.plot(np.linspace(a, b, len(seg)), seg, color="grey", lw=1.2)
        ax.axvspan(a, b, color=COLORS[5], alpha=0.06, lw=0)
        ax.text(
            (a + b) / 2,
            1.0,
            name,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            color=COLORS[5],
        )
        closes.append((b, seg[-1]))
        ax.plot([b], [seg[-1]], "o", color="black", ms=4)

    lo, hi = px.min(), px.max()
    span = hi - lo
    # t 收盘：算出信号
    (xc, yc) = closes[1]
    ax.axvline(xc, color="black", lw=0.8, ls=":")
    ax.annotate(
        r"close $t$: $P_t$ known," + "\n" + r"compute signal $s_t$",
        xy=(xc, yc),
        xytext=(xc - 0.12, hi + 0.38 * span),
        ha="right",
        va="center",
        arrowprops={"arrowstyle": "-", "lw": 0.8},
    )

    def bracket(a: float, b: float, y: float, color: str, label: str) -> None:
        ax.annotate(
            "",
            xy=(a, y),
            xytext=(b, y),
            arrowprops={"arrowstyle": "|-|,widthA=0.3,widthB=0.3", "color": color, "lw": 1.4},
        )
        ax.text((a + b) / 2, y - 0.06 * span, label, ha="center", va="top", color=color)

    y_br = lo - 0.18 * span
    bracket(
        closes[0][0],
        closes[1][0],
        y_br,
        COLORS[4],
        r"$r_t$: already happened" + "\n" + r"$s_t \cdot r_t$ = look-ahead",
    )
    bracket(
        closes[1][0],
        closes[2][0],
        y_br,
        COLORS[2],
        r"$r_{t+1}$: first return $s_t$ can earn" + "\n" + r"$s_t \cdot r_{t+1}$ = shift(1)",
    )

    ax.set_ylim(lo - 0.62 * span, hi + 0.85 * span)
    ax.set_xlim(-0.1, 3.8)
    ax.set_xticks([b for b, _ in closes])
    ax.set_xticklabels([r"close $t-1$", r"close $t$", r"close $t+1$"])
    ax.set_yticks([])
    ax.spines[["left"]].set_visible(False)
    ax.set_title("When is the signal known, and which return can it earn?")
    fig.tight_layout()
    save_svg(fig, OUT / "timeline.svg")
    plt.close(fig)
    print("[timeline] drawn")


def lookahead() -> None:
    """§2：零漂移 GBM（没有任何可预测性）上，两种前视偏差的净值，与正确版本对比。"""
    px = data.gbm_prices(2520, mu=0.0, sigma=0.2, seed=0)
    r = px.pct_change().fillna(0.0)

    sig = bt.momentum_signal(px, 20)
    right = bt.backtest(px, sig)["net"]
    no_shift = (sig * r).iloc[1:]

    sma = px.rolling(21, center=True).mean()
    centered = bt.backtest(px, np.sign(sma.diff()).fillna(0.0))["net"]

    years = np.arange(len(right)) / 252
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 3.4))
    curves = [
        (no_shift, COLORS[4], "momentum, forgot shift(1)"),
        (centered, COLORS[1], "21-day centered moving average, shifted"),
        (right, COLORS[0], "momentum, correctly shifted"),
    ]
    for ret, color, label in curves:
        ax.plot(
            years[: len(ret)],
            (1 + ret).cumprod().to_numpy(),
            color=color,
            label=f"{label}: Sharpe {perf.sharpe(ret):.2f}",
        )
    ax.set_yscale("log")
    ax.set_yticks([0.5, 1, 10, 100, 1000])
    ax.set_yticklabels(["0.5", "1", "10", "100", "1000"])
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("years")
    ax.set_ylabel("wealth (log scale)")
    ax.set_title("Zero-drift random prices: nothing to predict")
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    save_svg(fig, OUT / "lookahead.svg")
    plt.close(fig)
    print(
        f"[lookahead] sharpe right {perf.sharpe(right):.2f}, no-shift {perf.sharpe(no_shift):.2f}, "
        f"centered {perf.sharpe(centered):.2f}; final wealth "
        f"{(1 + right).prod():.2f} / {(1 + no_shift).prod():.3g} / {(1 + centered).prod():.3g}"
    )


if __name__ == "__main__":
    use_lecture_style()
    timeline()
    lookahead()
