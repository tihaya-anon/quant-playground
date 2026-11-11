"""生成 lab08 讲义里的图。

    PYTHONPATH=src:. uv run --frozen python labs/lab08_event_driven_engine/figures/make.py

风格（字号、字体、SVG 文字设置）见 quantlab.plotting。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from quantlab.engine import bounce_ohlcv
from quantlab.plotting import COLORS, FULL_WIDTH, save_svg, use_lecture_style

OUT = Path(__file__).resolve().parent


def bid_ask_bounce() -> None:
    """收盘价落在买价或卖价上：真实价格没有可预测性，收盘价却出现虚假的负自相关。"""
    df = bounce_ohlcv(n=5000, spread_bps=80.0, seed=1)
    mid, close = df["mid"], df["close"]
    half = 80e-4 / 2

    def acf(x, lags):
        x = x - x.mean()
        return [float((x[k:] * x[:-k]).sum() / (x * x).sum()) for k in lags]

    lags = [1, 2, 3, 4, 5]
    r_close = np.log(close).diff().dropna().to_numpy()
    r_mid = np.log(mid).diff().dropna().to_numpy()
    ac_close, ac_mid = acf(r_close, lags), acf(r_mid, lags)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(FULL_WIDTH, 3.4), gridspec_kw={"width_ratios": [1.5, 1]}
    )

    # 左：截取 15 天，画真实 mid、买卖价区间和观测到的收盘价
    w = slice(200, 215)
    days = np.arange(15)
    m = mid.iloc[w].to_numpy()
    c = close.iloc[w].to_numpy()
    ax1.fill_between(
        days, m * (1 - half), m * (1 + half), color=COLORS[0], alpha=0.15, lw=0, label="bid–ask"
    )
    ax1.plot(days, m, color=COLORS[0], label="true mid $m_t$")
    ax1.plot(days, c, "o-", color=COLORS[1], ms=4, lw=1.0, label="close $p_t$")
    ax1.set_title("Closes jump between bid and ask")
    ax1.set_xlabel("day")
    ax1.set_ylabel("price")
    ax1.legend(loc="upper left", frameon=False, ncol=3, fontsize=9)
    lo, hi = ax1.get_ylim()
    ax1.set_ylim(lo, hi + 0.25 * (hi - lo))

    # 右：日收益的自相关
    x = np.arange(len(lags))
    ax2.bar(x - 0.2, ac_close, width=0.4, color=COLORS[1], label="close")
    ax2.bar(x + 0.2, ac_mid, width=0.4, color=COLORS[0], label="mid")
    ax2.axhline(0, color="grey", lw=0.8)
    band = 2 / np.sqrt(len(r_mid))
    ax2.axhspan(-band, band, color="grey", alpha=0.12, lw=0)
    ax2.set_xticks(x, [str(k) for k in lags])
    ax2.set_xlabel("lag (days)")
    ax2.set_title("Autocorrelation of daily returns")
    ax2.legend(loc="lower right", frameon=False)

    fig.tight_layout()
    save_svg(fig, OUT / "bid_ask_bounce.svg")
    plt.close(fig)

    print(f"lag-1 autocorr: close {ac_close[0]:+.3f}, mid {ac_mid[0]:+.3f}; ±2SE band {band:.3f}")
    print(f"Roll prediction -s^2/4 / Var(r): {-((80e-4) ** 2) / 4 / np.var(r_close):+.3f}")


if __name__ == "__main__":
    use_lecture_style()
    bid_ask_bounce()
