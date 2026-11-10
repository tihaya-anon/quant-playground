"""向量化回测的基本部件：信号 -> 仓位 -> 收益 -> 成本，以及几个常见信号。

讲解见 labs/lab02_vectorized_backtest/README.md。时间约定：signal.loc[t] 在 t 收盘时才可知，
最早只能吃到 t+1 的收益，所以仓位 = signal.shift(1)。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def signal_to_position(signal: pd.Series) -> pd.Series:
    """t 收盘的信号成为 t+1 的仓位：shift(1)，第一期为 0。"""
    return signal.shift(1).fillna(0.0)


def turnover(positions: pd.Series) -> pd.Series:
    """每期换手 |pos_t - pos_{t-1}|，期初仓位视为 0。"""
    return positions.diff().fillna(positions).abs()


def backtest(prices: pd.Series, signal: pd.Series, cost_bps: float = 0.0) -> pd.DataFrame:
    """单资产向量化回测。返回列 position, asset_ret, gross, turnover, cost, net（去掉第一行）。"""
    signal = signal.reindex(prices.index).fillna(0.0)
    asset_ret = prices.pct_change().fillna(0.0)
    pos = signal_to_position(signal)
    to = turnover(pos)
    gross = pos * asset_ret
    cost = to * cost_bps / 1e4
    out = pd.DataFrame(
        {
            "position": pos,
            "asset_ret": asset_ret,
            "gross": gross,
            "turnover": to,
            "cost": cost,
            "net": gross - cost,
        }
    )
    return out.iloc[1:]


def sma_crossover_signal(
    prices: pd.Series, fast: int = 20, slow: int = 100, long_only: bool = False
) -> pd.Series:
    """均线交叉：快线在慢线上方为 1，否则 -1（long_only 时为 0）；慢线未就绪时为 0。"""
    f = prices.rolling(fast).mean()
    s = prices.rolling(slow).mean()
    down = 0.0 if long_only else -1.0
    sig = pd.Series(np.where(f > s, 1.0, down), index=prices.index)
    sig[s.isna()] = 0.0
    return sig


def momentum_signal(prices: pd.Series, lookback: int = 60) -> pd.Series:
    """时间序列动量：过去 lookback 期收益的符号。"""
    past = prices / prices.shift(lookback) - 1
    return np.sign(past).fillna(0.0)


def vol_target_signal(
    signal: pd.Series,
    prices: pd.Series,
    target_vol: float = 0.10,
    window: int = 60,
    max_leverage: float = 3.0,
) -> pd.Series:
    """把信号按 目标波动 / 近期实现波动 缩放，杠杆上限 max_leverage。"""
    vol = prices.pct_change().rolling(window).std() * np.sqrt(TRADING_DAYS)
    scale = (target_vol / vol).clip(upper=max_leverage)
    return (signal * scale).fillna(0.0)


def annualized_turnover(result: pd.DataFrame, periods: int = TRADING_DAYS) -> float:
    """年化换手（单边）。"""
    return float(result["turnover"].mean() * periods)


def breakeven_cost_bps(result: pd.DataFrame) -> float:
    """让净收益归零的单边成本（bps）：毛收益均值 / 换手均值。"""
    return float(result["gross"].mean() / result["turnover"].mean() * 1e4)


def universe_with_delistings(
    n_assets: int = 30,
    n_days: int = 2520,
    delist_level: float = 0.25,
    seed: int | None = 0,
) -> pd.DataFrame:
    """一组 GBM 股票，跌破初始价 delist_level 倍后退市（之后为 NaN）。

    每只股票的漂移在 [-0.15, 0.15] 之间随机，年化波动 35%。退市当天的价格保留，
    即退市前最后一段下跌是可以被交易到的。用来演示幸存者偏差。
    """
    from quantlab import data

    rng = np.random.default_rng(seed)
    mus = rng.uniform(-0.15, 0.15, n_assets)
    cols = {}
    for i, mu in enumerate(mus):
        px = data.gbm_prices(n_days, mu=float(mu), sigma=0.35, seed=int(rng.integers(1 << 31)))
        below = np.flatnonzero((px < 100.0 * delist_level).to_numpy())
        if below.size:
            px.iloc[below[0] + 1 :] = np.nan
        cols[f"S{i:02d}"] = px
    return pd.DataFrame(cols)
