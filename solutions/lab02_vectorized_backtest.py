"""Lab 02 参考答案。"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def signal_to_position(signal: pd.Series) -> pd.Series:
    return signal.shift(1).fillna(0.0)


def turnover(positions: pd.Series) -> pd.Series:
    return positions.diff().fillna(positions).abs()


def backtest(prices: pd.Series, signal: pd.Series, cost_bps: float = 0.0) -> pd.DataFrame:
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
    f = prices.rolling(fast).mean()
    s = prices.rolling(slow).mean()
    down = 0.0 if long_only else -1.0
    sig = pd.Series(np.where(f > s, 1.0, down), index=prices.index)
    sig[s.isna()] = 0.0
    return sig


def momentum_signal(prices: pd.Series, lookback: int = 60) -> pd.Series:
    past = prices / prices.shift(lookback) - 1
    return np.sign(past).fillna(0.0)


def vol_target_signal(
    signal: pd.Series,
    prices: pd.Series,
    target_vol: float = 0.10,
    window: int = 60,
    max_leverage: float = 3.0,
) -> pd.Series:
    vol = prices.pct_change().rolling(window).std() * np.sqrt(TRADING_DAYS)
    scale = (target_vol / vol).clip(upper=max_leverage)
    return (signal * scale).fillna(0.0)


def annualized_turnover(result: pd.DataFrame, periods: int = TRADING_DAYS) -> float:
    return float(result["turnover"].mean() * periods)


def breakeven_cost_bps(result: pd.DataFrame) -> float:
    return float(result["gross"].mean() / result["turnover"].mean() * 1e4)
