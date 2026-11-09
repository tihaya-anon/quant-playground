"""Lab 02 · 向量化回测。

把每个 `raise NotImplementedError` 换成你的实现，然后 `make test LAB=02`。

时间约定（整个课程通用，务必记住）：
- prices.loc[t] 是 t 日收盘价；asset_ret.loc[t] = prices[t] / prices[t-1] - 1。
- signal.loc[t] 是用 **t 日收盘及以前** 的信息算出来的目标仓位。
- 我们在 t 日收盘按 signal[t] 调仓，于是 t+1 日的收益由 signal[t] 决定：
      position[t+1] = signal[t]       ← 这就是那个 shift(1)
      gross[t+1]    = position[t+1] * asset_ret[t+1]
- 仓位是“权益的倍数”：1 = 满仓做多，-1 = 满仓做空，0.5 = 半仓。
"""

from __future__ import annotations

import numpy as np  # noqa: F401
import pandas as pd

TRADING_DAYS = 252


def signal_to_position(signal: pd.Series) -> pd.Series:
    """把信号变成实际持仓：position[t] = signal[t-1]。

    第一期（没有前一天信号）为 0。与 signal 同索引。
    """
    raise NotImplementedError


def turnover(positions: pd.Series) -> pd.Series:
    """每期换手 |position[t] - position[t-1]|，假设第一期之前的仓位为 0。与 positions 同索引。

    从 +1 翻到 -1 的换手是 2。
    """
    raise NotImplementedError


def backtest(prices: pd.Series, signal: pd.Series, cost_bps: float = 0.0) -> pd.DataFrame:
    """单资产向量化回测。

    1. signal 先 reindex 到 prices.index，缺失填 0。
    2. asset_ret = prices.pct_change()（第一期填 0）。
    3. position = signal_to_position(signal)；turnover = turnover(position)。
    4. gross = position * asset_ret；cost = turnover * cost_bps / 1e4；net = gross - cost。
       （成本按成交名义金额收取，cost_bps=10 表示每交易 1 倍权益收 0.1%。）
    5. 返回 DataFrame，列依次为 position, asset_ret, gross, turnover, cost, net，
       **去掉第一行**（第一行没有收益），索引为 prices.index[1:]。
    """
    raise NotImplementedError


def sma_crossover_signal(
    prices: pd.Series, fast: int = 20, slow: int = 100, long_only: bool = False
) -> pd.Series:
    """双均线信号：快均线 > 慢均线时 +1，否则 -1（long_only=True 时为 0）。

    均线用简单移动平均 prices.rolling(n).mean()（包含当日收盘，这在 t 日收盘时是已知的）。
    慢均线还没有值的预热期，信号为 0。与 prices 同索引，dtype float。
    """
    raise NotImplementedError


def momentum_signal(prices: pd.Series, lookback: int = 60) -> pd.Series:
    """时间序列动量：sign(prices[t] / prices[t - lookback] - 1)，预热期为 0。与 prices 同索引。"""
    raise NotImplementedError


def vol_target_signal(
    signal: pd.Series,
    prices: pd.Series,
    target_vol: float = 0.10,
    window: int = 60,
    max_leverage: float = 3.0,
) -> pd.Series:
    """波动率目标化：把方向信号缩放到目标年化波动。

    realized_vol[t] = std(最近 window 个日收益，含 t 日) * sqrt(252)
    scaled[t]       = signal[t] * min(target_vol / realized_vol[t], max_leverage)
    预热期为 0。注意 realized_vol 只用到 t 日及以前，所以把结果交给 backtest（它会 shift）不会前视。
    """
    raise NotImplementedError


def annualized_turnover(result: pd.DataFrame, periods: int = TRADING_DAYS) -> float:
    """年化换手：backtest 结果中 turnover 列的均值 * periods。“一年换手 20 倍”就是 20.0。"""
    raise NotImplementedError


def breakeven_cost_bps(result: pd.DataFrame) -> float:
    """盈亏平衡成本（bps）：多高的单边成本会把毛收益正好吃光。

    mean(gross) - mean(turnover) * c / 1e4 = 0  =>  c = mean(gross) / mean(turnover) * 1e4
    """
    raise NotImplementedError
