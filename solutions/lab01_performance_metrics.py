"""Lab 01 审计题参考答案：audit.py 修正后的版本。

修正清单：
1. daily_returns：净值按日历日给出，周末沿用周五净值，产生大量 0 收益。它们把日波动摊薄，
   再乘 sqrt(252) 年化，波动率被低估约 sqrt(5/7)。只保留交易日（这里用工作日近似）。
2. sortino：下行偏差是 sqrt(mean(min(r, 0)^2))，分母是全部天数、偏离的基准是 0（或 MAR），
   不是“亏损日收益的标准差”。后者围绕亏损日自己的均值计算，偏小，Sortino 被高估。
3. max_drawdown：只看月末净值会漏掉月中的最低点。回撤是路径的性质，要用全部观测。
4. var_cvar：正态参数化在肥尾收益上会低估尾部，CVaR 尤其严重。用历史分位数和尾部均值
   （或者用肥尾分布拟合）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab import risk

TRADING_DAYS = 252
ALPHA = 0.01


def daily_returns(nav: pd.Series) -> pd.Series:
    trading = nav[nav.index.day_of_week < 5]
    return trading.pct_change().dropna()


def ann_vol(returns: pd.Series) -> float:
    return float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS))


def sharpe(returns: pd.Series) -> float:
    return float(returns.mean() / returns.std(ddof=1) * np.sqrt(TRADING_DAYS))


def sortino(returns: pd.Series) -> float:
    downside = np.sqrt(np.mean(np.minimum(returns.to_numpy(), 0.0) ** 2))
    return float(returns.mean() * TRADING_DAYS / (downside * np.sqrt(TRADING_DAYS)))


def max_drawdown(nav: pd.Series) -> float:
    return float(-(nav / nav.cummax() - 1).min())


def var_cvar(returns: pd.Series, alpha: float = ALPHA) -> tuple[float, float]:
    return risk.historical_var(returns, alpha), risk.historical_cvar(returns, alpha)


def dashboard(nav: pd.Series) -> dict:
    r = daily_returns(nav)
    var, cvar = var_cvar(r)
    return {
        "ann_vol": ann_vol(r),
        "sharpe": sharpe(r),
        "sortino": sortino(r),
        "max_drawdown": max_drawdown(nav),
        "var_99": var,
        "cvar_99": cvar,
    }
