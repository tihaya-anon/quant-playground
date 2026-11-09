"""参考绩效函数。

这些就是 lab01 要你亲手实现的东西。lab01 之后的 lab 直接 `from quantlab import perf`
使用它们，这样后面的测试不会因为前面某个 lab 没写完而连带失败。

建议：先做完 lab01 再来看这个文件。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def to_returns(prices: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    return prices.pct_change().iloc[1:]


def annualized_return(returns: pd.Series, periods: int = TRADING_DAYS) -> float:
    """几何年化收益（CAGR）。"""
    r = returns.dropna()
    growth = float((1 + r).prod())
    return growth ** (periods / len(r)) - 1


def annualized_vol(returns: pd.Series, periods: int = TRADING_DAYS) -> float:
    return float(returns.dropna().std(ddof=1) * np.sqrt(periods))


def sharpe(returns: pd.Series, rf: float = 0.0, periods: int = TRADING_DAYS) -> float:
    """年化 Sharpe：算术均值超额收益 / 标准差 * sqrt(periods)。rf 为年化无风险利率。"""
    ex = returns.dropna() - rf / periods
    sd = ex.std(ddof=1)
    # 容差而不是 == 0：常数序列的样本标准差在浮点下是 1e-19 量级而不是精确的 0。
    if not np.isfinite(sd) or sd <= 1e-12 * max(1.0, abs(ex.mean())):
        return float("nan")
    return float(ex.mean() / sd * np.sqrt(periods))


def drawdown(returns: pd.Series) -> pd.Series:
    """回撤序列（<= 0），基于复利净值。期初资金 1.0 也算一个高点：第一天就亏也是回撤。"""
    wealth = (1 + returns.fillna(0)).cumprod()
    peak = wealth.cummax().clip(lower=1.0)
    return wealth / peak - 1


def max_drawdown(returns: pd.Series) -> float:
    """最大回撤，返回正数（例如 0.35 表示 -35%）。"""
    return float(-drawdown(returns).min())


def summary(returns: pd.Series, periods: int = TRADING_DAYS) -> pd.Series:
    return pd.Series(
        {
            "ann_return": annualized_return(returns, periods),
            "ann_vol": annualized_vol(returns, periods),
            "sharpe": sharpe(returns, periods=periods),
            "max_drawdown": max_drawdown(returns),
        }
    )
