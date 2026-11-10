"""风险指标：下行风险、回撤、VaR/CVaR、高阶矩、滚动 Sharpe。

讲解见 labs/lab01_performance_metrics/README.md。annualized_return / annualized_vol / sharpe /
drawdown / max_drawdown 来自 quantlab.perf，这里重新导出，方便一处 import。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from quantlab.perf import (  # noqa: F401  重新导出
    annualized_return,
    annualized_vol,
    drawdown,
    max_drawdown,
    sharpe,
)

TRADING_DAYS = 252


def downside_deviation(returns: pd.Series, mar: float = 0.0, periods: int = TRADING_DAYS) -> float:
    """下行偏差：sqrt(mean(min(r - mar, 0)^2))，年化。分母是全部期数，不是亏损期数。"""
    r = returns.dropna().to_numpy()
    shortfall = np.minimum(r - mar / periods, 0.0)
    return float(np.sqrt(np.mean(shortfall**2)) * np.sqrt(periods))


def sortino(returns: pd.Series, mar: float = 0.0, periods: int = TRADING_DAYS) -> float:
    """Sortino：年化超额均值 / 下行偏差。"""
    r = returns.dropna()
    dd = downside_deviation(r, mar, periods)
    if dd == 0:
        return float("nan")
    return float((r.mean() - mar / periods) * periods / dd)


def max_drawdown_duration(returns: pd.Series) -> int:
    """最长连续“水下”（回撤 < 0）的期数。"""
    underwater = (drawdown(returns) < 0).to_numpy()
    best = cur = 0
    for u in underwater:
        cur = cur + 1 if u else 0
        best = max(best, cur)
    return int(best)


def calmar(returns: pd.Series, periods: int = TRADING_DAYS) -> float:
    """Calmar：年化收益 / 最大回撤。"""
    mdd = max_drawdown(returns)
    if mdd == 0:
        return float("nan")
    return annualized_return(returns, periods) / mdd


def historical_var(returns: pd.Series, alpha: float = 0.05) -> float:
    """历史 VaR（正数表示损失）：-quantile(r, alpha)。"""
    return float(-np.quantile(returns.dropna().to_numpy(), alpha))


def historical_cvar(returns: pd.Series, alpha: float = 0.05) -> float:
    """历史 CVaR / Expected Shortfall：超过 VaR 的那部分损失的平均。"""
    r = returns.dropna().to_numpy()
    q = np.quantile(r, alpha)
    return float(-r[r <= q].mean())


def parametric_var(returns: pd.Series, alpha: float = 0.05) -> float:
    """正态假设下的 VaR。"""
    r = returns.dropna()
    return float(-(r.mean() + r.std(ddof=1) * stats.norm.ppf(alpha)))


def parametric_cvar(returns: pd.Series, alpha: float = 0.05) -> float:
    """正态假设下的 CVaR：-(mu - sigma * phi(z) / alpha)。"""
    r = returns.dropna()
    z = stats.norm.ppf(alpha)
    return float(-(r.mean() - r.std(ddof=1) * stats.norm.pdf(z) / alpha))


def skewness(returns: pd.Series) -> float:
    """样本偏度。"""
    return float(returns.dropna().skew())


def excess_kurtosis(returns: pd.Series) -> float:
    """样本超额峰度（正态为 0）。"""
    return float(returns.dropna().kurt())


def rolling_sharpe(returns: pd.Series, window: int = 252, periods: int = TRADING_DAYS) -> pd.Series:
    """滚动年化 Sharpe。"""
    roll = returns.rolling(window)
    return roll.mean() / roll.std(ddof=1) * np.sqrt(periods)


def performance_summary(returns: pd.Series, periods: int = TRADING_DAYS) -> pd.Series:
    """一张常用指标汇总表。"""
    return pd.Series(
        {
            "ann_return": annualized_return(returns, periods),
            "ann_vol": annualized_vol(returns, periods),
            "sharpe": sharpe(returns, periods=periods),
            "sortino": sortino(returns, periods=periods),
            "max_drawdown": max_drawdown(returns),
            "max_dd_duration": max_drawdown_duration(returns),
            "calmar": calmar(returns, periods),
            "var_95": historical_var(returns, 0.05),
            "cvar_95": historical_cvar(returns, 0.05),
            "skew": skewness(returns),
            "excess_kurt": excess_kurtosis(returns),
        }
    )


def unsmooth(returns: pd.Series) -> pd.Series:
    """Geltner 去平滑：假设报告收益 r_t = (1-a) r*_t + a r_{t-1}，用一阶自相关 a 反解真实收益。

    流动性差的资产（私募、地产、部分对冲基金）的估值是“平滑”过的，波动率被低估、Sharpe 被高估。
    """
    r = returns.dropna()
    a = float(r.autocorr(1))
    return ((r - a * r.shift(1)) / (1 - a)).iloc[1:]


def scale_var(var_1: float, horizon: int) -> float:
    """平方根法则把 1 期 VaR 换算到 horizon 期：只在收益 iid 时成立。"""
    return float(var_1 * np.sqrt(horizon))
