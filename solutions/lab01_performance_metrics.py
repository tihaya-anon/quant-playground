"""Lab 01 参考答案。"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS = 252


def annualized_return(returns: pd.Series, periods: int = TRADING_DAYS) -> float:
    r = returns.dropna()
    return float((1 + r).prod()) ** (periods / len(r)) - 1


def annualized_vol(returns: pd.Series, periods: int = TRADING_DAYS) -> float:
    return float(returns.dropna().std(ddof=1) * np.sqrt(periods))


def sharpe(returns: pd.Series, rf: float = 0.0, periods: int = TRADING_DAYS) -> float:
    ex = returns.dropna() - rf / periods
    sd = ex.std(ddof=1)
    if not np.isfinite(sd) or sd <= 1e-12 * max(1.0, abs(ex.mean())):
        return float("nan")
    return float(ex.mean() / sd * np.sqrt(periods))


def downside_deviation(returns: pd.Series, mar: float = 0.0, periods: int = TRADING_DAYS) -> float:
    r = returns.dropna().to_numpy()
    shortfall = np.minimum(r - mar / periods, 0.0)
    return float(np.sqrt(np.mean(shortfall**2)) * np.sqrt(periods))


def sortino(returns: pd.Series, mar: float = 0.0, periods: int = TRADING_DAYS) -> float:
    r = returns.dropna()
    dd = downside_deviation(r, mar, periods)
    if dd == 0:
        return float("nan")
    return float((r.mean() - mar / periods) * periods / dd)


def drawdown(returns: pd.Series) -> pd.Series:
    wealth = (1 + returns.fillna(0)).cumprod()
    peak = wealth.cummax().clip(lower=1.0)
    return wealth / peak - 1


def max_drawdown(returns: pd.Series) -> float:
    return float(-drawdown(returns).min())


def max_drawdown_duration(returns: pd.Series) -> int:
    underwater = (drawdown(returns) < 0).to_numpy()
    best = cur = 0
    for u in underwater:
        cur = cur + 1 if u else 0
        best = max(best, cur)
    return int(best)


def calmar(returns: pd.Series, periods: int = TRADING_DAYS) -> float:
    mdd = max_drawdown(returns)
    if mdd == 0:
        return float("nan")
    return annualized_return(returns, periods) / mdd


def historical_var(returns: pd.Series, alpha: float = 0.05) -> float:
    return float(-np.quantile(returns.dropna().to_numpy(), alpha))


def historical_cvar(returns: pd.Series, alpha: float = 0.05) -> float:
    r = returns.dropna().to_numpy()
    q = np.quantile(r, alpha)
    return float(-r[r <= q].mean())


def parametric_var(returns: pd.Series, alpha: float = 0.05) -> float:
    r = returns.dropna()
    return float(-(r.mean() + r.std(ddof=1) * stats.norm.ppf(alpha)))


def parametric_cvar(returns: pd.Series, alpha: float = 0.05) -> float:
    r = returns.dropna()
    z = stats.norm.ppf(alpha)
    return float(-(r.mean() - r.std(ddof=1) * stats.norm.pdf(z) / alpha))


def skewness(returns: pd.Series) -> float:
    return float(returns.dropna().skew())


def excess_kurtosis(returns: pd.Series) -> float:
    return float(returns.dropna().kurt())


def rolling_sharpe(returns: pd.Series, window: int = 252, periods: int = TRADING_DAYS) -> pd.Series:
    roll = returns.rolling(window)
    return roll.mean() / roll.std(ddof=1) * np.sqrt(periods)


def performance_summary(returns: pd.Series, periods: int = TRADING_DAYS) -> pd.Series:
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
