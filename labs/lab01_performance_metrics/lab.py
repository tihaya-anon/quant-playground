"""Lab 01 · 绩效与风险指标。

把每个 `raise NotImplementedError` 换成你的实现，然后 `make test LAB=01`。
约定：
- returns 是简单收益的 pd.Series（通常是日频），可能含 NaN，计算前先 dropna()。
- periods 是每年的期数（日频 252、周频 52、月频 12）。
- “损失类”指标（最大回撤、VaR、CVaR）一律返回 **正数**，例如 0.05 表示亏 5%。
"""

from __future__ import annotations

import numpy as np  # noqa: F401
import pandas as pd
from scipy import stats  # noqa: F401

TRADING_DAYS = 252


def annualized_return(returns: pd.Series, periods: int = TRADING_DAYS) -> float:
    """几何年化收益（CAGR）：(prod(1 + r)) ** (periods / n) - 1，n 为有效样本数。"""
    raise NotImplementedError


def annualized_vol(returns: pd.Series, periods: int = TRADING_DAYS) -> float:
    """年化波动率：样本标准差（ddof=1）* sqrt(periods)。

    为什么是 sqrt？iid 假设下方差随时间线性累加，标准差随 sqrt(时间)。
    """
    raise NotImplementedError


def sharpe(returns: pd.Series, rf: float = 0.0, periods: int = TRADING_DAYS) -> float:
    """年化 Sharpe 比率。

    rf 是 **年化** 无风险利率，每期扣 rf / periods 得到超额收益 ex。
    SR = mean(ex) / std(ex, ddof=1) * sqrt(periods)。
    注意这里用的是 **算术** 均值，不是 CAGR。标准差为 0 时返回 nan。
    """
    raise NotImplementedError


def downside_deviation(returns: pd.Series, mar: float = 0.0, periods: int = TRADING_DAYS) -> float:
    """年化下行偏差。

    mar 是年化的最低可接受收益（minimum acceptable return），每期门槛 mar / periods。
    DD = sqrt( mean( min(r - mar/periods, 0)^2 ) ) * sqrt(periods)
    注意分母是 **全部** 样本数 n，而不是只数亏损的那些（这是 Sortino 的标准定义）。
    """
    raise NotImplementedError


def sortino(returns: pd.Series, mar: float = 0.0, periods: int = TRADING_DAYS) -> float:
    """Sortino 比率 = (mean(r) - mar/periods) * periods / downside_deviation。

    下行偏差为 0 时返回 nan。
    """
    raise NotImplementedError


def drawdown(returns: pd.Series) -> pd.Series:
    """回撤序列 D_t = W_t / max(1, max_{s<=t} W_s) - 1（<= 0），W 为复利净值（NaN 视作 0 收益）。

    期初资金 W_0 = 1 也算一个高点：如果第一天就亏损，那一天就已经处于回撤中。

    与 returns 同索引。
    """
    raise NotImplementedError


def max_drawdown(returns: pd.Series) -> float:
    """最大回撤（正数）。"""
    raise NotImplementedError


def max_drawdown_duration(returns: pd.Series) -> int:
    """最长“水下”持续期：连续处于回撤（D_t < 0）中的最多期数。

    例：净值 1.0, 1.1, 1.0, 0.9, 1.05, 1.2, 1.1 → 水下期为第 3、4、5 个点（1.0, 0.9, 1.05 都
    低于 1.1），持续期 3；最后一点又水下 1 期。答案为 3。
    从来没回撤过返回 0。
    """
    raise NotImplementedError


def calmar(returns: pd.Series, periods: int = TRADING_DAYS) -> float:
    """Calmar 比率 = 年化收益（CAGR）/ 最大回撤。最大回撤为 0 时返回 nan。"""
    raise NotImplementedError


def historical_var(returns: pd.Series, alpha: float = 0.05) -> float:
    """历史模拟 VaR：-quantile(returns, alpha)，用 np.quantile 的默认线性插值。"""
    raise NotImplementedError


def historical_cvar(returns: pd.Series, alpha: float = 0.05) -> float:
    """历史 CVaR（Expected Shortfall）：所有 <= alpha 分位数的收益的均值，取负号。"""
    raise NotImplementedError


def parametric_var(returns: pd.Series, alpha: float = 0.05) -> float:
    """正态假设下的 VaR：-(mu + sigma * z_alpha)，mu、sigma 为样本均值与标准差（ddof=1），
    z_alpha = Phi^{-1}(alpha)（scipy.stats.norm.ppf）。"""
    raise NotImplementedError


def parametric_cvar(returns: pd.Series, alpha: float = 0.05) -> float:
    """正态假设下的 CVaR。

    对 X ~ N(mu, sigma^2)：E[X | X <= mu + sigma z_alpha] = mu - sigma * phi(z_alpha) / alpha，
    phi 是标准正态密度。返回它的相反数。（自己推一遍：对截断正态求期望。）
    """
    raise NotImplementedError


def skewness(returns: pd.Series) -> float:
    """样本偏度（与 pandas Series.skew() 一致，即调整后的无偏估计）。"""
    raise NotImplementedError


def excess_kurtosis(returns: pd.Series) -> float:
    """样本超额峰度（与 pandas Series.kurt() 一致；正态分布为 0）。"""
    raise NotImplementedError


def rolling_sharpe(returns: pd.Series, window: int = 252, periods: int = TRADING_DAYS) -> pd.Series:
    """滚动年化 Sharpe（rf = 0），窗口长度 window。前 window-1 个值为 NaN，与 returns 同索引。"""
    raise NotImplementedError


def performance_summary(returns: pd.Series, periods: int = TRADING_DAYS) -> pd.Series:
    """汇总表。返回 pd.Series，键依次为：

    ann_return, ann_vol, sharpe, sortino, max_drawdown, max_dd_duration, calmar,
    var_95, cvar_95（历史法，alpha=0.05）, skew, excess_kurt
    """
    raise NotImplementedError
