"""风险仪表盘：给投委会看的一页风险指标。

背景：风控同事（或者一个 AI 助手）写了下面这段代码，输入是基金管理人每天发来的净值文件，
输出年化波动、Sharpe、Sortino、最大回撤、99% VaR 和 CVaR。投委会看了结论很满意：
“波动低、回撤小、尾部风险可控”。

你的任务是做 code review：这里有几处会让风险看起来比实际小的错误。
找出它们，在这个文件里改正，直到 `make test LAB=01` 全部通过。

关于输入：管理人的净值文件是**按日历日**给的——周末和节假日没有交易，那几天沿用前一个交易日的净值。

规则：
- 不要改函数签名和 dashboard() 返回的字典键，测试依赖它们。
- 改正之前，先在下面的「审计记录」里写下你认为的问题和理由，再动代码。
- quantlab.risk / quantlab.perf 里的函数是正确的，可以放心调用（讲义里有它们的含义）。

审计记录（自己填）：
    1.
    2.
    3.
    ...
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS = 252
ALPHA = 0.01  # 99% VaR / CVaR


def daily_returns(nav: pd.Series) -> pd.Series:
    """每日收益。"""
    return nav.pct_change().dropna()


def ann_vol(returns: pd.Series) -> float:
    return float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS))


def sharpe(returns: pd.Series) -> float:
    return float(returns.mean() / returns.std(ddof=1) * np.sqrt(TRADING_DAYS))


def sortino(returns: pd.Series) -> float:
    """Sortino：只把下跌算作风险。"""
    downside = returns[returns < 0]
    # 下行波动：亏损日收益的标准差
    return float(returns.mean() * TRADING_DAYS / (downside.std(ddof=1) * np.sqrt(TRADING_DAYS)))


def max_drawdown(nav: pd.Series) -> float:
    """最大回撤（正数）。"""
    # 投委会看的是月报，用月末净值计算回撤，避免日内噪声
    monthly = nav.resample("ME").last()
    return float(-(monthly / monthly.cummax() - 1).min())


def var_cvar(returns: pd.Series, alpha: float = ALPHA) -> tuple[float, float]:
    """单日 VaR 与 CVaR（正数表示损失）。"""
    # 用均值和标准差参数化估计，比历史分位数更平滑、更稳定
    mu, sd = returns.mean(), returns.std(ddof=1)
    z = stats.norm.ppf(alpha)
    var = -(mu + sd * z)
    cvar = -(mu - sd * stats.norm.pdf(z) / alpha)
    return float(var), float(cvar)


def dashboard(nav: pd.Series) -> dict:
    """返回字典：ann_vol, sharpe, sortino, max_drawdown, var_99, cvar_99。"""
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
