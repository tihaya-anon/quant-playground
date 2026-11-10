"""Lab 00 审计题参考答案：audit.py 修正后的版本。

修正清单：
1. net_of_fees：费用按日历日 365 摊，但数据只有交易日（一年约 252 行），一年只扣了
   252/365 ≈ 69% 的费用。按交易日摊：(1 - fee)^(1/252) 的复利形式，或近似 fee/252。
2. total_return：简单收益在时间上不可加，要连乘。+50% 再 -50% 加起来是 0，实际亏 25%。
   波动越大，“相加”高估得越多。
3. years_covered：交易日数据一年 252 行，不是 365。按 365 算会把 10 年说成 6.9 年，
   年化收益被放大。
4. cagr：日均收益 × 252 是算术年化，比几何年化（真实复利增长率）大约高 sigma^2/2，
   而且对一个净值在下降的产品也可能是正数。CAGR 必须满足 (1+CAGR)^年数 = 期末/期初。
5. monthly_returns：月度收益同样要复利聚合，不能把日收益相加。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252
MANAGEMENT_FEE = 0.015


def daily_returns(nav: pd.Series) -> pd.Series:
    return nav.pct_change().dropna()


def net_of_fees(returns: pd.Series, annual_fee: float = MANAGEMENT_FEE) -> pd.Series:
    daily_keep = (1 - annual_fee) ** (1 / TRADING_DAYS)
    return (1 + returns) * daily_keep - 1


def total_return(returns: pd.Series) -> float:
    return float((1 + returns).prod() - 1)


def years_covered(returns: pd.Series) -> float:
    return len(returns) / TRADING_DAYS


def cagr(returns: pd.Series) -> float:
    return float((1 + total_return(returns)) ** (1 / years_covered(returns)) - 1)


def annual_vol(returns: pd.Series) -> float:
    return float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS))


def monthly_returns(returns: pd.Series) -> pd.Series:
    return (1 + returns).resample("ME").prod() - 1


def report(nav: pd.Series, annual_fee: float = MANAGEMENT_FEE) -> dict:
    r = net_of_fees(daily_returns(nav), annual_fee)
    monthly = monthly_returns(r)
    return {
        "total_return": total_return(r),
        "years": years_covered(r),
        "cagr": cagr(r),
        "ann_vol": annual_vol(r),
        "best_month": float(monthly.max()),
        "worst_month": float(monthly.min()),
        "monthly": monthly,
    }
