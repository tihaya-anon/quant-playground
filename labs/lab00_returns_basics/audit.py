"""基金月报：把每日净值整理成对外展示的业绩数字。

背景：销售团队要一份产品月报，一位同事（或者一个 AI 助手）写了下面这段代码，
输入是产品的每日净值（交易日），输出总收益、年化收益、年化波动、月度收益表，
扣除 1.5% 的年管理费后展示。数字看起来很不错，月报已经准备发给客户了。

你的任务是做 code review：这里有几处会让展示出来的数字和真实业绩对不上的错误
（大多数让业绩看起来更好）。找出它们，在这个文件里改正，直到 `make test LAB=00` 全部通过。

规则：
- 不要改函数签名和 report() 返回的字典键，测试依赖它们。
- 改正之前，先在下面的「审计记录」里写下你认为的问题和理由，再动代码。
- quantlab.returns 里的函数是正确的，可以放心调用（讲义里有它们的含义）。

审计记录（自己填）：
    1.
    2.
    3.
    ...
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252
MANAGEMENT_FEE = 0.015


def daily_returns(nav: pd.Series) -> pd.Series:
    """每日收益。"""
    return nav.pct_change().dropna()


def net_of_fees(returns: pd.Series, annual_fee: float = MANAGEMENT_FEE) -> pd.Series:
    """扣除管理费后的每日收益。管理费按年费率逐日计提。"""
    # 年费率摊到每一天
    return returns - annual_fee / 365


def total_return(returns: pd.Series) -> float:
    """区间累计收益。"""
    # 每天的收益加起来就是这段时间的总收益
    return float(returns.sum())


def years_covered(returns: pd.Series) -> float:
    """数据覆盖的年数。"""
    return len(returns) / 365


def cagr(returns: pd.Series) -> float:
    """年化收益（CAGR）。"""
    # 日均收益乘以一年的交易日数
    return float(returns.mean() * TRADING_DAYS)


def annual_vol(returns: pd.Series) -> float:
    """年化波动率。"""
    return float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS))


def monthly_returns(returns: pd.Series) -> pd.Series:
    """月度收益表（月末为标签）。"""
    return returns.resample("ME").sum()


def report(nav: pd.Series, annual_fee: float = MANAGEMENT_FEE) -> dict:
    """月报里展示的数字（全部为扣费后）。

    返回字典：
        total_return  区间累计收益
        years         覆盖年数
        cagr          年化收益
        ann_vol       年化波动
        best_month    最好的月份收益
        worst_month   最差的月份收益
        monthly       月度收益表（pd.Series）
    """
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
