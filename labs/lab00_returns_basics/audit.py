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
    1. 混用交易日和自然日
    2. 时间数据计算累计收益率不能简单相加，而是相乘或转为对数收益率
    3. 年化收益的几何平均才是日收益，不是算数平均
    4. （审阅后判定不是问题）年化波动用简单收益还是对数收益：ln(1+r) ≈ r - r²/2，
       日收益很小，两者标准差只差 r² 这一高阶小量，可以忽略。按惯例保留简单收益，
       与 Sharpe（简单收益均值 / 简单收益标准差）口径一致。√252 真正依赖的假设是
       日收益相互独立，与用哪种收益无关。
    5. 月度收益表把当月日收益相加（resample("ME").sum()），与第 2 条同一个错误：
       月内也要复利连乘。检验办法：12 个月的月收益连乘回去，必须等于全年累计收益。
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
    return returns - annual_fee / TRADING_DAYS


def total_return(returns: pd.Series) -> float:
    """区间累计收益。"""
    # 每天的收益乘起来就是这段时间的总收益
    returns = returns.map(lambda x: x + 1)
    return float(returns.prod()) - 1


def years_covered(returns: pd.Series) -> float:
    """数据覆盖的年数。"""
    return len(returns) / TRADING_DAYS


def cagr(returns: pd.Series) -> float:
    """年化收益（CAGR）。"""
    # (1 + r)^years = 1 + total_r
    return np.power((1 + total_return(returns)), (1 / years_covered(returns))) - 1


def annual_vol(returns: pd.Series) -> float:
    """年化波动率。"""
    return float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS))


def monthly_returns(returns: pd.Series) -> pd.Series:
    """月度收益表（月末为标签）。"""
    # resample("ME") 按自然月分组（标签取月末），每组内的日收益要连乘，和 total_return 同理
    return (1 + returns).resample("ME").prod() - 1


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
