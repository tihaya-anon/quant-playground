"""研究报告：每月卖出 delta 对冲的平值跨式期权，“稳定赚取波动率风险溢价”。

背景：一位同事（或者一个 AI 助手）交来了这份回测和结论——

    "15 年回测：每月月初卖出 1 个月平值跨式，每日 delta 对冲到期。
     年化 Sharpe 约 2，t 值远大于 2，最差月份亏损很小，尾部风险在风控限额之内，建议上线。"

你的任务是做 code review：这份回测里有若干处会让结论偏乐观的错误。
找出它们，在这个文件里改正，直到 `make test LAB=10` 全部通过。

规则：
- 不要改函数签名和 evaluate() 返回的字典键，测试依赖它们。
- 改正之前，先在下面的「审计记录」里写下你认为的问题和理由，再动代码。
- quantlab.options 里的函数是正确的，可以放心调用。

审计记录（自己填）：
    1.
    2.
    3.
    ...
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from quantlab import options

TRADING_DAYS = 252
HOLD = 21  # 每份期权持有 21 个交易日（约 1 个月）到期
VRP = 1.10  # 隐含波动率相对已实现波动率的溢价（市场常见水平）
HALF_SPREAD = 0.02  # 期权买卖价差的一半，占权利金的比例
RISK_LIMIT = 0.04  # 风控：最差 5% 月份的平均亏损不得超过名义本金的 4%


def clean_returns(returns: pd.Series) -> pd.Series:
    """数据清洗：超过 5 个标准差的日收益通常是数据错误（错价、拆股未复权），置为 0。"""
    z = (returns - returns.mean()) / returns.std()
    return returns.where(z.abs() < 5, 0.0)


def selling_vol(returns: pd.Series) -> pd.Series:
    """每个交易日卖出期权时的隐含波动率（年化）。

    历史隐含波动率数据缺失较多，用已实现波动率乘以 VRP 近似。期权的价值取决于
    它存续期内的波动，所以用从当天开始、未来 HOLD 天的已实现波动率。
    """
    fwd = returns[::-1].rolling(HOLD).std()[::-1]
    return fwd * np.sqrt(TRADING_DAYS) * VRP


def _straddle(S, T, sigma):
    """平值（K=1）跨式期权的价格与 delta（向量化）。"""
    price = options.bs_price(S, 1.0, T, 0.0, sigma, kind="call") + options.bs_price(
        S, 1.0, T, 0.0, sigma, kind="put"
    )
    d1, _ = options.d1_d2(S, 1.0, T, 0.0, sigma)
    return price, 2 * norm.cdf(d1) - 1


def monthly_pnl(returns: pd.Series, half_spread: float = HALF_SPREAD) -> pd.Series:
    """每月卖出一份平值跨式（名义本金 1），每日 delta 对冲，持有到期。

    返回每份期权的 P&L（以名义本金为单位），索引为卖出日。
    """
    r = clean_returns(returns)
    vol = selling_vol(r)
    starts = np.arange(HOLD, len(r) - HOLD, HOLD)
    rr = r.to_numpy()

    paths = np.ones((len(starts), HOLD + 1))
    for j in range(HOLD):
        paths[:, j + 1] = paths[:, j] * (1 + rr[starts + j])

    sigma = vol.to_numpy()[starts]
    mid, _ = _straddle(1.0, HOLD / TRADING_DAYS, sigma)
    # 卖出时加上半个价差
    premium = mid * (1 + half_spread)

    hedge = np.zeros(len(starts))
    for j in range(HOLD):
        _, delta = _straddle(paths[:, j], (HOLD - j) / TRADING_DAYS, sigma)
        hedge += delta * (paths[:, j + 1] - paths[:, j])

    payoff = np.abs(paths[:, -1] - 1.0)
    return pd.Series(premium - payoff + hedge, index=r.index[starts], name="pnl")


def tail_risk(pnl: pd.Series, level: float = 0.05) -> float:
    """CVaR：最差 level 比例的月份的平均亏损（正数）。"""
    a = level
    return float(-(pnl.mean() - pnl.std() * norm.pdf(norm.ppf(a)) / a))


def evaluate(returns: pd.Series) -> dict:
    """完整评估，返回研究结论。

    返回字典：
        sharpe       年化 Sharpe（月度 P&L）
        tstat        均值 P&L 的 t 统计量
        skew         月度 P&L 的偏度
        worst_month  最差一个月的 P&L
        cvar_95      tail_risk(pnl, 0.05)
        verdict      "deploy" 或 "reject"
    """
    pnl = monthly_pnl(returns)
    sr = pnl.mean() / pnl.std()
    t = sr * np.sqrt(len(pnl))
    cvar = tail_risk(pnl)
    return {
        "sharpe": float(sr * np.sqrt(12)),
        "tstat": float(t),
        "skew": float(pnl.skew()),
        "worst_month": float(pnl.min()),
        "cvar_95": cvar,
        "verdict": "deploy" if (t > 2.0 and cvar < RISK_LIMIT) else "reject",
    }
