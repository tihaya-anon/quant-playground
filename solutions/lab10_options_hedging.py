"""Lab 10 审计题参考答案：audit.py 修正后的版本。

修正清单：
1. selling_vol 前视偏差：用期权存续期（未来 21 天）的已实现波动率给期权定价，等于卖出时就知道
   未来会有多波动——每份期权的隐含波动率都恰好比未来实现的高 10%，卖方几乎稳赚。
   卖出时只能用当时已知的信息：过去 21 天的已实现波动率（再 shift 一天）。
2. clean_returns 把崩盘当成数据错误删掉了：卖波动率策略的风险恰恰集中在这些日子里。
   超过 5 个标准差的日收益在真实市场里不罕见（肥尾），不能仅凭大小判定为错误；
   要核对数据源（是否除权、是否错价），而不是按阈值一刀切。
3. 价差方向错了：卖出成交在买价（bid），权利金是中间价减去半个价差，不是加上。
4. tail_risk 用正态分布算 CVaR：卖保险型策略的收益严重负偏，正态假设系统性低估尾部亏损。
   用历史 CVaR（最差 5% 月份的实际平均亏损）。
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
    """大收益不是数据错误的证据；只处理真正的缺失值。"""
    return returns.fillna(0.0)


def selling_vol(returns: pd.Series) -> pd.Series:
    """第 t 天卖出时的隐含波动率：用截至 t-1 的过去 HOLD 天已实现波动率乘以 VRP。"""
    trailing = returns.rolling(HOLD).std().shift(1)
    return trailing * np.sqrt(TRADING_DAYS) * VRP


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
    premium = mid * (1 - half_spread)  # 卖出成交在买价

    hedge = np.zeros(len(starts))
    for j in range(HOLD):
        _, delta = _straddle(paths[:, j], (HOLD - j) / TRADING_DAYS, sigma)
        hedge += delta * (paths[:, j + 1] - paths[:, j])

    payoff = np.abs(paths[:, -1] - 1.0)
    return pd.Series(premium - payoff + hedge, index=r.index[starts], name="pnl")


def tail_risk(pnl: pd.Series, level: float = 0.05) -> float:
    """历史 CVaR：最差 level 比例的月份的实际平均亏损（正数）。"""
    q = pnl.quantile(level)
    return float(-pnl[pnl <= q].mean())


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
