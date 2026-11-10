"""Lab 11 审计题参考答案：audit.py 修正后的版本。

修正清单：
1. estimate_adv 用均值：成交额分布右偏，少数放量日（指数调仓、消息日）把均值拉高约 1.6 倍，
   而你不能指望每天都是放量日。用中位数（或剔除放量日后的均值）。
2. calibrate_impact 以市场 VWAP 为基准：我们自己的交易推动了价格，
   当天其他人也在被推动后的价格上成交，VWAP 跟着我们一起移动，
   于是“相对 VWAP 的滑点”几乎看不到我们自己的冲击。
   执行成本（implementation shortfall）要相对决策时的价格（arrival price）来衡量。
3. daily_turnover 除以 2：调仓时买入和卖出各自都要付价差和冲击，成本按双边成交额计算。
4. alpha_capture 假设当天全部成交：参与率上限 5% 意味着更大的单子要分几天执行，
   而 alpha 在以 2 天的半衰期衰减。延后成交的部分只能吃到衰减后的 alpha。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.execution import StrategyMarket

TRADING_DAYS = 252
HALF_SPREAD_BPS = 3.0  # 半个买卖价差
PARTICIPATION_CAP = 0.05  # 风控：单只股票每天的成交不超过当日成交额的 5%
ALPHA_HALF_LIFE = 2.0  # 信号的 alpha 半衰期（交易日）
TARGET_SHARPE = 1.0  # 上线门槛：净 Sharpe 不低于 1.0
AUM_GRID = np.geomspace(1e7, 1e10, 31)


def estimate_adv(volume: pd.DataFrame) -> pd.Series:
    """每只股票“典型的一天”的成交额：中位数，不被少数放量日主导。"""
    return volume.median()


def calibrate_impact(exec_log: pd.DataFrame) -> float:
    """用历史成交记录校准平方根冲击模型：成本 = Y * sigma * sqrt(participation)。

    成本用相对决策价格（arrival price）的 implementation shortfall 衡量。
    最小二乘、不带截距，返回 Y。
    """
    slip = exec_log["side"] * (exec_log["avg_price"] / exec_log["arrival"] - 1)
    x = exec_log["sigma"] * np.sqrt(exec_log["participation"])
    return float((x * slip).sum() / (x * x).sum())


def daily_turnover(weights: pd.DataFrame) -> pd.Series:
    """每日成交额占 AUM 的比例（双边：买入和卖出都要付成本）。"""
    return weights.diff().abs().sum(axis=1).fillna(0.0)


def alpha_capture(participation: float) -> float:
    """在给定参与率下，实际能吃到的 alpha 比例。

    单票每日成交不超过 PARTICIPATION_CAP，所以参与率为 p 的单子要分 ceil(p / cap) 天执行，
    每天执行相同份额；第 k 天成交的部分只能吃到 0.5 ** (k / 半衰期) 的 alpha。
    """
    days = max(1, int(np.ceil(participation / PARTICIPATION_CAP)))
    return float(np.mean(0.5 ** (np.arange(days) / ALPHA_HALF_LIFE)))


def net_returns(market: StrategyMarket, aum: float) -> pd.Series:
    """给定 AUM 下扣除成本后的日收益。"""
    adv = estimate_adv(market.volume)
    coef = calibrate_impact(market.exec_log)
    turnover = daily_turnover(market.weights)
    n_names = market.weights.shape[1]

    # 每天每只股票的典型成交额 / 该股 ADV
    participation = (turnover * aum / n_names).to_numpy()[:, None] / adv.to_numpy()[None, :]
    impact_bps = coef * market.sigma.to_numpy()[None, :] * np.sqrt(participation) * 1e4
    cost_bps = (HALF_SPREAD_BPS + impact_bps).mean(axis=1)
    cost = turnover * cost_bps / 1e4

    capture = alpha_capture(float(np.median(participation)))
    return market.gross * capture - cost


def annualized_sharpe(r: pd.Series) -> float:
    return float(r.mean() / r.std(ddof=1) * np.sqrt(TRADING_DAYS))


def capacity(market: StrategyMarket, aums=AUM_GRID) -> float:
    """净 Sharpe 仍不低于 TARGET_SHARPE 的最大 AUM（网格上）。不满足时返回 0。"""
    ok = [a for a in aums if annualized_sharpe(net_returns(market, a)) >= TARGET_SHARPE]
    return float(max(ok)) if ok else 0.0


def evaluate(market: StrategyMarket, aum: float = 5e8) -> dict:
    """完整评估。

    返回字典：
        gross_sharpe  纸面 Sharpe
        net_sharpe    给定 AUM 下的净 Sharpe
        cost_annual   给定 AUM 下每年的成本（占 AUM 比例）
        impact_coef   校准出的冲击系数 Y
        capacity      容量（净 Sharpe >= TARGET_SHARPE 的最大 AUM）
        verdict       "deploy" 或 "reject"
    """
    net = net_returns(market, aum)
    net_sr = annualized_sharpe(net)
    return {
        "gross_sharpe": annualized_sharpe(market.gross),
        "net_sharpe": net_sr,
        "cost_annual": float((market.gross - net).mean() * TRADING_DAYS),
        "impact_coef": calibrate_impact(market.exec_log),
        "capacity": capacity(market),
        "verdict": "deploy" if net_sr >= TARGET_SHARPE else "reject",
    }
