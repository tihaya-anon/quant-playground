"""研究报告：高换手选股策略的交易成本与容量。

背景：一位同事（或者一个 AI 助手）交来了这份容量分析和结论——

    "策略纸面 Sharpe 1.5。用我们自己的历史成交记录校准了冲击成本模型，扣除价差和冲击之后，
     在 5 亿规模下净 Sharpe 仍有 1.4 以上；容量超过 20 亿。建议以 5 亿规模上线。"

你的任务是做 code review：这份分析里有若干处会让成本被低估、容量被高估的错误。
找出它们，在这个文件里改正，直到 `make test LAB=11` 全部通过。

规则：
- 不要改函数签名和 evaluate() 返回的字典键，测试依赖它们。
- 改正之前，先在下面的「审计记录」里写下你认为的问题和理由，再动代码。
- quantlab.execution 里的函数是正确的，可以放心调用。

审计记录（自己填）：
    1.
    2.
    3.
    ...
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
    """每只股票的日均成交额（ADV）。"""
    return volume.mean()


def calibrate_impact(exec_log: pd.DataFrame) -> float:
    """用历史成交记录校准平方根冲击模型：成本 = Y * sigma * sqrt(participation)。

    成本用成交均价相对市场 VWAP 的滑点衡量（行业通用的执行质量基准）。
    最小二乘、不带截距，返回 Y。
    """
    slip = exec_log["side"] * (exec_log["avg_price"] / exec_log["market_vwap"] - 1)
    x = exec_log["sigma"] * np.sqrt(exec_log["participation"])
    return float((x * slip).sum() / (x * x).sum())


def daily_turnover(weights: pd.DataFrame) -> pd.Series:
    """每日换手率（单边）：买入和卖出是同一笔调仓的两面，只算一边。"""
    return weights.diff().abs().sum(axis=1).fillna(0.0) / 2


def alpha_capture(participation: float) -> float:
    """在给定参与率下，实际能吃到的 alpha 比例。

    风控要求单票每日成交不超过 PARTICIPATION_CAP。我们用算法单在收盘前完成全部成交，
    信号当天就兑现，所以不存在 alpha 损失。
    """
    return 1.0


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
