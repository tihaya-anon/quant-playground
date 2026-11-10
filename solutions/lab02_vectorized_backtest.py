"""Lab 02 审计题参考答案：audit.py 修正后的版本。

修正清单：
1. clean_universe：剔除“数据不完整”的股票，等于只保留了活到最后的股票——幸存者偏差。
   一只股票是否会退市，在回测的每一个时点都是未来信息。退市前它正常参与，退市后自然没有仓位。
2. trend_score：mom.mean() / mom.std() 是按列对**整段历史**求的，用到了未来数据。
   要么横截面标准化（每天在股票之间），要么只用截至当天的数据（expanding/rolling）。这里用横截面。
3. positions：用 t 日收盘价算出的信号，最早只能在 t 日收盘之后成交，吃到 t+1 的收益。
   直接把当天的权重乘当天的收益，就是在用收盘价预测已经发生的收益。要 shift(1)。
4. transaction_costs：卖出同样要付佣金、冲击成本（A 股还有印花税）。换手是 |Δ持仓|，买卖都算。
"""

from __future__ import annotations

import pandas as pd

from quantlab import perf

LOOKBACK = 120
COST_BPS = 10.0


def clean_universe(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.dropna(axis=1, how="all")


def trend_score(prices: pd.DataFrame, lookback: int = LOOKBACK) -> pd.DataFrame:
    mom = prices / prices.shift(lookback) - 1
    return mom.sub(mom.mean(axis=1), axis=0).div(mom.std(axis=1), axis=0)


def target_weights(score: pd.DataFrame) -> pd.DataFrame:
    long = score.clip(lower=0)
    return long.div(long.sum(axis=1), axis=0).fillna(0.0)


def positions(weights: pd.DataFrame) -> pd.DataFrame:
    return weights.shift(1).fillna(0.0)


def transaction_costs(pos: pd.DataFrame, cost_bps: float = COST_BPS) -> pd.Series:
    turnover = pos.diff().fillna(pos).abs().sum(axis=1)
    return turnover * cost_bps / 1e4


def strategy_positions(prices: pd.DataFrame) -> pd.DataFrame:
    universe = clean_universe(prices)
    return positions(target_weights(trend_score(universe)))


def backtest(prices: pd.DataFrame, cost_bps: float = COST_BPS) -> pd.DataFrame:
    pos = strategy_positions(prices)
    asset_ret = (prices[pos.columns] / prices[pos.columns].shift(1) - 1).fillna(0.0)
    gross = (pos * asset_ret).sum(axis=1)
    cost = transaction_costs(pos, cost_bps)
    return pd.DataFrame({"gross": gross, "cost": cost, "net": gross - cost}).iloc[LOOKBACK:]


def report(prices: pd.DataFrame) -> dict:
    res = backtest(prices)
    return {
        "sharpe_gross": perf.sharpe(res["gross"]),
        "sharpe_net": perf.sharpe(res["net"]),
        "ann_cost": float(res["cost"].mean() * 252),
    }
