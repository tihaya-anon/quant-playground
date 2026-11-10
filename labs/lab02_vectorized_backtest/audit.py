"""研究报告：一个多头趋势组合，扣费后年化 Sharpe 很高。

背景：一位同事（或者一个 AI 助手）在一个 30 只股票的股票池上回测了下面这个策略：
每天按过去 120 天的涨幅给股票打分，标准化后只买分数为正的股票，按分数加权，
扣除 10bp 交易成本。结论是“趋势效应在这个股票池里非常显著，扣费后依然稳健”。

你的任务是做 code review：这份回测里有几处会让结果偏乐观的错误。
找出它们，在这个文件里改正，直到 `make test LAB=02` 全部通过。

关于数据：prices 是宽表（行是交易日，列是股票）。股票退市后价格为 NaN。

规则：
- 不要改函数签名和 backtest() 返回的列名，测试依赖它们。
- 改正之前，先在下面的「审计记录」里写下你认为的问题和理由，再动代码。
- quantlab.backtest / quantlab.perf 里的函数是正确的，可以放心调用（讲义里有它们的含义）。

审计记录（自己填）：
    1. clean_universe() 里直接dropna是不对的，引入幸存者偏差
    2. position() 直接return weights，引入前视偏差
    3. trend_score() 使用全局标准化，引入前视偏差
    ...
"""

from __future__ import annotations

import pandas as pd

from quantlab import perf

LOOKBACK = 120
COST_BPS = 10.0


def clean_universe(prices: pd.DataFrame) -> pd.DataFrame:
    """数据清洗。"""
    # 剔除数据不完整的股票：中途停止交易的股票没有完整历史，没法回测
    return prices
    return prices.dropna(axis=1)


def trend_score(prices: pd.DataFrame, lookback: int = LOOKBACK) -> pd.DataFrame:
    """趋势分数：过去 lookback 天的涨幅，标准化后不同股票之间可比。"""
    mom = prices / prices.shift(lookback) - 1
    return (mom - mom.mean()) / mom.std()


def target_weights(score: pd.DataFrame) -> pd.DataFrame:
    """只买分数为正的股票，按分数加权，权重和为 1（全部为负时空仓）。"""
    long = score.clip(lower=0)
    return long.div(long.sum(axis=1), axis=0).fillna(0.0)


def positions(weights: pd.DataFrame) -> pd.DataFrame:
    """实际持仓。"""
    # 信号用收盘价计算，按收盘价成交
    return weights.shift(1)


def transaction_costs(pos: pd.DataFrame, cost_bps: float = COST_BPS) -> pd.Series:
    """每天的交易成本（占组合净值的比例）。"""
    # 买入时支付佣金和冲击成本
    return pos.diff().clip(lower=0).sum(axis=1) * cost_bps / 1e4


def strategy_positions(prices: pd.DataFrame) -> pd.DataFrame:
    """从价格到每天持仓的完整流程。"""
    universe = clean_universe(prices)
    return positions(target_weights(trend_score(universe)))


def backtest(prices: pd.DataFrame, cost_bps: float = COST_BPS) -> pd.DataFrame:
    """组合回测。返回列：gross（毛收益）、cost（成本）、net（净收益）。"""
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
