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
    2. position() 直接return weights，引入前视偏差。
       t 收盘才算出的权重被当天持有，吃到了算出信号的那天的收益。修正：shift(1)。
       shift 后第一行是 NaN，但期初持仓是确定的 0（空仓），要 fillna(0)：
       NaN 表示“不知道”，而这里是“知道，是 0”。
    3. trend_score() 使用全局标准化，引入前视偏差。
       mom.mean() 默认 axis=0，每只股票用自己全部历史（含未来）的均值和标准差标准化。
       同时方向也错了：docstring 的目的是“股票之间可比”，应做横截面标准化——每天在所有
       股票之间减均值、除标准差，只用当天数据，天然没有前视。
       实现陷阱：mom - mom.mean(axis=1) 会把以日期为索引的 Series 对齐到列名（股票代码），
       结果全为 NaN，再被 fillna(0) 静默变成“永远空仓”，不报错。要写
       mom.sub(mom.mean(axis=1), axis=0).div(mom.std(axis=1), axis=0)。
       检查办法：打印输出的形状，看是否全为 NaN。
    4. transaction_costs() 只对买入收费（clip(lower=0)），卖出免费：换手被低估一半，
       高换手策略的净收益被高估。修正：换手取 abs()，买卖双边都收费。
       另一个隐蔽处：diff() 第 0 天是 NaN，sum() 会静默跳过，期初从现金建仓的那笔交易
       没有收费。第 0 天的换手应等于第 0 天的持仓：pos.diff().fillna(pos)。
       原则：函数要自己对“期初空仓”负责，不能依赖上游恰好把第一行填成 0。
    经验：diff()/shift()/pct_change() 都在开头产生 NaN，sum()/mean() 默认跳过 NaN；
    每次都要问这个 NaN 实际代表什么。注释与代码不一致也要一并修正。
"""

from __future__ import annotations

import pandas as pd

from quantlab import perf

LOOKBACK = 120
COST_BPS = 10.0


def clean_universe(prices: pd.DataFrame) -> pd.DataFrame:
    """数据清洗。"""
    return prices


def trend_score(prices: pd.DataFrame, lookback: int = LOOKBACK) -> pd.DataFrame:
    """趋势分数：过去 lookback 天的涨幅，标准化后不同股票之间可比。"""
    mom = prices / prices.shift(lookback) - 1
    return mom.sub(mom.mean(axis=1), axis=0).div(mom.std(axis=1), axis=0)


def target_weights(score: pd.DataFrame) -> pd.DataFrame:
    """只买分数为正的股票，按分数加权，权重和为 1（全部为负时空仓）。"""
    long = score.clip(lower=0)
    return long.div(long.sum(axis=1), axis=0).fillna(0.0)


def positions(weights: pd.DataFrame) -> pd.DataFrame:
    """实际持仓。"""
    # t 收盘算出的目标权重，t+1 才持有；第 0 天没有信号，期初空仓
    return weights.shift(1).fillna(0)


def transaction_costs(pos: pd.DataFrame, cost_bps: float = COST_BPS) -> pd.Series:
    """每天的交易成本（占组合净值的比例）。"""
    # 买卖双边都支付佣金和冲击成本，换手 = |今天持仓 - 昨天持仓|
    # 第 0 天没有“昨天”，diff() 给出 NaN；期初是空仓，所以第 0 天的换手就是第 0 天的持仓本身
    turnover = pos.diff().fillna(pos).abs()
    return turnover.sum(axis=1) * cost_bps / 1e4


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
