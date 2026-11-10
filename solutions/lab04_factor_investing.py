"""Lab 04 审计题参考答案：audit.py 修正后的版本。

修正清单：
1. daily_ic 同日对齐：5 日动量里包含了当天的收益，和当天收益算相关性是“用答案预测答案”。
   t 日的因子只能和 t+1 日的收益比（quantlab.factors.information_coefficient）。
2. quantile_buckets 用全样本分位数分组：分组边界由未来的因子分布决定，是前视。
   而且因子分布随时间漂移时，某些天会整组为空。每天按当天横截面排名分组。
3. clean_universe 只保留全程有数据的股票：退市股票整只消失，连同退市前的暴跌一起——
   幸存者偏差。股票应留在回测里直到退市那天，退市当天的亏损必须计入。
4. long_short 忽略成本：组合每天都在换仓，成本 = 换手 × 单边费率，必须扣除。
5. factor_alpha_tstat 把多空平均收益当 alpha：多空组合对冲了市场，但没有对冲小盘因子。
   因子里混入了小盘暴露，赚的是已知的小盘溢价。alpha 要控制市场与小盘因子后回归得到。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab import factors
from quantlab.factors import Universe

TRADING_DAYS = 252
N_QUANTILES = 5
COST_BPS = 10.0


def build_factor(returns: pd.DataFrame, signal: pd.DataFrame) -> pd.DataFrame:
    """复合因子 = 横截面标准化的基本面信号 + 横截面标准化的 5 日动量。"""
    momentum = returns.rolling(5).sum()
    return factors.cross_sectional_zscore(signal) + factors.cross_sectional_zscore(momentum)


def clean_universe(
    returns: pd.DataFrame, factor: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """数据清洗：保留全部股票，退市后的 NaN 在排序与持仓里自然被跳过。"""
    return returns, factor


def daily_ic(factor: pd.DataFrame, returns: pd.DataFrame) -> pd.Series:
    """每天的横截面 Spearman IC。"""
    return factors.information_coefficient(factor, returns).dropna()


def quantile_buckets(factor: pd.DataFrame, n: int = N_QUANTILES) -> pd.DataFrame:
    """把因子值分成 1..n 组（n 为因子最高组），NaN 保持 NaN。"""
    pct = factor.rank(axis=1, pct=True)
    return np.ceil(pct * n).clip(1, n)


def long_short(
    factor: pd.DataFrame, returns: pd.DataFrame, cost_bps: float = COST_BPS
) -> tuple[pd.Series, pd.Series]:
    """做多最高组、做空最低组（组内等权）。返回 (毛收益, 净收益)。"""
    held = quantile_buckets(factor).shift(1)  # t 收盘分组，t+1 持有
    is_long = (held == N_QUANTILES).astype(float)
    is_short = (held == 1).astype(float)
    w = is_long.div(is_long.sum(axis=1), axis=0) - is_short.div(is_short.sum(axis=1), axis=0)
    w = w.fillna(0.0)
    gross = (w * returns.fillna(0.0)).sum(axis=1).iloc[N_QUANTILES + 1 :]
    turnover = w.diff().abs().sum(axis=1).iloc[N_QUANTILES + 1 :]
    net = gross - turnover * cost_bps / 1e4
    return gross, net


def factor_alpha_tstat(ls: pd.Series, market: pd.Series, size_factor: pd.Series) -> float:
    """多空组合 alpha 的 t 值。"""
    df = pd.concat([ls, market, size_factor], axis=1, join="inner").dropna()
    res = factors.ols(df.iloc[:, 0], df.iloc[:, 1:])
    return float(res["tstats"][0])


def evaluate(u: Universe, cost_bps: float = COST_BPS) -> dict:
    """完整的因子评估流程。

    返回字典：
        ic_mean, ic_t        IC 均值与 t 值
        ls_sharpe_gross      多空组合年化 Sharpe（扣成本前）
        ls_sharpe_net        多空组合年化 Sharpe（扣成本后）
        alpha_t              控制已知因子后的 alpha t 值
        universe_size        回测中平均每天参与排序的股票数
        verdict              "alpha" 或 "no alpha"
    """
    factor = build_factor(u.returns, u.signal)
    returns, factor = clean_universe(u.returns, factor)

    ic = daily_ic(factor, returns)
    gross, net = long_short(factor, returns, cost_bps)
    alpha_t = factor_alpha_tstat(net, u.market, u.size_factor)

    def ann_sharpe(r: pd.Series) -> float:
        return float(r.mean() / r.std(ddof=1) * np.sqrt(TRADING_DAYS))

    return {
        "ic_mean": float(ic.mean()),
        "ic_t": float(ic.mean() / ic.std(ddof=1) * np.sqrt(len(ic))),
        "ls_sharpe_gross": ann_sharpe(gross),
        "ls_sharpe_net": ann_sharpe(net),
        "alpha_t": alpha_t,
        "universe_size": float(factor.notna().sum(axis=1).mean()),
        "verdict": "alpha" if (alpha_t > 3.0 and ann_sharpe(net) > 0) else "no alpha",
    }
