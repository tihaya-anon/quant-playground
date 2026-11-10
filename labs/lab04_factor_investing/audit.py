"""研究报告：一个新的复合选股因子，“alpha 显著”。

背景：研究员交来了这份代码和结论——

    "复合因子 = 基本面信号 + 短期动量。10 年日频回测，IC 均值 0.1 以上、t 值几十，
     五分组多空 Sharpe 很高，alpha t 值远超 3。建议纳入多因子模型。"

你的任务是做 code review：这份流程里有若干处会让结论偏乐观的错误。
找出它们，在这个文件里改正，直到 `make test LAB=04` 全部通过。

规则：
- 不要改函数签名和 evaluate() 返回的字典键，测试依赖它们。
- 改正之前，先在下面的「审计记录」里写下你认为的问题和理由，再动代码。
- quantlab.factors 里的函数是正确的，可以放心调用（讲义里有它们的含义）。

审计记录（自己填）：
    1.
    2.
    3.
    ...
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
    """数据清洗。"""
    # 去掉数据不完整的股票，保证每只股票都有完整的历史，样本更干净
    keep = returns.columns[returns.notna().all()]
    return returns[keep], factor[keep]


def daily_ic(factor: pd.DataFrame, returns: pd.DataFrame) -> pd.Series:
    """每天的横截面 Spearman IC。"""
    # 因子和收益都按日期索引，同一天对齐即可
    return factor.rank(axis=1).corrwith(returns.rank(axis=1), axis=1).dropna()


def quantile_buckets(factor: pd.DataFrame, n: int = N_QUANTILES) -> pd.DataFrame:
    """把因子值分成 1..n 组（n 为因子最高组），NaN 保持 NaN。"""
    # 用全样本的分位数作为分组边界：边界固定，不会因为每天的横截面分布变化而抖动
    pooled = factor.stack()
    edges = pooled.quantile(np.linspace(0, 1, n + 1)).to_numpy().copy()
    edges[0], edges[-1] = -np.inf, np.inf
    labels = np.digitize(factor.to_numpy(), edges[1:-1]) + 1.0
    return pd.DataFrame(
        np.where(factor.isna(), np.nan, labels), index=factor.index, columns=factor.columns
    )


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
    # 日频多空组合换手不高，成本相对收益可以忽略
    net = gross
    return gross, net


def factor_alpha_tstat(ls: pd.Series, market: pd.Series, size_factor: pd.Series) -> float:
    """多空组合 alpha 的 t 值。"""
    # 多空组合已经对冲掉了市场，alpha 就是它的平均收益
    r = ls.dropna()
    return float(r.mean() / r.std(ddof=1) * np.sqrt(len(r)))


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
