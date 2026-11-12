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
    1. clean_universe() 引入幸存者偏差：只保留全程有数据的股票，退市股票连同退市前的暴跌
       一起消失。股票要留在回测里直到退市那天，退市当天的亏损必须计入。
    2. long_short() 不能忽略成本：“日频多空换手不高”不成立，组合每天都在换仓。
       成本 = 换手（买卖双边）× 单边费率。
    3. factor_alpha_tstat() 还有小盘因子不能忽略：多空组合对冲了市场，但没有对冲小盘因子，
       因子里混入的小盘暴露赚的是已知溢价。alpha 要对市场 + 小盘因子回归取截距。
    4. （复盘时由测试发现）daily_ic 同日对齐：5 日动量包含当天收益，和当天收益算相关性是
       用答案预测答案。t 日因子只能和 t+1 日收益比。
    5. （复盘时由测试发现）quantile_buckets 用全样本分位数分组：边界由未来的因子分布决定，
       是前视。每天只按当天横截面排名分组。
    复盘：4、5 都属于“时间隔离”这一类机制，lab02、lab03 都处理过，这次因为注释把它们
    说成了“对齐”和“稳定”而没认出来。检查清单：凡是全样本统计量（stack/quantile/mean/std）
    和同日对齐，都要问一句“t 时刻真的知道这个吗”。
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
    """数据清洗：保留全部股票。退市后的 NaN 在排序与持仓里自然被跳过，退市当天的亏损照常计入。"""
    return returns, factor


def daily_ic(factor: pd.DataFrame, returns: pd.DataFrame) -> pd.Series:
    """每天的横截面 Spearman IC。"""
    # t 日的因子只能和 t+1 日的收益比（information_coefficient 内部做了错位）
    return factors.information_coefficient(factor, returns).dropna()


def quantile_buckets(factor: pd.DataFrame, n: int = N_QUANTILES) -> pd.DataFrame:
    """把因子值分成 1..n 组（n 为因子最高组），NaN 保持 NaN。"""
    # 每天只按当天的横截面排名分组，边界不能用到未来的因子分布
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
    # 成本 = 换手（买卖双边）× 单边费率
    turnover = w.diff().abs().sum(axis=1).iloc[N_QUANTILES + 1 :]
    net = gross - turnover * cost_bps / 1e4
    return gross, net


def factor_alpha_tstat(ls: pd.Series, market: pd.Series, size_factor: pd.Series) -> float:
    """多空组合 alpha 的 t 值。"""
    # 对市场和小盘因子做时间序列回归，截距才是控制了已知因子之后的 alpha
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
