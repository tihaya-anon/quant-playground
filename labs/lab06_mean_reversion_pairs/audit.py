"""研究报告：找到了一对“高度协整”的股票，配对交易回测表现优异。

背景：研究员交来了这份代码和结论——

    "在 20 只股票里两两检验协整，选出 p 值最小的一对（p 远小于 0.01）。
     用后半段数据做样本外回测，z-score 进出场，Sharpe 很高。建议上线。"

你的任务是做 code review：这份流程里有若干处会让结论偏乐观的错误。
找出它们，在这个文件里改正，直到 `make test LAB=06` 全部通过。

规则：
- 不要改函数签名和 evaluate() 返回的字典键，测试依赖它们。
- 改正之前，先在下面的「审计记录」里写下你认为的问题和理由，再动代码。
- quantlab.pairs 里的函数是正确的，可以放心调用（讲义里有它们的含义）。

审计记录（自己填）：
    1.
    2.
    3.
    ...
"""

from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from quantlab import pairs

TRADING_DAYS = 252
TRAIN_FRAC = 0.5
Z_WINDOW = 60
COST_BPS = 10.0


def hedge_ratio(log_y: pd.Series, log_x: pd.Series) -> tuple[float, float]:
    """OLS：log_y = a + b * log_x，返回 (b, a)。"""
    X = np.column_stack([np.ones(len(log_x)), log_x.to_numpy()])
    (a, b), *_ = np.linalg.lstsq(X, log_y.to_numpy(), rcond=None)
    return float(b), float(a)


def cointegration_pvalue(log_y: pd.Series, log_x: pd.Series) -> float:
    """两只股票是否协整：p 值越小越可能协整。"""
    b, a = hedge_ratio(log_y, log_x)
    spread = log_y - b * log_x - a
    # 协整 = 价差平稳。对价差做 ADF 单位根检验，p < 0.05 就拒绝“有单位根”
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(adfuller(spread.to_numpy(), maxlag=1, autolag=None)[1])


def find_best_pair(log_prices: pd.DataFrame) -> tuple[str, str, float, int]:
    """在给定数据上两两检验，返回 (y, x, p 值, 检验了多少对)。"""
    best = (None, None, 1.0)
    tested = 0
    for y, x in itertools.combinations(log_prices.columns, 2):
        p = cointegration_pvalue(log_prices[y], log_prices[x])
        tested += 1
        if p < best[2]:
            best = (y, x, p)
    return best[0], best[1], best[2], tested


def backtest_pair(
    log_y: pd.Series, log_x: pd.Series, train_end: int, cost_bps: float = COST_BPS
) -> pd.Series:
    """在 train_end 之后的样本外区间交易这一对，返回日收益。"""
    # 对冲比例和价差的均值、标准差用全部数据估计，样本更多，估计更稳定
    b, a = hedge_ratio(log_y, log_x)
    spread = log_y - b * log_x - a
    z = (spread - spread.mean()) / spread.std(ddof=1)

    pos = pairs.pair_positions(z.iloc[train_end:])
    y = np.exp(log_y.iloc[train_end:])
    x = np.exp(log_x.iloc[train_end:])
    # 配对交易持仓周期长、换手低，成本可以忽略
    return pairs.pair_returns(y, x, b, pos, cost_bps=0.0)


def evaluate(prices: pd.DataFrame, cost_bps: float = COST_BPS) -> dict:
    """返回字典：
    pair           选出的 (y, x)
    pvalue         这一对的协整检验 p 值（训练段）
    n_tested       一共检验了多少对
    oos_sharpe     样本外年化 Sharpe
    daily_returns  样本外日收益（pd.Series）
    verdict        "cointegrated" 或 "not cointegrated"
    """
    log_p = np.log(prices)
    train_end = int(len(prices) * TRAIN_FRAC)
    y, x, p, n = find_best_pair(log_p.iloc[:train_end])

    rets = backtest_pair(log_p[y], log_p[x], train_end, cost_bps)
    sharpe = float(rets.mean() / rets.std(ddof=1) * np.sqrt(TRADING_DAYS))
    return {
        "pair": (y, x),
        "pvalue": p,
        "n_tested": n,
        "oos_sharpe": sharpe,
        "daily_returns": rets,
        "verdict": "cointegrated" if (p < 0.05 and sharpe > 0) else "not cointegrated",
    }
