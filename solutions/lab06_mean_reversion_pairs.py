"""Lab 06 审计题参考答案：audit.py 修正后的版本。

修正清单：
1. cointegration_pvalue 对估计出的价差直接做 ADF：对冲比例是用 OLS 挑出来的，
   OLS 恰好选了让残差方差最小、看起来最平稳的那个组合，所以普通 ADF 临界值过于宽松
   （两只独立随机游走，约 18% 会被判为“协整”）。要用 Engle-Granger 临界值（statsmodels coint）。
2. find_best_pair 扫了上百对，却用 p < 0.05 判断：这是多重检验。在没有任何协整关系的股票池里，
   最小的 p 值几乎总是小于 0.05。至少要用 Bonferroni 校正：p < 0.05 / 检验次数。
3. backtest_pair 用全样本估计对冲比例和 z-score 的均值、标准差：样本外区间的数据参与了参数估计，
   是前视。对冲比例只能用训练段估计；z-score 用滚动窗口（只用过去）。
4. 忽略成本：配对交易一次进出场要交易两只股票、四笔单子，成本不可忽略。
"""

from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(coint(log_y.to_numpy(), log_x.to_numpy(), maxlag=1, autolag=None)[1])


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
    b, a = hedge_ratio(log_y.iloc[:train_end], log_x.iloc[:train_end])
    spread = log_y - b * log_x - a
    z = pairs.rolling_zscore(spread, Z_WINDOW)

    pos = pairs.pair_positions(z.iloc[train_end:])
    y = np.exp(log_y.iloc[train_end:])
    x = np.exp(log_x.iloc[train_end:])
    return pairs.pair_returns(y, x, b, pos, cost_bps=cost_bps)


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
        "verdict": "cointegrated" if (p < 0.05 / n and sharpe > 0) else "not cointegrated",
    }
