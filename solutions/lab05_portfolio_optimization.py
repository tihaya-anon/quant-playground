"""Lab 05 审计题参考答案：audit.py 修正后的版本。

修正清单：
1. backtest 用全部历史一次估计参数：每个再平衡日用到了它之后的数据，是前视。
   切点组合在样本内“恰好”是最优的，所以样本内 Sharpe 一定好看。
   每个再平衡日只能用它之前 WINDOW 天的数据估计。
2. tangency 用 x / x.sum() 归一化：当 1'Σ^{-1}mu < 0 时（估计的 mu 普遍为负，或多空抵消），
   除以负数会把整个组合翻过来，变成预期收益为负的组合。按总敞口归一化：x / |x|.sum()。
3. 忽略成本：每月重新估计的切点组合权重变化很大，换手很高。按换手扣成本。
4. 报告的波动率是优化器的事前估计 w'Σw：优化器恰恰挑中了估计误差最有利的方向，
   事前风险系统性地低估事后风险。报告实际实现的波动率。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252
WINDOW = 126  # 用最近半年的数据估计参数，反映最新的市场状况
COST_BPS = 10.0


def estimate_inputs(window: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """由一段日收益估计年化的期望收益向量与协方差矩阵。"""
    mu = window.mean().to_numpy() * TRADING_DAYS
    cov = window.cov().to_numpy() * TRADING_DAYS
    return mu, cov


def tangency(mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """切点组合（最大 Sharpe）：w ∝ Σ^{-1} mu。"""
    x = np.linalg.solve(cov, mu)
    return x / np.abs(x).sum()


def rebalance_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """每月最后一个交易日。"""
    s = pd.Series(index, index=index)
    return pd.DatetimeIndex(s.groupby([index.year, index.month]).max().to_numpy())


def backtest(
    returns: pd.DataFrame, cost_bps: float = COST_BPS
) -> tuple[pd.Series, pd.DataFrame, float]:
    """每月再平衡的切点组合。返回 (日净收益, 每个再平衡日的目标权重, 组合的年化波动率)。"""
    dates = [d for d in rebalance_dates(returns.index) if returns.index.get_loc(d) >= WINDOW]

    rows, predicted = [], []
    for d in dates:
        window = returns.loc[:d].iloc[-WINDOW:]
        mu, cov = estimate_inputs(window)
        w = tangency(mu, cov)
        rows.append(w)
        predicted.append(float(np.sqrt(w @ cov @ w)))  # 事前风险：只用于对比，不对外报告
    targets = pd.DataFrame(rows, index=pd.DatetimeIndex(dates), columns=returns.columns)

    held = targets.reindex(returns.index).ffill().shift(1)
    held = held.loc[held.notna().all(axis=1)]
    gross = (held * returns.loc[held.index]).sum(axis=1)
    turnover = held.diff().abs().sum(axis=1).fillna(held.iloc[0].abs().sum())
    net = gross - turnover * cost_bps / 1e4

    vol = float(net.std(ddof=1) * np.sqrt(TRADING_DAYS))
    return net, targets, vol


def equal_weight(returns: pd.DataFrame) -> pd.Series:
    """每日再平衡的等权组合（对照组）。"""
    return returns.mean(axis=1)


def evaluate(returns: pd.DataFrame, cost_bps: float = COST_BPS) -> dict:
    """返回字典：
    sharpe         策略年化 Sharpe（扣成本后）
    vol            报告的组合年化波动率
    ew_sharpe      同期等权组合的年化 Sharpe
    daily_returns  策略日收益（pd.Series）
    weights        每个再平衡日的目标权重（pd.DataFrame）
    verdict        "beats 1/N" 或 "no better than 1/N"
    """
    net, targets, vol = backtest(returns, cost_bps)
    ew = equal_weight(returns).loc[net.index]

    def ann_sharpe(r: pd.Series) -> float:
        return float(r.mean() / r.std(ddof=1) * np.sqrt(TRADING_DAYS))

    s, s_ew = ann_sharpe(net), ann_sharpe(ew)
    return {
        "sharpe": s,
        "vol": vol,
        "ew_sharpe": s_ew,
        "daily_returns": net,
        "weights": targets,
        "verdict": "beats 1/N" if s > s_ew + 0.3 else "no better than 1/N",
    }
