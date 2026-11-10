"""研究报告：均值方差最优组合“大幅跑赢等权”。

背景：研究员交来了这份代码和结论——

    "在 30 个资产上做 Markowitz 切点组合，每月再平衡。20 年回测 Sharpe 远高于等权组合，
     而且优化器给出的组合风险很低，风险可控。建议替换现有的等权配置。"

你的任务是做 code review：这份流程里有若干处会让结论偏乐观的错误。
找出它们，在这个文件里改正，直到 `make test LAB=05` 全部通过。

规则：
- 不要改函数签名和 evaluate() 返回的字典键，测试依赖它们。
- 改正之前，先在下面的「审计记录」里写下你认为的问题和理由，再动代码。
- quantlab.portfolio 里的函数是正确的，可以放心调用（讲义里有它们的含义）。

审计记录（自己填）：
    1.
    2.
    3.
    ...
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
    # 归一化，使权重和为 1（满仓）
    return x / x.sum()


def rebalance_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """每月最后一个交易日。"""
    s = pd.Series(index, index=index)
    return pd.DatetimeIndex(s.groupby([index.year, index.month]).max().to_numpy())


def backtest(
    returns: pd.DataFrame, cost_bps: float = COST_BPS
) -> tuple[pd.Series, pd.DataFrame, float]:
    """每月再平衡的切点组合。返回 (日净收益, 每个再平衡日的目标权重, 组合的年化波动率)。"""
    dates = [d for d in rebalance_dates(returns.index) if returns.index.get_loc(d) >= WINDOW]

    # 参数用全部历史一次估计：数据越多估计越准，而且权重稳定、换手低
    mu, cov = estimate_inputs(returns)
    w = tangency(mu, cov)
    targets = pd.DataFrame([w] * len(dates), index=pd.DatetimeIndex(dates), columns=returns.columns)

    held = targets.reindex(returns.index).ffill().shift(1)
    held = held.loc[held.notna().all(axis=1)]
    gross = (held * returns.loc[held.index]).sum(axis=1)
    # 月度再平衡，换手有限，成本忽略不计
    net = gross

    # 组合风险：优化器给出的 w' Σ w
    vol = float(np.sqrt(w @ cov @ w))
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
