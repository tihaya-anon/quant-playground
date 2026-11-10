"""收益的算术：简单/对数收益、复利、组合收益、GBM 模拟与参数估计。

讲解见 labs/lab00_returns_basics/README.md。约定：一年 252 个交易日；“收益”不加说明时指简单收益。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def simple_returns(prices: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """简单收益 P_t / P_{t-1} - 1，去掉第一行。"""
    return prices.pct_change().iloc[1:]


def log_returns(prices: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """对数收益 ln(P_t / P_{t-1})，去掉第一行。"""
    return np.log(prices / prices.shift(1)).iloc[1:]


def simple_to_log(r):
    """简单收益 -> 对数收益：log1p(r)。"""
    return np.log1p(r)


def log_to_simple(lr):
    """对数收益 -> 简单收益：expm1(lr)。"""
    return np.expm1(lr)


def cumulative_return(returns: pd.Series) -> float:
    """一串简单收益的累计收益：prod(1 + r) - 1。"""
    return float((1 + returns).prod() - 1)


def wealth_index(returns: pd.Series | pd.DataFrame, start: float = 1.0):
    """净值曲线 start * cumprod(1 + r)，与 returns 同索引。"""
    return start * (1 + returns).cumprod()


def resample_returns(returns: pd.Series | pd.DataFrame, rule: str = "ME"):
    """把日收益复利聚合到更低频率（"ME" 月末、"QE"、"YE"、"W-FRI"）。"""
    return (1 + returns).resample(rule).prod() - 1


def rebalanced_portfolio_returns(asset_returns: pd.DataFrame, weights) -> pd.Series:
    """每期再平衡到固定权重的组合收益；权重和小于 1 的部分是收益为 0 的现金。"""
    w = np.asarray(weights, dtype=float)
    return pd.Series(asset_returns.to_numpy() @ w, index=asset_returns.index, name="portfolio")


def buy_and_hold_portfolio_returns(asset_returns: pd.DataFrame, weights) -> pd.Series:
    """期初按权重买入、之后不交易的组合每期收益（权重随涨跌漂移）。"""
    w0 = np.asarray(weights, dtype=float)
    growth = (1 + asset_returns).cumprod()
    cash = 1.0 - w0.sum()
    value = growth.to_numpy() @ w0 + cash
    value = np.concatenate([[1.0], value])
    return pd.Series(value[1:] / value[:-1] - 1, index=asset_returns.index, name="portfolio")


def simulate_gbm(
    s0: float, mu: float, sigma: float, n_days: int, n_paths: int, seed: int | None = None
) -> np.ndarray:
    """GBM 精确解模拟，返回 (n_days + 1, n_paths)，第 0 行为 s0。"""
    rng = np.random.default_rng(seed)
    dt = 1.0 / TRADING_DAYS
    z = rng.standard_normal((n_days, n_paths))
    log_inc = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    out = np.empty((n_days + 1, n_paths))
    out[0] = s0
    out[1:] = s0 * np.exp(np.cumsum(log_inc, axis=0))
    return out


def annualize_log_stats(log_rets: pd.Series, periods: int = TRADING_DAYS) -> tuple[float, float]:
    """由对数收益估计 GBM 的年化 (mu, sigma)；注意 E[log ret] = (mu - sigma^2/2) dt。"""
    m = float(log_rets.mean())
    s = float(log_rets.std(ddof=1))
    sigma = s * np.sqrt(periods)
    mu = m * periods + 0.5 * sigma**2
    return mu, sigma


def mu_standard_error(sigma: float, n_days: int, periods: int = TRADING_DAYS) -> float:
    """mu 估计量的年化标准误 sigma / sqrt(年数)：只取决于数据覆盖了多少年。"""
    return sigma / np.sqrt(n_days / periods)


def growth_rate(mu: float, sigma: float, leverage: float = 1.0) -> float:
    """以固定杠杆 L 持有 GBM 资产（每日再平衡）的长期几何增长率：L*mu - (L*sigma)^2 / 2。"""
    return leverage * mu - 0.5 * (leverage * sigma) ** 2
