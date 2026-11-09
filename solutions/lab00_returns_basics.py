"""Lab 00 参考答案。"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def simple_returns(prices: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    return prices.pct_change().iloc[1:]


def log_returns(prices: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    return np.log(prices / prices.shift(1)).iloc[1:]


def simple_to_log(r):
    return np.log1p(r)


def log_to_simple(lr):
    return np.expm1(lr)


def cumulative_return(returns: pd.Series) -> float:
    return float((1 + returns).prod() - 1)


def wealth_index(returns: pd.Series | pd.DataFrame, start: float = 1.0):
    return start * (1 + returns).cumprod()


def resample_returns(returns: pd.Series | pd.DataFrame, rule: str = "ME"):
    return (1 + returns).resample(rule).prod() - 1


def rebalanced_portfolio_returns(asset_returns: pd.DataFrame, weights) -> pd.Series:
    w = np.asarray(weights, dtype=float)
    return pd.Series(asset_returns.to_numpy() @ w, index=asset_returns.index, name="portfolio")


def buy_and_hold_portfolio_returns(asset_returns: pd.DataFrame, weights) -> pd.Series:
    w0 = np.asarray(weights, dtype=float)
    growth = (1 + asset_returns).cumprod()
    cash = 1.0 - w0.sum()
    value = growth.to_numpy() @ w0 + cash
    value = np.concatenate([[1.0], value])
    return pd.Series(value[1:] / value[:-1] - 1, index=asset_returns.index, name="portfolio")


def simulate_gbm(
    s0: float, mu: float, sigma: float, n_days: int, n_paths: int, seed: int | None = None
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dt = 1.0 / TRADING_DAYS
    z = rng.standard_normal((n_days, n_paths))
    log_inc = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    out = np.empty((n_days + 1, n_paths))
    out[0] = s0
    out[1:] = s0 * np.exp(np.cumsum(log_inc, axis=0))
    return out


def annualize_log_stats(log_rets: pd.Series, periods: int = TRADING_DAYS) -> tuple[float, float]:
    m = float(log_rets.mean())
    s = float(log_rets.std(ddof=1))
    sigma = s * np.sqrt(periods)
    mu = m * periods + 0.5 * sigma**2
    return mu, sigma


def mu_standard_error(sigma: float, n_days: int, periods: int = TRADING_DAYS) -> float:
    return sigma / np.sqrt(n_days / periods)
