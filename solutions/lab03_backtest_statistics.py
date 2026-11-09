"""Lab 03 参考答案。"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS = 252
EULER_GAMMA = 0.5772156649015329


def momentum_strategy_returns(prices: pd.Series, lookback: int) -> pd.Series:
    """（给定的辅助函数，与 lab.py 中相同）时间序列动量策略的日收益，已正确 shift。"""
    sig = np.sign(prices / prices.shift(lookback) - 1).fillna(0.0)
    return (sig.shift(1).fillna(0.0) * prices.pct_change().fillna(0.0)).iloc[1:]


def per_period_sharpe(returns) -> float:
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    return float(r.mean() / r.std(ddof=1))


def sharpe_se_iid(sr: float, n: int) -> float:
    return float(np.sqrt((1 + 0.5 * sr**2) / n))


def sharpe_se_mertens(sr: float, n: int, skew: float, kurt: float) -> float:
    return float(np.sqrt((1 - skew * sr + (kurt - 1) / 4 * sr**2) / (n - 1)))


def bootstrap_sharpe(
    returns, n_boot: int = 2000, seed: int | None = 0, periods: int = TRADING_DAYS
) -> np.ndarray:
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(r), size=(n_boot, len(r)))
    s = r[idx]
    return s.mean(axis=1) / s.std(axis=1, ddof=1) * np.sqrt(periods)


def block_bootstrap_sharpe(
    returns,
    block_size: int = 20,
    n_boot: int = 2000,
    seed: int | None = 0,
    periods: int = TRADING_DAYS,
) -> np.ndarray:
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    n = len(r)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_size))
    starts = rng.integers(0, n, size=(n_boot, n_blocks))
    offsets = np.arange(block_size)
    idx = ((starts[:, :, None] + offsets[None, None, :]) % n).reshape(n_boot, -1)[:, :n]
    s = r[idx]
    return s.mean(axis=1) / s.std(axis=1, ddof=1) * np.sqrt(periods)


def percentile_ci(samples, alpha: float = 0.05) -> tuple[float, float]:
    lo, hi = np.quantile(np.asarray(samples), [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def probabilistic_sharpe_ratio(returns, sr_benchmark: float = 0.0) -> float:
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    n = len(r)
    sr = r.mean() / r.std(ddof=1)
    g3 = stats.skew(r)
    g4 = stats.kurtosis(r, fisher=False)
    denom = np.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr**2)
    return float(stats.norm.cdf((sr - sr_benchmark) * np.sqrt(n - 1) / denom))


def expected_max_sharpe(n_trials: int, sr_variance: float, sr_mean: float = 0.0) -> float:
    z1 = stats.norm.ppf(1 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1 - 1.0 / (n_trials * np.e))
    return float(sr_mean + np.sqrt(sr_variance) * ((1 - EULER_GAMMA) * z1 + EULER_GAMMA * z2))


def deflated_sharpe_ratio(returns, trial_sharpes) -> float:
    trials = np.asarray(trial_sharpes, dtype=float)
    sr0 = expected_max_sharpe(len(trials), float(trials.var(ddof=1)))
    return probabilistic_sharpe_ratio(returns, sr0)


def sweep_returns(prices: pd.Series, lookbacks) -> pd.DataFrame:
    return pd.DataFrame({lb: momentum_strategy_returns(prices, lb) for lb in lookbacks})


def walk_forward(
    prices: pd.Series, lookbacks, train_size: int = 756, test_size: int = 126
) -> pd.DataFrame:
    rets = sweep_returns(prices, lookbacks)
    pieces = []
    for start in range(train_size, len(rets), test_size):
        train = rets.iloc[start - train_size : start]
        test = rets.iloc[start : start + test_size]
        sr = train.mean() / train.std(ddof=1)
        best = sr.idxmax()
        pieces.append(pd.DataFrame({"ret": test[best], "lookback": best}, index=test.index))
    return pd.concat(pieces)
