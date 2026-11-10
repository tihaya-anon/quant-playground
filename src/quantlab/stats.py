"""回测的统计学：Sharpe 的抽样分布、bootstrap、多重检验校正（PSR/DSR）、walk-forward。

讲解见 labs/lab03_backtest_statistics/README.md。这里是现成实现，lab 里直接调用。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS = 252
EULER_GAMMA = 0.5772156649015329


def momentum_strategy_returns(prices: pd.Series, lookback: int) -> pd.Series:
    """时间序列动量：过去 lookback 天涨则做多、跌则做空。

    t 收盘算信号，t+1 才持仓（已 shift，无前视）。返回日收益，索引为 prices.index[1:]。
    """
    sig = np.sign(prices / prices.shift(lookback) - 1).fillna(0.0)
    return (sig.shift(1).fillna(0.0) * prices.pct_change().fillna(0.0)).iloc[1:]


def per_period_sharpe(returns) -> float:
    """每期（未年化）Sharpe：mean / std(ddof=1)。"""
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    return float(r.mean() / r.std(ddof=1))


def sharpe_se_iid(sr: float, n: int) -> float:
    """iid 正态收益下每期 Sharpe 的渐近标准误（Lo, 2002）。"""
    return float(np.sqrt((1 + 0.5 * sr**2) / n))


def sharpe_se_mertens(sr: float, n: int, skew: float, kurt: float) -> float:
    """考虑偏度与（非超额）峰度的 Sharpe 标准误（Mertens, 2002）。"""
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
    """PSR：真实（每期）Sharpe 超过 sr_benchmark 的概率（Bailey & López de Prado, 2012）。"""
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    n = len(r)
    sr = r.mean() / r.std(ddof=1)
    g3 = stats.skew(r)
    g4 = stats.kurtosis(r, fisher=False)
    denom = np.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr**2)
    return float(stats.norm.cdf((sr - sr_benchmark) * np.sqrt(n - 1) / denom))


def expected_max_sharpe(n_trials: int, sr_variance: float, sr_mean: float = 0.0) -> float:
    """n_trials 个真实 Sharpe 都为 sr_mean 的策略里，样本 Sharpe 最大值的期望（近似）。"""
    z1 = stats.norm.ppf(1 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1 - 1.0 / (n_trials * np.e))
    return float(sr_mean + np.sqrt(sr_variance) * ((1 - EULER_GAMMA) * z1 + EULER_GAMMA * z2))


def deflated_sharpe_ratio(returns, trial_sharpes) -> float:
    """DSR：以“试了这么多次之后，最好的那个在纯噪声下本该有多好”为基准的 PSR。

    trial_sharpes 为所有尝试过的策略的每期 Sharpe。
    """
    trials = np.asarray(trial_sharpes, dtype=float)
    sr0 = expected_max_sharpe(len(trials), float(trials.var(ddof=1)))
    return probabilistic_sharpe_ratio(returns, sr0)


def sweep_returns(prices: pd.Series, lookbacks) -> pd.DataFrame:
    """每个 lookback 一列的动量策略日收益。"""
    return pd.DataFrame({lb: momentum_strategy_returns(prices, lb) for lb in lookbacks})


def walk_forward(
    prices: pd.Series, lookbacks, train_size: int = 756, test_size: int = 126
) -> pd.DataFrame:
    """滚动地用前 train_size 天选最优 lookback，在之后 test_size 天上交易。返回列 ret、lookback。"""
    rets = sweep_returns(prices, lookbacks)
    pieces = []
    for start in range(train_size, len(rets), test_size):
        train = rets.iloc[start - train_size : start]
        test = rets.iloc[start : start + test_size]
        sr = train.mean() / train.std(ddof=1)
        best = sr.idxmax()
        pieces.append(pd.DataFrame({"ret": test[best], "lookback": best}, index=test.index))
    return pd.concat(pieces)
