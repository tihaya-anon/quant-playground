"""合成数据生成器。

所有 lab 的测试都跑在这里生成的数据上：离线、确定（固定 seed）、并且“真相已知”——
我们知道真实的 mu/sigma、真实的 beta、真实的半衰期、真实的 GARCH 参数，于是可以检验
你的估计量到底估得准不准。这是用真实行情做不到的。

约定：
- 价格是 pd.Series / pd.DataFrame，索引为工作日 DatetimeIndex（freq="B"）。
- 收益默认为简单收益 r_t = P_t / P_{t-1} - 1；对数收益会在函数名里写明。
- 一年按 252 个交易日计。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252
DEFAULT_START = "2015-01-02"


def business_index(n: int, start: str = DEFAULT_START) -> pd.DatetimeIndex:
    """n 个工作日组成的索引。"""
    return pd.bdate_range(start=start, periods=n, name="date")


def gbm_prices(
    n_days: int = 2520,
    mu: float = 0.08,
    sigma: float = 0.20,
    s0: float = 100.0,
    seed: int | None = 0,
    start: str = DEFAULT_START,
    name: str = "close",
) -> pd.Series:
    """几何布朗运动价格路径。

    dS/S = mu dt + sigma dW，离散化为精确解：
        S_{t+1} = S_t * exp((mu - sigma^2/2) dt + sigma sqrt(dt) Z)
    其中 dt = 1/252，mu/sigma 都是年化参数。返回长度为 n_days 的序列（第一个值为 s0）。
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / TRADING_DAYS
    z = rng.standard_normal(n_days - 1)
    log_ret = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    path = s0 * np.exp(np.concatenate([[0.0], np.cumsum(log_ret)]))
    return pd.Series(path, index=business_index(n_days, start), name=name)


def correlated_gbm(
    n_assets: int = 5,
    n_days: int = 2520,
    mu: np.ndarray | float = 0.07,
    sigma: np.ndarray | float = 0.20,
    corr: np.ndarray | float = 0.3,
    seed: int | None = 0,
    start: str = DEFAULT_START,
) -> pd.DataFrame:
    """多资产相关 GBM。corr 可以是标量（所有资产两两相同相关系数）或完整相关矩阵。"""
    rng = np.random.default_rng(seed)
    mu = np.broadcast_to(np.asarray(mu, dtype=float), (n_assets,))
    sigma = np.broadcast_to(np.asarray(sigma, dtype=float), (n_assets,))
    if np.isscalar(corr) or np.ndim(corr) == 0:
        c = np.full((n_assets, n_assets), float(corr))
        np.fill_diagonal(c, 1.0)
    else:
        c = np.asarray(corr, dtype=float)
    chol = np.linalg.cholesky(c)
    dt = 1.0 / TRADING_DAYS
    z = rng.standard_normal((n_days - 1, n_assets)) @ chol.T
    log_ret = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    paths = 100.0 * np.exp(np.vstack([np.zeros(n_assets), np.cumsum(log_ret, axis=0)]))
    cols = [f"A{i:02d}" for i in range(n_assets)]
    return pd.DataFrame(paths, index=business_index(n_days, start), columns=cols)


def regime_switching_prices(
    n_days: int = 2520,
    seed: int | None = 0,
    trend_vol: float = 0.15,
    trend_drift: float = 0.25,
    regime_len: int = 120,
    start: str = DEFAULT_START,
) -> pd.Series:
    """在“上涨趋势 / 下跌趋势 / 横盘”三种状态之间切换的价格。

    状态持续时间服从均值为 regime_len 的几何分布。趋势跟踪策略在这种数据上应该有正收益，
    而在纯 GBM（无可预测性）上不应该——这是 lab02/lab03 用来对照的“有信号 vs 没信号”。
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / TRADING_DAYS
    drifts = np.array([trend_drift, -trend_drift, 0.0])
    rets = np.empty(n_days - 1)
    i = 0
    while i < n_days - 1:
        length = rng.geometric(1.0 / regime_len)
        state = rng.integers(0, 3)
        j = min(i + length, n_days - 1)
        rets[i:j] = drifts[state] * dt + trend_vol * np.sqrt(dt) * rng.standard_normal(j - i)
        i = j
    path = 100.0 * np.exp(np.concatenate([[0.0], np.cumsum(rets)]))
    return pd.Series(path, index=business_index(n_days, start), name="close")


@dataclass(frozen=True)
class FactorPanel:
    """横截面面板数据（lab04/lab09）。

    returns.loc[t]  是 t-1 收盘到 t 收盘的简单收益。
    signal.loc[t]   是 t 收盘时就已知的信号，它预测的是 returns.loc[t+1]。
    beta / size / sector 是静态的资产属性。market 是市场因子收益。
    """

    returns: pd.DataFrame
    signal: pd.DataFrame
    market: pd.Series
    beta: pd.Series
    size: pd.Series
    sector: pd.Series
    ic: float


def factor_panel(
    n_assets: int = 200,
    n_days: int = 1260,
    ic: float = 0.05,
    n_sectors: int = 8,
    seed: int | None = 0,
    start: str = DEFAULT_START,
) -> FactorPanel:
    """单因子市场模型 + 行业因子 + 一个带有预测能力的 alpha 信号。

        r_{i,t+1} = beta_i * m_{t+1} + s_{sector(i),t+1} + alpha_{i,t+1} + eps_{i,t+1}

    其中 signal_{i,t} 与 alpha_{i,t+1} 的横截面相关系数约为 ic / (alpha 在特异收益中的占比)，
    使得 signal 与 *特异* 收益的 Spearman IC 大约等于 ic。signal 还故意混入了 size 暴露，
    让 lab04 的“中性化”有事可做。
    """
    rng = np.random.default_rng(seed)
    idx = business_index(n_days, start)
    cols = [f"S{i:03d}" for i in range(n_assets)]
    beta = pd.Series(rng.normal(1.0, 0.3, n_assets), index=cols, name="beta")
    size = pd.Series(rng.normal(0.0, 1.0, n_assets), index=cols, name="size")
    sector = pd.Series(rng.integers(0, n_sectors, n_assets), index=cols, name="sector")

    daily_vol_idio = 0.02
    m = rng.normal(0.0003, 0.01, n_days)
    s = rng.normal(0.0, 0.006, (n_days, n_sectors))

    # 信号的“纯”部分：标准正态，每天重新抽
    pure = rng.standard_normal((n_days, n_assets))
    # 下一期的特异收益 = ic * vol * pure_{t} + sqrt(1-ic^2) * vol * noise
    idio = np.empty((n_days, n_assets))
    idio[0] = rng.normal(0, daily_vol_idio, n_assets)
    noise = rng.standard_normal((n_days, n_assets))
    idio[1:] = daily_vol_idio * (ic * pure[:-1] + np.sqrt(1 - ic**2) * noise[1:])

    rets = beta.to_numpy() * m[:, None] + s[:, sector.to_numpy()] + idio
    # 观测到的信号 = 纯信号 + size 暴露（一个“脏”的、需要中性化的信号）
    observed = pure + 0.8 * size.to_numpy()[None, :]

    return FactorPanel(
        returns=pd.DataFrame(rets, index=idx, columns=cols),
        signal=pd.DataFrame(observed, index=idx, columns=cols),
        market=pd.Series(m, index=idx, name="market"),
        beta=beta,
        size=size,
        sector=sector,
        ic=ic,
    )


def cointegrated_pair(
    n_days: int = 1500,
    hedge_ratio: float = 1.5,
    half_life: float = 15.0,
    spread_vol: float = 0.02,
    seed: int | None = 0,
    start: str = DEFAULT_START,
) -> pd.DataFrame:
    """协整的一对对数价格 (y, x)。

        x_t        随机游走（对数价格）
        spread_t   OU 过程（离散 AR(1)），phi = exp(-ln2 / half_life)
        y_t        = hedge_ratio * x_t + spread_t + const

    返回 DataFrame，列为 "x"、"y"（价格，非对数），以及 "spread"（真实的对数价差，供检验用）。
    """
    rng = np.random.default_rng(seed)
    phi = np.exp(-np.log(2) / half_life)
    log_x = np.log(50.0) + np.cumsum(rng.normal(0.0002, 0.015, n_days))
    spread = np.empty(n_days)
    spread[0] = 0.0
    eps = rng.normal(0, spread_vol, n_days)
    for t in range(1, n_days):
        spread[t] = phi * spread[t - 1] + eps[t]
    log_y = 0.5 + hedge_ratio * log_x + spread
    return pd.DataFrame(
        {"x": np.exp(log_x), "y": np.exp(log_y), "spread": spread},
        index=business_index(n_days, start),
    )


def garch_returns(
    n_days: int = 3000,
    omega: float = 2e-6,
    alpha: float = 0.08,
    beta: float = 0.90,
    mu: float = 0.0003,
    seed: int | None = 0,
    start: str = DEFAULT_START,
) -> pd.DataFrame:
    """GARCH(1,1) 日收益（正态新息）。

        r_t = mu + e_t,  e_t = sigma_t z_t
        sigma_t^2 = omega + alpha e_{t-1}^2 + beta sigma_{t-1}^2

    返回列 "ret"（日收益）与 "sigma"（真实条件波动率）。
    """
    if alpha + beta >= 1:
        raise ValueError("需要 alpha + beta < 1 才是协方差平稳的")
    rng = np.random.default_rng(seed)
    var = np.empty(n_days)
    e = np.empty(n_days)
    var[0] = omega / (1 - alpha - beta)
    z = rng.standard_normal(n_days)
    e[0] = np.sqrt(var[0]) * z[0]
    for t in range(1, n_days):
        var[t] = omega + alpha * e[t - 1] ** 2 + beta * var[t - 1]
        e[t] = np.sqrt(var[t]) * z[t]
    idx = business_index(n_days, start)
    return pd.DataFrame({"ret": mu + e, "sigma": np.sqrt(var)}, index=idx)


def ohlcv_from_close(close: pd.Series, seed: int | None = 0) -> pd.DataFrame:
    """由收盘价伪造 OHLCV：open 为前收盘加一点跳空，high/low 包住 open/close。"""
    rng = np.random.default_rng(seed)
    prev = close.shift(1).fillna(close.iloc[0])
    daily_vol = close.pct_change().std()
    if not np.isfinite(daily_vol):
        daily_vol = 0.01
    open_ = prev * np.exp(rng.normal(0, 0.25 * daily_vol, len(close)))
    hi_ext = np.abs(rng.normal(0, 0.5 * daily_vol, len(close)))
    lo_ext = np.abs(rng.normal(0, 0.5 * daily_vol, len(close)))
    high = np.maximum(open_, close) * (1 + hi_ext)
    low = np.minimum(open_, close) * (1 - lo_ext)
    volume = rng.lognormal(mean=13.0, sigma=0.4, size=len(close)).round()
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=close.index,
    )
