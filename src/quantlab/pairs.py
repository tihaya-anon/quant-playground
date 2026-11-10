"""均值回归与配对交易工具。

AR(1)/OU、半衰期、ADF、Engle-Granger、z-score、配对回测、Kalman 对冲比例。

讲解见 labs/lab06_mean_reversion_pairs/README.md。这里是现成实现，lab 里直接调用。
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint


def fit_ar1(x: pd.Series) -> tuple[float, float, float]:
    """OLS 拟合 x_t = c + phi x_{t-1} + e_t，返回 (c, phi, sigma_e)。"""
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    y, lag = x[1:], x[:-1]
    X = np.column_stack([np.ones_like(lag), lag])
    (c, phi), *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ np.array([c, phi])
    sigma = float(np.sqrt(resid @ resid / (len(y) - 2)))
    return float(c), float(phi), sigma


def half_life(phi: float) -> float:
    """AR(1) 半衰期 -ln2 / ln(phi)；phi 不在 (0,1) 时返回 inf。"""
    if not 0 < phi < 1:
        return float("inf")
    return float(-np.log(2) / np.log(phi))


def ou_from_ar1(
    c: float, phi: float, sigma_eps: float, dt: float = 1.0
) -> tuple[float, float, float]:
    """离散 AR(1) 参数换算成 OU 过程的 (kappa, theta, sigma)。"""
    kappa = -np.log(phi) / dt
    theta = c / (1 - phi)
    sigma = sigma_eps * np.sqrt(2 * kappa / (1 - phi**2))
    return float(kappa), float(theta), float(sigma)


def adf_pvalue(x: pd.Series) -> float:
    """ADF 单位根检验的 p 值（小 = 拒绝单位根 = 平稳）。"""
    return float(
        adfuller(
            np.asarray(pd.Series(x).dropna(), dtype=float), autolag="AIC", result_object=False
        )[1]
    )


def engle_granger(log_y: pd.Series, log_x: pd.Series) -> dict:
    """Engle-Granger 两步法：OLS 求对冲比例，再检验残差（价差）是否平稳。

    p 值来自 coint（使用 EG 临界值，而不是普通 ADF 临界值）。
    """
    X = np.column_stack([np.ones(len(log_x)), log_x.to_numpy()])
    (a, b), *_ = np.linalg.lstsq(X, log_y.to_numpy(), rcond=None)
    spread = (log_y - b * log_x - a).rename("spread")
    with warnings.catch_warnings():
        # statsmodels 内部调用 adfuller 时的 FutureWarning，与我们无关
        warnings.simplefilter("ignore", FutureWarning)
        pvalue = float(coint(log_y, log_x, trend="c", autolag="AIC")[1])
    return {"hedge_ratio": float(b), "intercept": float(a), "spread": spread, "pvalue": pvalue}


def rolling_zscore(spread: pd.Series, window: int = 60) -> pd.Series:
    """价差的滚动 z-score，窗口包含当期。"""
    m = spread.rolling(window).mean()
    s = spread.rolling(window).std(ddof=1)
    return ((spread - m) / s).rename("z")


def pair_positions(z: pd.Series, entry: float = 2.0, exit: float = 0.5) -> pd.Series:
    """z-score 进出场：z > entry 做空价差，z < -entry 做多价差，回到 ±exit 内平仓。

    返回 t 收盘后的仓位。
    """
    pos = np.zeros(len(z))
    cur = 0.0
    for i, v in enumerate(z.to_numpy()):
        if np.isnan(v):
            cur = 0.0
        elif cur == 0.0:
            if v > entry:
                cur = -1.0
            elif v < -entry:
                cur = 1.0
        elif cur == 1.0 and v > -exit:
            cur = 0.0
        elif cur == -1.0 and v < exit:
            cur = 0.0
        pos[i] = cur
    return pd.Series(pos, index=z.index, name="position")


def pair_returns(
    y: pd.Series, x: pd.Series, hedge_ratio, positions: pd.Series, cost_bps: float = 0.0
) -> pd.Series:
    """配对组合收益：t 的仓位吃 t+1 的收益（内部已 shift），按总敞口 1+|beta| 归一，扣成本。"""
    ry = y.pct_change()
    rx = x.pct_change()
    beta = hedge_ratio if np.isscalar(hedge_ratio) else pd.Series(hedge_ratio, index=y.index)
    held = positions.shift(1).fillna(0.0)
    beta_held = beta if np.isscalar(beta) else beta.shift(1)
    gross = 1 + np.abs(beta_held)
    ret = held * (ry - beta_held * rx) / gross
    turnover = positions.diff().abs().fillna(positions.abs())
    ret = ret - turnover.shift(1).fillna(0.0) * cost_bps / 1e4
    return ret.iloc[1:].fillna(0.0).rename("pair")


def kalman_hedge_ratio(
    log_y: pd.Series, log_x: pd.Series, delta: float = 1e-4, obs_var: float = 1e-3
) -> pd.DataFrame:
    """用 Kalman 滤波在线估计随时间变化的对冲比例与截距（每步只用到当期及以前的数据）。"""
    yv = log_y.to_numpy()
    xv = log_x.to_numpy()
    n = len(yv)
    state = np.zeros(2)  # [beta, alpha]
    P = np.eye(2)
    Q = delta / (1 - delta) * np.eye(2)
    out = np.empty((n, 2))
    for t in range(n):
        P = P + Q
        H = np.array([xv[t], 1.0])
        e = yv[t] - H @ state
        S = H @ P @ H + obs_var
        K = P @ H / S
        state = state + K * e
        P = P - np.outer(K, H) @ P
        out[t] = state
    return pd.DataFrame(out, index=log_y.index, columns=["beta", "alpha"])


# ---------------------------------------------------------------------------------------------
# 模拟一个用来“找配对”的股票池（lab06 的实验与审计用）


def simulate_universe(
    n_assets: int = 20,
    n_days: int = 1500,
    cointegrated_pair: bool = False,
    half_life: float = 15.0,
    hedge_ratio: float = 1.2,
    seed: int | None = 0,
) -> pd.DataFrame:
    """n_assets 只股票的价格，彼此独立的随机游走（对数价格日波动 1.5%）。

    cointegrated_pair=True 时，前两列（P00、P01）被替换成一对真正协整的股票：
    log P00 = const + hedge_ratio * log P01 + OU 价差（给定半衰期）。
    """
    from quantlab import data

    rng = np.random.default_rng(seed)
    log_p = np.log(50.0) + np.cumsum(rng.normal(0.0002, 0.015, (n_days, n_assets)), axis=0)
    idx = pd.bdate_range("2014-01-02", periods=n_days, name="date")
    prices = pd.DataFrame(np.exp(log_p), index=idx, columns=[f"P{i:02d}" for i in range(n_assets)])
    if cointegrated_pair:
        pair = data.cointegrated_pair(
            n_days=n_days, hedge_ratio=hedge_ratio, half_life=half_life, seed=seed
        )
        prices["P00"] = pair["y"].to_numpy()
        prices["P01"] = pair["x"].to_numpy()
    return prices
