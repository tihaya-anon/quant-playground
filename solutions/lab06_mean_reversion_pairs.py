"""Lab 06 参考答案。"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint


def fit_ar1(x: pd.Series) -> tuple[float, float, float]:
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    y, lag = x[1:], x[:-1]
    X = np.column_stack([np.ones_like(lag), lag])
    (c, phi), *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ np.array([c, phi])
    sigma = float(np.sqrt(resid @ resid / (len(y) - 2)))
    return float(c), float(phi), sigma


def half_life(phi: float) -> float:
    if not 0 < phi < 1:
        return float("inf")
    return float(-np.log(2) / np.log(phi))


def ou_from_ar1(
    c: float, phi: float, sigma_eps: float, dt: float = 1.0
) -> tuple[float, float, float]:
    kappa = -np.log(phi) / dt
    theta = c / (1 - phi)
    sigma = sigma_eps * np.sqrt(2 * kappa / (1 - phi**2))
    return float(kappa), float(theta), float(sigma)


def adf_pvalue(x: pd.Series) -> float:
    return float(
        adfuller(
            np.asarray(pd.Series(x).dropna(), dtype=float), autolag="AIC", result_object=False
        )[1]
    )


def engle_granger(log_y: pd.Series, log_x: pd.Series) -> dict:
    X = np.column_stack([np.ones(len(log_x)), log_x.to_numpy()])
    (a, b), *_ = np.linalg.lstsq(X, log_y.to_numpy(), rcond=None)
    spread = (log_y - b * log_x - a).rename("spread")
    with warnings.catch_warnings():
        # statsmodels 内部调用 adfuller 时的 FutureWarning，与我们无关
        warnings.simplefilter("ignore", FutureWarning)
        pvalue = float(coint(log_y, log_x, trend="c", autolag="AIC")[1])
    return {"hedge_ratio": float(b), "intercept": float(a), "spread": spread, "pvalue": pvalue}


def rolling_zscore(spread: pd.Series, window: int = 60) -> pd.Series:
    m = spread.rolling(window).mean()
    s = spread.rolling(window).std(ddof=1)
    return ((spread - m) / s).rename("z")


def pair_positions(z: pd.Series, entry: float = 2.0, exit: float = 0.5) -> pd.Series:
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
