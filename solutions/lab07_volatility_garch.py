"""Lab 07 参考答案。"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

TRADING_DAYS = 252


def realized_vol(returns: pd.Series, window: int = 21, periods: int = TRADING_DAYS) -> pd.Series:
    return returns.rolling(window).std(ddof=1) * np.sqrt(periods)


def ewma_variance(returns: pd.Series, lam: float = 0.94, init: float | None = None) -> pd.Series:
    r = returns.to_numpy(dtype=float)
    out = np.empty(len(r))
    out[0] = float(np.mean(r[:20] ** 2)) if init is None else init
    for t in range(1, len(r)):
        out[t] = lam * out[t - 1] + (1 - lam) * r[t - 1] ** 2
    return pd.Series(out, index=returns.index, name="ewma_var")


def garch_variance(params, eps: np.ndarray) -> np.ndarray:
    omega, alpha, beta = params
    eps = np.asarray(eps, dtype=float)
    var = np.empty(len(eps))
    var[0] = np.var(eps)
    for t in range(1, len(eps)):
        var[t] = omega + alpha * eps[t - 1] ** 2 + beta * var[t - 1]
    return var


def garch_neg_loglik(params, eps: np.ndarray) -> float:
    var = garch_variance(params, eps)
    if np.any(var <= 0) or not np.all(np.isfinite(var)):
        return 1e10
    eps = np.asarray(eps, dtype=float)
    return float(0.5 * np.sum(np.log(2 * np.pi) + np.log(var) + eps**2 / var))


def _unpack(theta: np.ndarray, sample_var: float) -> tuple[float, float, float]:
    # 重参数化：persistence = sigmoid(a)，alpha 占比 = sigmoid(b)，长期方差 = sample_var * exp(c)
    sig = lambda x: 1.0 / (1.0 + np.exp(-x))  # noqa: E731
    pers = sig(theta[0])
    frac = sig(theta[1])
    alpha = pers * frac
    beta = pers * (1 - frac)
    long_run = sample_var * np.exp(theta[2])
    omega = long_run * (1 - pers)
    return omega, alpha, beta


def fit_garch(returns: pd.Series) -> dict[str, float]:
    r = returns.to_numpy(dtype=float)
    mu = float(r.mean())
    eps = r - mu
    v = float(np.var(eps))
    # 初值：persistence 0.95, alpha 占比 0.1
    theta0 = np.array([np.log(0.95 / 0.05), np.log(0.1 / 0.9), 0.0])
    res = minimize(
        lambda th: garch_neg_loglik(_unpack(th, v), eps),
        theta0,
        method="Nelder-Mead",
        options={"xatol": 1e-8, "fatol": 1e-8, "maxiter": 5000},
    )
    res = minimize(lambda th: garch_neg_loglik(_unpack(th, v), eps), res.x, method="BFGS")
    omega, alpha, beta = _unpack(res.x, v)
    return {
        "mu": mu,
        "omega": omega,
        "alpha": alpha,
        "beta": beta,
        "loglik": -garch_neg_loglik((omega, alpha, beta), eps),
    }


def long_run_variance(omega: float, alpha: float, beta: float) -> float:
    return omega / (1 - alpha - beta)


def garch_forecast(
    omega, alpha, beta, last_var: float, last_eps: float, horizon: int
) -> np.ndarray:
    vl = long_run_variance(omega, alpha, beta)
    v1 = omega + alpha * last_eps**2 + beta * last_var
    h = np.arange(horizon)
    return vl + (alpha + beta) ** h * (v1 - vl)


def half_life(alpha: float, beta: float) -> float:
    return float(np.log(0.5) / np.log(alpha + beta))


def vol_target_weights(
    sigma_daily: pd.Series,
    target_vol: float = 0.10,
    max_leverage: float = 2.0,
    periods: int = TRADING_DAYS,
) -> pd.Series:
    w = target_vol / (sigma_daily * np.sqrt(periods))
    return w.clip(upper=max_leverage)


def vol_target_returns(
    returns: pd.Series, target_vol: float = 0.10, lam: float = 0.94, max_leverage: float = 2.0
) -> pd.Series:
    sigma = np.sqrt(ewma_variance(returns, lam))
    w = vol_target_weights(sigma, target_vol, max_leverage)
    return (w * returns).rename("vol_target")
