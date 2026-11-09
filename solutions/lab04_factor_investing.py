"""Lab 04 参考答案。"""

from __future__ import annotations

import numpy as np
import pandas as pd


def ols(y, X, add_const: bool = True) -> dict:
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    if add_const:
        X = np.column_stack([np.ones(len(X)), X])
    n, k = X.shape
    xtx_inv = np.linalg.inv(X.T @ X)
    params = xtx_inv @ X.T @ y
    resid = y - X @ params
    s2 = resid @ resid / (n - k)
    se = np.sqrt(np.diag(xtx_inv) * s2)
    tss = ((y - y.mean()) ** 2).sum()
    return {
        "params": params,
        "se": se,
        "tstats": params / se,
        "resid": resid,
        "r2": float(1 - resid @ resid / tss),
    }


def capm(asset: pd.Series, market: pd.Series, rf: float = 0.0, periods: int = 252) -> dict:
    df = pd.concat([asset, market], axis=1, join="inner").dropna()
    y = df.iloc[:, 0].to_numpy() - rf / periods
    x = df.iloc[:, 1].to_numpy() - rf / periods
    res = ols(y, x)
    return {
        "alpha": float(res["params"][0]),
        "beta": float(res["params"][1]),
        "t_alpha": float(res["tstats"][0]),
        "t_beta": float(res["tstats"][1]),
        "r2": res["r2"],
        "alpha_annual": float(res["params"][0] * periods),
    }


def rolling_beta(asset: pd.Series, market: pd.Series, window: int = 60) -> pd.Series:
    cov = asset.rolling(window).cov(market)
    var = market.rolling(window).var()
    return (cov / var).rename("beta")


def information_coefficient(signal: pd.DataFrame, returns: pd.DataFrame) -> pd.Series:
    fwd = returns.shift(-1)
    s_rank = signal.rank(axis=1)
    f_rank = fwd.rank(axis=1)
    ic = s_rank.corrwith(f_rank, axis=1)
    return ic.iloc[:-1].rename("ic")


def ic_summary(ic: pd.Series, periods: int = 252) -> dict:
    ic = ic.dropna()
    mean, std = float(ic.mean()), float(ic.std(ddof=1))
    return {
        "mean": mean,
        "std": std,
        "icir": mean / std,
        "icir_annual": mean / std * np.sqrt(periods),
        "t_stat": mean / std * np.sqrt(len(ic)),
        "hit_rate": float((ic > 0).mean()),
    }


def quantile_returns(
    signal: pd.DataFrame, returns: pd.DataFrame, n_quantiles: int = 5
) -> pd.DataFrame:
    ranks = signal.rank(axis=1, pct=True)
    buckets = np.ceil(ranks * n_quantiles).clip(1, n_quantiles)
    buckets = buckets.shift(1)  # t 的分组用于 t+1 的收益
    out = {}
    for q in range(1, n_quantiles + 1):
        out[q] = returns.where(buckets == q).mean(axis=1)
    return pd.DataFrame(out).iloc[1:]


def long_short_returns(
    signal: pd.DataFrame, returns: pd.DataFrame, n_quantiles: int = 5
) -> pd.Series:
    q = quantile_returns(signal, returns, n_quantiles)
    return (q[n_quantiles] - q[1]).rename("long_short")


def cross_sectional_zscore(signal: pd.DataFrame) -> pd.DataFrame:
    return signal.sub(signal.mean(axis=1), axis=0).div(signal.std(axis=1, ddof=1), axis=0)


def neutralize(signal: pd.DataFrame, size: pd.Series, sector: pd.Series) -> pd.DataFrame:
    cols = signal.columns
    dummies = pd.get_dummies(sector.loc[cols]).astype(float).to_numpy()
    X = np.column_stack([size.loc[cols].to_numpy(), dummies])  # 行业哑变量已经包含截距
    hat = X @ np.linalg.pinv(X)
    m = np.eye(len(cols)) - hat
    return pd.DataFrame(signal.to_numpy() @ m.T, index=signal.index, columns=cols)
