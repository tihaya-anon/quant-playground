"""因子研究工具：OLS/CAPM、rolling beta、横截面 IC、分组收益、多空组合、中性化。

讲解见 labs/lab04_factor_investing/README.md。这里是现成实现，lab 里直接调用。

约定：signal.loc[t] 是 t 收盘时已知的信号，它预测 returns.loc[t+1]。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def ols(y, X, add_const: bool = True) -> dict:
    """OLS 回归。返回 params、se（同方差假设）、tstats、resid、r2。

    add_const 时第 0 个参数是截距。
    """
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
    """CAPM 时间序列回归：(r_i - rf) = alpha + beta (r_m - rf) + e。

    alpha 为每期，alpha_annual 已年化。
    """
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
    """滚动窗口 beta = Cov(r_i, r_m) / Var(r_m)，窗口包含当期。"""
    cov = asset.rolling(window).cov(market)
    var = market.rolling(window).var()
    return (cov / var).rename("beta")


def information_coefficient(signal: pd.DataFrame, returns: pd.DataFrame) -> pd.Series:
    """每天的横截面 Spearman IC：t 日信号的排名与 t+1 日收益排名的相关系数。"""
    fwd = returns.shift(-1)
    s_rank = signal.rank(axis=1)
    f_rank = fwd.rank(axis=1)
    ic = s_rank.corrwith(f_rank, axis=1)
    return ic.iloc[:-1].rename("ic")


def ic_summary(ic: pd.Series, periods: int = 252) -> dict:
    """IC 序列的均值、标准差、ICIR（均值/标准差）、t 值、胜率。"""
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
    """按 t 日信号分 n 组，t+1 日各组等权平均收益（列 1..n，n 为信号最高组）。"""
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
    """最高组减最低组的多空收益。"""
    q = quantile_returns(signal, returns, n_quantiles)
    return (q[n_quantiles] - q[1]).rename("long_short")


def cross_sectional_zscore(signal: pd.DataFrame) -> pd.DataFrame:
    """每天横截面标准化：减当天均值、除以当天标准差。"""
    return signal.sub(signal.mean(axis=1), axis=0).div(signal.std(axis=1, ddof=1), axis=0)


def neutralize(signal: pd.DataFrame, size: pd.Series, sector: pd.Series) -> pd.DataFrame:
    """每天对 size 与行业哑变量做横截面回归，返回残差（与这些暴露正交的那部分信号）。"""
    cols = signal.columns
    dummies = pd.get_dummies(sector.loc[cols]).astype(float).to_numpy()
    X = np.column_stack([size.loc[cols].to_numpy(), dummies])  # 行业哑变量已经包含截距
    hat = X @ np.linalg.pinv(X)
    m = np.eye(len(cols)) - hat
    return pd.DataFrame(signal.to_numpy() @ m.T, index=signal.index, columns=cols)


# ---------------------------------------------------------------------------------------------
# 模拟一个带“已知风险溢价”和退市的股票横截面（lab04 的实验与审计用）


@dataclass(frozen=True)
class Universe:
    """一个模拟的股票池。

    returns     T x N 日收益；股票退市后为 NaN，退市当天是一笔大亏损。
    signal      T x N，t 收盘时已知的“基本面”信号。
    market      市场因子日收益。
    size_factor 小盘因子日收益（做多小盘、做空大盘）。
    smallness   每只股票对小盘因子的暴露（静态，越大越“小盘”）。
    alive       T x N 布尔，当天是否仍在交易（退市当天仍为 True）。
    """

    returns: pd.DataFrame
    signal: pd.DataFrame
    market: pd.Series
    size_factor: pd.Series
    smallness: pd.Series
    alive: pd.DataFrame


def simulate_universe(
    n_assets: int = 300,
    n_days: int = 2520,
    ic: float = 0.0,
    size_premium: float = 0.0,
    size_tilt: float = 0.0,
    signal_persistence: float = 0.95,
    delist_rate: float = 0.0,
    delist_signal_link: float = 0.0,
    seed: int | None = 0,
) -> Universe:
    """生成一个股票池。

        r_{i,t+1} = beta_i m_{t+1} + smallness_i f_{t+1} + ic * vol * z_{i,t} + e_{i,t+1}

    - z 是“纯 alpha”成分，按 AR(1)（系数 signal_persistence）随时间缓慢变化；
      观测到的 signal = z + size_tilt * smallness（size_tilt > 0 时信号里混入小盘暴露）。
    - 小盘因子 f 的年化期望为 size_premium。
    - delist_rate 是年化退市概率；delist_signal_link > 0 时，信号越高越容易退市
      （比如“便宜”的股票里有更多濒临破产的公司）。退市当天收益为 -50%，之后为 NaN。
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2012-01-02", periods=n_days, name="date")
    cols = [f"S{i:03d}" for i in range(n_assets)]
    vol = 0.02

    beta = rng.normal(1.0, 0.3, n_assets)
    smallness = rng.normal(0.0, 1.0, n_assets)
    m = rng.normal(0.0003, 0.01, n_days)
    f = rng.normal(size_premium / 252, 0.004, n_days)

    z = np.empty((n_days, n_assets))
    z[0] = rng.standard_normal(n_assets)
    innov = np.sqrt(1 - signal_persistence**2)
    for t in range(1, n_days):
        z[t] = signal_persistence * z[t - 1] + innov * rng.standard_normal(n_assets)

    rets = (
        beta * m[:, None] + smallness * f[:, None] + vol * rng.standard_normal((n_days, n_assets))
    )
    rets[1:] += ic * vol * z[:-1]
    signal = z + size_tilt * smallness

    alive = np.ones((n_days, n_assets), dtype=bool)
    if delist_rate > 0:
        base = delist_rate / 252
        dead = np.zeros(n_assets, dtype=bool)
        for t in range(1, n_days):
            alive[t] = ~dead
            hazard = base * np.exp(delist_signal_link * signal[t - 1])
            dies = ~dead & (rng.random(n_assets) < hazard)
            rets[t, dies] = -0.5
            dead |= dies
        rets = np.where(alive, rets, np.nan)
        signal = np.where(alive, signal, np.nan)

    return Universe(
        returns=pd.DataFrame(rets, index=idx, columns=cols),
        signal=pd.DataFrame(signal, index=idx, columns=cols),
        market=pd.Series(m, index=idx, name="market"),
        size_factor=pd.Series(f, index=idx, name="size"),
        smallness=pd.Series(smallness, index=cols, name="smallness"),
        alive=pd.DataFrame(alive, index=idx, columns=cols),
    )
