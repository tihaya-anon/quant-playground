"""期权：Black-Scholes 定价与 Greeks、隐含波动率、蒙特卡洛、离散 delta 对冲模拟。

讲解见 labs/lab10_options_hedging/README.md。这里是现成实现，lab 里直接调用。
约定：T 以年计，r / q / sigma 为年化连续复利；theta = -dV/dT（每年），vega 对应 sigma 变化 1.0。
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def d1_d2(S, K, T, r, sigma, q=0.0):
    """BS 公式里的 d1, d2（支持 ndarray 广播）。"""
    S, K, T, sigma = (np.asarray(v, dtype=float) for v in (S, K, T, sigma))
    vs = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / vs
    return d1, d1 - vs


def bs_price(S, K, T, r, sigma, q=0.0, kind="call"):
    """欧式期权 Black-Scholes 价格，kind 为 "call" 或 "put"，q 为连续股息率。"""
    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    df_q, df_r = np.exp(-q * T), np.exp(-r * T)
    if kind == "call":
        out = S * df_q * norm.cdf(d1) - K * df_r * norm.cdf(d2)
    elif kind == "put":
        out = K * df_r * norm.cdf(-d2) - S * df_q * norm.cdf(-d1)
    else:
        raise ValueError(kind)
    return float(out) if np.ndim(out) == 0 else out


def put_call_parity_gap(call, put, S, K, T, r, q=0.0):
    """C - P - (S e^{-qT} - K e^{-rT})；无套利时为 0。"""
    return call - put - (S * np.exp(-q * T) - K * np.exp(-r * T))


def greeks(S, K, T, r, sigma, q=0.0, kind="call"):
    """解析 Greeks：返回 dict(delta, gamma, vega, theta, rho)。"""
    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    d1, d2 = float(d1), float(d2)
    df_q, df_r = np.exp(-q * T), np.exp(-r * T)
    pdf = norm.pdf(d1)
    gamma = df_q * pdf / (S * sigma * np.sqrt(T))
    vega = S * df_q * pdf * np.sqrt(T)
    common = -S * df_q * pdf * sigma / (2 * np.sqrt(T))
    if kind == "call":
        delta = df_q * norm.cdf(d1)
        theta = common + q * S * df_q * norm.cdf(d1) - r * K * df_r * norm.cdf(d2)
        rho = K * T * df_r * norm.cdf(d2)
    elif kind == "put":
        delta = -df_q * norm.cdf(-d1)
        theta = common - q * S * df_q * norm.cdf(-d1) + r * K * df_r * norm.cdf(-d2)
        rho = -K * T * df_r * norm.cdf(-d2)
    else:
        raise ValueError(kind)
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


def implied_vol(price, S, K, T, r, q=0.0, kind="call", tol=1e-10, max_iter=50):
    """由期权价格反解 BS 隐含波动率：先 Newton，失败时 brentq 兜底；超出无套利边界抛 ValueError。"""
    lo_bound = bs_price(S, K, T, r, 1e-9, q, kind)
    hi_bound = bs_price(S, K, T, r, 10.0, q, kind)
    if not (lo_bound - tol <= price <= hi_bound + tol):
        raise ValueError("price outside no-arbitrage bounds")
    sigma = 0.2
    for _ in range(max_iter):
        diff = bs_price(S, K, T, r, sigma, q, kind) - price
        if abs(diff) < tol:
            return sigma
        vega = greeks(S, K, T, r, sigma, q, kind)["vega"]
        if vega < 1e-8:
            break
        new = sigma - diff / vega
        if not (1e-6 < new < 10.0):
            break
        sigma = new
    return brentq(lambda s: bs_price(S, K, T, r, s, q, kind) - price, 1e-9, 10.0, xtol=1e-14)


def mc_price(S, K, T, r, sigma, q=0.0, kind="call", n_paths=100_000, seed=None, method="plain"):
    """蒙特卡洛定价（风险中性 GBM）。method: plain / antithetic / control。返回 (价格, 标准误)。"""
    rng = np.random.default_rng(seed)
    disc = np.exp(-r * T)

    def terminal(z):
        return S * np.exp((r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)

    def payoff(st):
        return np.maximum(st - K, 0.0) if kind == "call" else np.maximum(K - st, 0.0)

    if method == "plain":
        y = disc * payoff(terminal(rng.standard_normal(n_paths)))
    elif method == "antithetic":
        z = rng.standard_normal(n_paths // 2)
        y = 0.5 * disc * (payoff(terminal(z)) + payoff(terminal(-z)))
    elif method == "control":
        st = terminal(rng.standard_normal(n_paths))
        y0 = disc * payoff(st)
        c = disc * st
        c_mean = S * np.exp(-q * T)
        cov = np.cov(y0, c)
        b = cov[0, 1] / cov[1, 1]
        y = y0 - b * (c - c_mean)
    else:
        raise ValueError(method)
    return float(y.mean()), float(y.std(ddof=1) / np.sqrt(len(y)))


def delta_hedge_pnl(S0, K, T, r, sigma_real, sigma_hedge, n_steps, n_paths, seed=None, mu=None):
    """卖出一份看涨期权（按 sigma_hedge 收取 BS 权利金），用 sigma_hedge 算的 delta 离散对冲。

    标的按真实波动 sigma_real、漂移 mu（默认 r）演化，对冲 n_steps 次。返回到期时每条路径的 P&L。
    """
    rng = np.random.default_rng(seed)
    mu = r if mu is None else mu
    dt = T / n_steps
    S = np.full(n_paths, float(S0))
    premium = bs_price(S0, K, T, r, sigma_hedge)
    delta = np.full(n_paths, greeks(S0, K, T, r, sigma_hedge)["delta"])
    cash = premium - delta * S
    for k in range(1, n_steps + 1):
        z = rng.standard_normal(n_paths)
        S = S * np.exp((mu - 0.5 * sigma_real**2) * dt + sigma_real * np.sqrt(dt) * z)
        cash = cash * np.exp(r * dt)
        if k < n_steps:
            tau = T - k * dt
            d1, _ = d1_d2(S, K, tau, r, sigma_hedge)
            new_delta = norm.cdf(d1)
            cash -= (new_delta - delta) * S
            delta = new_delta
    return cash + delta * S - np.maximum(S - K, 0.0)


def jump_garch_returns(
    n_days: int = 3780,
    seed: int | None = 0,
    omega: float = 2e-6,
    alpha: float = 0.08,
    beta: float = 0.90,
    jump_prob: float = 1 / 400,
    jump_mean: float = -0.06,
    jump_sd: float = 0.03,
):
    """带向下跳跃的 GARCH(1,1) 日收益：波动率会聚集，偶尔出现崩盘日，崩盘后波动率升高。

    长期年化波动约 16%；平均每 400 个交易日一次跳跃，跳幅 N(jump_mean, jump_sd)。
    这是“卖波动率”策略要面对的世界：平时平静，偶尔一次大亏，大亏之后更不平静。
    """
    import pandas as pd

    from quantlab.data import business_index

    rng = np.random.default_rng(seed)
    r = np.empty(n_days)
    var = omega / (1 - alpha - beta)
    for t in range(n_days):
        e = np.sqrt(var) * rng.standard_normal()
        if rng.random() < jump_prob:
            e += rng.normal(jump_mean, jump_sd)
        r[t] = e
        var = omega + alpha * e * e + beta * var
    return pd.Series(r, index=business_index(n_days), name="ret")
