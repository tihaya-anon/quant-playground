"""Lab 10 参考答案。"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def d1_d2(S, K, T, r, sigma, q=0.0):
    S, K, T, sigma = (np.asarray(v, dtype=float) for v in (S, K, T, sigma))
    vs = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / vs
    return d1, d1 - vs


def bs_price(S, K, T, r, sigma, q=0.0, kind="call"):
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
    return call - put - (S * np.exp(-q * T) - K * np.exp(-r * T))


def greeks(S, K, T, r, sigma, q=0.0, kind="call"):
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
