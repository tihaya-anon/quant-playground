"""Lab 11 参考答案。"""

from __future__ import annotations

import numpy as np
import pandas as pd


def twap_schedule(X: float, N: int) -> np.ndarray:
    return np.full(N, X / N)


def vwap_schedule(X: float, volume_profile) -> np.ndarray:
    v = np.asarray(volume_profile, dtype=float)
    if np.any(v < 0) or v.sum() <= 0:
        raise ValueError("volume profile must be non-negative with positive sum")
    return X * v / v.sum()


def holdings_from_trades(trades) -> np.ndarray:
    n = np.asarray(trades, dtype=float)
    X = n.sum()
    return np.concatenate([[X], X - np.cumsum(n)])


def trades_from_holdings(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x[:-1] - x[1:]


def ac_kappa(lam: float, sigma: float, eta: float, gamma: float, tau: float) -> float:
    eta_tilde = eta - 0.5 * gamma * tau
    if eta_tilde <= 0:
        raise ValueError("need eta > gamma * tau / 2")
    kt2 = lam * sigma**2 / eta_tilde
    return float(np.arccosh(1.0 + 0.5 * kt2 * tau**2) / tau)


def ac_trajectory(X, T, N, sigma, eta, gamma, lam) -> np.ndarray:
    tau = T / N
    t = np.arange(N + 1) * tau
    kappa = ac_kappa(lam, sigma, eta, gamma, tau)
    if kappa * T < 1e-8:
        return X * (1 - t / T)
    return X * np.sinh(kappa * (T - t)) / np.sinh(kappa * T)


def expected_cost(x, sigma, eta, gamma, epsilon, tau) -> float:
    x = np.asarray(x, dtype=float)
    n = trades_from_holdings(x)
    eta_tilde = eta - 0.5 * gamma * tau
    return float(
        0.5 * gamma * x[0] ** 2 + epsilon * np.abs(n).sum() + eta_tilde / tau * (n**2).sum()
    )


def cost_variance(x, sigma, tau) -> float:
    x = np.asarray(x, dtype=float)
    return float(sigma**2 * tau * (x[1:] ** 2).sum())


def efficient_frontier(X, T, N, sigma, eta, gamma, epsilon, lams) -> pd.DataFrame:
    tau = T / N
    rows = []
    for lam in lams:
        x = ac_trajectory(X, T, N, sigma, eta, gamma, lam)
        rows.append(
            {
                "lam": lam,
                "expected_cost": expected_cost(x, sigma, eta, gamma, epsilon, tau),
                "variance": cost_variance(x, sigma, tau),
            }
        )
    return pd.DataFrame(rows).set_index("lam")


def simulate_shortfall(x, S0, sigma, eta, gamma, epsilon, tau, n_paths, seed=None) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    n = trades_from_holdings(x)
    N = len(n)
    rng = np.random.default_rng(seed)
    xi = rng.standard_normal((n_paths, N))
    S_prev = np.full(n_paths, float(S0))
    revenue = np.zeros(n_paths)
    for k in range(N):
        exec_price = S_prev - epsilon * np.sign(n[k]) - eta * n[k] / tau
        revenue += n[k] * exec_price
        S_prev = S_prev + sigma * np.sqrt(tau) * xi[:, k] - gamma * n[k]
    return x[0] * S0 - revenue
