"""Lab 05 参考答案。"""

from __future__ import annotations

import cvxpy as cp
import numpy as np


def portfolio_vol(w, cov) -> float:
    w = np.asarray(w, dtype=float)
    return float(np.sqrt(w @ cov @ w))


def min_variance_weights(cov) -> np.ndarray:
    ones = np.ones(len(cov))
    x = np.linalg.solve(cov, ones)
    return x / x.sum()


def tangency_weights(mu, cov, rf: float = 0.0) -> np.ndarray:
    ex = np.asarray(mu, dtype=float) - rf
    x = np.linalg.solve(cov, ex)
    return x / x.sum()


def frontier_weights(mu, cov, target: float) -> np.ndarray:
    mu = np.asarray(mu, dtype=float)
    ones = np.ones(len(mu))
    inv_1 = np.linalg.solve(cov, ones)
    inv_mu = np.linalg.solve(cov, mu)
    a = ones @ inv_1
    b = ones @ inv_mu
    c = mu @ inv_mu
    d = a * c - b**2
    lam = (c - b * target) / d
    gam = (a * target - b) / d
    return lam * inv_1 + gam * inv_mu


def efficient_frontier(mu, cov, targets) -> np.ndarray:
    return np.array([portfolio_vol(frontier_weights(mu, cov, t), cov) for t in targets])


def risk_contributions(w, cov) -> np.ndarray:
    w = np.asarray(w, dtype=float)
    marginal = cov @ w
    rc = w * marginal
    return rc / rc.sum()


def risk_parity_weights(cov, tol: float = 1e-10, max_iter: int = 10_000) -> np.ndarray:
    cov = np.asarray(cov, dtype=float)
    w = 1 / np.sqrt(np.diag(cov))
    w /= w.sum()
    for _ in range(max_iter):
        # 固定点迭代：w_i ∝ 1 / (Σw)_i 的平方根形式（Spinu/Chaves 风格）
        m = cov @ w
        w_new = np.sqrt(w / m)
        w_new /= w_new.sum()
        if np.max(np.abs(w_new - w)) < tol:
            w = w_new
            break
        w = w_new
    return w


def ledoit_wolf(returns) -> tuple[np.ndarray, float]:
    X = np.asarray(returns, dtype=float)
    X = X - X.mean(axis=0)
    n, p = X.shape
    emp = X.T @ X / n
    mu = np.trace(emp) / p
    X2 = X**2
    phi = (X2.T @ X2).sum() / n - (emp**2).sum()
    gamma = ((emp - mu * np.eye(p)) ** 2).sum()
    shrink = 0.0 if gamma == 0 else min(1.0, max(0.0, phi / n / gamma))
    return (1 - shrink) * emp + shrink * mu * np.eye(p), float(shrink)


def mean_variance_cvxpy(
    mu, cov, risk_aversion: float, long_only: bool = True, max_weight=None
) -> np.ndarray:
    n = len(mu)
    w = cp.Variable(n)
    obj = cp.Maximize(np.asarray(mu) @ w - 0.5 * risk_aversion * cp.quad_form(w, cp.psd_wrap(cov)))
    cons = [cp.sum(w) == 1]
    if long_only:
        cons.append(w >= 0)
    if max_weight is not None:
        cons.append(w <= max_weight)
    cp.Problem(obj, cons).solve()
    return np.asarray(w.value).ravel()


def oos_min_variance(
    train, test, estimator: str = "sample", periods: int = 252
) -> tuple[float, float]:
    train = np.asarray(train, dtype=float)
    test = np.asarray(test, dtype=float)
    if estimator == "sample":
        cov = np.cov(train, rowvar=False)
    elif estimator == "ledoit_wolf":
        cov, _ = ledoit_wolf(train)
    else:
        raise ValueError(estimator)
    w = min_variance_weights(cov)
    predicted = portfolio_vol(w, cov) * np.sqrt(periods)
    realized = float((test @ w).std(ddof=1) * np.sqrt(periods))
    return predicted, realized
