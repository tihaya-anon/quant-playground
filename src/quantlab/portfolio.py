"""组合构建工具：均值方差闭式解、有效前沿、风险平价、Ledoit-Wolf 收缩、cvxpy 约束优化。

讲解见 labs/lab05_portfolio_optimization/README.md。这里是现成实现，lab 里直接调用。

约定：mu 是期望收益向量，cov 是协方差矩阵，权重和为 1。
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np
import pandas as pd


def portfolio_vol(w, cov) -> float:
    """组合波动率 sqrt(w' Σ w)。"""
    w = np.asarray(w, dtype=float)
    return float(np.sqrt(w @ cov @ w))


def min_variance_weights(cov) -> np.ndarray:
    """全局最小方差组合：w ∝ Σ^{-1} 1（允许做空）。"""
    ones = np.ones(len(cov))
    x = np.linalg.solve(cov, ones)
    return x / x.sum()


def tangency_weights(mu, cov, rf: float = 0.0) -> np.ndarray:
    """切点（最大 Sharpe）组合：w ∝ Σ^{-1} (mu - rf)（允许做空）。"""
    ex = np.asarray(mu, dtype=float) - rf
    x = np.linalg.solve(cov, ex)
    return x / x.sum()


def frontier_weights(mu, cov, target: float) -> np.ndarray:
    """有效前沿上期望收益为 target 的最小方差组合（两基金定理的闭式解）。"""
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
    """一组目标收益对应的前沿波动率。"""
    return np.array([portfolio_vol(frontier_weights(mu, cov, t), cov) for t in targets])


def risk_contributions(w, cov) -> np.ndarray:
    """各资产的风险贡献占比 w_i (Σw)_i / w'Σw，和为 1。"""
    w = np.asarray(w, dtype=float)
    marginal = cov @ w
    rc = w * marginal
    return rc / rc.sum()


def risk_parity_weights(cov, tol: float = 1e-10, max_iter: int = 10_000) -> np.ndarray:
    """风险平价（等风险贡献）组合，固定点迭代求解，只做多。"""
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
    """Ledoit-Wolf 收缩协方差（目标为 μI），返回 (协方差, 收缩强度)。"""
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
    """带约束的均值方差：max mu'w - λ/2 w'Σw，s.t. Σw=1、可选只做多与权重上限。"""
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
    """在 train 上估计协方差、构建最小方差组合，返回 (预测年化波动, test 上实现年化波动)。"""
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


# ---------------------------------------------------------------------------------------------
# 模拟资产收益（lab05 的实验与审计用）


def simulate_assets(
    n_assets: int = 10,
    n_days: int = 5040,
    sharpe=0.4,
    vol_range: tuple[float, float] = (0.10, 0.40),
    corr: float = 0.3,
    seed: int | None = 0,
) -> pd.DataFrame:
    """单因子相关结构的日收益，每个资产的真实年化 Sharpe 由 sharpe 给定（标量或数组）。

        r_i = mu_i + vol_i * (sqrt(corr) * f + sqrt(1 - corr) * e_i)

    默认所有资产真实 Sharpe 相同：此时“按预期收益加权”没有任何真实信息可用，
    理论上的切点组合就是按 Σ^{-1} vol 加权，而估计出来的 mu 全是噪声。
    """
    rng = np.random.default_rng(seed)
    vols = rng.uniform(*vol_range, n_assets)
    sr = np.broadcast_to(np.asarray(sharpe, dtype=float), (n_assets,))
    mu = sr * vols / 252
    d_vol = vols / np.sqrt(252)
    f = rng.standard_normal(n_days)[:, None]
    e = rng.standard_normal((n_days, n_assets))
    rets = mu + d_vol * (np.sqrt(corr) * f + np.sqrt(1 - corr) * e)
    idx = pd.bdate_range("2006-01-02", periods=n_days, name="date")
    return pd.DataFrame(rets, index=idx, columns=[f"A{i:02d}" for i in range(n_assets)])
