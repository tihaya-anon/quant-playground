"""Lab 05 · 组合优化。

把每个 `raise NotImplementedError` 换成你的实现，然后 `make test LAB=05`。

约定：mu 是长度 N 的期望收益向量，cov 是 N x N 协方差矩阵（两者同一频率，通常年化）。
权重 w 是长度 N 的 ndarray，除非特别说明都满足 sum(w) = 1，允许做空（负权重）。
求解线性方程组请用 np.linalg.solve，不要显式求逆。
"""

from __future__ import annotations

import numpy as np


def portfolio_vol(w, cov) -> float:
    """组合波动率 sqrt(w' Σ w)。"""
    raise NotImplementedError


# ---------------------------------------------------------------------------
# 1. 均值-方差的闭式解
# ---------------------------------------------------------------------------


def min_variance_weights(cov) -> np.ndarray:
    """全局最小方差组合：min w'Σw  s.t. 1'w = 1。

    拉格朗日一阶条件给出 w = Σ^{-1} 1 / (1' Σ^{-1} 1)。
    """
    raise NotImplementedError


def tangency_weights(mu, cov, rf: float = 0.0) -> np.ndarray:
    """切点组合（最大 Sharpe）：w ∝ Σ^{-1}(mu - rf)，再归一化到 sum = 1。"""
    raise NotImplementedError


def frontier_weights(mu, cov, target: float) -> np.ndarray:
    """给定目标收益的最小方差组合：min w'Σw  s.t. 1'w = 1, mu'w = target。

    记 A = 1'Σ^{-1}1, B = 1'Σ^{-1}mu, C = mu'Σ^{-1}mu, D = AC - B^2，则
        w = λ Σ^{-1}1 + γ Σ^{-1}mu,  λ = (C - B·target)/D,  γ = (A·target - B)/D
    （“两基金分离”：前沿上任何组合都是两个固定组合的线性组合。）
    """
    raise NotImplementedError


def efficient_frontier(mu, cov, targets) -> np.ndarray:
    """对每个目标收益算 frontier_weights 的波动率，返回与 targets 等长的 ndarray。

    理论上 σ^2(t) = (A t^2 - 2 B t + C) / D，是 (σ, t) 平面上的双曲线。
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# 2. 风险平价
# ---------------------------------------------------------------------------


def risk_contributions(w, cov) -> np.ndarray:
    """各资产的风险贡献占比。

    σ_p = sqrt(w'Σw) 是一次齐次函数，由欧拉定理 σ_p = Σ_i w_i ∂σ_p/∂w_i，
    第 i 项 RC_i = w_i (Σw)_i / σ_p 就是资产 i 的风险贡献。返回 RC_i / σ_p（加总为 1）。
    """
    raise NotImplementedError


def risk_parity_weights(cov, tol: float = 1e-10, max_iter: int = 10_000) -> np.ndarray:
    """风险平价（等风险贡献）组合：只做多，sum(w) = 1，所有 RC_i 相等。

    没有闭式解（对角协方差时 w_i ∝ 1/σ_i 除外），需要数值求解。可选做法：
        - 固定点迭代：w_i ← sqrt(w_i / (Σw)_i)，再归一化，直到收敛；
        - 或者最小化凸函数 0.5 w'Σw - (1/N) Σ log w_i（其解归一化后就是风险平价）；
        - 或者 scipy.optimize。
    结果的风险贡献需与 1/N 相差 < 1e-6。
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# 3. 协方差估计
# ---------------------------------------------------------------------------


def ledoit_wolf(returns) -> tuple[np.ndarray, float]:
    """Ledoit-Wolf (2004) 收缩协方差，目标为 μI（μ = 样本协方差的平均方差）。

    returns: n x p 收益矩阵（行是时间）。先去均值得到 X，S = X'X / n（注意除以 n 不是 n-1）。
        μ      = tr(S) / p
        γ      = ||S - μI||_F^2
        φ      = (1/n) Σ_t ||x_t x_t' - S||_F^2
               = Σ_{ij} (X^2)'(X^2) / n  -  ||S||_F^2     （逐元素平方，便于向量化）
        δ*     = clip(φ / (n γ), 0, 1)
        Σ_LW   = (1 - δ*) S + δ* μ I

    返回 (Σ_LW, δ*)。应与 sklearn.covariance.LedoitWolf().fit(returns) 的 covariance_ 和
    shrinkage_ 一致。直觉：样本越少（n 相对 p 越小），S 的噪声越大，δ* 越大。
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# 4. 带约束的优化（cvxpy）
# ---------------------------------------------------------------------------


def mean_variance_cvxpy(
    mu, cov, risk_aversion: float, long_only: bool = True, max_weight=None
) -> np.ndarray:
    """用 cvxpy 解  max  mu'w - (risk_aversion / 2) w'Σw
                    s.t. sum(w) = 1
                         w >= 0            （long_only 时）
                         w <= max_weight   （max_weight 不为 None 时）

    返回最优 w（一维 ndarray）。提示：cp.quad_form(w, cp.psd_wrap(cov))。
    无约束时你可以用拉格朗日乘子推出闭式解来验证（测试就是这么做的）。
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# 5. 样本内 vs 样本外
# ---------------------------------------------------------------------------


def oos_min_variance(
    train, test, estimator: str = "sample", periods: int = 252
) -> tuple[float, float]:
    """在 train 上估协方差、求最小方差组合，然后在 test 上看实际波动。

    train/test: 日收益矩阵（行时间，列资产）。
    estimator: "sample"（np.cov，ddof=1）或 "ledoit_wolf"（你上面的实现）；其他值 raise ValueError。

    返回 (predicted_vol, realized_vol)，都年化（乘 sqrt(periods)）：
        predicted = 用 train 估计的协方差算出的组合波动率
        realized  = 组合在 test 期间日收益 (test @ w) 的样本标准差（ddof=1）

    要观察的现象：资产数接近样本数时，样本协方差的最小方差组合“预测”波动很低，
    实际却高得多——优化器专门挑了估计误差最有利的方向（error maximization）。
    """
    raise NotImplementedError
