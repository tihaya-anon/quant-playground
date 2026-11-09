"""Lab 07 · 波动率建模：rolling、EWMA、GARCH(1,1) 与波动率目标。

把每个 `raise NotImplementedError` 换成你的实现，然后 `make test LAB=07`。
约定：returns 为日简单收益（数值很小时与对数收益差别可忽略）；一年 252 个交易日。
“方差/波动率”不加说明时都是 *日* 频的；带 ann 的才是年化。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def realized_vol(returns: pd.Series, window: int = 21, periods: int = TRADING_DAYS) -> pd.Series:
    """滚动窗口年化波动率：rolling std(ddof=1) * sqrt(periods)。前 window-1 个值为 NaN。"""
    raise NotImplementedError


def ewma_variance(returns: pd.Series, lam: float = 0.94, init: float | None = None) -> pd.Series:
    """RiskMetrics EWMA 方差 *预测*。

        s2[0] = init（默认：前 20 个收益平方的均值）
        s2[t] = lam * s2[t-1] + (1 - lam) * r[t-1]^2

    关键：s2[t] 只用到 t-1 及之前的收益，它是“站在 t-1 收盘、对第 t 天方差的预测”。
    返回与 returns 同索引的 Series。
    """
    raise NotImplementedError


def garch_variance(params, eps: np.ndarray) -> np.ndarray:
    """GARCH(1,1) 条件方差递推。

    params = (omega, alpha, beta)；eps 是去均值后的残差。
        var[0] = np.var(eps)                 （用样本方差初始化）
        var[t] = omega + alpha * eps[t-1]^2 + beta * var[t-1]
    返回 ndarray，长度同 eps。
    """
    raise NotImplementedError


def garch_neg_loglik(params, eps: np.ndarray) -> float:
    """正态新息下 GARCH(1,1) 的负对数似然：

        NLL = 0.5 * sum_t [ log(2 pi) + log var_t + eps_t^2 / var_t ]

    若出现非正或非有限方差，返回一个很大的数（如 1e10），让优化器远离。
    """
    raise NotImplementedError


def fit_garch(returns: pd.Series) -> dict[str, float]:
    """用 scipy.optimize.minimize 做 GARCH(1,1) 极大似然估计。

    步骤：mu = 样本均值，eps = r - mu；最小化 garch_neg_loglik。
    约束：omega > 0, alpha >= 0, beta >= 0, alpha + beta < 1。
    两种常见做法（任选）：
      1. 有界 + 不等式约束：method="SLSQP"，bounds + constraints。
      2. 重参数化成无约束问题（推荐）：例如 persistence = sigmoid(a)，
         alpha = persistence * sigmoid(b)，beta = persistence - alpha，
         omega = 长期方差 * (1 - persistence)，长期方差 = 样本方差 * exp(c)。
    提示：omega 的量级 ~1e-6，直接优化它数值上很糟糕——这正是重参数化的意义。

    返回 {"mu", "omega", "alpha", "beta", "loglik"}（loglik 为对数似然，即 -NLL）。
    """
    raise NotImplementedError


def long_run_variance(omega: float, alpha: float, beta: float) -> float:
    """无条件（长期）方差 V_L = omega / (1 - alpha - beta)。"""
    raise NotImplementedError


def garch_forecast(
    omega, alpha, beta, last_var: float, last_eps: float, horizon: int
) -> np.ndarray:
    """站在 T 收盘，对 T+1, ..., T+horizon 的条件方差预测。

        E[var_{T+1}] = omega + alpha * eps_T^2 + beta * var_T
        E[var_{T+h}] = V_L + (alpha + beta)^(h-1) * (E[var_{T+1}] - V_L)

    返回长度 horizon 的 ndarray。
    """
    raise NotImplementedError


def half_life(alpha: float, beta: float) -> float:
    """波动率冲击的半衰期（天）：(alpha + beta)^h = 1/2 解出 h。"""
    raise NotImplementedError


def vol_target_weights(
    sigma_daily: pd.Series,
    target_vol: float = 0.10,
    max_leverage: float = 2.0,
    periods: int = TRADING_DAYS,
) -> pd.Series:
    """波动率目标仓位：w_t = target_vol / (sigma_daily_t * sqrt(periods))，上限 max_leverage。"""
    raise NotImplementedError


def vol_target_returns(
    returns: pd.Series, target_vol: float = 0.10, lam: float = 0.94, max_leverage: float = 2.0
) -> pd.Series:
    """波动率目标策略的日收益：w_t * r_t。

    w_t 必须只依赖 t-1 及以前的信息——用 ewma_variance 的输出（它本身就是对 t 的预测），
    *不要*再额外 shift，也绝不能用包含 r_t 的波动率估计。
    """
    raise NotImplementedError
