"""Lab 11 · 最优执行：TWAP、VWAP 与 Almgren-Chriss。

把每个 `raise NotImplementedError` 换成你的实现，然后 `make test LAB=11`。

记号（Almgren & Chriss 2000，离散版本）：
- 要在 [0, T] 内卖出 X 股，分成 N 个区间，tau = T / N，t_k = k tau。
- 持仓轨迹 x_0 = X, x_1, ..., x_N = 0（长度 N+1）；第 k 笔交易 n_k = x_{k-1} - x_k（长度 N）。
  卖出时 n_k > 0；买入可以用负的持仓表示（x_0 = -X），所有公式照样成立。
- 价格是**算术**随机游走（单位：美元），sigma 是每单位时间的价格波动。
- 永久冲击 g(v) = gamma v，临时冲击 h(v) = epsilon sgn(n) + eta v，交易速率 v = n_k / tau。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def twap_schedule(X: float, N: int) -> np.ndarray:
    """TWAP：N 个区间平均分配，返回长度 N 的交易量数组 n_k。"""
    raise NotImplementedError


def vwap_schedule(X: float, volume_profile) -> np.ndarray:
    """VWAP：按预期成交量占比分配，n_k = X * v_k / sum(v)。

    volume_profile 有负数或总和不为正时抛 ValueError。
    """
    raise NotImplementedError


def holdings_from_trades(trades) -> np.ndarray:
    """交易序列 n（长度 N）-> 持仓轨迹 x（长度 N+1），x_0 = sum(n)，x_N = 0。"""
    raise NotImplementedError


def trades_from_holdings(x) -> np.ndarray:
    """持仓轨迹 x（长度 N+1）-> 交易序列 n_k = x_{k-1} - x_k（长度 N）。"""
    raise NotImplementedError


def ac_kappa(lam: float, sigma: float, eta: float, gamma: float, tau: float) -> float:
    """Almgren-Chriss 的衰减率 kappa（单位：1/时间）。

        eta_tilde   = eta - gamma * tau / 2         （必须 > 0，否则抛 ValueError）
        kappa_tilde^2 = lam * sigma^2 / eta_tilde
        cosh(kappa * tau) = 1 + kappa_tilde^2 * tau^2 / 2

    当 tau -> 0 时 kappa -> kappa_tilde。lam = 0 时 kappa = 0。
    """
    raise NotImplementedError


def ac_trajectory(X, T, N, sigma, eta, gamma, lam) -> np.ndarray:
    """最小化 E[cost] + lam * Var[cost] 的最优持仓轨迹（长度 N+1）：

        x_j = X * sinh(kappa (T - t_j)) / sinh(kappa T)

    kappa -> 0（lam -> 0，风险中性）时上式的极限是线性轨迹 X (1 - t_j / T)，即 TWAP；
    注意数值上要单独处理 kappa * T 极小的情况，不能 0/0。
    """
    raise NotImplementedError


def expected_cost(x, sigma, eta, gamma, epsilon, tau) -> float:
    """给定轨迹的期望执行成本（implementation shortfall 的期望）：

        E = 0.5 gamma X^2 + epsilon * sum |n_k| + (eta_tilde / tau) * sum n_k^2

    其中 X = x_0，eta_tilde = eta - gamma tau / 2。sigma 不参与（保留参数是为了接口统一）。
    推导提示：永久冲击成本 sum_k x_k * gamma * n_k，用 sum x_k n_k 的“分部求和”化简。
    """
    raise NotImplementedError


def cost_variance(x, sigma, tau) -> float:
    """执行成本的方差：V = sigma^2 * tau * sum_{k=1}^{N} x_k^2。

    直觉：第 k 个区间里你还拿着 x_k 股，承受 sigma sqrt(tau) 的价格风险。
    """
    raise NotImplementedError


def efficient_frontier(X, T, N, sigma, eta, gamma, epsilon, lams) -> pd.DataFrame:
    """对每个风险厌恶系数 lam 计算最优轨迹的 (E, V)。

    返回 DataFrame：索引为 lam（顺序与输入相同，索引名 "lam"），
    列至少包括 "expected_cost" 与 "variance"。
    """
    raise NotImplementedError


def simulate_shortfall(x, S0, sigma, eta, gamma, epsilon, tau, n_paths, seed=None) -> np.ndarray:
    """蒙特卡洛模拟执行，返回每条路径的 implementation shortfall（长度 n_paths）。

    动态（k = 1..N，xi_k ~ N(0, 1) 独立，用 np.random.default_rng(seed)，
    一次性抽 shape (n_paths, N) 的随机数，保证可复现）：

        成交价   S~_k = S_{k-1} - epsilon * sgn(n_k) - eta * n_k / tau
        价格演化 S_k  = S_{k-1} + sigma sqrt(tau) xi_k - gamma * n_k

    收入 = sum_k n_k S~_k，IS = x_0 * S0 - 收入（正数代表成本）。
    样本均值/方差应分别逼近 expected_cost / cost_variance。
    """
    raise NotImplementedError
