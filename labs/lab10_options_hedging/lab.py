"""Lab 10 · 期权定价与 Delta 对冲。

把每个 `raise NotImplementedError` 换成你的实现，然后 `make test LAB=10`。

全局约定：
- S 现价，K 行权价，T 剩余期限（年），r 连续复利无风险利率，sigma 年化波动率，q 连续股息率。
- kind 取 "call" 或 "put"，其它值抛 ValueError。
- 可以用 scipy.stats.norm（cdf / pdf）与 scipy.optimize.brentq。
"""

from __future__ import annotations

import numpy as np


def d1_d2(S, K, T, r, sigma, q=0.0):
    """返回 (d1, d2)。

        d1 = [ln(S/K) + (r - q + sigma^2/2) T] / (sigma sqrt(T))
        d2 = d1 - sigma sqrt(T)

    参数可以是标量或 ndarray（按 numpy 广播）。
    """
    raise NotImplementedError


def bs_price(S, K, T, r, sigma, q=0.0, kind="call"):
    """Black-Scholes-Merton 欧式期权价格（含连续股息率 q）。

        call = S e^{-qT} N(d1) - K e^{-rT} N(d2)
        put  = K e^{-rT} N(-d2) - S e^{-qT} N(-d1)

    标量输入返回 float；任一参数为数组时返回 ndarray（广播）。
    """
    raise NotImplementedError


def put_call_parity_gap(call, put, S, K, T, r, q=0.0):
    """Put-call parity 的偏离：C - P - (S e^{-qT} - K e^{-rT})。

    无套利时为 0。一个正的 gap 意味着“卖看涨、买看跌、买现货、借钱”可以锁定利润。
    """
    raise NotImplementedError


def greeks(S, K, T, r, sigma, q=0.0, kind="call") -> dict:
    """解析 Greeks，返回 dict，键为 "delta", "gamma", "vega", "theta", "rho"（标量输入）。

    单位约定（测试会用有限差分逐一核对）：
    - delta = dV/dS，gamma = d²V/dS²
    - vega  = dV/dsigma（sigma 变化 1.0，即 100 个波动率点；不是 1%）
    - theta = dV/dt，t 是日历时间，即 **-dV/dT**，单位为“每年”。多头期权通常为负。
    - rho   = dV/dr

    提示：call 与 put 的 gamma、vega 相同；delta_call - delta_put = e^{-qT}。
    """
    raise NotImplementedError


def implied_vol(price, S, K, T, r, q=0.0, kind="call", tol=1e-10, max_iter=50) -> float:
    """由市场价反推隐含波动率。

    要求：
    1. 先检查无套利边界：price 必须落在 [bs_price(sigma→0), bs_price(sigma 很大, 如 10)] 之间
       （可以放宽 tol），否则抛 ValueError。
    2. 先用 Newton 法：sigma_{n+1} = sigma_n - (BS(sigma_n) - price) / vega(sigma_n)，初值 0.2。
    3. Newton 在 vega 很小（深度虚值/实值）时会飞出去。一旦 vega 太小或迭代跑出 (0, 10)，
       或者 max_iter 内没收敛，改用 brentq 在 [1e-9, 10] 上求根。
    4. 返回 float，使得 bs_price(..., iv, ...) 与 price 之差 < 1e-8 量级。
    """
    raise NotImplementedError


def mc_price(
    S, K, T, r, sigma, q=0.0, kind="call", n_paths=100_000, seed=None, method="plain"
) -> tuple[float, float]:
    """风险中性蒙特卡洛定价，返回 (价格估计, 标准误)。

    S_T = S exp((r - q - sigma^2/2) T + sigma sqrt(T) Z)，价格 = e^{-rT} E[payoff(S_T)]。
    用 np.random.default_rng(seed)。method：

    - "plain"：n_paths 个独立 Z。标准误 = 样本标准差 / sqrt(n_paths)。
    - "antithetic"：抽 n_paths // 2 个 Z，每个 Z 与 -Z 组成一对，把每对的平均贴现收益当作一个
      样本（这样样本之间才独立），标准误用这些“对平均”计算。
    - "control"：以贴现后的 S_T 作控制变量 C = e^{-rT} S_T，已知 E[C] = S e^{-qT}。
      Y_cv = Y - b (C - E[C])，b = Cov(Y, C) / Var(C)（从样本估计）。

    同一个 seed 必须给出完全相同的结果。
    """
    raise NotImplementedError


def delta_hedge_pnl(
    S0, K, T, r, sigma_real, sigma_hedge, n_steps, n_paths, seed=None, mu=None
) -> np.ndarray:
    """**卖出**一份欧式看涨期权并做离散 delta 对冲，返回每条路径在到期日的 P&L（长度 n_paths）。

    设定（q = 0）：
    - 标的真实动态：GBM，漂移 mu（None 时取 r），波动率 sigma_real，dt = T / n_steps，
      用精确解逐步模拟。
    - 你按 sigma_hedge（“隐含波动率”）收取权利金 V0 = bs_price(S0, K, T, r, sigma_hedge)，
      并且始终用 sigma_hedge 计算 BS delta。
    - t=0：收到 V0，买入 delta_0 股，现金 = V0 - delta_0 * S0（可为负，即借钱）。
    - 每一步：现金先按 e^{r dt} 计息；价格走一步；若还没到期（k < n_steps），按新的剩余期限
      T - k dt 重算 delta 并调仓，调仓的钱从现金账户出入。
    - 到期：P&L = 现金 + delta * S_T - max(S_T - K, 0)。

    你应该观察到：
    - sigma_real == sigma_hedge 时，P&L 均值约 0，标准差 ∝ 1/sqrt(n_steps)，且与 mu 无关。
    - 卖方是 short gamma：P&L ≈ ∫ 0.5 Γ S² (sigma_hedge² - sigma_real²) dt，
      真实波动大于隐含时亏钱。
    """
    raise NotImplementedError
