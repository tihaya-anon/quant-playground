"""Lab 06 · 均值回归与配对交易。

把每个 `raise NotImplementedError` 换成你的实现，然后 `make test LAB=06`。

数据约定（quantlab.data.cointegrated_pair）：列 "x"、"y" 是价格，"spread" 是真实的对数价差。
    log y_t = a + β log x_t + spread_t，spread 是 AR(1)/OU 过程。
"""

from __future__ import annotations

import pandas as pd

# ---------------------------------------------------------------------------
# 1. AR(1) 与 OU
# ---------------------------------------------------------------------------


def fit_ar1(x: pd.Series) -> tuple[float, float, float]:
    """用 OLS 拟合 x_t = c + φ x_{t-1} + ε_t。

    返回 (c, φ, σ_ε)，σ_ε 为残差标准差，自由度 n - 2（n 为回归样本数）。
    先 dropna。
    """
    raise NotImplementedError


def half_life(phi: float) -> float:
    """均值回归半衰期：偏离衰减一半所需期数，φ^h = 1/2  =>  h = -ln2 / ln φ。

    φ 不在 (0, 1) 内（不回归或振荡）时返回 float("inf")。
    """
    raise NotImplementedError


def ou_from_ar1(
    c: float, phi: float, sigma_eps: float, dt: float = 1.0
) -> tuple[float, float, float]:
    """把 AR(1) 参数换算成 OU 过程 dX = κ(θ - X)dt + σ dW 的参数。

    OU 的精确离散化：X_{t+dt} = θ(1 - e^{-κdt}) + e^{-κdt} X_t + ε，
                     Var(ε) = σ^2 (1 - e^{-2κdt}) / (2κ)。
    据此反解：κ = -ln φ / dt，θ = c / (1 - φ)，σ = σ_ε sqrt(2κ / (1 - φ^2))。
    返回 (κ, θ, σ)。
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# 2. 平稳性与协整
# ---------------------------------------------------------------------------


def adf_pvalue(x: pd.Series) -> float:
    """Augmented Dickey-Fuller 检验的 p 值（原假设：有单位根，即不平稳）。

    用 statsmodels.tsa.stattools.adfuller，autolag="AIC"，默认含常数项。先 dropna。
    """
    raise NotImplementedError


def engle_granger(log_y: pd.Series, log_x: pd.Series) -> dict:
    """Engle-Granger 两步法。

    第一步：OLS  log_y = a + β log_x + u。
    第二步：检验残差 u 是否平稳。注意 u 是 *估计* 出来的（OLS 天然让它看起来更平稳），
           所以不能直接用普通 ADF 临界值，要用 MacKinnon 的协整临界值——
           statsmodels.tsa.stattools.coint(log_y, log_x) 就是这么做的，p 值用它。

    返回 dict：
        "hedge_ratio": β
        "intercept":   a
        "spread":      u = log_y - β log_x - a（pd.Series，与输入同索引）
        "pvalue":      coint 的 p 值
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# 3. 交易规则与回测
# ---------------------------------------------------------------------------


def rolling_zscore(spread: pd.Series, window: int = 60) -> pd.Series:
    """z_t = (s_t - 滚动均值_t) / 滚动标准差_t（ddof=1），窗口包含 t 本身。前 window-1 个为 NaN。"""
    raise NotImplementedError


def pair_positions(z: pd.Series, entry: float = 2.0, exit: float = 0.5) -> pd.Series:
    """z-score 进出场状态机，返回每日收盘后的目标仓位 ∈ {-1, 0, +1}（float）。

    仓位是“价差”的仓位：+1 = 做多价差（买 y、卖 β 份 x），-1 = 做空价差。
    逐日处理（状态依赖于前一天，所以需要循环）：
        - z 为 NaN：仓位 0（强制平仓）
        - 空仓时：z > entry 开空(-1)；z < -entry 开多(+1)；否则保持 0
        - 持多(+1)时：z > -exit 平仓到 0；否则保持
        - 持空(-1)时：z < exit 平仓到 0；否则保持
    注意平仓当天不会同时反手开新仓；第二天再按空仓规则判断。
    """
    raise NotImplementedError


def pair_returns(
    y: pd.Series, x: pd.Series, hedge_ratio, positions: pd.Series, cost_bps: float = 0.0
) -> pd.Series:
    """配对组合的日收益。

    positions[t] 是 t 日收盘后的仓位，所以它赚的是 t+1 的收益（必须错开一期！）：
        ret[t] = positions[t-1] * (r_y[t] - β r_x[t]) / (1 + |β|)
    r_y、r_x 为简单收益；除以 (1 + |β|) 是把总敞口归一化为 1。
    hedge_ratio 可以是标量，也可以是与 y 同索引的 Series（时变 β，用 t-1 的值）。

    交易成本：在 t 日调仓 |positions[t] - positions[t-1]|（第一天相对 0），
    成本 = 调仓量 * cost_bps / 1e4，计入 t+1 日的收益（与持仓收益同一天扣）。

    返回索引为 y.index[1:] 的 Series，NaN 填 0。
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# 4. 进阶：Kalman 滤波动态对冲比例
# ---------------------------------------------------------------------------


def kalman_hedge_ratio(
    log_y: pd.Series, log_x: pd.Series, delta: float = 1e-4, obs_var: float = 1e-3
) -> pd.DataFrame:
    """用 Kalman 滤波估计时变的 (β_t, α_t)。

    状态方程（随机游走）：θ_t = θ_{t-1} + w_t,  w_t ~ N(0, Q),  Q = delta / (1 - delta) · I_2
    观测方程：           log_y_t = H_t θ_t + v_t,  H_t = [log_x_t, 1],  v_t ~ N(0, obs_var)
    θ = [β, α]，初值 θ_0 = 0，P_0 = I。

    每一步（t = 0..n-1）：
        预测：P ← P + Q                       （θ 的预测就是上一步的 θ）
        新息：e = y_t - H θ，S = H P H' + obs_var
        增益：K = P H' / S
        更新：θ ← θ + K e，P ← P - K H P
    记录更新后的 θ。

    返回 DataFrame，列 ["beta", "alpha"]，与输入同索引。
    delta 越大，越相信 β 在变化（跟得快但噪声大）。
    """
    raise NotImplementedError
