"""执行：TWAP / VWAP 排程、Almgren-Chriss 最优轨迹、成本的期望与方差、执行模拟。

讲解见 labs/lab11_optimal_execution/README.md。这里是现成实现，lab 里直接调用。
约定：卖出 X 股（X > 0；买入用负数），分 N 个区间，每区间长 tau = T / N。
x 为持仓轨迹（长度 N+1，x[0] = X，x[N] = 0），n 为每区间卖出量 n_k = x_{k-1} - x_k。
线性冲击：临时冲击 eta * n / tau（只影响本次成交价），永久冲击 gamma * n（之后一直存在），
epsilon 为半个买卖价差。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def twap_schedule(X: float, N: int) -> np.ndarray:
    """等分到 N 个区间。"""
    return np.full(N, X / N)


def vwap_schedule(X: float, volume_profile) -> np.ndarray:
    """按成交量曲线的比例分配。"""
    v = np.asarray(volume_profile, dtype=float)
    if np.any(v < 0) or v.sum() <= 0:
        raise ValueError("volume profile must be non-negative with positive sum")
    return X * v / v.sum()


def holdings_from_trades(trades) -> np.ndarray:
    """每区间交易量 -> 持仓轨迹（长度 N+1）。"""
    n = np.asarray(trades, dtype=float)
    X = n.sum()
    return np.concatenate([[X], X - np.cumsum(n)])


def trades_from_holdings(x) -> np.ndarray:
    """持仓轨迹 -> 每区间交易量。"""
    x = np.asarray(x, dtype=float)
    return x[:-1] - x[1:]


def ac_kappa(lam: float, sigma: float, eta: float, gamma: float, tau: float) -> float:
    """Almgren-Chriss 的 kappa（离散版本）：轨迹衰减速度，随风险厌恶 lam 增大。"""
    eta_tilde = eta - 0.5 * gamma * tau
    if eta_tilde <= 0:
        raise ValueError("need eta > gamma * tau / 2")
    kt2 = lam * sigma**2 / eta_tilde
    return float(np.arccosh(1.0 + 0.5 * kt2 * tau**2) / tau)


def ac_trajectory(X, T, N, sigma, eta, gamma, lam) -> np.ndarray:
    """最优持仓轨迹 x_k = X sinh(kappa (T - t_k)) / sinh(kappa T)；lam -> 0 退化为 TWAP。"""
    tau = T / N
    t = np.arange(N + 1) * tau
    kappa = ac_kappa(lam, sigma, eta, gamma, tau)
    if kappa * T < 1e-8:
        return X * (1 - t / T)
    return X * np.sinh(kappa * (T - t)) / np.sinh(kappa * T)


def expected_cost(x, sigma, eta, gamma, epsilon, tau) -> float:
    """轨迹 x 的期望执行成本（implementation shortfall 的期望，单位：美元）。"""
    x = np.asarray(x, dtype=float)
    n = trades_from_holdings(x)
    eta_tilde = eta - 0.5 * gamma * tau
    return float(
        0.5 * gamma * x[0] ** 2 + epsilon * np.abs(n).sum() + eta_tilde / tau * (n**2).sum()
    )


def cost_variance(x, sigma, tau) -> float:
    """轨迹 x 的执行成本方差（来自执行期间的价格波动）。"""
    x = np.asarray(x, dtype=float)
    return float(sigma**2 * tau * (x[1:] ** 2).sum())


def efficient_frontier(X, T, N, sigma, eta, gamma, epsilon, lams) -> pd.DataFrame:
    """不同风险厌恶下的 (期望成本, 方差)，索引为 lam。"""
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
    """按轨迹 x 模拟执行，返回每条路径的 implementation shortfall（X S0 - 实际收入）。"""
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


# ---------------------------------------------------------------- 容量研究用的模拟市场

from dataclasses import dataclass  # noqa: E402


@dataclass(frozen=True)
class StrategyMarket:
    """一个高换手策略和它交易的市场（lab11 审计题）。

    gross       策略的纸面日收益（不含任何成本），年化 Sharpe 恰为 gross_sharpe
    weights     每天收盘的目标权重（天 × 股票，占 AUM 的比例）
    volume      每只股票每天的成交额（天 × 股票，货币单位）；偶有指数调仓之类的放量日
    sigma       每只股票的日波动率
    exec_log    历史大单的执行记录（每行一笔母单），用来校准冲击成本：
                    participation  母单金额 / 当日成交额
                    sigma          该股日波动率
                    side           +1 买入，-1 卖出
                    arrival        决策时（下单前）的价格
                    avg_price      我们的成交均价
                    market_vwap    当天全市场的成交均价（包含我们自己的成交）
    impact_coef 生成执行记录时用的真实冲击系数 Y：期望执行成本 = Y * sigma * sqrt(participation)
    """

    gross: pd.Series
    weights: pd.DataFrame
    volume: pd.DataFrame
    sigma: pd.Series
    exec_log: pd.DataFrame
    impact_coef: float


def strategy_market(
    n_days: int = 1260,
    n_names: int = 50,
    daily_turnover: float = 0.10,
    gross_sharpe: float = 1.5,
    gross_vol: float = 0.10,
    median_adv: float = 5e7,
    impact_coef: float = 0.8,
    n_orders: int = 400,
    seed: int | None = 0,
) -> StrategyMarket:
    """生成 StrategyMarket。

    执行记录的生成方式（平方根冲击律）：一笔参与率为 p 的母单，在一天里均匀执行。
    它造成的永久冲击随执行进度累积到 1.6 Y sigma sqrt(p)，我们每笔成交还额外付出
    0.2 Y sigma sqrt(p) 的临时冲击。于是相对决策价格的平均成本约为 Y sigma sqrt(p)；
    其他人在同一天按已被我们推高（或压低）的价格成交，所以全市场 VWAP 也被我们带着走。
    """
    from quantlab.data import business_index

    rng = np.random.default_rng(seed)
    idx = business_index(n_days)
    names = [f"S{i:02d}" for i in range(n_names)]

    # 纸面收益：标准化后恰好是 gross_sharpe / gross_vol（研究报告里“回测出来”的那条曲线）
    z = rng.standard_normal(n_days)
    z = (z - z.mean()) / z.std()
    daily_vol = gross_vol / np.sqrt(252)
    gross = pd.Series(
        gross_sharpe * daily_vol / np.sqrt(252) + daily_vol * z, index=idx, name="gross"
    )

    step = daily_turnover / (n_names * np.sqrt(2 / np.pi))
    weights = pd.DataFrame(
        np.cumsum(rng.normal(0, step, (n_days, n_names)), axis=0), index=idx, columns=names
    )

    base = median_adv * rng.lognormal(0, 0.3, n_names)
    vol = base * rng.lognormal(0, 0.4, (n_days, n_names))
    spikes = rng.random((n_days, n_names)) < 0.03
    vol[spikes] *= rng.uniform(10, 30, spikes.sum())
    volume = pd.DataFrame(vol, index=idx, columns=names)

    sigma = pd.Series(rng.uniform(0.012, 0.03, n_names), index=names, name="sigma")

    p = rng.uniform(0.005, 0.25, n_orders)
    s = rng.uniform(0.012, 0.03, n_orders)
    side = rng.choice([-1.0, 1.0], n_orders)
    arrival = rng.uniform(20, 200, n_orders)
    total = impact_coef * s * np.sqrt(p)
    drift = s * rng.standard_normal(n_orders) / np.sqrt(3)  # 执行期间与我们无关的价格变动
    prevailing = arrival * (1 + side * 0.8 * total + drift)  # 平均“当时价格”（含我们的永久冲击）
    avg_price = prevailing * (1 + side * 0.2 * total)
    market_vwap = p * avg_price + (1 - p) * prevailing
    exec_log = pd.DataFrame(
        {
            "participation": p,
            "sigma": s,
            "side": side,
            "arrival": arrival,
            "avg_price": avg_price,
            "market_vwap": market_vwap,
        }
    )
    return StrategyMarket(gross, weights, volume, sigma, exec_log, impact_coef)
