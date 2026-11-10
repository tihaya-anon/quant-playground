import numpy as np
import pandas as pd
import pytest

from quantlab import execution as lab

# Almgren & Chriss (2000) 原文的数值例子：卖出 100 万股，5 天，每半天交易一次。
P = dict(
    X=1e6,
    T=5.0,
    N=10,
    sigma=0.95,  # 每天的价格波动（美元）
    eta=2.5e-6,  # 临时冲击
    gamma=2.5e-7,  # 永久冲击
    epsilon=0.0625,  # 半个买卖价差
    S0=50.0,
)
TAU = P["T"] / P["N"]
LAM = 1e-6


def traj(lam=LAM, **kw):
    p = {**P, **kw}
    return lab.ac_trajectory(p["X"], p["T"], p["N"], p["sigma"], p["eta"], p["gamma"], lam)


# ---------- 排程 ----------


def test_twap():
    n = lab.twap_schedule(1000.0, 8)
    assert n.shape == (8,)
    assert np.allclose(n, 125.0)


def test_vwap_follows_profile():
    # 典型的 U 型日内成交量
    profile = np.array([3, 2, 1.5, 1, 1, 1.5, 2, 4], dtype=float)
    n = lab.vwap_schedule(8000.0, profile)
    assert n.sum() == pytest.approx(8000.0)
    assert np.allclose(n / n.sum(), profile / profile.sum())
    with pytest.raises(ValueError):
        lab.vwap_schedule(100.0, [1.0, -1.0, 2.0])


def test_holdings_trades_roundtrip():
    n = np.array([10.0, 20.0, 30.0, 40.0])
    x = lab.holdings_from_trades(n)
    assert np.allclose(x, [100, 90, 70, 40, 0])
    assert np.allclose(lab.trades_from_holdings(x), n)


# ---------- Almgren-Chriss 轨迹 ----------


def test_kappa_matches_paper():
    # 原文：lambda = 1e-6 时 kappa 约 0.6（每天）
    k = lab.ac_kappa(LAM, P["sigma"], P["eta"], P["gamma"], TAU)
    assert k == pytest.approx(0.6, abs=0.02)
    # 小 tau 极限：kappa -> sqrt(lambda sigma^2 / eta_tilde)
    k_small = lab.ac_kappa(LAM, P["sigma"], P["eta"], 0.0, 1e-4)
    assert k_small == pytest.approx(np.sqrt(LAM * P["sigma"] ** 2 / P["eta"]), rel=1e-4)


def test_kappa_rejects_bad_params():
    with pytest.raises(ValueError):
        lab.ac_kappa(LAM, 1.0, 1e-7, 1e-6, 1.0)  # eta_tilde < 0


def test_trajectory_boundary():
    x = traj()
    assert x.shape == (P["N"] + 1,)
    assert x[0] == pytest.approx(P["X"])
    assert x[-1] == pytest.approx(0.0, abs=1e-6)
    assert np.all(np.diff(x) < 0)  # 卖出时持仓单调下降


def test_trajectory_lambda_zero_is_twap():
    x = traj(lam=0.0)
    assert np.allclose(x, P["X"] * (1 - np.arange(P["N"] + 1) / P["N"]))
    x_tiny = traj(lam=1e-14)
    assert np.allclose(x_tiny, x, rtol=1e-4, atol=1.0)


def test_risk_aversion_front_loads():
    x_low, x_mid, x_high = traj(lam=1e-8), traj(lam=1e-6), traj(lam=1e-5)
    # 越怕风险，越早卖：第一期后的剩余持仓越少
    assert x_high[1] < x_mid[1] < x_low[1]
    # 第一笔交易比最后一笔大
    n = lab.trades_from_holdings(x_mid)
    assert n[0] > n[-1]


def test_trajectory_is_optimal():
    # 在 E + lambda V 意义下，AC 轨迹应优于随机扰动后的轨迹
    x = traj()

    def objective(h):
        return lab.expected_cost(
            h, P["sigma"], P["eta"], P["gamma"], P["epsilon"], TAU
        ) + LAM * lab.cost_variance(h, P["sigma"], TAU)

    best = objective(x)
    rng = np.random.default_rng(0)
    for _ in range(50):
        h = x.copy()
        h[1:-1] += rng.normal(0, 2e4, P["N"] - 1)
        h[1:-1] = np.clip(h[1:-1], 0, P["X"])
        h[1:-1] = np.sort(h[1:-1])[::-1]
        assert objective(h) >= best - 1e-6
    assert objective(x) < objective(traj(lam=0.0))


# ---------- 成本与方差 ----------


def test_expected_cost_twap_closed_form():
    x = traj(lam=0.0)
    E = lab.expected_cost(x, P["sigma"], P["eta"], P["gamma"], P["epsilon"], TAU)
    X, T, eta, g, eps = P["X"], P["T"], P["eta"], P["gamma"], P["epsilon"]
    expected = 0.5 * g * X**2 + eps * X + (eta - 0.5 * g * TAU) * X**2 / T
    assert E == pytest.approx(expected, rel=1e-10)


def test_variance_twap_closed_form():
    x = traj(lam=0.0)
    V = lab.cost_variance(x, P["sigma"], TAU)
    X, T, N = P["X"], P["T"], P["N"]
    expected = (1 / 3) * P["sigma"] ** 2 * X**2 * T * (1 - 1 / N) * (1 - 1 / (2 * N))
    assert V == pytest.approx(expected, rel=1e-10)


def test_efficient_frontier_monotone():
    lams = [0.0, 1e-8, 1e-7, 1e-6, 1e-5]
    ef = lab.efficient_frontier(
        P["X"], P["T"], P["N"], P["sigma"], P["eta"], P["gamma"], P["epsilon"], lams
    )
    assert isinstance(ef, pd.DataFrame)
    assert list(ef.index) == lams
    assert {"expected_cost", "variance"} <= set(ef.columns)
    assert np.all(np.diff(ef["expected_cost"].to_numpy()) > 0)
    assert np.all(np.diff(ef["variance"].to_numpy()) < 0)


# ---------- 模拟 ----------


@pytest.mark.parametrize("lam", [0.0, 1e-6])
def test_simulation_matches_analytics(lam):
    x = traj(lam=lam)
    n_paths = 100_000
    is_ = lab.simulate_shortfall(
        x, P["S0"], P["sigma"], P["eta"], P["gamma"], P["epsilon"], TAU, n_paths, seed=7
    )
    assert is_.shape == (n_paths,)
    E = lab.expected_cost(x, P["sigma"], P["eta"], P["gamma"], P["epsilon"], TAU)
    V = lab.cost_variance(x, P["sigma"], TAU)
    assert abs(is_.mean() - E) < 4 * np.sqrt(V / n_paths)
    assert is_.var() == pytest.approx(V, rel=0.03)


def test_simulation_zero_vol_is_deterministic():
    x = traj(lam=0.0)
    is_ = lab.simulate_shortfall(
        x, P["S0"], 0.0, P["eta"], P["gamma"], P["epsilon"], TAU, 5, seed=0
    )
    E = lab.expected_cost(x, 0.0, P["eta"], P["gamma"], P["epsilon"], TAU)
    assert np.allclose(is_, E)


def test_simulation_buy_order():
    # 买入：持仓为负数（x 从 -X 到 0），成本仍应为正
    x = -traj(lam=0.0)
    is_ = lab.simulate_shortfall(
        x, P["S0"], 0.0, P["eta"], P["gamma"], P["epsilon"], TAU, 3, seed=0
    )
    assert np.all(is_ > 0)
    E = lab.expected_cost(x, 0.0, P["eta"], P["gamma"], P["epsilon"], TAU)
    assert np.allclose(is_, E)


def test_simulation_seed_reproducible():
    x = traj()
    a = lab.simulate_shortfall(x, 50.0, 0.95, P["eta"], P["gamma"], P["epsilon"], TAU, 100, seed=1)
    b = lab.simulate_shortfall(x, 50.0, 0.95, P["eta"], P["gamma"], P["epsilon"], TAU, 100, seed=1)
    assert np.array_equal(a, b)
