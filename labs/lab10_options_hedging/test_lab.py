import numpy as np
import pytest

from labs._harness import load_impl

lab = load_impl(__file__)

BASE = dict(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2)


# ---------- 定价 ----------


def test_bs_known_values():
    # Hull 教科书经典数值
    assert lab.bs_price(**BASE, kind="call") == pytest.approx(10.4506, abs=1e-4)
    assert lab.bs_price(**BASE, kind="put") == pytest.approx(5.5735, abs=1e-4)


def test_bs_with_dividend():
    # q 等价于把现货换成 S e^{-qT}、r 不变
    p_q = lab.bs_price(100.0, 95.0, 0.5, 0.03, 0.25, q=0.02, kind="call")
    assert p_q == pytest.approx(9.83195, abs=1e-4)
    assert p_q == pytest.approx(
        lab.bs_price(100.0 * np.exp(-0.01), 95.0, 0.5, 0.03, 0.25), abs=1e-10
    )
    assert p_q < lab.bs_price(100.0, 95.0, 0.5, 0.03, 0.25, q=0.0, kind="call")


def test_d1_d2():
    d1, d2 = lab.d1_d2(100.0, 100.0, 1.0, 0.05, 0.2)
    assert float(d1) == pytest.approx(0.35)
    assert float(d2) == pytest.approx(0.15)


@pytest.mark.parametrize("K", [60.0, 90.0, 100.0, 120.0, 180.0])
@pytest.mark.parametrize("q", [0.0, 0.03])
def test_put_call_parity(K, q):
    args = dict(S=100.0, K=K, T=0.75, r=0.04, sigma=0.3, q=q)
    c = lab.bs_price(**args, kind="call")
    p = lab.bs_price(**args, kind="put")
    gap = lab.put_call_parity_gap(c, p, 100.0, K, 0.75, 0.04, q)
    assert gap == pytest.approx(0.0, abs=1e-10)
    # 违反 parity 时 gap 不为 0
    assert lab.put_call_parity_gap(c + 1, p, 100.0, K, 0.75, 0.04, q) == pytest.approx(1.0)


def test_price_bounds_and_limits():
    S, K, T, r = 100.0, 90.0, 1.0, 0.05
    c = lab.bs_price(S, K, T, r, 0.3)
    assert max(S - K * np.exp(-r * T), 0) < c < S
    # sigma -> 0：价格趋于远期内在价值
    assert lab.bs_price(S, K, T, r, 1e-6) == pytest.approx(S - K * np.exp(-r * T), abs=1e-6)


def test_bs_vectorized():
    K = np.array([80.0, 100.0, 120.0])
    out = lab.bs_price(100.0, K, 1.0, 0.05, 0.2)
    assert np.shape(out) == (3,)
    assert out[1] == pytest.approx(10.4506, abs=1e-4)
    assert np.all(np.diff(out) < 0)


# ---------- Greeks ----------


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("K", [80.0, 100.0, 125.0])
def test_greeks_vs_finite_differences(kind, K):
    S, T, r, sig, q = 100.0, 0.8, 0.03, 0.25, 0.01
    g = lab.greeks(S, K, T, r, sig, q, kind)

    def V(S=S, T=T, r=r, sig=sig):
        return lab.bs_price(S, K, T, r, sig, q, kind)

    h = 1e-3
    assert g["delta"] == pytest.approx((V(S=S + h) - V(S=S - h)) / (2 * h), rel=1e-5, abs=1e-7)
    hs = 1e-2
    assert g["gamma"] == pytest.approx(
        (V(S=S + hs) - 2 * V() + V(S=S - hs)) / hs**2, rel=1e-4, abs=1e-7
    )
    assert g["vega"] == pytest.approx((V(sig=sig + h) - V(sig=sig - h)) / (2 * h), rel=1e-5)
    # theta = dV/dt（日历时间往前走，剩余期限 T 变短）
    assert g["theta"] == pytest.approx((V(T=T - h) - V(T=T + h)) / (2 * h), rel=1e-4, abs=1e-6)
    assert g["rho"] == pytest.approx((V(r=r + h) - V(r=r - h)) / (2 * h), rel=1e-5, abs=1e-6)


def test_greeks_relations():
    g_c = lab.greeks(100.0, 105.0, 1.0, 0.05, 0.2, 0.0, "call")
    g_p = lab.greeks(100.0, 105.0, 1.0, 0.05, 0.2, 0.0, "put")
    assert g_c["delta"] - g_p["delta"] == pytest.approx(1.0)
    assert g_c["gamma"] == pytest.approx(g_p["gamma"])
    assert g_c["vega"] == pytest.approx(g_p["vega"])
    # BS PDE: theta + 0.5 sigma^2 S^2 gamma + r S delta - r V = 0
    S, r, sig = 100.0, 0.05, 0.2
    V = lab.bs_price(S, 105.0, 1.0, r, sig)
    pde = g_c["theta"] + 0.5 * sig**2 * S**2 * g_c["gamma"] + r * S * g_c["delta"] - r * V
    assert pde == pytest.approx(0.0, abs=1e-10)


# ---------- 隐含波动率 ----------


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("K", [50.0, 80.0, 100.0, 130.0, 200.0])
@pytest.mark.parametrize("sigma", [0.05, 0.2, 0.6, 1.5])
def test_implied_vol_roundtrip(kind, K, sigma):
    S, T, r = 100.0, 0.5, 0.02
    price = lab.bs_price(S, K, T, r, sigma, 0.0, kind)
    # 时间价值几乎为 0（深度虚值或低波动深度实值）时，隐含波动率在数值上无法识别
    if price - lab.bs_price(S, K, T, r, 1e-9, 0.0, kind) < 1e-6:
        pytest.skip("time value below numerical resolution")
    iv = lab.implied_vol(price, S, K, T, r, 0.0, kind)
    assert lab.bs_price(S, K, T, r, iv, 0.0, kind) == pytest.approx(price, abs=1e-8)
    assert iv == pytest.approx(sigma, rel=1e-4)


def test_implied_vol_rejects_arbitrage():
    with pytest.raises(ValueError):
        lab.implied_vol(150.0, 100.0, 100.0, 1.0, 0.05)  # 看涨期权价格不能高于现货
    with pytest.raises(ValueError):
        lab.implied_vol(1.0, 100.0, 50.0, 1.0, 0.05)  # 低于内在价值


# ---------- 蒙特卡洛 ----------


@pytest.mark.parametrize("method", ["plain", "antithetic", "control"])
@pytest.mark.parametrize("kind", ["call", "put"])
def test_mc_matches_bs(method, kind):
    true = lab.bs_price(**BASE, kind=kind)
    est, se = lab.mc_price(**BASE, kind=kind, n_paths=200_000, seed=11, method=method)
    assert se > 0
    assert abs(est - true) < 4 * se + 1e-3


def test_variance_reduction():
    _, se_plain = lab.mc_price(**BASE, n_paths=200_000, seed=3, method="plain")
    _, se_anti = lab.mc_price(**BASE, n_paths=200_000, seed=3, method="antithetic")
    _, se_ctrl = lab.mc_price(**BASE, n_paths=200_000, seed=3, method="control")
    assert se_anti < 0.85 * se_plain
    assert se_ctrl < 0.6 * se_plain


def test_mc_seed_reproducible():
    a = lab.mc_price(**BASE, n_paths=1000, seed=5)
    b = lab.mc_price(**BASE, n_paths=1000, seed=5)
    assert a == b


# ---------- Delta 对冲 ----------


def test_hedge_pnl_shape_and_unbiased():
    pnl = lab.delta_hedge_pnl(100.0, 100.0, 0.5, 0.03, 0.2, 0.2, n_steps=50, n_paths=20_000, seed=1)
    assert pnl.shape == (20_000,)
    # 正确的波动率对冲：P&L 均值约为 0，且远小于期权价格
    premium = lab.bs_price(100.0, 100.0, 0.5, 0.03, 0.2)
    assert abs(pnl.mean()) < 0.05
    assert pnl.std() < 0.2 * premium


def test_hedge_error_scales_inverse_sqrt_steps():
    args = dict(S0=100.0, K=100.0, T=0.5, r=0.03, sigma_real=0.2, sigma_hedge=0.2, n_paths=20_000)
    s25 = lab.delta_hedge_pnl(**args, n_steps=25, seed=2).std()
    s100 = lab.delta_hedge_pnl(**args, n_steps=100, seed=2).std()
    s400 = lab.delta_hedge_pnl(**args, n_steps=400, seed=2).std()
    assert s25 / s100 == pytest.approx(2.0, rel=0.15)
    assert s100 / s400 == pytest.approx(2.0, rel=0.15)


def test_hedge_pnl_not_dependent_on_drift():
    args = dict(S0=100.0, K=100.0, T=0.5, r=0.03, sigma_real=0.2, sigma_hedge=0.2, n_steps=200)
    a = lab.delta_hedge_pnl(**args, n_paths=20_000, seed=4, mu=0.03)
    b = lab.delta_hedge_pnl(**args, n_paths=20_000, seed=4, mu=0.40)
    # 漂移项对冲掉了：两者均值都接近 0
    assert abs(a.mean()) < 0.05
    assert abs(b.mean()) < 0.1


@pytest.mark.parametrize("sigma_real,sign", [(0.30, -1), (0.12, +1)])
def test_wrong_vol_pnl_direction(sigma_real, sign):
    # 卖出期权（short gamma）：实际波动率 > 隐含 时亏钱，< 隐含 时赚钱
    sigma_impl, S0, T = 0.2, 100.0, 0.5
    pnl = lab.delta_hedge_pnl(
        S0, 100.0, T, 0.0, sigma_real, sigma_impl, n_steps=200, n_paths=20_000, seed=9
    )
    assert np.sign(pnl.mean()) == sign
    # 量级粗略检验：0.5 * Gamma * S^2 * (sigma_impl^2 - sigma_real^2) * T，Gamma 取初始值
    gamma0 = lab.greeks(S0, 100.0, T, 0.0, sigma_impl)["gamma"]
    approx = 0.5 * gamma0 * S0**2 * (sigma_impl**2 - sigma_real**2) * T
    assert pnl.mean() == pytest.approx(approx, rel=0.5)
