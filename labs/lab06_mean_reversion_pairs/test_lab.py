import numpy as np
import pandas as pd
import pytest

from labs._harness import load_impl
from quantlab import data, perf

lab = load_impl(__file__)

HALF_LIFE = 15.0
HEDGE = 1.5


@pytest.fixture(scope="module")
def pair():
    return data.cointegrated_pair(hedge_ratio=HEDGE, half_life=HALF_LIFE, seed=0)


def ar1(n, c, phi, sigma, seed):
    rng = np.random.default_rng(seed)
    x = np.empty(n)
    x[0] = c / (1 - phi)
    for t in range(1, n):
        x[t] = c + phi * x[t - 1] + sigma * rng.standard_normal()
    return pd.Series(x, index=data.business_index(n))


# ---------- AR(1) / OU ----------


def test_fit_ar1_recovers_params():
    x = ar1(20_000, c=0.5, phi=0.9, sigma=0.1, seed=1)
    c, phi, sigma = lab.fit_ar1(x)
    assert phi == pytest.approx(0.9, abs=0.01)
    assert c / (1 - phi) == pytest.approx(5.0, abs=0.05)
    assert sigma == pytest.approx(0.1, rel=0.02)


def test_fit_ar1_on_spread(pair):
    _, phi, _ = lab.fit_ar1(pair["spread"])
    assert phi == pytest.approx(np.exp(-np.log(2) / HALF_LIFE), abs=0.02)


def test_half_life():
    assert lab.half_life(0.5) == pytest.approx(1.0)
    assert lab.half_life(np.exp(-np.log(2) / 10)) == pytest.approx(10.0)
    assert lab.half_life(1.0) == np.inf
    assert lab.half_life(1.02) == np.inf


def test_ou_from_ar1():
    kappa, theta, sigma = lab.ou_from_ar1(c=0.2, phi=0.8, sigma_eps=0.1, dt=1.0)
    assert kappa == pytest.approx(-np.log(0.8))
    assert theta == pytest.approx(1.0)
    # OU 的平稳方差 sigma^2/(2 kappa) 应等于 AR(1) 的平稳方差 sigma_eps^2/(1-phi^2)
    assert sigma**2 / (2 * kappa) == pytest.approx(0.1**2 / (1 - 0.8**2))
    k2, _, _ = lab.ou_from_ar1(c=0.2, phi=0.8, sigma_eps=0.1, dt=1 / 252)
    assert k2 == pytest.approx(kappa * 252)


# ---------- 平稳性 / 协整 ----------


def test_adf(pair):
    assert lab.adf_pvalue(pair["spread"]) < 0.01
    assert lab.adf_pvalue(np.log(pair["x"])) > 0.05


def test_engle_granger(pair):
    ly, lx = np.log(pair["y"]), np.log(pair["x"])
    eg = lab.engle_granger(ly, lx)
    assert eg["hedge_ratio"] == pytest.approx(HEDGE, abs=0.07)
    assert eg["pvalue"] < 0.01
    assert isinstance(eg["spread"], pd.Series)
    assert np.allclose(eg["spread"], ly - eg["hedge_ratio"] * lx - eg["intercept"])
    assert abs(eg["spread"].mean()) < 1e-10


def test_engle_granger_spurious():
    rng = np.random.default_rng(0)
    idx = data.business_index(1500)
    a = pd.Series(np.cumsum(rng.normal(0, 0.015, 1500)), index=idx)
    b = pd.Series(np.cumsum(rng.normal(0, 0.015, 1500)), index=idx)
    assert lab.engle_granger(a, b)["pvalue"] > 0.1


# ---------- 交易规则 ----------


def test_rolling_zscore(pair):
    s = pair["spread"]
    z = lab.rolling_zscore(s, window=20)
    assert z.index.equals(s.index)
    assert z.iloc[:19].isna().all()
    win = s.iloc[30:50]
    assert z.iloc[49] == pytest.approx((s.iloc[49] - win.mean()) / win.std(ddof=1))


def test_pair_positions_state_machine():
    z = pd.Series([np.nan, 0.0, 2.5, 1.5, 0.6, 0.4, -1.0, -2.1, -1.0, -0.6, -0.4, 3.0, -3.0, 0.0])
    expected = [0, 0, -1, -1, -1, 0, 0, 1, 1, 1, 0, -1, 0, 0]
    pos = lab.pair_positions(z, entry=2.0, exit=0.5)
    assert pos.index.equals(z.index)
    assert pos.tolist() == expected


def test_pair_returns_alignment(pair):
    y, x = pair["y"], pair["x"]
    pos = pd.Series(0.0, index=y.index)
    pos.iloc[10] = 1.0  # 第 10 天收盘建仓，持有一天
    r = lab.pair_returns(y, x, HEDGE, pos)
    # r.iloc[10] 对应第 11 天（r 的索引从 y.index[1] 开始）
    assert r.index.equals(y.index[1:])
    ry, rx = y.pct_change(), x.pct_change()
    assert r.iloc[10] == pytest.approx((ry.iloc[11] - HEDGE * rx.iloc[11]) / (1 + HEDGE))
    assert (r.drop(r.index[10]) == 0).all()


def test_pair_returns_no_lookahead(pair):
    y, x = pair["y"].copy(), pair["x"]
    eg = lab.engle_granger(np.log(y), np.log(x))
    pos = lab.pair_positions(lab.rolling_zscore(eg["spread"], 60))
    r1 = lab.pair_returns(y, x, eg["hedge_ratio"], pos)
    y2 = y.copy()
    y2.iloc[-1] *= 1.5
    r2 = lab.pair_returns(y2, x, eg["hedge_ratio"], pos)
    assert np.allclose(r1.iloc[:-1], r2.iloc[:-1])


def test_pair_returns_costs(pair):
    y, x = pair["y"], pair["x"]
    eg = lab.engle_granger(np.log(y), np.log(x))
    pos = lab.pair_positions(lab.rolling_zscore(eg["spread"], 60))
    gross = lab.pair_returns(y, x, eg["hedge_ratio"], pos)
    net = lab.pair_returns(y, x, eg["hedge_ratio"], pos, cost_bps=10)
    n_trades = pos.diff().abs().fillna(pos.abs()).sum()
    assert (gross - net).sum() == pytest.approx(n_trades * 10 / 1e4, rel=1e-6)


def test_pair_strategy_profitable(pair):
    y, x = pair["y"], pair["x"]
    eg = lab.engle_granger(np.log(y), np.log(x))
    pos = lab.pair_positions(lab.rolling_zscore(eg["spread"], 60))
    r = lab.pair_returns(y, x, eg["hedge_ratio"], pos, cost_bps=2)
    assert perf.sharpe(r) > 0.8


# ---------- Kalman ----------


def test_kalman_static(pair):
    k = lab.kalman_hedge_ratio(np.log(pair["y"]), np.log(pair["x"]))
    assert list(k.columns) == ["beta", "alpha"]
    assert k.index.equals(pair.index)
    assert k["beta"].iloc[-300:].mean() == pytest.approx(HEDGE, abs=0.15)


def test_kalman_tracks_drift():
    n = 1500
    rng = np.random.default_rng(3)
    idx = data.business_index(n)
    lx = np.log(50) + np.cumsum(rng.normal(0, 0.015, n))
    beta = np.linspace(1.0, 2.0, n)
    spread = np.zeros(n)
    for t in range(1, n):
        spread[t] = 0.9 * spread[t - 1] + rng.normal(0, 0.01)
    ly = beta * lx + spread
    k = lab.kalman_hedge_ratio(
        pd.Series(ly, index=idx), pd.Series(lx, index=idx), delta=1e-4, obs_var=1e-3
    )
    static = np.polyfit(lx, ly, 1)[0]
    late = slice(n // 2, n)
    err_kalman = np.abs(k["beta"].to_numpy()[late] - beta[late]).mean()
    err_static = np.abs(static - beta[late]).mean()
    assert err_kalman < 0.5 * err_static
