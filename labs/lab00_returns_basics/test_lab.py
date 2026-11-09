import numpy as np
import pandas as pd
import pytest

from labs._harness import load_impl
from quantlab import data

lab = load_impl(__file__)


@pytest.fixture(scope="module")
def prices():
    return data.gbm_prices(n_days=2520, mu=0.08, sigma=0.2, seed=7)


@pytest.fixture(scope="module")
def assets():
    return data.correlated_gbm(n_assets=3, n_days=500, corr=0.2, seed=3)


def test_simple_returns_series(prices):
    r = lab.simple_returns(prices)
    assert isinstance(r, pd.Series)
    assert len(r) == len(prices) - 1
    assert r.index[0] == prices.index[1]
    assert r.iloc[0] == pytest.approx(prices.iloc[1] / prices.iloc[0] - 1)
    assert not r.isna().any()


def test_simple_returns_frame(assets):
    r = lab.simple_returns(assets)
    assert isinstance(r, pd.DataFrame)
    assert r.shape == (len(assets) - 1, assets.shape[1])


def test_log_returns_sum_to_total(prices):
    lr = lab.log_returns(prices)
    assert lr.sum() == pytest.approx(np.log(prices.iloc[-1] / prices.iloc[0]))


@pytest.mark.parametrize("r", [0.0, 0.01, -0.5, 1.0, 1e-12])
def test_simple_log_roundtrip(r):
    assert lab.log_to_simple(lab.simple_to_log(r)) == pytest.approx(r, abs=1e-15, rel=1e-12)


def test_simple_to_log_precision():
    # 1 + 1e-17 在 float64 里就是 1，np.log(1 + r) 会给出 0
    assert lab.simple_to_log(1e-17) == pytest.approx(1e-17, rel=1e-6)


def test_simple_to_log_vectorized(prices):
    r = lab.simple_returns(prices)
    assert np.allclose(lab.simple_to_log(r), lab.log_returns(prices))


def test_cumulative_return(prices):
    r = lab.simple_returns(prices)
    assert lab.cumulative_return(r) == pytest.approx(prices.iloc[-1] / prices.iloc[0] - 1)
    assert lab.cumulative_return(pd.Series([0.5, -0.5])) == pytest.approx(-0.25)


def test_wealth_index(prices):
    r = lab.simple_returns(prices)
    w = lab.wealth_index(r, start=100.0)
    assert len(w) == len(r)
    assert np.allclose(w.to_numpy(), prices.iloc[1:].to_numpy() / prices.iloc[0] * 100.0)


def test_resample_monthly(prices):
    r = lab.simple_returns(prices)
    m = lab.resample_returns(r, "ME")
    month_end = prices.resample("ME").last()
    # 第一个月从第二天起算，其余月份应等于月末价格之比
    expected = month_end.pct_change().iloc[1:]
    assert np.allclose(m.iloc[1:].to_numpy(), expected.to_numpy())
    assert lab.cumulative_return(m) == pytest.approx(lab.cumulative_return(r))


def test_rebalanced_portfolio(assets):
    r = lab.simple_returns(assets)
    w = [0.5, 0.3, 0.2]
    p = lab.rebalanced_portfolio_returns(r, w)
    assert isinstance(p, pd.Series)
    assert p.index.equals(r.index)
    assert p.iloc[10] == pytest.approx(float(r.iloc[10] @ np.array(w)))


def test_rebalanced_not_log_weighted(assets):
    # 用对数收益加权是常见错误；两者结果不应相同
    r = lab.simple_returns(assets)
    w = np.array([0.5, 0.3, 0.2])
    p = lab.rebalanced_portfolio_returns(r, w)
    wrong = np.expm1(np.log1p(r).to_numpy() @ w)
    assert not np.allclose(p.to_numpy(), wrong)


def test_buy_and_hold(assets):
    r = lab.simple_returns(assets)
    w = np.array([0.5, 0.3, 0.2])
    p = lab.buy_and_hold_portfolio_returns(r, w)
    assert p.index.equals(r.index)
    final_value = float((assets.iloc[-1] / assets.iloc[0]).to_numpy() @ w)
    assert lab.cumulative_return(p) == pytest.approx(final_value - 1)
    # 第一天的收益与再平衡组合相同（还没漂移）
    assert p.iloc[0] == pytest.approx(float(r.iloc[0] @ w))


def test_buy_and_hold_with_cash(assets):
    r = lab.simple_returns(assets)
    w = np.array([0.3, 0.3, 0.0])  # 40% 现金
    p = lab.buy_and_hold_portfolio_returns(r, w)
    final_value = float((assets.iloc[-1] / assets.iloc[0]).to_numpy() @ w) + 0.4
    assert lab.cumulative_return(p) == pytest.approx(final_value - 1)


def test_gbm_shape_and_start():
    paths = lab.simulate_gbm(100.0, 0.1, 0.3, n_days=252, n_paths=50, seed=0)
    assert paths.shape == (253, 50)
    assert np.all(paths[0] == 100.0)
    assert np.all(paths > 0)


def test_gbm_moments():
    mu, sigma, T = 0.1, 0.3, 1.0
    paths = lab.simulate_gbm(100.0, mu, sigma, n_days=252, n_paths=200_000, seed=1)
    st = paths[-1]
    # 均值 e^{mu T}，中位数 e^{(mu - sigma^2/2) T}
    assert st.mean() == pytest.approx(100 * np.exp(mu * T), rel=0.005)
    assert np.median(st) == pytest.approx(100 * np.exp((mu - 0.5 * sigma**2) * T), rel=0.005)
    lr = np.log(st / 100.0)
    assert lr.std() == pytest.approx(sigma * np.sqrt(T), rel=0.01)


def test_gbm_seed_reproducible():
    a = lab.simulate_gbm(1.0, 0.0, 0.2, 10, 3, seed=42)
    b = lab.simulate_gbm(1.0, 0.0, 0.2, 10, 3, seed=42)
    assert np.array_equal(a, b)


def test_annualize_log_stats_recovers_sigma():
    sigmas, mus = [], []
    for seed in range(30):
        px = data.gbm_prices(n_days=2520, mu=0.08, sigma=0.2, seed=seed)
        mu_hat, sigma_hat = lab.annualize_log_stats(lab.log_returns(px))
        mus.append(mu_hat)
        sigmas.append(sigma_hat)
    # sigma 估得非常准
    assert np.mean(sigmas) == pytest.approx(0.2, abs=0.003)
    assert np.std(sigmas) < 0.005
    # mu 无偏，但离散度巨大——约等于 mu_standard_error
    assert np.mean(mus) == pytest.approx(0.08, abs=0.03)
    assert np.std(mus) == pytest.approx(lab.mu_standard_error(0.2, 2520), rel=0.35)


def test_mu_standard_error_depends_only_on_years():
    se_daily = lab.mu_standard_error(0.2, n_days=2520, periods=252)
    se_minute = lab.mu_standard_error(0.2, n_days=2520 * 240, periods=252 * 240)
    assert se_daily == pytest.approx(0.2 / np.sqrt(10))
    assert se_minute == pytest.approx(se_daily)
