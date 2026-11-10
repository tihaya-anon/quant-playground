import numpy as np
import pytest
from sklearn.covariance import LedoitWolf

from quantlab import data
from quantlab import portfolio as lab


@pytest.fixture(scope="module")
def universe():
    """5 个资产的年化 mu 与协方差（“真实”参数，不是估计出来的）。"""
    sig = np.array([0.15, 0.20, 0.25, 0.30, 0.18])
    corr = np.array(
        [
            [1.0, 0.3, 0.2, 0.1, 0.4],
            [0.3, 1.0, 0.5, 0.2, 0.3],
            [0.2, 0.5, 1.0, 0.4, 0.1],
            [0.1, 0.2, 0.4, 1.0, 0.2],
            [0.4, 0.3, 0.1, 0.2, 1.0],
        ]
    )
    cov = np.outer(sig, sig) * corr
    mu = np.array([0.06, 0.08, 0.10, 0.12, 0.07])
    return mu, cov


@pytest.fixture(scope="module")
def big_returns():
    rng = np.random.default_rng(0)
    sig = rng.uniform(0.15, 0.4, 50)
    px = data.correlated_gbm(n_assets=50, n_days=1000, sigma=sig, corr=0.3, seed=5)
    return px.pct_change().dropna().to_numpy()


def random_weights(n, k=2000, seed=0):
    rng = np.random.default_rng(seed)
    w = rng.normal(size=(k, n))
    return w / w.sum(axis=1, keepdims=True)


def test_portfolio_vol(universe):
    _, cov = universe
    w = np.full(5, 0.2)
    assert lab.portfolio_vol(w, cov) == pytest.approx(np.sqrt(w @ cov @ w))


def test_min_variance(universe):
    _, cov = universe
    w = lab.min_variance_weights(cov)
    assert w.sum() == pytest.approx(1.0)
    # 一阶条件：Σw ∝ 1
    g = cov @ w
    assert np.allclose(g, g[0])
    v = lab.portfolio_vol(w, cov)
    others = np.sqrt(np.einsum("ij,jk,ik->i", random_weights(5), cov, random_weights(5)))
    assert np.all(others >= v - 1e-12)


def test_tangency(universe):
    mu, cov = universe
    rf = 0.02
    w = lab.tangency_weights(mu, cov, rf=rf)
    assert w.sum() == pytest.approx(1.0)
    g = cov @ w
    ratio = g / (mu - rf)
    assert np.allclose(ratio, ratio[0])
    sr = (w @ mu - rf) / lab.portfolio_vol(w, cov)
    W = random_weights(5)
    srs = (W @ mu - rf) / np.sqrt(np.einsum("ij,jk,ik->i", W, cov, W))
    assert np.all(srs <= sr + 1e-12)


def test_frontier_weights(universe):
    mu, cov = universe
    for target in [0.05, 0.08, 0.15]:
        w = lab.frontier_weights(mu, cov, target)
        assert w.sum() == pytest.approx(1.0)
        assert w @ mu == pytest.approx(target)
    w_mv = lab.min_variance_weights(cov)
    assert np.allclose(lab.frontier_weights(mu, cov, w_mv @ mu), w_mv)


def test_efficient_frontier_hyperbola(universe):
    mu, cov = universe
    targets = np.linspace(0.04, 0.16, 13)
    vols = lab.efficient_frontier(mu, cov, targets)
    assert vols.shape == (13,)
    inv = np.linalg.inv(cov)
    one = np.ones(5)
    a, b, c = one @ inv @ one, one @ inv @ mu, mu @ inv @ mu
    d = a * c - b**2
    expected = np.sqrt((a * targets**2 - 2 * b * targets + c) / d)
    assert np.allclose(vols, expected)
    assert vols.min() >= lab.portfolio_vol(lab.min_variance_weights(cov), cov) - 1e-12


def test_risk_contributions(universe):
    _, cov = universe
    w = np.array([0.3, 0.2, 0.1, 0.25, 0.15])
    rc = lab.risk_contributions(w, cov)
    assert rc.sum() == pytest.approx(1.0)
    expected = w * (cov @ w) / (w @ cov @ w)
    assert np.allclose(rc, expected)


def test_risk_parity_equal_contrib(universe):
    _, cov = universe
    w = lab.risk_parity_weights(cov)
    assert w.sum() == pytest.approx(1.0)
    assert np.all(w > 0)
    assert np.allclose(lab.risk_contributions(w, cov), 0.2, atol=1e-6)


def test_risk_parity_diagonal():
    sig = np.array([0.1, 0.2, 0.4])
    w = lab.risk_parity_weights(np.diag(sig**2))
    expected = (1 / sig) / (1 / sig).sum()
    assert np.allclose(w, expected, atol=1e-6)


def test_ledoit_wolf_matches_sklearn(big_returns):
    X = big_returns[:200, :20]
    cov, shrink = lab.ledoit_wolf(X)
    ref = LedoitWolf().fit(X)
    assert shrink == pytest.approx(ref.shrinkage_, rel=1e-8)
    assert np.allclose(cov, ref.covariance_)


def test_ledoit_wolf_shrinks_more_with_less_data(big_returns):
    _, s_small = lab.ledoit_wolf(big_returns[:60])
    _, s_big = lab.ledoit_wolf(big_returns[:900])
    assert 0 <= s_big < s_small <= 1


def test_mean_variance_unconstrained_matches_closed_form(universe):
    mu, cov = universe
    gamma = 4.0
    w = lab.mean_variance_cvxpy(mu, cov, gamma, long_only=False)
    # 闭式解：w = Σ^{-1}(mu - eta 1) / gamma，eta 使 sum(w) = 1
    inv = np.linalg.inv(cov)
    one = np.ones(5)
    eta = (one @ inv @ mu - gamma) / (one @ inv @ one)
    expected = inv @ (mu - eta * one) / gamma
    assert np.allclose(w, expected, atol=1e-4)


def test_mean_variance_constraints(universe):
    mu, cov = universe
    w = lab.mean_variance_cvxpy(mu, cov, 1.0, long_only=True, max_weight=0.3)
    assert w.sum() == pytest.approx(1.0, abs=1e-6)
    assert np.all(w >= -1e-6)
    assert np.all(w <= 0.3 + 1e-6)


def test_mean_variance_risk_aversion(universe):
    mu, cov = universe
    vols = [lab.portfolio_vol(lab.mean_variance_cvxpy(mu, cov, g), cov) for g in [1.0, 5.0, 50.0]]
    assert vols[0] > vols[1] > vols[2]


def test_oos_sample_cov_is_overoptimistic(big_returns):
    train, test = big_returns[:120], big_returns[120:]
    pred, real = lab.oos_min_variance(train, test, estimator="sample")
    assert real > 1.5 * pred


def test_oos_shrinkage_helps(big_returns):
    train, test = big_returns[:120], big_returns[120:]
    _, real_sample = lab.oos_min_variance(train, test, estimator="sample")
    _, real_lw = lab.oos_min_variance(train, test, estimator="ledoit_wolf")
    assert real_lw < real_sample
    with pytest.raises(ValueError):
        lab.oos_min_variance(train, test, estimator="nope")
