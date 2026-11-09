import numpy as np
import pandas as pd
import pytest
from scipy import stats

from labs._harness import load_impl
from quantlab import data, perf

lab = load_impl(__file__)

LOOKBACKS = list(range(5, 255, 10))


def ar1_returns(n, phi, mu, sigma, seed):
    rng = np.random.default_rng(seed)
    e = rng.normal(0, sigma, n)
    x = np.empty(n)
    x[0] = e[0]
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return pd.Series(mu + x, index=data.business_index(n))


# ---------------------------------------------------------------- 标准误


def test_per_period_sharpe():
    r = pd.Series([0.01, -0.02, np.nan, 0.03, 0.0])
    x = r.dropna()
    assert lab.per_period_sharpe(r) == pytest.approx(x.mean() / x.std(ddof=1))
    assert lab.per_period_sharpe(x.to_numpy()) == pytest.approx(x.mean() / x.std(ddof=1))


def test_se_formulas_closed_form():
    assert lab.sharpe_se_iid(0.1, 252) == pytest.approx(np.sqrt(1.005 / 252))
    # 正态（skew 0, kurt 3）时 Mertens 与 Lo 只差 n vs n-1
    assert lab.sharpe_se_mertens(0.1, 252, 0.0, 3.0) == pytest.approx(np.sqrt(1.005 / 251))
    # 负偏度放大标准误
    assert lab.sharpe_se_mertens(0.2, 252, -2.0, 10.0) > lab.sharpe_se_mertens(0.2, 252, 0.0, 3.0)


def test_se_iid_matches_monte_carlo():
    rng = np.random.default_rng(0)
    n, sr = 252, 0.1
    r = rng.normal(sr * 0.01, 0.01, (5000, n))
    est = r.mean(1) / r.std(1, ddof=1)
    assert est.std() == pytest.approx(lab.sharpe_se_iid(sr, n), rel=0.05)


def test_se_mertens_beats_iid_under_skew():
    rng = np.random.default_rng(1)
    n = 252
    mu_ln = np.exp(0.32)
    sd_ln = np.sqrt((np.exp(0.64) - 1) * np.exp(0.64))
    r = -rng.lognormal(0, 0.8, (4000, n)) + mu_ln + 0.2 * sd_ln  # 每期 SR = 0.2，强负偏
    est = r.mean(1) / r.std(1, ddof=1)
    big = -rng.lognormal(0, 0.8, 2_000_000)
    g3, g4 = stats.skew(big), stats.kurtosis(big, fisher=False)
    mert = lab.sharpe_se_mertens(0.2, n, g3, g4)
    iid = lab.sharpe_se_iid(0.2, n)
    assert est.std() == pytest.approx(mert, rel=0.1)
    assert est.std() > 1.25 * iid


# ---------------------------------------------------------------- bootstrap


def test_bootstrap_shape_and_seed():
    r = np.random.default_rng(2).normal(0.0005, 0.01, 500)
    a = lab.bootstrap_sharpe(r, n_boot=300, seed=3)
    b = lab.bootstrap_sharpe(r, n_boot=300, seed=3)
    assert a.shape == (300,)
    assert np.array_equal(a, b)


def test_bootstrap_matches_analytic_se():
    r = np.random.default_rng(4).normal(0.0005, 0.01, 2520)
    boots = lab.bootstrap_sharpe(r, n_boot=3000, seed=0)
    sr = r.mean() / r.std(ddof=1)
    assert boots.std() == pytest.approx(lab.sharpe_se_iid(sr, len(r)) * np.sqrt(252), rel=0.1)
    assert boots.mean() == pytest.approx(sr * np.sqrt(252), abs=0.1)


def test_percentile_ci():
    x = np.arange(1001)
    assert lab.percentile_ci(x, 0.1) == pytest.approx((50.0, 950.0))


def test_block_bootstrap_is_wider_with_autocorrelation():
    r = ar1_returns(2520, phi=0.3, mu=0.0004, sigma=0.01, seed=5)
    iid = lab.bootstrap_sharpe(r, n_boot=2000, seed=1)
    blk = lab.block_bootstrap_sharpe(r, block_size=25, n_boot=2000, seed=1)
    assert blk.shape == (2000,)
    lo_i, hi_i = lab.percentile_ci(iid)
    lo_b, hi_b = lab.percentile_ci(blk)
    # AR(1) phi=0.3 时，均值的长期方差放大 (1+phi)/(1-phi) ≈ 1.86 倍，标准差约 1.36 倍
    assert (hi_b - lo_b) / (hi_i - lo_i) > 1.2
    assert lo_b < perf.sharpe(r) < hi_b


def test_block_bootstrap_block_one_is_iid_like():
    r = np.random.default_rng(6).normal(0.0005, 0.01, 1000)
    blk = lab.block_bootstrap_sharpe(r, block_size=1, n_boot=3000, seed=2)
    iid = lab.bootstrap_sharpe(r, n_boot=3000, seed=2)
    assert blk.std() == pytest.approx(iid.std(), rel=0.1)


def test_block_bootstrap_wraps_around():
    # block_size = n 时，每个重抽样本都是原序列的一个循环移位：均值、标准差不变
    r = np.random.default_rng(12).normal(0.001, 0.01, 300)
    blk = lab.block_bootstrap_sharpe(r, block_size=300, n_boot=50, seed=0)
    assert np.allclose(blk, r.mean() / r.std(ddof=1) * np.sqrt(252))


# ---------------------------------------------------------------- PSR / DSR


def test_psr_formula():
    r = np.random.default_rng(7).standard_t(5, 1000) * 0.01 + 0.0008
    sr = r.mean() / r.std(ddof=1)
    g3, g4 = stats.skew(r), stats.kurtosis(r, fisher=False)
    bench = 0.02
    z = (sr - bench) * np.sqrt(999) / np.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr**2)
    assert lab.probabilistic_sharpe_ratio(r, bench) == pytest.approx(stats.norm.cdf(z))


def test_psr_at_own_sharpe_is_half():
    r = np.random.default_rng(8).normal(0.001, 0.01, 500)
    assert lab.probabilistic_sharpe_ratio(r, r.mean() / r.std(ddof=1)) == pytest.approx(0.5)


def test_expected_max_sharpe_monte_carlo():
    rng = np.random.default_rng(9)
    n_trials, v = 100, 0.004
    maxes = rng.normal(0, np.sqrt(v), (20000, n_trials)).max(1)
    assert lab.expected_max_sharpe(n_trials, v) == pytest.approx(maxes.mean(), rel=0.03)
    assert lab.expected_max_sharpe(1000, v) > lab.expected_max_sharpe(10, v)
    assert lab.expected_max_sharpe(10, v, sr_mean=0.1) == pytest.approx(
        0.1 + lab.expected_max_sharpe(10, v)
    )


def test_dsr_deflates_lucky_noise():
    """200 个纯噪声“策略”里挑最好的：PSR 说它显著，DSR 说它是运气。"""
    rng = np.random.default_rng(10)
    noise = rng.normal(0, 0.01, (756, 200))
    srs = noise.mean(0) / noise.std(0, ddof=1)
    best = noise[:, srs.argmax()]
    assert srs.max() * np.sqrt(252) > 1.5  # 年化 Sharpe 看起来很漂亮
    assert lab.probabilistic_sharpe_ratio(best, 0.0) > 0.95
    assert lab.deflated_sharpe_ratio(best, srs) < 0.6


def test_dsr_keeps_real_edge():
    rng = np.random.default_rng(11)
    good = rng.normal(2.0 / np.sqrt(252) * 0.01, 0.01, 2520)  # 真实年化 Sharpe 2
    trials = rng.normal(0, np.sqrt(1 / 2520), 100)
    trials[0] = good.mean() / good.std(ddof=1)
    assert lab.deflated_sharpe_ratio(good, trials) > 0.95


# ---------------------------------------------------------------- 扫描与 walk-forward


def test_sweep_returns():
    px = data.gbm_prices(n_days=600, seed=1)
    sw = lab.sweep_returns(px, [5, 20, 60])
    assert list(sw.columns) == [5, 20, 60]
    assert sw.index.equals(px.index[1:])
    assert np.allclose(sw[20], lab.momentum_strategy_returns(px, 20))


def test_walk_forward_structure():
    px = data.regime_switching_prices(n_days=2000, seed=3)
    wf = lab.walk_forward(px, [10, 50, 100], train_size=500, test_size=100)
    assert list(wf.columns) == ["ret", "lookback"]
    sw = lab.sweep_returns(px, [10, 50, 100])
    assert wf.index.equals(sw.index[500:])
    # 每个测试块内参数不变，且等于训练窗口里 Sharpe 最高的
    first = sw.iloc[0:500]
    best = (first.mean() / first.std(ddof=1)).idxmax()
    assert (wf["lookback"].iloc[:100] == best).all()
    assert np.allclose(wf["ret"].iloc[:100], sw[best].iloc[500:600])


def _wf_stats(gen, seeds):
    ins, oos = [], []
    for seed in seeds:
        px = gen(seed)
        sw = lab.sweep_returns(px, LOOKBACKS)
        ins.append(float((sw.mean() / sw.std() * np.sqrt(252)).max()))
        oos.append(perf.sharpe(lab.walk_forward(px, LOOKBACKS)["ret"]))
    return np.mean(ins), np.mean(oos)


def test_overfitting_on_noise():
    ins, oos = _wf_stats(lambda s: data.gbm_prices(mu=0.0, seed=200 + s), range(6))
    assert ins > 0.1  # 全样本挑最优参数：看起来有正 Sharpe
    assert oos < ins - 0.15  # 样本外打回原形
    assert abs(oos) < 0.35


def test_walk_forward_keeps_real_signal():
    _, oos = _wf_stats(lambda s: data.regime_switching_prices(seed=s), range(6))
    assert oos > 0.15
