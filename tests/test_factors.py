import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy import stats

from quantlab import data
from quantlab import factors as lab


@pytest.fixture(scope="module")
def panel():
    return data.factor_panel(seed=0)


@pytest.fixture(scope="module")
def capm_data(panel):
    rng = np.random.default_rng(11)
    m = panel.market
    asset = pd.Series(0.0004 + 1.3 * m.to_numpy() + rng.normal(0, 0.01, len(m)), index=m.index)
    return asset, m


# ---------- OLS / CAPM ----------


def test_ols_matches_statsmodels():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((300, 3))
    y = 0.5 + X @ np.array([1.0, -2.0, 0.3]) + rng.standard_normal(300)
    res = lab.ols(y, X)
    ref = sm.OLS(y, sm.add_constant(X)).fit()
    assert np.allclose(res["params"], ref.params)
    assert np.allclose(res["se"], ref.bse)
    assert np.allclose(res["tstats"], ref.tvalues)
    assert np.allclose(res["resid"], ref.resid)
    assert res["r2"] == pytest.approx(ref.rsquared)


def test_ols_no_const():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(200)
    y = 2 * x + rng.standard_normal(200)
    res = lab.ols(y, x, add_const=False)
    ref = sm.OLS(y, x[:, None]).fit()
    assert res["params"].shape == (1,)
    assert np.allclose(res["params"], ref.params)
    assert np.allclose(res["tstats"], ref.tvalues)


def test_capm_recovers_beta(capm_data):
    asset, m = capm_data
    res = lab.capm(asset, m)
    ref = sm.OLS(asset.to_numpy(), sm.add_constant(m.to_numpy())).fit()
    assert res["beta"] == pytest.approx(ref.params[1])
    assert res["alpha"] == pytest.approx(ref.params[0])
    assert res["t_beta"] == pytest.approx(ref.tvalues[1])
    assert res["t_alpha"] == pytest.approx(ref.tvalues[0])
    assert res["r2"] == pytest.approx(ref.rsquared)
    assert res["alpha_annual"] == pytest.approx(ref.params[0] * 252)
    assert res["beta"] == pytest.approx(1.3, abs=0.1)  # 标准误约 0.028


def test_capm_rf_shifts_alpha(capm_data):
    asset, m = capm_data
    a0 = lab.capm(asset, m)
    a1 = lab.capm(asset, m, rf=0.03)
    assert a1["beta"] == pytest.approx(a0["beta"])
    expected = a0["alpha"] - 0.03 / 252 * (1 - a0["beta"])
    assert a1["alpha"] == pytest.approx(expected)


def test_capm_aligns_index(capm_data):
    asset, m = capm_data
    res_full = lab.capm(asset.iloc[5:], m)  # 索引不同，需要对齐
    res_ref = lab.capm(asset.iloc[5:], m.iloc[5:])
    assert res_full["beta"] == pytest.approx(res_ref["beta"])


def test_rolling_beta(capm_data):
    asset, m = capm_data
    rb = lab.rolling_beta(asset, m, window=60)
    assert rb.index.equals(asset.index)
    assert rb.iloc[:59].isna().all()
    a, b = asset.iloc[100:160], m.iloc[100:160]
    assert rb.iloc[159] == pytest.approx(np.cov(a, b)[0, 1] / b.var())
    exact = lab.rolling_beta(2 * m, m, window=30).dropna()
    assert np.allclose(exact, 2.0)


# ---------- IC ----------


def test_ic_definition(panel):
    ic = lab.information_coefficient(panel.signal, panel.returns)
    assert isinstance(ic, pd.Series)
    assert len(ic) == len(panel.signal) - 1
    assert ic.index.equals(panel.signal.index[:-1])
    t = 37
    expected = stats.spearmanr(panel.signal.iloc[t], panel.returns.iloc[t + 1]).statistic
    assert ic.iloc[t] == pytest.approx(expected)


def test_ic_level(panel):
    ic = lab.information_coefficient(panel.signal, panel.returns)
    assert 0.025 < ic.mean() < 0.045


def test_ic_no_lookahead_on_noise(panel):
    # 拿“当天收益”本身当信号：如果正确地和 *下一天* 收益比，IC 约为 0；
    # 如果忘了错开一期，就会得到 IC = 1。
    cheat = panel.returns.copy()
    ic = lab.information_coefficient(cheat, panel.returns)
    assert abs(ic.mean()) < 0.02


def test_ic_summary():
    ic = pd.Series([0.1, -0.05, 0.02, 0.08, 0.0])
    s = lab.ic_summary(ic, periods=252)
    assert s["mean"] == pytest.approx(0.03)
    assert s["std"] == pytest.approx(ic.std(ddof=1))
    assert s["icir"] == pytest.approx(0.03 / ic.std(ddof=1))
    assert s["icir_annual"] == pytest.approx(s["icir"] * np.sqrt(252))
    assert s["t_stat"] == pytest.approx(s["icir"] * np.sqrt(5))
    assert s["hit_rate"] == pytest.approx(0.6)


# ---------- 分位数组合 ----------


def test_quantile_returns_shape(panel):
    q = lab.quantile_returns(panel.signal, panel.returns, n_quantiles=5)
    assert list(q.columns) == [1, 2, 3, 4, 5]
    assert q.index.equals(panel.returns.index[1:])


def test_quantile_returns_exact(panel):
    q = lab.quantile_returns(panel.signal, panel.returns, n_quantiles=5)
    t = 100
    s = panel.signal.iloc[t]
    top = s.sort_values().index[-40:]
    bottom = s.sort_values().index[:40]
    nxt = panel.returns.iloc[t + 1]
    assert q.iloc[t, 4] == pytest.approx(nxt[top].mean())
    assert q.iloc[t, 0] == pytest.approx(nxt[bottom].mean())


def test_quantile_monotone(panel):
    means = lab.quantile_returns(panel.signal, panel.returns, n_quantiles=5).mean()
    assert np.all(np.diff(means.to_numpy()) > 0)


def test_long_short(panel):
    ls = lab.long_short_returns(panel.signal, panel.returns, n_quantiles=5)
    q = lab.quantile_returns(panel.signal, panel.returns, n_quantiles=5)
    assert np.allclose(ls, q[5] - q[1])
    t_stat = ls.mean() / ls.std() * np.sqrt(len(ls))
    assert t_stat > 5


# ---------- 中性化 ----------


def test_zscore(panel):
    z = lab.cross_sectional_zscore(panel.signal)
    assert np.allclose(z.mean(axis=1), 0, atol=1e-12)
    assert np.allclose(z.std(axis=1, ddof=1), 1)


def test_neutralize_orthogonal(panel):
    n = lab.neutralize(panel.signal, panel.size, panel.sector)
    assert n.shape == panel.signal.shape
    row = n.iloc[10]
    assert abs(np.dot(row, panel.size)) < 1e-8
    sector_means = row.groupby(panel.sector).mean()
    assert np.allclose(sector_means, 0, atol=1e-10)


def test_neutralize_idempotent(panel):
    sig = panel.signal.iloc[:50]
    once = lab.neutralize(sig, panel.size, panel.sector)
    twice = lab.neutralize(once, panel.size, panel.sector)
    assert np.allclose(once, twice)


def test_neutralize_improves_ic(panel):
    raw = lab.information_coefficient(panel.signal, panel.returns).mean()
    neu = lab.information_coefficient(
        lab.neutralize(panel.signal, panel.size, panel.sector), panel.returns
    ).mean()
    assert neu > raw + 0.004
    assert neu == pytest.approx(panel.ic, abs=0.012)
