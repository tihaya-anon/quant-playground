import numpy as np
import pandas as pd
import pytest
from scipy import stats

from labs._harness import load_impl
from quantlab import data, perf

lab = load_impl(__file__)


@pytest.fixture(scope="module")
def rets():
    return data.gbm_prices(n_days=2520, mu=0.08, sigma=0.2, seed=11).pct_change().dropna()


@pytest.fixture(scope="module")
def normal_rets():
    rng = np.random.default_rng(5)
    idx = data.business_index(200_000)
    return pd.Series(rng.normal(0.0004, 0.01, len(idx)), index=idx)


def series(values):
    return pd.Series(values, index=data.business_index(len(values)), dtype=float)


# ---------------------------------------------------------------- 收益与波动


def test_annualized_return_matches_perf(rets):
    assert lab.annualized_return(rets) == pytest.approx(perf.annualized_return(rets))


def test_annualized_return_closed_form():
    # 每天 +0.1%，252 天后正好是 1.001^252 - 1
    r = series([0.001] * 504)
    assert lab.annualized_return(r) == pytest.approx(1.001**252 - 1)
    # 月频
    assert lab.annualized_return(series([0.01] * 24), periods=12) == pytest.approx(1.01**12 - 1)


def test_annualized_return_ignores_nan():
    r = series([np.nan, 0.001, 0.001])
    assert lab.annualized_return(r) == pytest.approx(1.001**252 - 1)


def test_annualized_vol(rets, normal_rets):
    assert lab.annualized_vol(rets) == pytest.approx(perf.annualized_vol(rets))
    assert lab.annualized_vol(normal_rets) == pytest.approx(0.01 * np.sqrt(252), rel=0.01)
    expected_monthly = 0.01 * np.sqrt(12)
    assert lab.annualized_vol(normal_rets, periods=12) == pytest.approx(expected_monthly, rel=0.01)


def test_sharpe_matches_perf(rets):
    assert lab.sharpe(rets) == pytest.approx(perf.sharpe(rets))
    assert lab.sharpe(rets, rf=0.03) == pytest.approx(perf.sharpe(rets, rf=0.03))


def test_sharpe_closed_form(normal_rets):
    assert lab.sharpe(normal_rets) == pytest.approx(0.0004 / 0.01 * np.sqrt(252), abs=0.05)


def test_sharpe_rf_lowers_it(rets):
    assert lab.sharpe(rets, rf=0.05) < lab.sharpe(rets)


def test_sharpe_zero_vol_is_nan():
    assert np.isnan(lab.sharpe(series([0.0] * 10)))


# ---------------------------------------------------------------- 下行风险


def test_downside_deviation_hand():
    r = series([0.02, -0.01, 0.03, -0.02])
    expected = np.sqrt((0.01**2 + 0.02**2) / 4) * np.sqrt(252)
    assert lab.downside_deviation(r) == pytest.approx(expected)


def test_downside_deviation_with_mar():
    r = series([0.02, -0.01, 0.03, -0.02])
    mar = 0.0252  # 每期 0.0001
    m = 0.0001
    sf = np.minimum(r.to_numpy() - m, 0)
    expected = np.sqrt(np.mean(sf**2)) * np.sqrt(252)
    assert lab.downside_deviation(r, mar=mar) == pytest.approx(expected)


def test_sortino_hand():
    r = series([0.02, -0.01, 0.03, -0.02])
    dd = np.sqrt((0.01**2 + 0.02**2) / 4) * np.sqrt(252)
    assert lab.sortino(r) == pytest.approx(0.005 * 252 / dd)


def test_sortino_normal_closed_form(normal_rets):
    # 对 N(m, s^2)，E[min(X,0)^2] 有闭式；m 很小时下行偏差 ≈ s/sqrt(2)
    m, s = 0.0004, 0.01
    z = -m / s
    second = (m**2 + s**2) * stats.norm.cdf(z) - m * s * stats.norm.pdf(z)
    expected = m * 252 / (np.sqrt(second) * np.sqrt(252))
    assert lab.sortino(normal_rets) == pytest.approx(expected, rel=0.08)


def test_sortino_no_losses_is_nan():
    assert np.isnan(lab.sortino(series([0.01, 0.02])))


# ---------------------------------------------------------------- 回撤


def test_drawdown_hand():
    wealth = np.array([1.0, 1.1, 1.0, 0.9, 1.05, 1.2, 1.1])
    r = series(wealth).pct_change().dropna()
    dd = lab.drawdown(r)
    assert dd.index.equals(r.index)
    expected = np.array([0, 1.0 / 1.1 - 1, 0.9 / 1.1 - 1, 1.05 / 1.1 - 1, 0, 1.1 / 1.2 - 1])
    assert np.allclose(dd.to_numpy(), expected)
    assert lab.max_drawdown(r) == pytest.approx(0.2 / 1.1)
    assert lab.max_drawdown_duration(r) == 3


def test_drawdown_matches_perf(rets):
    assert np.allclose(lab.drawdown(rets).to_numpy(), perf.drawdown(rets).to_numpy())
    assert lab.max_drawdown(rets) == pytest.approx(perf.max_drawdown(rets))


def test_no_drawdown():
    r = series([0.01, 0.02, 0.0, 0.01])
    assert lab.max_drawdown(r) == 0
    assert lab.max_drawdown_duration(r) == 0
    assert np.isnan(lab.calmar(r))


def test_max_drawdown_is_positive(rets):
    assert 0 < lab.max_drawdown(rets) < 1


def test_calmar(rets):
    assert lab.calmar(rets) == pytest.approx(perf.annualized_return(rets) / perf.max_drawdown(rets))


# ---------------------------------------------------------------- VaR / CVaR


def test_historical_var_cvar_hand():
    r = series(np.linspace(-0.10, 0.09, 20))  # -0.10, -0.09, ..., 0.09
    q = np.quantile(r.to_numpy(), 0.1)
    assert lab.historical_var(r, 0.1) == pytest.approx(-q)
    assert lab.historical_cvar(r, 0.1) == pytest.approx(-r[r <= q].mean())
    assert lab.historical_cvar(r, 0.1) >= lab.historical_var(r, 0.1)


def test_parametric_closed_form():
    r = series([0.01, -0.02, 0.015, 0.0, -0.005, 0.02, -0.01])
    mu, sd = r.mean(), r.std(ddof=1)
    z = stats.norm.ppf(0.05)
    assert lab.parametric_var(r, 0.05) == pytest.approx(-(mu + sd * z))
    assert lab.parametric_cvar(r, 0.05) == pytest.approx(-(mu - sd * stats.norm.pdf(z) / 0.05))


@pytest.mark.parametrize("alpha", [0.01, 0.05])
def test_historical_converges_to_parametric_under_normality(normal_rets, alpha):
    assert lab.historical_var(normal_rets, alpha) == pytest.approx(
        lab.parametric_var(normal_rets, alpha), rel=0.02
    )
    assert lab.historical_cvar(normal_rets, alpha) == pytest.approx(
        lab.parametric_cvar(normal_rets, alpha), rel=0.03
    )


def test_fat_tails_break_parametric_var():
    # 学生 t(3) 收益：正态 VaR 在 1% 尾部严重低估损失
    rng = np.random.default_rng(9)
    r = series(0.01 * rng.standard_t(3, 200_000) / np.sqrt(3))
    assert lab.historical_cvar(r, 0.01) > 1.2 * lab.parametric_cvar(r, 0.01)
    assert lab.excess_kurtosis(r) > 3


# ---------------------------------------------------------------- 高阶矩与滚动


def test_skew_kurt_match_pandas(rets):
    assert lab.skewness(rets) == pytest.approx(rets.skew())
    assert lab.excess_kurtosis(rets) == pytest.approx(rets.kurt())


def test_skew_of_lognormal_is_positive():
    rng = np.random.default_rng(1)
    r = series(np.exp(rng.normal(0, 0.5, 50_000)) - 1)
    assert lab.skewness(r) > 1


def test_rolling_sharpe(rets):
    rs = lab.rolling_sharpe(rets, window=126)
    assert rs.index.equals(rets.index)
    assert rs.iloc[:125].isna().all()
    assert rs.iloc[125] == pytest.approx(perf.sharpe(rets.iloc[:126]))
    assert rs.iloc[-1] == pytest.approx(perf.sharpe(rets.iloc[-126:]))


def test_performance_summary(rets):
    s = lab.performance_summary(rets)
    keys = [
        "ann_return", "ann_vol", "sharpe", "sortino", "max_drawdown", "max_dd_duration",
        "calmar", "var_95", "cvar_95", "skew", "excess_kurt",
    ]  # fmt: skip
    assert list(s.index) == keys
    assert s["sharpe"] == pytest.approx(perf.sharpe(rets))
    assert s["var_95"] == pytest.approx(lab.historical_var(rets, 0.05))


def test_drawdown_counts_initial_capital():
    # 第一天就亏 10%：期初的 1.0 是高点，这一天已经在回撤中
    r = series([-0.1, 0.05, 0.1])
    dd = lab.drawdown(r)
    assert dd.iloc[0] == pytest.approx(-0.1)
    assert lab.max_drawdown(r) == pytest.approx(0.1)


def test_sharpe_constant_nonzero_is_nan():
    # 常数收益的样本标准差在浮点下不是精确的 0，不能用 == 0 判断
    assert np.isnan(lab.sharpe(series([0.001] * 50)))
