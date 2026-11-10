import numpy as np
import pandas as pd
import pytest

from quantlab import data
from quantlab import vol as lab

TRUE = {"omega": 2e-6, "alpha": 0.08, "beta": 0.90}


@pytest.fixture(scope="module")
def garch_df():
    return data.garch_returns(n_days=4000, seed=11, **TRUE)


@pytest.fixture(scope="module")
def fitted(garch_df):
    return lab.fit_garch(garch_df["ret"])


def test_realized_vol(garch_df):
    r = garch_df["ret"]
    rv = lab.realized_vol(r, window=21)
    assert rv.index.equals(r.index)
    assert rv.iloc[:20].isna().all()
    assert rv.iloc[20] == pytest.approx(r.iloc[:21].std() * np.sqrt(252))


def test_ewma_recursion_and_init():
    r = pd.Series([0.01, -0.02, 0.03, 0.0], index=data.business_index(4))
    s2 = lab.ewma_variance(r, lam=0.9, init=1e-4)
    assert s2.iloc[0] == pytest.approx(1e-4)
    assert s2.iloc[1] == pytest.approx(0.9 * 1e-4 + 0.1 * 0.01**2)
    assert s2.iloc[2] == pytest.approx(0.9 * s2.iloc[1] + 0.1 * 0.02**2)
    default = lab.ewma_variance(r, lam=0.9)
    assert default.iloc[0] == pytest.approx(np.mean(r.to_numpy() ** 2))


def test_ewma_no_lookahead(garch_df):
    r = garch_df["ret"].copy()
    base = lab.ewma_variance(r)
    r.iloc[500] = 0.2
    bumped = lab.ewma_variance(r)
    assert bumped.iloc[500] == pytest.approx(base.iloc[500])
    assert bumped.iloc[501] > base.iloc[501]


def test_ewma_tracks_true_sigma(garch_df):
    s = np.sqrt(lab.ewma_variance(garch_df["ret"]))
    corr = np.corrcoef(s.iloc[100:], garch_df["sigma"].iloc[100:])[0, 1]
    assert corr > 0.8


def test_garch_variance_recursion():
    eps = np.array([0.01, -0.02, 0.005])
    v = lab.garch_variance((1e-6, 0.1, 0.8), eps)
    assert v[0] == pytest.approx(np.var(eps))
    assert v[1] == pytest.approx(1e-6 + 0.1 * 0.01**2 + 0.8 * v[0])
    assert v[2] == pytest.approx(1e-6 + 0.1 * 0.02**2 + 0.8 * v[1])


def test_neg_loglik_value_and_penalty():
    eps = np.array([0.01, -0.02, 0.005, 0.0])
    p = (1e-6, 0.1, 0.8)
    v = lab.garch_variance(p, eps)
    expected = 0.5 * np.sum(np.log(2 * np.pi) + np.log(v) + eps**2 / v)
    assert lab.garch_neg_loglik(p, eps) == pytest.approx(expected)
    assert lab.garch_neg_loglik((-1.0, 0.1, 0.8), eps) >= 1e9


def test_true_params_beat_wrong_params(garch_df):
    eps = (garch_df["ret"] - garch_df["ret"].mean()).to_numpy()
    true = lab.garch_neg_loglik((TRUE["omega"], TRUE["alpha"], TRUE["beta"]), eps)
    wrong = lab.garch_neg_loglik((2e-5, 0.05, 0.50), eps)
    assert true < wrong


def test_fit_recovers_parameters(fitted):
    assert fitted["alpha"] == pytest.approx(TRUE["alpha"], abs=0.03)
    assert fitted["beta"] == pytest.approx(TRUE["beta"], abs=0.04)
    assert fitted["alpha"] + fitted["beta"] == pytest.approx(0.98, abs=0.015)
    assert fitted["alpha"] + fitted["beta"] < 1
    lr = lab.long_run_variance(fitted["omega"], fitted["alpha"], fitted["beta"])
    true_lr = lab.long_run_variance(**TRUE)
    assert lr == pytest.approx(true_lr, rel=0.25)


def test_fit_matches_arch_package(garch_df, fitted):
    from arch import arch_model

    res = arch_model(
        garch_df["ret"] * 100, mean="Constant", vol="GARCH", p=1, q=1, dist="normal"
    ).fit(disp="off")
    assert fitted["alpha"] == pytest.approx(res.params["alpha[1]"], abs=0.01)
    assert fitted["beta"] == pytest.approx(res.params["beta[1]"], abs=0.015)
    # arch 用百分比收益：omega 差 1e4 倍
    assert fitted["omega"] * 1e4 == pytest.approx(res.params["omega"], rel=0.3)


def test_fit_loglik_is_optimal(garch_df, fitted):
    eps = (garch_df["ret"] - fitted["mu"]).to_numpy()
    best = lab.garch_neg_loglik((fitted["omega"], fitted["alpha"], fitted["beta"]), eps)
    assert fitted["loglik"] == pytest.approx(-best)
    true = lab.garch_neg_loglik((TRUE["omega"], TRUE["alpha"], TRUE["beta"]), eps)
    assert best <= true + 1e-6


def test_forecast_converges_to_long_run():
    o, a, b = 2e-6, 0.08, 0.90
    vl = lab.long_run_variance(o, a, b)
    assert vl == pytest.approx(1e-4)
    f = lab.garch_forecast(o, a, b, last_var=4e-4, last_eps=0.03, horizon=500)
    assert f[0] == pytest.approx(o + a * 0.03**2 + b * 4e-4)
    assert f[1] - vl == pytest.approx((a + b) * (f[0] - vl))
    assert f[-1] == pytest.approx(vl, rel=1e-3)
    assert np.all(np.diff(f) < 0)  # 从高位单调回落


def test_half_life():
    assert lab.half_life(0.08, 0.90) == pytest.approx(np.log(0.5) / np.log(0.98))
    assert 30 < lab.half_life(0.08, 0.90) < 40


def test_vol_target_weights():
    s = pd.Series([0.01, 0.002, 0.02])
    w = lab.vol_target_weights(s, target_vol=0.10, max_leverage=2.0)
    assert w.iloc[0] == pytest.approx(0.10 / (0.01 * np.sqrt(252)))
    assert w.iloc[1] == pytest.approx(2.0)
    assert w.iloc[2] == pytest.approx(0.10 / (0.02 * np.sqrt(252)))


def test_vol_target_stabilizes_vol(garch_df):
    r = garch_df["ret"]
    vt = lab.vol_target_returns(r, target_vol=0.10, max_leverage=5.0)
    assert vt.index.equals(r.index)
    vt = vt.iloc[50:]
    ann = vt.std() * np.sqrt(252)
    assert ann == pytest.approx(0.10, rel=0.15)
    # 年度波动率的离散程度应明显小于原始资产
    yearly_raw = r.iloc[50:].groupby(r.iloc[50:].index.year).std()
    yearly_vt = vt.groupby(vt.index.year).std()
    cv_raw = yearly_raw.std() / yearly_raw.mean()
    cv_vt = yearly_vt.std() / yearly_vt.mean()
    assert cv_vt < 0.6 * cv_raw


def test_vol_target_no_lookahead(garch_df):
    r = garch_df["ret"].copy()
    base = lab.vol_target_returns(r)
    r.iloc[700] = -0.15
    bumped = lab.vol_target_returns(r)
    # 当天的仓位不应该因为当天的收益而改变
    w_base = base.iloc[700] / garch_df["ret"].iloc[700]
    w_bumped = bumped.iloc[700] / -0.15
    assert w_bumped == pytest.approx(w_base)
