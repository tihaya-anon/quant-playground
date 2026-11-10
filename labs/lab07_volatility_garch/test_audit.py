"""审计 audit.py 的自动检查。

建议先自己读 audit.py、写下审计记录，再跑这些测试——测试的名字本身就是提示。
"""

import numpy as np
import pandas as pd
import pytest

from labs._harness import load_impl
from quantlab import data

audit = load_impl(__file__, module="audit")

DAYS = 252


def normal_returns(mu: float, sigma: float, years: float, seed: int) -> pd.Series:
    """iid 正态日收益：没有波动聚集。mu、sigma 为年化。"""
    n = int(years * DAYS)
    r = np.random.default_rng(seed).normal(mu / DAYS, sigma / np.sqrt(DAYS), n)
    return pd.Series(r, index=data.business_index(n))


def garch_returns(seed: int) -> pd.Series:
    return data.garch_returns(3000, seed=seed)["ret"]


# ------------------------------------------------------------ 检查一：仓位只能用过去的信息


def test_todays_weight_does_not_depend_on_todays_return():
    """第 t 天的仓位在 t 开盘前就要定下来：改动第 t 天及之后的收益，t 天的仓位不能变。"""
    r = garch_returns(0)
    t = 1500
    other = r.copy()
    other.iloc[t:] = garch_returns(1).iloc[t:].to_numpy() * 3
    a = audit.vol_target_weights(r).iloc[: t + 1]
    b = audit.vol_target_weights(other).iloc[: t + 1]
    assert np.allclose(a.to_numpy(), b.to_numpy())


def test_vol_targeting_does_not_help_without_vol_clustering():
    """波动率目标的价值来自“波动率可预测”。在没有波动聚集的 iid 数据上，它不应显著提高 Sharpe。"""
    gains = []
    for s in range(6):
        rep = audit.evaluate(normal_returns(0.08, 0.2, 20, seed=s))
        gains.append(rep["sharpe_vol_target"] - rep["sharpe_buy_hold"])
    assert abs(np.mean(gains)) < 0.1


def test_vol_targeting_still_stabilizes_vol_under_clustering():
    """修正不能矫枉过正：在有波动聚集的数据上，波动率目标依然应该让波动率更稳定。"""
    for s in range(3):
        rep = audit.evaluate(garch_returns(s))
        assert rep["vol_of_vol_target"] < 0.7 * rep["vol_of_vol_buy_hold"]


# ------------------------------------------------------------ 检查二：没有免费的钱


@pytest.mark.parametrize("w", [0.5, 1.0, 2.0, 4.0])
def test_leverage_on_the_risk_free_asset_earns_the_risk_free_rate(w):
    """对一个只赚无风险利率的资产，不论加多少杠杆，组合都只应赚无风险利率。"""
    idx = data.business_index(500)
    rf_daily = audit.RF / DAYS
    r = pd.Series(rf_daily, index=idx)
    out = audit.levered_returns(r, pd.Series(w, index=idx), audit.RF)
    assert np.allclose(out.to_numpy(), rf_daily)


# ------------------------------------------------------------ 检查三：预测要和实现对得上


@pytest.mark.parametrize("lev", [1.0, 2.0, 3.5])
def test_expected_growth_matches_simulated_compounding(lev):
    """“预期年化增长”要和真的按这个杠杆逐日复利得到的增长率对得上。

    用这段样本自己的均值和波动代入公式，排除估计误差，只检查公式本身。
    """
    rf = audit.RF
    r = normal_returns(0.08, 0.20, 50, seed=11)
    lev_r = lev * r - (lev - 1) * rf / DAYS
    realized = np.log1p(lev_r).sum() / 50
    mu_hat = r.mean() * DAYS
    sigma_hat = r.std(ddof=1) * np.sqrt(DAYS)
    assert audit.expected_growth(lev, mu_hat, sigma_hat, rf) == pytest.approx(realized, abs=0.005)


# ------------------------------------------------------------ 检查四：下注规模要承认自己不知道 mu


def test_leverage_choice_has_positive_growth_despite_estimation_error():
    """用 5 年数据估参数再定杠杆。考虑到 mu 的估计误差，平均而言这样选出的杠杆
    应该带来明显为正的超额增长（真实参数：mu-rf = 6%，sigma = 20%）。"""
    mu, sigma, rf = 0.08, 0.20, audit.RF

    def true_excess_growth(f):
        return f * (mu - rf) - 0.5 * f**2 * sigma**2

    g = [
        true_excess_growth(audit.kelly_leverage(normal_returns(mu, sigma, 5, seed=1000 + s)))
        for s in range(400)
    ]
    assert np.mean(g) > 0.017
