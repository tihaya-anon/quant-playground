"""审计 audit.py 的自动检查。

建议先自己读 audit.py、写下审计记录，再跑这些测试——测试的名字本身就是提示。
核对业绩数字最可靠的办法只有一个：任何汇总数字都必须能还原出期初到期末的净值变化。
"""

import numpy as np
import pandas as pd
import pytest

from labs._harness import load_impl
from quantlab import data

audit = load_impl(__file__, module="audit")


def nav_path(seed=0, years=10, mu=0.06, sigma=0.30):
    """一条 years 年的交易日净值。"""
    return data.gbm_prices(int(years * 252) + 1, mu=mu, sigma=sigma, seed=seed, name="nav")


def returns_from(values):
    return pd.Series(values, index=data.business_index(len(values)), dtype=float)


# ------------------------------------------------------------ 检查一：累计数字能还原净值


def test_total_return_reproduces_terminal_nav():
    nav = nav_path(seed=1)
    r = audit.daily_returns(nav)
    assert audit.total_return(r) == pytest.approx(nav.iloc[-1] / nav.iloc[0] - 1, rel=1e-9)


def test_round_trip_is_a_loss():
    """先涨 50% 再跌 50%，钱是变少了的。"""
    assert audit.total_return(returns_from([0.5, -0.5])) == pytest.approx(-0.25)


# ------------------------------------------------------------ 检查二：时间口径


def test_ten_years_of_trading_days_is_ten_years():
    r = audit.daily_returns(nav_path(years=10))
    assert audit.years_covered(r) == pytest.approx(10.0, rel=0.01)


# ------------------------------------------------------------ 检查三：年化收益的定义


@pytest.mark.parametrize("seed", [2, 3, 4])
def test_cagr_compounds_back_to_terminal_nav(seed):
    """(1 + CAGR) ^ 年数 必须等于 期末净值 / 期初净值。"""
    nav = nav_path(seed=seed, years=10)
    r = audit.daily_returns(nav)
    growth = nav.iloc[-1] / nav.iloc[0]
    assert (1 + audit.cagr(r)) ** 10 == pytest.approx(growth, rel=0.01)


def test_cagr_is_negative_when_nav_shrinks():
    """每天交替 +10% / -10%：日均收益为 0，但净值在持续缩水。"""
    r = returns_from([0.10, -0.10] * 126)
    assert audit.cagr(r) < -0.1


# ------------------------------------------------------------ 检查四：费用


def test_annual_fee_is_fully_charged():
    """净值完全不动的产品，收 2% 管理费 10 年，扣费后净值应约为 0.98^10。"""
    r = returns_from([0.0] * 2520)
    net = audit.net_of_fees(r, annual_fee=0.02)
    assert (1 + net).prod() == pytest.approx(0.98**10, rel=0.005)


# ------------------------------------------------------------ 检查五：分段数字与整体一致


def test_monthly_table_compounds_to_total():
    r = audit.daily_returns(nav_path(seed=5, years=3))
    monthly = audit.monthly_returns(r)
    assert (1 + monthly).prod() - 1 == pytest.approx((1 + r).prod() - 1, rel=1e-9)


def test_report_is_internally_consistent():
    nav = nav_path(seed=6, years=10)
    rep = audit.report(nav, annual_fee=0.015)
    assert (1 + rep["cagr"]) ** rep["years"] == pytest.approx(1 + rep["total_return"], rel=1e-6)
    assert (1 + rep["monthly"]).prod() - 1 == pytest.approx(rep["total_return"], rel=1e-6)
    gross = nav.iloc[-1] / nav.iloc[0]
    assert 1 + rep["total_return"] == pytest.approx(gross * 0.985 ** rep["years"], rel=0.005)
    assert np.isfinite(rep["ann_vol"])
