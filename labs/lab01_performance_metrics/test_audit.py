"""审计 audit.py 的自动检查。

建议先自己读 audit.py、写下审计记录，再跑这些测试——测试的名字本身就是提示。
每个测试都是检查任何风险报表时都值得做的一类事情。
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from labs._harness import load_impl
from quantlab import data

audit = load_impl(__file__, module="audit")


def fat_tailed_returns(n, seed=0, vol=0.01, df=3):
    """学生 t 分布的日收益（肥尾），标准差为 vol，交易日索引。"""
    rng = np.random.default_rng(seed)
    t = rng.standard_t(df, n) / np.sqrt(df / (df - 2))
    return pd.Series(0.0003 + vol * t, index=data.business_index(n))


def nav_from(returns):
    return 100 * (1 + returns).cumprod()


def as_calendar_file(nav):
    """管理人的格式：日历日，非交易日沿用上一个交易日的净值。"""
    full = pd.date_range(nav.index[0], nav.index[-1], freq="D")
    return nav.reindex(full).ffill()


@pytest.fixture(scope="module")
def nav():
    return nav_from(fat_tailed_returns(2520, seed=1))


# ------------------------------------------------------------ 检查一：数据格式不应改变结论


def test_volatility_does_not_depend_on_file_format(nav):
    """同一段业绩，按交易日给还是按日历日给，年化波动和 Sharpe 应该一样。"""
    a = audit.dashboard(nav)
    b = audit.dashboard(as_calendar_file(nav))
    assert b["ann_vol"] == pytest.approx(a["ann_vol"], rel=0.02)
    assert b["sharpe"] == pytest.approx(a["sharpe"], rel=0.02)


def test_volatility_matches_the_generating_process(nav):
    assert audit.dashboard(as_calendar_file(nav))["ann_vol"] == pytest.approx(
        0.01 * np.sqrt(252), rel=0.08
    )


# ------------------------------------------------------------ 检查二：回撤是路径的性质


def test_max_drawdown_sees_every_observation(nav):
    """最大回撤不能小于净值路径上任何一个“前高到后低”的跌幅。"""
    worst = float((1 - nav / nav.cummax()).max())
    assert audit.max_drawdown(nav) == pytest.approx(worst, rel=1e-9)


def test_intra_month_crash_counts():
    """月中跌 30%、月底前收复：这依然是一次 30% 的回撤。"""
    idx = data.business_index(60)
    values = np.full(60, 100.0)
    values[25:30] = 70.0
    nav = pd.Series(values, index=idx)
    assert audit.max_drawdown(nav) == pytest.approx(0.30)


# ------------------------------------------------------------ 检查三：下行风险的定义


def test_sortino_matches_downside_deviation_definition():
    """正态收益的下行偏差 sqrt(E[min(r,0)^2]) 有闭式解，Sortino 应与之一致。"""
    rng = np.random.default_rng(3)
    mu, sd = 0.001, 0.01
    r = pd.Series(rng.normal(mu, sd, 1_000_000))
    a = -mu / sd
    lower_pm2 = (mu**2 + sd**2) * stats.norm.cdf(a) - mu * sd * stats.norm.pdf(a)
    expected = mu * 252 / (np.sqrt(lower_pm2) * np.sqrt(252))
    assert audit.sortino(r) == pytest.approx(expected, rel=0.03)


# ------------------------------------------------------------ 检查四：尾部风险要回测


def test_var_is_breached_about_as_often_as_promised():
    """99% VaR 的意思是：大约 1% 的日子损失超过它。在样本外检验这一点。"""
    r = fat_tailed_returns(400_000, seed=4)
    var, _ = audit.var_cvar(r.iloc[:200_000])
    breaches = (r.iloc[200_000:] < -var).mean()
    assert 0.007 < breaches < 0.0125


def test_cvar_matches_the_average_tail_loss():
    """CVaR 是“最坏 1% 的日子里平均亏多少”。拿实际发生的尾部损失来核对。"""
    r = fat_tailed_returns(400_000, seed=5)
    _, cvar = audit.var_cvar(r.iloc[:200_000])
    test = r.iloc[200_000:]
    tail = -test[test <= test.quantile(0.01)].mean()
    assert cvar == pytest.approx(tail, rel=0.08)
