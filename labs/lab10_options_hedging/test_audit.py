"""审计 audit.py 的自动检查。

建议先自己读 audit.py、写下审计记录，再跑这些测试——测试的名字本身就是提示。
"""

import numpy as np
import pandas as pd
import pytest

from labs._harness import load_impl
from quantlab import data, options

audit = load_impl(__file__, module="audit")

N_DAYS = 3780  # 15 年
SEEDS = range(8)


def world(seed):
    """波动率聚集、偶有崩盘的日收益。"""
    return options.jump_garch_returns(N_DAYS, seed=seed)


@pytest.fixture(scope="module")
def reports():
    return [audit.evaluate(world(s)) for s in SEEDS]


# ------------------------------------------------------------ 检查一：定价只能用当时已知的信息


def test_selling_vol_does_not_depend_on_the_future():
    """第 t 天卖出时用的波动率，不能因为第 t 天及之后的收益改变而改变。"""
    r = world(0)
    t = 1500
    other = r.copy()
    other.iloc[t:] = world(99).iloc[t:].to_numpy() * 3.0
    a = audit.selling_vol(r).iloc[: t + 1]
    b = audit.selling_vol(other).iloc[: t + 1]
    assert np.allclose(a.to_numpy(), b.to_numpy(), equal_nan=True)


# ------------------------------------------------------------ 检查二：风险事件必须进入回测


def test_a_crash_shows_up_in_the_pnl():
    """在某个持有期中间放进一次 -15% 的暴跌，那一份期权的 P&L 必须明显变差。

    （只看那一份：暴跌之后已实现波动率飙升，下一份期权会卖出很高的价格，
    总 P&L 里的损失会被部分抵消——这本身也值得想一想。）
    """
    r = world(1)
    t = audit.HOLD * 40 + 10
    crash_day = r.index[t]
    shocked = r.copy()
    shocked.iloc[t] = -0.15
    base = audit.monthly_pnl(r).loc[:crash_day].iloc[-1]
    hit = audit.monthly_pnl(shocked).loc[:crash_day].iloc[-1]
    assert hit < base - 0.08


# ------------------------------------------------------------ 检查三：成交价


def test_wider_spread_means_lower_pnl():
    """买卖价差越大，卖方实际收到的权利金越少。"""
    r = world(2)
    tight = audit.monthly_pnl(r, half_spread=0.0).mean()
    wide = audit.monthly_pnl(r, half_spread=0.05).mean()
    assert wide < tight


# ------------------------------------------------------------ 检查四：尾部风险


def test_tail_risk_is_not_rosier_than_what_happened():
    """报告的尾部风险不能比历史上真实发生过的更乐观。

    用一个典型的“卖保险”型 P&L 分布：平时小赚，偶尔大亏。
    """
    rng = np.random.default_rng(0)
    n = 600
    pnl = rng.normal(0.004, 0.01, n)
    crash = rng.random(n) < 0.04
    pnl[crash] -= rng.uniform(0.05, 0.12, crash.sum())
    pnl = pd.Series(pnl, index=data.business_index(n))
    worst = np.sort(pnl.to_numpy())[: int(0.05 * n)]
    realized = -worst.mean()
    assert audit.tail_risk(pnl, 0.05) >= 0.9 * realized


# ------------------------------------------------------------ 检查五：结论


def test_short_vol_pnl_is_negatively_skewed(reports):
    """卖波动率在有崩盘的世界里，月度 P&L 应该是负偏的。"""
    assert np.median([r["skew"] for r in reports]) < -0.3


def test_worst_month_reflects_crash_risk(reports):
    assert np.median([r["worst_month"] for r in reports]) < -0.05


def test_verdict_respects_the_risk_limit(reports):
    """在这个世界里，诚实的回测大多数情况下不会同时通过显著性和风险限额。"""
    assert sum(r["verdict"] == "deploy" for r in reports) <= 1


def test_tstat_is_plausible(reports):
    """溢价只有 10%、还要承担崩盘，t 值不可能远超 3。"""
    assert np.median([r["tstat"] for r in reports]) < 3
