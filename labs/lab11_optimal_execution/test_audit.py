"""审计 audit.py 的自动检查。

建议先自己读 audit.py、写下审计记录，再跑这些测试——测试的名字本身就是提示。
"""

import numpy as np
import pandas as pd
import pytest

from labs._harness import load_impl
from quantlab import data, execution

audit = load_impl(__file__, module="audit")


@pytest.fixture(scope="module")
def market():
    return execution.strategy_market(seed=0)


@pytest.fixture(scope="module")
def report(market):
    return audit.evaluate(market, aum=5e8)


# ------------------------------------------------------------ 检查一：流动性估计


def test_liquidity_estimate_reflects_a_typical_day():
    """ADV 应该代表“典型的一天”能成交多少，而不是被少数放量日拉高。"""
    rng = np.random.default_rng(0)
    idx = data.business_index(500)
    typical = 1e7
    vol = typical * rng.lognormal(0, 0.2, (500, 3))
    spikes = rng.random((500, 3)) < 0.03
    vol[spikes] *= 25
    adv = audit.estimate_adv(pd.DataFrame(vol, index=idx, columns=["a", "b", "c"]))
    assert np.all(np.abs(adv.to_numpy() / typical - 1) < 0.15)


# ------------------------------------------------------------ 检查二：成本模型能复现真实成本


def test_calibrated_impact_reproduces_what_we_actually_paid(market):
    """执行记录里我们实际付出的成本，是从决定交易那一刻算起的。校准出的系数应接近真实值。"""
    coef = audit.calibrate_impact(market.exec_log)
    assert coef == pytest.approx(market.impact_coef, rel=0.2)


# ------------------------------------------------------------ 检查三：成交额的口径


def test_round_trip_trades_twice_the_position():
    """从空仓建立 100% 总敞口、再全部平掉，一共成交了 2 倍 AUM。"""
    idx = data.business_index(3)
    w = pd.DataFrame({"a": [0.0, 0.5, 0.0], "b": [0.0, -0.5, 0.0]}, index=idx)
    assert audit.daily_turnover(w).sum() == pytest.approx(2.0)


# ------------------------------------------------------------ 检查四：执行需要时间，alpha 不等人


def test_alpha_capture_falls_when_orders_take_days():
    cap = audit.PARTICIPATION_CAP
    assert audit.alpha_capture(cap / 2) == pytest.approx(1.0)
    big = [audit.alpha_capture(k * cap) for k in (2, 4, 8)]
    assert big[0] < 1.0
    assert big[0] > big[1] > big[2]


# ------------------------------------------------------------ 检查五：结论


def test_costs_grow_with_size(market):
    small = audit.net_returns(market, 1e7).mean()
    large = audit.net_returns(market, 5e9).mean()
    assert large < small


def test_costs_are_material_at_the_proposed_size(report):
    """5 亿规模、每天 10% 双边换手的策略，每年的成本不可能只有几十个基点。"""
    assert report["cost_annual"] > 0.03


def test_verdict_at_the_proposed_size(report):
    assert report["verdict"] == "reject"


def test_capacity_is_not_overstated(report):
    assert report["capacity"] < 1e9
