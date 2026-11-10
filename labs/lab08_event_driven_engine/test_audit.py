"""审计 audit.py 的自动检查。

建议先自己读 audit.py、写下审计记录，再跑这些测试——测试的名字本身就是提示。
每个测试都是审查任何成交模型时值得做的一类检查。
"""

import numpy as np
import pandas as pd
import pytest

from labs._harness import load_impl
from quantlab import engine

audit = load_impl(__file__, module="audit")

TS = pd.Timestamp("2020-01-02")
BAR = {"open": 100.0, "high": 102.0, "low": 98.0, "close": 101.0, "volume": 1_000_000.0}


def no_edge(seed):
    """真实价格是随机游走，只有买卖价差反弹。"""
    return engine.bounce_ohlcv(1500, sigma=0.15, spread_bps=80, volume=20_000, seed=seed)


def real_reversal(seed):
    """真实价格本身带有均值回复。"""
    return engine.bounce_ohlcv(
        1500, sigma=0.15, spread_bps=80, volume=20_000, mid_ar=-0.25, seed=seed
    )


def fill_one(order, bar=BAR, **broker_kwargs):
    broker = audit.AuditBroker(**broker_kwargs)
    broker.submit(order)
    return broker.process_bar(TS, {order.symbol: bar}), broker


# ------------------------------------------------------------ 检查一：成交时点


class _Probe(engine.Strategy):
    """每 10 根 bar 交替买入/卖出，记下每张订单的下单时间。"""

    def __init__(self):
        self.created = {}
        self.i = 0

    def on_bar(self, timestamp, bars, portfolio):
        self.i += 1
        if self.i % 10:
            return []
        o = engine.Order("ASSET", 100 if (self.i // 10) % 2 else -100, created_at=timestamp)
        self.created[o.id] = timestamp
        return [o]


def test_orders_fill_no_earlier_than_the_next_bar():
    """收盘时才知道的信号，最早只能在下一根 bar 成交。"""
    probe = _Probe()
    eng = audit.AuditEngine(no_edge(0).iloc[:300], probe, broker=audit.AuditBroker())
    eng.run()
    fills = eng.portfolio.fills
    assert fills
    assert all(f.timestamp > probe.created[f.order_id] for f in fills)


# ------------------------------------------------------------ 检查二：成本总是成本


@pytest.mark.parametrize("qty", [100, 5_000, 50_000])
def test_commission_is_symmetric_and_never_below_minimum(qty):
    """买和卖付同样的佣金，而且不会低于最低佣金。"""
    b = audit.AuditBroker()
    buy, sell = b.commission(qty, 100.0), b.commission(-qty, 100.0)
    assert buy == pytest.approx(sell)
    assert buy >= b.min_commission


def test_slippage_hurts_in_both_directions():
    """滑点永远对你不利：买得更贵，卖得更便宜。"""
    (buy,), _ = fill_one(engine.Order("X", 100))
    (sell,), _ = fill_one(engine.Order("X", -100))
    assert buy.price > BAR["open"]
    assert sell.price < BAR["open"]


# ------------------------------------------------------------ 检查三：限价单


def test_limit_order_merely_touched_is_not_assumed_filled():
    """价格只是碰到你的限价，并不能证明你排在队列里成交了；保守的假设是没有成交。"""
    bar = dict(BAR, low=98.0, high=102.0)
    fills, _ = fill_one(engine.Order("X", 100, "limit", limit_price=98.0), bar)
    assert fills == []
    fills, _ = fill_one(engine.Order("X", -100, "limit", limit_price=102.0), bar)
    assert fills == []


def test_limit_order_traded_through_is_filled_at_the_limit():
    bar = dict(BAR, low=97.0, high=103.0)
    (f,), _ = fill_one(engine.Order("X", 100, "limit", limit_price=98.0), bar)
    assert f.price == pytest.approx(98.0)
    (f,), _ = fill_one(engine.Order("X", -100, "limit", limit_price=102.0), bar)
    assert f.price == pytest.approx(102.0)


# ------------------------------------------------------------ 检查四：成交量


def test_fills_respect_the_participation_limit():
    """一根 bar 内的成交不能超过该 bar 成交量的 participation 比例，剩下的留到之后。"""
    bar = dict(BAR, volume=10_000.0)
    order = engine.Order("X", 5_000)
    fills, broker = fill_one(order, bar, participation=0.1)
    assert sum(f.quantity for f in fills) <= 0.1 * 10_000 + 1e-9
    total = sum(f.quantity for f in fills)
    for _ in range(10):
        total += sum(f.quantity for f in broker.process_bar(TS, {"X": bar}))
    assert total == pytest.approx(5_000)


# ------------------------------------------------------------ 检查五：结论


def test_no_edge_after_realistic_fills_on_pure_bounce():
    """真实价格不可预测、只有价差反弹时，扣掉真实成本后不应该有显著的正 Sharpe。"""
    srs = [audit.evaluate(no_edge(s))["sharpe"] for s in range(6)]
    assert np.mean(srs) < 0.2


def test_real_reversal_is_still_found():
    """修正不能矫枉过正：价格本身真的均值回复时，策略应该依然赚钱。"""
    srs = [audit.evaluate(real_reversal(s))["sharpe"] for s in range(6)]
    assert np.mean(srs) > 0.15
