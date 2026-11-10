from collections import deque

import numpy as np
import pandas as pd
import pytest

from quantlab import data
from quantlab import engine as lab

TS = pd.Timestamp("2020-01-02")
BAR = {"open": 100.0, "high": 105.0, "low": 95.0, "close": 102.0, "volume": 1e6}


@pytest.fixture(scope="module")
def ohlcv():
    close = data.gbm_prices(n_days=600, mu=0.05, sigma=0.25, seed=5)
    return data.ohlcv_from_close(close, seed=5)


def sma_targets(ohlcv, fast=10, slow=40, shares=100):
    """向量化地算出“t 收盘时想要的持仓”。"""
    c = ohlcv["close"]
    return pd.Series(
        np.where(c.rolling(fast).mean() > c.rolling(slow).mean(), shares, 0), index=c.index
    )


def vectorized_equity(ohlcv, targets, cash0):
    """lab02 风格的向量化参考：t 收盘决定的目标，在 t+1 开盘成交，无成本。"""
    held = targets.shift(1).fillna(0)  # bar t 开盘成交后实际持有的股数
    trades = held.diff().fillna(held)
    cash = cash0 - (trades * ohlcv["open"]).cumsum()
    return cash + held * ohlcv["close"]


# ---------------------------------------------------------------- Broker


def test_market_order_fills_at_open_with_slippage():
    b = lab.Broker(slippage_bps=10)
    b.submit(lab.Order("X", 10))
    b.submit(lab.Order("X", -5))
    fills = b.process_bar(TS, {"X": BAR})
    assert [f.quantity for f in fills] == [10, -5]
    assert fills[0].price == pytest.approx(100.0 * 1.001)
    assert fills[1].price == pytest.approx(100.0 * 0.999)
    assert fills[0].timestamp == TS
    assert b.open_orders == []


def test_commission_with_minimum():
    b = lab.Broker(commission_rate=0.001, min_commission=5.0)
    assert b.commission(10, 100.0) == pytest.approx(5.0)
    assert b.commission(-1000, 100.0) == pytest.approx(100.0)
    b.submit(lab.Order("X", 1000))
    (f,) = b.process_bar(TS, {"X": BAR})
    assert f.commission == pytest.approx(100.0)


def test_limit_orders():
    b = lab.Broker(slippage_bps=50)
    buy_hit = lab.Order("X", 1, "limit", limit_price=96.0)
    buy_miss = lab.Order("X", 1, "limit", limit_price=90.0)
    buy_gap = lab.Order("X", 1, "limit", limit_price=110.0)  # 开盘就低于限价
    sell_hit = lab.Order("X", -1, "limit", limit_price=104.0)
    sell_gap = lab.Order("X", -1, "limit", limit_price=90.0)
    for o in (buy_hit, buy_miss, buy_gap, sell_hit, sell_gap):
        b.submit(o)
    fills = {f.order_id: f.price for f in b.process_bar(TS, {"X": BAR})}
    assert fills == {
        buy_hit.id: pytest.approx(96.0),
        buy_gap.id: pytest.approx(100.0),
        sell_hit.id: pytest.approx(104.0),
        sell_gap.id: pytest.approx(100.0),
    }
    assert [o.id for o in b.open_orders] == [buy_miss.id]  # GTC
    low_bar = dict(BAR, open=91.0, low=89.0)
    (f,) = b.process_bar(TS + pd.Timedelta(days=1), {"X": low_bar})
    assert f.order_id == buy_miss.id and f.price == pytest.approx(90.0)


def test_missing_symbol_keeps_order():
    b = lab.Broker()
    b.submit(lab.Order("Y", 1))
    assert b.process_bar(TS, {"X": BAR}) == []
    assert len(b.open_orders) == 1
    b.cancel_all()
    assert b.open_orders == []


# ---------------------------------------------------------------- Portfolio


def fill(q, px, comm=0.0, sym="X"):
    return lab.Fill(TS, sym, q, px, comm, 0)


def test_portfolio_long_round_trip():
    p = lab.Portfolio(cash=10_000)
    p.apply_fill(fill(10, 100, comm=1))
    p.apply_fill(fill(10, 110, comm=1))
    assert p.position("X") == 20
    assert p.avg_cost["X"] == pytest.approx(105)
    assert p.cash == pytest.approx(10_000 - 1000 - 1100 - 2)
    p.apply_fill(fill(-5, 120, comm=1))
    assert p.realized_pnl == pytest.approx(5 * 15)
    assert p.avg_cost["X"] == pytest.approx(105)  # 减仓不改均价
    p.apply_fill(fill(-15, 100))
    assert p.position("X") == 0
    assert p.realized_pnl == pytest.approx(75 - 75)
    assert p.total_commission == pytest.approx(3)
    assert p.position("NOPE") == 0.0


def test_portfolio_short_and_flip():
    p = lab.Portfolio(cash=0)
    p.apply_fill(fill(-10, 50))
    assert p.cash == pytest.approx(500)
    assert p.unrealized_pnl({"X": 40}) == pytest.approx(100)  # 空头赚钱
    p.apply_fill(fill(15, 40))  # 平 10 空，再开 5 多
    assert p.realized_pnl == pytest.approx(100)
    assert p.position("X") == 5
    assert p.avg_cost["X"] == pytest.approx(40)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_accounting_identity_random_fills(seed):
    rng = np.random.default_rng(seed)
    p = lab.Portfolio(cash=1e5)
    prices = {"X": 100.0, "Y": 20.0}
    for _ in range(300):
        sym = rng.choice(["X", "Y"])
        prices[sym] *= np.exp(rng.normal(0, 0.02))
        q = int(rng.integers(-50, 51)) or 1
        p.apply_fill(fill(q, prices[sym], comm=abs(q) * 0.01, sym=sym))
        eq = p.equity(prices)
        assert eq == pytest.approx(p.cash + sum(p.position(s) * prices[s] for s in prices))
        assert eq - p.initial_cash == pytest.approx(
            p.realized_pnl + p.unrealized_pnl(prices) - p.total_commission, abs=1e-6
        )


# ---------------------------------------------------------------- Engine


class SpyStrategy(lab.Strategy):
    """在第 k 根 bar 收盘时买 1 股，并记录 on_bar 时看到的成交情况。"""

    def __init__(self, k):
        self.k = k
        self.i = 0
        self.positions_seen = []

    def on_bar(self, timestamp, bars, portfolio):
        self.positions_seen.append(portfolio.position("ASSET"))
        self.i += 1
        if self.i - 1 == self.k:
            return [lab.Order("ASSET", 1)]
        return []


def test_engine_no_lookahead(ohlcv):
    k = 50
    strat = SpyStrategy(k)
    pf = lab.Portfolio(cash=0)
    lab.Engine(ohlcv, strat, portfolio=pf).run()
    (f,) = pf.fills
    assert f.timestamp == ohlcv.index[k + 1]
    assert f.price == pytest.approx(ohlcv["open"].iloc[k + 1])
    assert strat.positions_seen[k] == 0 and strat.positions_seen[k + 1] == 1


def test_engine_matches_vectorized(ohlcv):
    targets = sma_targets(ohlcv)
    pf = lab.Portfolio(cash=100_000)
    eq = lab.Engine(ohlcv, lab.TargetPositionStrategy("ASSET", targets), portfolio=pf).run()
    assert isinstance(eq, pd.Series) and eq.index.equals(ohlcv.index)
    ref = vectorized_equity(ohlcv, targets, 100_000)
    assert np.allclose(eq.to_numpy(), ref.to_numpy())
    assert len(pf.fills) > 5


def test_engine_costs_reduce_equity(ohlcv):
    targets = sma_targets(ohlcv)
    free = lab.Engine(
        ohlcv, lab.TargetPositionStrategy("ASSET", targets), portfolio=lab.Portfolio(1e5)
    ).run()
    pf = lab.Portfolio(1e5)
    broker = lab.Broker(commission_rate=0.0005, min_commission=5, slippage_bps=5)
    costly = lab.Engine(ohlcv, lab.TargetPositionStrategy("ASSET", targets), broker, pf).run()
    assert costly.iloc[-1] < free.iloc[-1]
    assert pf.total_commission >= 5 * len(pf.fills)
    last = {"ASSET": ohlcv["close"].iloc[-1]}
    assert costly.iloc[-1] == pytest.approx(pf.equity(last))


class RollingBreakout(lab.Strategy):
    """事件驱动版的 20 日突破：自己维护历史窗口。"""

    def __init__(self):
        self.window = deque(maxlen=20)

    def on_bar(self, timestamp, bars, portfolio):
        c = bars["ASSET"]["close"]
        orders = []
        if len(self.window) == 20:
            pos = portfolio.position("ASSET")
            if c > max(self.window) and pos <= 0:
                orders.append(lab.Order("ASSET", 10 - pos))
            elif c < min(self.window) and pos >= 0:
                orders.append(lab.Order("ASSET", -10 - pos))
        self.window.append(c)
        return orders


def test_engine_multi_symbol_and_identity(ohlcv):
    other = data.ohlcv_from_close(data.gbm_prices(n_days=600, seed=9), seed=9).iloc[::2]  # 稀疏行情
    pf = lab.Portfolio(1e5)
    eq = lab.Engine({"ASSET": ohlcv, "B": other}, RollingBreakout(), portfolio=pf).run()
    assert len(eq) == len(ohlcv)
    last = {"ASSET": ohlcv["close"].iloc[-1], "B": other["close"].iloc[-1]}
    assert eq.iloc[-1] - pf.initial_cash == pytest.approx(
        pf.realized_pnl + pf.unrealized_pnl(last) - pf.total_commission
    )
    assert all(f.symbol == "ASSET" for f in pf.fills)
