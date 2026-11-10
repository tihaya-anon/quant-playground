"""一个小而完整的事件驱动回测引擎：Order / Fill / Broker / Portfolio / Strategy / Engine。

讲解见 labs/lab08_event_driven_engine/README.md。

每根 bar 内的事件顺序（整个引擎最重要的约定）：

    bar t 开盘 ──► Broker 撮合之前提交的订单（市价单按 open_t 加滑点成交）
             ──► Portfolio 记账
    bar t 收盘 ──► Portfolio 按 close_t 记录净值
             ──► Strategy 看到 bar t（含 close_t）并下单
             ──► 订单进入 Broker，最早在 bar t+1 撮合

Broker 的成交模型是可替换的：子类化并重写 `_fill_price`（以及需要时 `process_bar`）
即可试验不同的成交假设。
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

_ids = itertools.count(1)


@dataclass
class Order:
    symbol: str
    quantity: float
    order_type: str = "market"
    limit_price: float | None = None
    created_at: pd.Timestamp | None = None
    id: int = field(default_factory=lambda: next(_ids))

    def __post_init__(self):
        if self.order_type not in ("market", "limit"):
            raise ValueError(f"unknown order_type {self.order_type!r}")
        if self.order_type == "limit" and self.limit_price is None:
            raise ValueError("limit order needs limit_price")
        if self.quantity == 0:
            raise ValueError("quantity must be non-zero")


@dataclass(frozen=True)
class Fill:
    timestamp: pd.Timestamp
    symbol: str
    quantity: float
    price: float
    commission: float
    order_id: int


class Broker:
    def __init__(
        self, commission_rate: float = 0.0, min_commission: float = 0.0, slippage_bps: float = 0.0
    ):
        self.commission_rate = commission_rate
        self.min_commission = min_commission
        self.slippage_bps = slippage_bps
        self._open: list[Order] = []

    @property
    def open_orders(self) -> list[Order]:
        return list(self._open)

    def submit(self, order: Order) -> None:
        self._open.append(order)

    def cancel_all(self) -> None:
        self._open.clear()

    def commission(self, quantity: float, price: float) -> float:
        return max(self.min_commission, self.commission_rate * abs(quantity) * price)

    def _fill_price(self, order: Order, bar: Mapping[str, float]) -> float | None:
        o, h, lo = bar["open"], bar["high"], bar["low"]
        side = np.sign(order.quantity)
        if order.order_type == "market":
            return o * (1 + side * self.slippage_bps * 1e-4)
        lim = order.limit_price
        if side > 0:
            return min(o, lim) if lo <= lim else None
        return max(o, lim) if h >= lim else None

    def process_bar(
        self, timestamp: pd.Timestamp, bars: Mapping[str, Mapping[str, float]]
    ) -> list[Fill]:
        fills, remaining = [], []
        for order in self._open:
            bar = bars.get(order.symbol)
            price = None if bar is None else self._fill_price(order, bar)
            if price is None:
                remaining.append(order)
                continue
            fills.append(
                Fill(
                    timestamp,
                    order.symbol,
                    order.quantity,
                    price,
                    self.commission(order.quantity, price),
                    order.id,
                )
            )
        self._open = remaining
        return fills


class Portfolio:
    def __init__(self, cash: float = 1_000_000.0):
        self.initial_cash = cash
        self.cash = cash
        self.positions: dict[str, float] = {}
        self.avg_cost: dict[str, float] = {}
        self.realized_pnl = 0.0
        self.total_commission = 0.0
        self.fills: list[Fill] = []
        self._history: list[tuple[pd.Timestamp, float]] = []

    def position(self, symbol: str) -> float:
        return self.positions.get(symbol, 0.0)

    def apply_fill(self, fill: Fill) -> None:
        sym, q, x = fill.symbol, fill.quantity, fill.price
        p = self.position(sym)
        c = self.avg_cost.get(sym, 0.0)
        if p == 0 or np.sign(p) == np.sign(q):
            new_p = p + q
            self.avg_cost[sym] = (p * c + q * x) / new_p
        else:
            closed = min(abs(q), abs(p))
            self.realized_pnl += closed * (x - c) * np.sign(p)
            new_p = p + q
            if new_p == 0:
                self.avg_cost[sym] = 0.0
            elif np.sign(new_p) != np.sign(p):
                self.avg_cost[sym] = x
        self.positions[sym] = new_p
        self.cash -= q * x + fill.commission
        self.total_commission += fill.commission
        self.fills.append(fill)

    def market_value(self, prices: Mapping[str, float]) -> float:
        return float(sum(q * prices[s] for s, q in self.positions.items() if q != 0))

    def equity(self, prices: Mapping[str, float]) -> float:
        return self.cash + self.market_value(prices)

    def unrealized_pnl(self, prices: Mapping[str, float]) -> float:
        return float(
            sum(q * (prices[s] - self.avg_cost[s]) for s, q in self.positions.items() if q != 0)
        )

    def record(self, timestamp: pd.Timestamp, prices: Mapping[str, float]) -> None:
        self._history.append((timestamp, self.equity(prices)))

    def equity_curve(self) -> pd.Series:
        if not self._history:
            return pd.Series(dtype=float, name="equity")
        ts, eq = zip(*self._history, strict=True)
        return pd.Series(eq, index=pd.DatetimeIndex(ts, name="date"), name="equity")


class Strategy:
    def on_bar(
        self, timestamp: pd.Timestamp, bars: Mapping[str, Mapping[str, float]], portfolio: Portfolio
    ) -> list[Order]:
        return []


class TargetPositionStrategy(Strategy):
    def __init__(self, symbol: str, targets: pd.Series):
        self.symbol = symbol
        self.targets = targets

    def on_bar(self, timestamp, bars, portfolio):
        target = float(self.targets.get(timestamp, 0.0))
        delta = target - portfolio.position(self.symbol)
        if delta == 0:
            return []
        return [Order(self.symbol, delta, created_at=timestamp)]


class Engine:
    def __init__(
        self,
        data: dict[str, pd.DataFrame] | pd.DataFrame,
        strategy: Strategy,
        broker: Broker | None = None,
        portfolio: Portfolio | None = None,
        symbol: str = "ASSET",
    ):
        self.data = {symbol: data} if isinstance(data, pd.DataFrame) else dict(data)
        self.strategy = strategy
        self.broker = broker or Broker()
        self.portfolio = portfolio or Portfolio()

    def run(self) -> pd.Series:
        index = sorted(set().union(*(df.index for df in self.data.values())))
        last_close: dict[str, float] = {}
        for ts in index:
            bars = {s: df.loc[ts].to_dict() for s, df in self.data.items() if ts in df.index}
            for fill in self.broker.process_bar(ts, bars):
                self.portfolio.apply_fill(fill)
            for s, b in bars.items():
                last_close[s] = b["close"]
            self.portfolio.record(ts, last_close)
            for order in self.strategy.on_bar(ts, bars, self.portfolio):
                if order.created_at is None:
                    order.created_at = ts
                self.broker.submit(order)
        return self.portfolio.equity_curve()


def bounce_ohlcv(
    n: int = 1500,
    sigma: float = 0.20,
    spread_bps: float = 20.0,
    steps: int = 20,
    volume: float = 50_000.0,
    mid_ar: float = 0.0,
    seed: int | None = 0,
    start: str = "2015-01-02",
) -> pd.DataFrame:
    """带买卖价差反弹（bid-ask bounce）的 OHLCV。

    mid_ar = 0 时，真实价格（mid）是零漂移的随机游走，没有任何可预测性；mid_ar < 0 时，
    mid 的日收益带有真实的均值回复（AR(1) 系数 mid_ar），短线反转策略确实有利可图。
    无论哪种情况，成交价都落在买价或卖价上：
    open / close 随机地等于 mid*(1 ± s/2)，high 是盘中最高 mid 加半个价差（卖价），
    low 是盘中最低 mid 减半个价差（买价）。

    结果：close-to-close 收益带有虚假的负自相关——“跌了就买”在收盘价上看起来能赚钱，
    但没人能以那个收盘价成交。列：open, high, low, close, volume, mid。
    """
    rng = np.random.default_rng(seed)
    half = spread_bps * 1e-4 / 2
    dt = 1.0 / (252 * steps)
    shocks = sigma * np.sqrt(dt) * rng.standard_normal((n, steps))
    log_mid = np.empty((n, steps))
    level, prev_ret = np.log(100.0), 0.0
    for i in range(n):
        path = level + np.cumsum(shocks[i] + mid_ar * prev_ret / steps)
        log_mid[i] = path
        prev_ret = path[-1] - level
        level = path[-1]
    mid = np.exp(log_mid)
    open_mid, close_mid = mid[:, 0], mid[:, -1]
    side_o = rng.choice([-1.0, 1.0], n)
    side_c = rng.choice([-1.0, 1.0], n)
    idx = pd.bdate_range(start=start, periods=n, name="date")
    return pd.DataFrame(
        {
            "open": open_mid * (1 + side_o * half),
            "high": mid.max(axis=1) * (1 + half),
            "low": mid.min(axis=1) * (1 - half),
            "close": close_mid * (1 + side_c * half),
            "volume": rng.lognormal(np.log(volume), 0.3, n).round(),
            "mid": close_mid,
        },
        index=idx,
    )
