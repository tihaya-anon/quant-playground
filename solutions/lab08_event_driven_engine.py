"""Lab 08 参考答案。"""

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
