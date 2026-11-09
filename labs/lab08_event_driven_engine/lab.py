"""Lab 08 · 事件驱动回测引擎。

这次不是写函数，而是设计一组相互协作的对象。数据类 Order / Fill 和 Strategy 基类已经给好，
你要实现 Broker、Portfolio、TargetPositionStrategy、Engine 里每个 `raise NotImplementedError`。
然后 `make test LAB=08`。

时间线（每根 bar 内的事件顺序，这是整个引擎最重要的约定）：

    bar t 开盘 ──► Broker 撮合 *之前* 提交的订单（市价单按 open_t 成交）
             ──► Portfolio 记账
    bar t 收盘 ──► Portfolio 按 close_t 记录净值
             ──► Strategy 看到 bar t（含 close_t）并下单
             ──► 订单进入 Broker，最早在 bar t+1 撮合

所以“t 收盘产生的信号”最早在 t+1 开盘成交——引擎在结构上就杜绝了前视偏差。
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np  # noqa: F401  (你大概会用到 np.sign)
import pandas as pd

_ids = itertools.count(1)


# ---------------------------------------------------------------------------
# 已给出：数据类
# ---------------------------------------------------------------------------


@dataclass
class Order:
    """一笔订单。quantity 带符号：正数买入，负数卖出（做空）。

    order_type: "market"（下一根 bar 开盘成交）或
                "limit"（价格触及 limit_price 才成交，否则一直挂着）。
    """

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
    """一笔成交。commission 是这笔成交付出的佣金（正数）。"""

    timestamp: pd.Timestamp
    symbol: str
    quantity: float
    price: float
    commission: float
    order_id: int


class Strategy:
    """策略基类。引擎在每根 bar 收盘后调用 on_bar，返回要提交的订单列表。

    bars: {symbol: {"open", "high", "low", "close", "volume"}}，只包含当前这根 bar。
    策略若需要历史，自己在实例里维护（这正是事件驱动和向量化的区别）。
    """

    def on_bar(
        self, timestamp: pd.Timestamp, bars: Mapping[str, Mapping[str, float]], portfolio: Portfolio
    ) -> list[Order]:
        return []


# ---------------------------------------------------------------------------
# 你来实现
# ---------------------------------------------------------------------------


class Broker:
    """模拟经纪商：保存挂单，在每根 bar 上撮合。

    成交规则：
    - 市价单：按该 bar 的 open 成交，并加滑点：买入价 open*(1+bps*1e-4)，卖出价 open*(1-bps*1e-4)。
    - 限价买单：若 low <= limit 则成交，价格 min(open, limit)（开盘就跳空低于限价时按开盘价）。
    - 限价卖单：若 high >= limit 则成交，价格 max(open, limit)。
    - 限价单不加滑点；未成交的限价单保留到下一根 bar（GTC）。
    - 某 symbol 在这根 bar 没有行情时，它的订单保留。
    - 佣金 = max(min_commission, commission_rate * |quantity| * price)。
    """

    def __init__(
        self, commission_rate: float = 0.0, min_commission: float = 0.0, slippage_bps: float = 0.0
    ):
        raise NotImplementedError

    @property
    def open_orders(self) -> list[Order]:
        """当前所有未成交订单（返回副本）。"""
        raise NotImplementedError

    def submit(self, order: Order) -> None:
        raise NotImplementedError

    def cancel_all(self) -> None:
        raise NotImplementedError

    def commission(self, quantity: float, price: float) -> float:
        raise NotImplementedError

    def process_bar(
        self, timestamp: pd.Timestamp, bars: Mapping[str, Mapping[str, float]]
    ) -> list[Fill]:
        """按提交顺序撮合所有挂单，返回成交列表（Fill.timestamp = 这根 bar 的 timestamp）。"""
        raise NotImplementedError


class Portfolio:
    """账户：现金、持仓、平均成本、已实现盈亏、佣金累计、净值历史。

    记账规则（平均成本法）：持仓 p、均价 c，成交 q @ x：
    - 开仓或加仓（p == 0 或 q 与 p 同号）：c <- (p*c + q*x) / (p+q)
    - 减仓/平仓/反手（q 与 p 异号）：平掉的数量 n = min(|q|, |p|)，
      realized_pnl += n * (x - c) * sign(p)；
      平到 0 时 c <- 0；反手（新持仓与 p 异号）时 c <- x；单纯减仓 c 不变。
    - 现金：cash -= q*x + commission。realized_pnl 不含佣金，佣金单独累计在 total_commission。

    恒等式（测试会检查）：
        equity = cash + Σ q_s * price_s
        equity - initial_cash = realized_pnl + unrealized_pnl - total_commission
    """

    def __init__(self, cash: float = 1_000_000.0):
        """需要的属性：initial_cash, cash, positions (dict), avg_cost (dict),
        realized_pnl, total_commission, fills (list)。"""
        raise NotImplementedError

    def position(self, symbol: str) -> float:
        """没有持仓时返回 0.0。"""
        raise NotImplementedError

    def apply_fill(self, fill: Fill) -> None:
        raise NotImplementedError

    def market_value(self, prices: Mapping[str, float]) -> float:
        raise NotImplementedError

    def equity(self, prices: Mapping[str, float]) -> float:
        raise NotImplementedError

    def unrealized_pnl(self, prices: Mapping[str, float]) -> float:
        """Σ q * (price - avg_cost)。"""
        raise NotImplementedError

    def record(self, timestamp: pd.Timestamp, prices: Mapping[str, float]) -> None:
        """记录一个净值点。"""
        raise NotImplementedError

    def equity_curve(self) -> pd.Series:
        """净值历史，索引为 DatetimeIndex，name="equity"。"""
        raise NotImplementedError


class TargetPositionStrategy(Strategy):
    """目标持仓策略：targets 是以时间为索引的目标股数（缺失视为 0）。

    在 bar t 收盘时，如果 targets[t] 与当前持仓不同，下一张市价单补齐差额。
    （因为成交在 t+1，下单时的持仓还没反映上一张单子——这里假设上一张单都已成交，
    用市价单时这总是成立的。）
    """

    def __init__(self, symbol: str, targets: pd.Series):
        raise NotImplementedError

    def on_bar(self, timestamp, bars, portfolio):
        raise NotImplementedError


class Engine:
    """主循环。

    data: 单个 OHLCV DataFrame（此时用 symbol 命名），或 {symbol: DataFrame}。
    run(): 遍历所有 symbol 索引的并集（排序后），对每个 timestamp 按模块顶部的时间线执行，
    返回 portfolio.equity_curve()。记录净值时，某 symbol 当天无行情就用它最近一次的收盘价。
    策略下单时若 order.created_at 为空，填上当前 timestamp。
    broker / portfolio 为 None 时使用默认构造。
    """

    def __init__(
        self,
        data: dict[str, pd.DataFrame] | pd.DataFrame,
        strategy: Strategy,
        broker: Broker | None = None,
        portfolio: Portfolio | None = None,
        symbol: str = "ASSET",
    ):
        raise NotImplementedError

    def run(self) -> pd.Series:
        raise NotImplementedError
