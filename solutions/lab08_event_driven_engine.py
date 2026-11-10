"""Lab 08 审计题参考答案：audit.py 修正后的版本。

修正清单：
1. AuditEngine 按信号 bar 的收盘价成交市价单：收盘价在收盘之后才知道，而且它本身带着
   买卖价差反弹——“跌了就买”恰好常常买在 bid 上，这部分“alpha”无法成交。
   删掉自定义的 run，用 quantlab.engine.Engine：订单最早在下一根 bar 开盘成交。
2. commission 用了带符号的 quantity：卖单的比例佣金为负，被 max() 换成了最低佣金，
   卖出几乎不付佣金。用 abs(quantity)。
3. 市价单滑点不分方向：卖出也按 open*(1+slip) 成交，等于卖在了更好的价格。
   滑点总是不利于你：买 open*(1+slip)，卖 open*(1-slip)。
4. 限价单“触及即成交”：价格刚好碰到你的限价时，排在队列后面的你多半没有成交；
   保守的假设是价格至少穿过限价一个 tick 才算成交。这同时刻画了逆向选择：你能成交的，
   恰恰是价格继续往不利方向走的那些情况；价格碰一下就反弹的“好成交”，你往往拿不到。
5. 没有成交量约束：单根 bar 最多成交 participation * volume，剩余部分留到之后的 bar。

原审计说明如下。

研究报告：小盘股短线反转，“扣费后 Sharpe 超过 1”。

背景：一位同事用事件驱动引擎回测了一个短线反转策略——

    "单日跌幅超过 1.5% 就在收盘买入，次日起挂 1% 的止盈限价单，最多持有 3 天。
     成交模型已经包含了滑点和佣金（含最低佣金），回测 6 年，扣费后 Sharpe 超过 1。
     建议在一篮子流动性较差的小盘股上上线。"

引擎（quantlab.engine）本身是对的。问题出在这份回测**配置的成交假设**上：
下面的 AuditBroker 和 AuditEngine 是同事为了“更贴近实盘”自己写的。

你的任务是做 code review：找出会让结论偏乐观的错误，在这个文件里改正，
直到 `make test LAB=08` 全部通过。

规则：
- 不要改类名、函数签名和 evaluate() 返回的字典键，测试依赖它们。
- 改正之前，先在下面的「审计记录」里写下你认为的问题和理由，再动代码。
- quantlab.engine 里的 Order / Fill / Broker / Portfolio / Engine 是正确的，可以放心复用。

审计记录（自己填）：
    1.
    2.
    3.
    ...
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from quantlab import engine, perf
from quantlab.engine import Fill, Order

TICK = 0.01


class AuditBroker(engine.Broker):
    """更贴近实盘的成交模型：滑点、佣金（含最低佣金）、限价单、成交量约束。

    participation: 单根 bar 内最多成交该 bar 成交量的这个比例。
    """

    def __init__(
        self,
        commission_rate: float = 3e-4,
        min_commission: float = 5.0,
        slippage_bps: float = 10.0,
        participation: float = 0.05,
    ):
        super().__init__(commission_rate, min_commission, slippage_bps)
        self.participation = participation

    def commission(self, quantity: float, price: float) -> float:
        return max(self.min_commission, self.commission_rate * abs(quantity) * price)

    def _fill_price(self, order: Order, bar: Mapping[str, float]) -> float | None:
        o, h, lo = bar["open"], bar["high"], bar["low"]
        side = 1.0 if order.quantity > 0 else -1.0
        if order.order_type == "market":
            return o * (1 + side * self.slippage_bps * 1e-4)
        lim = order.limit_price
        if side > 0:
            return min(o, lim) if lo < lim - TICK else None
        return max(o, lim) if h > lim + TICK else None

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
            cap = self.participation * bar["volume"]
            qty = (
                order.quantity
                if abs(order.quantity) <= cap
                else cap * (1 if order.quantity > 0 else -1)
            )
            fills.append(
                Fill(timestamp, order.symbol, qty, price, self.commission(qty, price), order.id)
            )
            left = order.quantity - qty
            if abs(left) > 1e-9:
                order.quantity = left
                remaining.append(order)
        self._open = remaining
        return fills


class AuditEngine(engine.Engine):
    """订单最早在下一根 bar 撮合：直接使用 quantlab.engine.Engine 的事件顺序。"""


class ReversalStrategy(engine.Strategy):
    """跌幅超过 threshold 就买入 shares 股；持仓后挂止盈限价单，最多持有 max_hold 根 bar。"""

    def __init__(
        self,
        symbol: str = "ASSET",
        shares: float = 2000,
        threshold: float = 0.015,
        take_profit: float = 0.01,
        max_hold: int = 3,
    ):
        self.symbol, self.shares = symbol, shares
        self.threshold, self.take_profit, self.max_hold = threshold, take_profit, max_hold
        self.prev_close: float | None = None
        self.held = 0
        self.tp_live = False

    def on_bar(self, timestamp, bars, portfolio):
        bar = bars.get(self.symbol)
        if bar is None:
            return []
        close = bar["close"]
        ret = None if self.prev_close is None else close / self.prev_close - 1
        self.prev_close = close
        pos = portfolio.position(self.symbol)
        orders: list[Order] = []
        if pos > 0:
            self.held += 1
            if not self.tp_live:
                cost = portfolio.avg_cost[self.symbol]
                orders.append(
                    Order(self.symbol, -pos, "limit", limit_price=cost * (1 + self.take_profit))
                )
                self.tp_live = True
            elif self.held >= self.max_hold:
                orders.append(Order(self.symbol, -pos))
        else:
            self.held, self.tp_live = 0, False
            if ret is not None and ret < -self.threshold:
                orders.append(Order(self.symbol, self.shares))
        return orders


def evaluate(ohlcv: pd.DataFrame, cash: float = 1_000_000.0) -> dict:
    """跑一遍回测，返回研究结论。

    返回字典：
        sharpe        日收益年化 Sharpe（扣费后）
        total_return  总收益
        n_fills       成交笔数
        commission    总佣金
    """
    broker = AuditBroker()
    eng = AuditEngine(ohlcv, ReversalStrategy(), broker=broker, portfolio=engine.Portfolio(cash))
    eq = eng.run()
    rets = eq.pct_change().dropna()
    return {
        "sharpe": perf.sharpe(rets),
        "total_return": float(eq.iloc[-1] / cash - 1),
        "n_fills": len(eng.portfolio.fills),
        "commission": eng.portfolio.total_commission,
    }
