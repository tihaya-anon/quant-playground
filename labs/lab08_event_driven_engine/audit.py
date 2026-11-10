"""研究报告：小盘股短线反转，“扣费后 Sharpe 超过 1”。

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
        return max(self.min_commission, self.commission_rate * quantity * price)

    def _fill_price(self, order: Order, bar: Mapping[str, float]) -> float | None:
        o, h, lo = bar["open"], bar["high"], bar["low"]
        if order.order_type == "market":
            return o * (1 + self.slippage_bps * 1e-4)
        lim = order.limit_price
        if order.quantity > 0:
            # 盘中最低价触及限价，说明有人在这个价格卖过，我们的买单就能成交
            return min(o, lim) if lo <= lim else None
        return max(o, lim) if h >= lim else None

    def process_bar(
        self, timestamp: pd.Timestamp, bars: Mapping[str, Mapping[str, float]]
    ) -> list[Fill]:
        # 订单量远小于日成交量，成交量约束在这里不会起作用，直接全部成交
        return super().process_bar(timestamp, bars)


class AuditEngine(engine.Engine):
    """收盘集合竞价下单：信号在收盘时产生，订单按当根 bar 的收盘价成交。"""

    def run(self) -> pd.Series:
        df = next(iter(self.data.values()))
        sym = next(iter(self.data))
        for ts in df.index:
            bar = df.loc[ts].to_dict()
            bars = {sym: bar}
            for fill in self.broker.process_bar(ts, bars):
                self.portfolio.apply_fill(fill)
            for order in self.strategy.on_bar(ts, bars, self.portfolio):
                order.created_at = ts
                if order.order_type == "market":
                    price = bar["close"]
                    self.portfolio.apply_fill(
                        Fill(
                            ts,
                            sym,
                            order.quantity,
                            price,
                            self.broker.commission(order.quantity, price),
                            order.id,
                        )
                    )
                else:
                    self.broker.submit(order)
            self.portfolio.record(ts, {sym: bar["close"]})
        return self.portfolio.equity_curve()


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
