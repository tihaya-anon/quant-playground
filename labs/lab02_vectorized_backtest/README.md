# Lab 02 · 向量化回测

> 难度 ★★☆☆☆ · 预计 2–3 小时 · 前置：Lab 00、Lab 01

回测 = 用历史数据回答“如果当时按这套规则交易，会怎样”。向量化回测把整段历史当作数组
一次算完，快、短、好调试，是研究阶段的主力工具。但它也最容易写出 **前视偏差（look-ahead bias）**
——用到了当时还不知道的信息——而且这种 bug 不会报错，只会给你一条漂亮得可疑的净值曲线。

## 你要学会的

1. **时间对齐**。$t$ 日收盘算出信号 $s_t$，按收盘价调仓，于是承担的是 $t+1$ 日的收益：
   $$\text{pos}_{t+1} = s_t,\qquad g_{t+1} = \text{pos}_{t+1}\cdot r_{t+1}.$$
   代码里就是一个 `shift(1)`。漏掉它，等于用今天的收益决定今天的仓位。
2. **交易成本**。换手 $\tau_t = |\text{pos}_t - \text{pos}_{t-1}|$，成本 $c\cdot\tau_t$（$c$ 以 bps 计）。
   多空翻转换手为 2。**盈亏平衡成本** $c^* = \bar g / \bar\tau$：策略每交易 1 单位能赚多少。
   高频信号常常毛收益很好，但 $c^*$ 只有 1–2 bps，实盘根本做不到。
3. **趋势跟踪两件套**：双均线交叉、时间序列动量 $\operatorname{sign}(P_t/P_{t-L}-1)$。
4. **有信号 vs 没信号**。同一个策略在 `regime_switching_prices`（有趋势）上赚钱，在 GBM
   （随机游走，原则上不可预测）上 Sharpe ≈ 0。任何在 GBM 上“稳定赚钱”的回测几乎一定有 bug。
5. **波动率目标化** $w_t = s_t\cdot\min(\sigma^*/\hat\sigma_t, L)$：让组合风险恒定，
   这是 CTA / 风险平价最基本的构件。

## 任务

实现 [`lab.py`](lab.py)，然后：

```bash
make test LAB=02
```

测试里有两个专门抓前视偏差的用例：
- `test_lookahead_is_not_profitable`：一个“偷看当日收益”的信号在正确回测里不应赚钱；
- `test_backtest_uses_only_past`：篡改未来价格，过去的回测结果不能变。

## 思考题

1. 用 `regime_switching_prices` 做一次 (fast, slow) 的网格扫描，画 Sharpe 热力图。
   最优参数附近是平滑的高原还是孤立的尖峰？哪一种更值得相信？（Lab 03 会定量回答。）
2. 假设实际只能在 **次日开盘** 成交（`quantlab.data.ohlcv_from_close` 能给你开盘价），
   回测应该怎么改？结果差多少？这叫执行滞后（implementation lag）。
3. 动量策略的回看期从 5 天到 250 天，盈亏平衡成本如何变化？画出来。
4. 对波动率目标化：为什么它在 GARCH 数据上能提高 Sharpe，在常波动率 GBM 上不能？

## 拓展：用真实数据

```python
from quantlab import realdata

px = realdata.index_daily("sh000300")["close"]
res = lab.backtest(px, lab.sma_crossover_signal(px, 20, 120, long_only=True), cost_bps=5)
```

A 股不能随意做空，所以先用 `long_only=True`。对比买入持有：收益、最大回撤、换手、盈亏平衡成本。
然后思考：这里 5 bps 的成本假设现实吗？（佣金、印花税只在卖出收、冲击成本。）
