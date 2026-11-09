# Lab 11 · 最优执行：TWAP、VWAP 与 Almgren-Chriss

> 难度 ★★★★☆ · 预计 3–5 小时 · 前置：lab00、基础变分法 / 二阶差分方程

回测里“按收盘价成交”是一个谎言。真实世界里，你下单本身会推动价格：卖得越快，冲击越大；
卖得越慢，暴露在价格波动里的时间越长。最优执行就是在这两种成本之间做权衡——
这和 Markowitz 的“收益 vs 风险”是同一个数学结构，只是变量从“权重”变成了“时间轨迹”。

## 你要学会的

1. **Implementation shortfall（IS）**：决策时的纸面价值 $X S_0$ 减去实际收入 $\sum n_k \tilde S_k$。
   它是衡量执行质量的标准度量。
2. **两类冲击**：
   - 永久冲击 $g(v)=\gamma v$：你的交易让市场相信信息，价格永久移动，后面所有成交都吃亏。
   - 临时冲击 $h(v)=\epsilon\,\mathrm{sgn}(n)+\eta v$：吃掉订单簿的流动性，只影响本次成交价。
3. **期望与方差**（线性冲击下的闭式解）：

   $$E[\text{IS}] = \tfrac12\gamma X^2 + \epsilon\sum_k|n_k| + \frac{\tilde\eta}{\tau}\sum_k n_k^2,\qquad
   \mathrm{Var}[\text{IS}] = \sigma^2\tau\sum_{k=1}^N x_k^2$$

   其中 $\tilde\eta=\eta-\tfrac12\gamma\tau$。注意永久冲击项 $\tfrac12\gamma X^2$ 与轨迹无关——
   线性永久冲击下，怎么卖都要付这笔钱。
4. **Almgren-Chriss 最优轨迹**：最小化 $E + \lambda V$，一阶条件是二阶差分方程
   $\frac{x_{k-1}-2x_k+x_{k+1}}{\tau^2} = \tilde\kappa^2 x_k$，解为

   $$x_j = X\,\frac{\sinh\big(\kappa(T-t_j)\big)}{\sinh(\kappa T)},\qquad
   \cosh(\kappa\tau) = 1 + \tfrac12\tilde\kappa^2\tau^2,\quad \tilde\kappa^2=\frac{\lambda\sigma^2}{\tilde\eta}$$

   - $\lambda\to0$：风险中性，退化为 **TWAP**（线性轨迹，冲击成本最小）。
   - $\lambda$ 越大：越“前置”（front-load），尽快卖掉以减少价格风险。
   - $1/\kappa$ 是交易的“半衰期”时间尺度，与总时长 $T$ 无关。
5. **有效前沿**：每个 $\lambda$ 给出前沿上的一点；在 $\lambda=0$ 处前沿斜率为 0，
   意味着从 TWAP 出发稍微加一点风险厌恶，几乎不增加期望成本就能显著降方差。
6. **VWAP**：按日内成交量曲线（通常是 U 型）分配，是最常见的被动基准；
   它最小化的是“相对 VWAP 的跟踪误差”，不是 IS。

## 任务

实现 [`lab.py`](lab.py) 中的函数。测试用的参数取自 Almgren & Chriss (2000) 原文例子
（卖 100 万股、5 天、每半天一次，$\lambda=10^{-6}$ 时 $\kappa\approx0.6$/天），可以拿来和论文对照。

```bash
make test LAB=11
make test LAB=11 K=simulation
```

## 思考题

1. 证明 $\sum_k x_k n_k = \tfrac12 X^2 - \tfrac12\sum_k n_k^2$，并由此推出期望成本公式。
2. 为什么线性永久冲击下 $\tfrac12\gamma X^2$ 与轨迹无关？如果永久冲击是非线性的（比如 $\propto\sqrt v$），
   还成立吗？（Huberman & Stanzl 指出非线性永久冲击会允许“价格操纵”式套利。）
3. 实证上临时冲击更接近**平方根律** $h(v)\propto\sigma\sqrt{v/V_{\text{daily}}}$。
   把 `simulate_shortfall` 改成平方根冲击，用数值优化（`scipy.optimize.minimize`）找最优轨迹，
   和 sinh 解比较形状。
4. 加入一个对价格的方向性预测（alpha，例如你认为价格会以 $\mu$ 漂移下跌），最优轨迹会如何变化？
   推一下带漂移项的 AC 解。
5. 执行成本对 lab02 的回测意味着什么？用这里的 $E[\text{IS}]$ 给一个“策略容量”的粗略估计：
   在多大资金规模下，年化 alpha 会被冲击成本吃光？

## 拓展

- 画出 $\lambda$ 从 $10^{-8}$ 到 $10^{-4}$ 的有效前沿（横轴 $\sqrt V$，纵轴 $E$），
  并在图上标出 TWAP 和一次性全部卖出（$n_1=X$）两个极端点。
- 读 Almgren, Thum, Hauptmann & Li (2005) "Direct Estimation of Equity Market Impact"，
  看看真实的冲击参数是怎么从成交数据里估出来的。
