# Lab 10 · 期权定价与 Delta 对冲

> 难度 ★★★★☆ · 预计 4–6 小时 · 前置：lab00（GBM）、伊藤引理、正态分布

这个 lab 把 Black-Scholes 从“一条公式”变成“一台可以拆开看的机器”：
你会亲手写定价、Greeks、隐含波动率求解器和蒙特卡洛，然后做一件最能建立直觉的事——
**真的去对冲一份卖出的期权**，看对冲误差从哪里来、P&L 到底在赌什么。

## 你要学会的

1. **BSM 公式**（连续股息率 $q$）：

   $$C = S e^{-qT} N(d_1) - K e^{-rT} N(d_2),\quad
   d_{1,2} = \frac{\ln(S/K) + (r - q \pm \tfrac12\sigma^2)T}{\sigma\sqrt T}$$

   注意漂移 $\mu$ 不在公式里：可复制性让定价只依赖风险中性测度。
2. **Put-call parity**：$C - P = S e^{-qT} - K e^{-rT}$，与模型无关，只依赖无套利。
3. **Greeks** 与 BS 偏微分方程：

   $$\Theta + \tfrac12 \sigma^2 S^2 \Gamma + (r-q) S \Delta - rV = 0$$

   这条式子说的是：一个 delta 中性的组合，时间损耗 $\Theta$ 恰好由 gamma 收益
   $\tfrac12\Gamma S^2\sigma^2$ 补偿。测试会用中心差分检验你的每个 Greek。
4. **隐含波动率**：价格对 $\sigma$ 单调（vega > 0），所以反解唯一。Newton 法二次收敛但在 vega≈0
   处发散——生产代码总是带一个区间法（brentq）兜底。
5. **蒙特卡洛与降方差**：标准误 $\propto 1/\sqrt{n}$，想多一位精度要 100 倍算力。
   对偶变量（$Z$ 与 $-Z$）利用了单调 payoff 的负相关；控制变量利用一个**已知期望**的相关量
   （这里是 $e^{-rT}S_T$，期望 $S e^{-qT}$），最优系数 $b^* = \mathrm{Cov}(Y,C)/\mathrm{Var}(C)$，
   方差缩减比例为 $1 - \rho_{YC}^2$。
6. **离散对冲的 P&L**：卖出期权并 delta 对冲，一步的 P&L 约为

   $$\delta\Pi \approx \tfrac12 \Gamma S^2 \left(\sigma_{\text{impl}}^2\,\delta t - (\delta S/S)^2\right)$$

   - 实现波动 = 隐含波动时，每步期望为 0，但离散采样带来噪声，总误差标准差 $\propto 1/\sqrt{N}$。
   - 实现波动 ≠ 隐含波动时，累积出 $\int \tfrac12\Gamma S^2(\sigma_{\text{impl}}^2 - \sigma_{\text{real}}^2)\,dt$：
     **卖期权 = 做空波动率**。这也是为什么 delta 对冲后的期权账户本质上是在交易
     “隐含 vs 实现”的波动率价差。
   - 漂移 $\mu$ 对 P&L 均值没有影响——被 delta 对冲掉了。

## 任务

实现 [`lab.py`](lab.py) 中的 6 个函数，docstring 就是规格：

| 函数 | 内容 |
| --- | --- |
| `d1_d2` / `bs_price` | BSM 定价（向量化） |
| `put_call_parity_gap` | 检查无套利关系 |
| `greeks` | 解析 Greeks（theta 按日历时间、vega 按 1.0 波动率） |
| `implied_vol` | Newton + brentq 兜底，含无套利边界检查 |
| `mc_price` | plain / antithetic / control 三种蒙特卡洛 |
| `delta_hedge_pnl` | 卖出看涨 + 离散 delta 对冲的到期 P&L |

```bash
make test LAB=10
make test LAB=10 K=greeks
```

## 思考题

1. 画出 ATM 看涨期权的 gamma 随剩余期限的变化。为什么临近到期的 ATM 期权对冲误差最大？
2. 固定总对冲次数，把对冲时点集中在临近到期（而非均匀）会怎样？为什么实务里有人用
   “按 delta 变化幅度触发”的对冲（如 Whalley-Wilmott 带宽）而不是固定时间间隔？
3. 加入每次调仓 5bp 的交易成本，对冲误差的标准差和均值各怎么变？对冲频率存在最优值吗？
   （Leland 1985 给出了一个“调整后的波动率”答案。）
4. 在 `delta_hedge_pnl` 里把 `sigma_hedge` 用于收权利金，但用 `sigma_real` 计算 delta，
   P&L 的路径依赖性会如何变化？（提示：此时 P&L 变成确定性的 $V(\sigma_{\text{impl}}) - V(\sigma_{\text{real}})$ 的终值。）
5. 控制变量换成“同一路径下的几何平均亚式期权”（有解析解）去定价算术平均亚式期权，方差能降多少？

## 拓展

- 用 `scipy.stats.t` 生成肥尾收益再做对冲实验：BS delta 还最优吗？
- 从真实期权链（例如用 akshare 拿 50ETF 期权）反推隐含波动率，画出波动率微笑——
  这正是 BS 常数波动率假设“错在哪里”的证据。
