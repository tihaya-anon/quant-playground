# Lab 01 · 绩效与风险指标

> 难度 ★☆☆☆☆ · 预计 2 小时 · 前置：Lab 00

一条收益曲线，怎么用几个数字说清楚它“好不好”“危不危险”？这个 lab 实现业界通用的一整套
指标，并搞清楚每个指标背后的假设——以及假设什么时候会骗你。

## 你要学会的

1. **年化**。收益用几何（CAGR）：$(\prod(1+r_t))^{P/n}-1$；波动用 $\hat\sigma\sqrt{P}$，
   因为 iid 时方差随时间线性累加。$P$ 是每年期数（日频 252）。
2. **Sharpe 比率** $SR = \dfrac{\overline{r - r_f}}{\hat\sigma}\sqrt{P}$。分子用 **算术** 均值：
   Sharpe 衡量的是“每单位风险的期望超额收益”，它和杠杆无关（加 2 倍杠杆，分子分母同乘 2）。
3. **Sortino 比率**：只惩罚下行波动。下行偏差
   $DD = \sqrt{\tfrac{1}{n}\sum_t \min(r_t - \text{MAR}, 0)^2}\cdot\sqrt P$，分母是 **全部** 样本数。
4. **回撤** $D_t = W_t / \max_{s\le t} W_s - 1$、最大回撤、最长水下期、Calmar = CAGR / MDD。
   回撤是投资者真正“感受到”的风险——一个 Sharpe 1.0 的策略，10 年里遇到 -30% 回撤并不稀奇。
5. **VaR / CVaR**。$\text{VaR}_\alpha = -q_\alpha(r)$；$\text{CVaR}_\alpha = -\mathbb E[r \mid r\le q_\alpha]$。
   正态假设下 $\text{CVaR}_\alpha = -\big(\mu - \sigma\,\varphi(z_\alpha)/\alpha\big)$（自己推一下截断正态的期望）。
   CVaR 是一致性风险度量（满足次可加性），VaR 不是。
6. **偏度与峰度**。真实收益几乎都是 **负偏 + 肥尾**，于是正态 VaR 会系统性低估尾部风险。
   测试里用 $t_3$ 分布演示：1% CVaR 被低估 20% 以上。

## 任务

实现 [`lab.py`](lab.py) 里的函数，然后：

```bash
make test LAB=01
```

做完以后，后续 lab 统一使用 `quantlab.perf` 里的参考实现（它们与你这里的定义一致）。

## 思考题

1. 同一个策略，用日收益和月收益分别算年化 Sharpe，结果一样吗？如果日收益存在正自相关呢？
   （提示：$\sqrt{P}$ 缩放只在收益不相关时成立，Lo (2002) 给出了修正。）
2. 一个卖出深度虚值期权的策略：99% 的天赚一点点，1% 的天亏一大笔。它的 Sharpe、Sortino、
   偏度、历史 VaR(95%)、CVaR(99%) 各自会讲出什么故事？用模拟数据验证你的猜测。
3. 为什么 `max_drawdown` 的数值对样本长度非常敏感，而 Sharpe 不那么敏感？
   在 GBM 上模拟 5 年 vs 20 年的最大回撤分布看看。
4. `rolling_sharpe` 窗口 1 年时，即便真实 Sharpe 恒为 0.5，滚动值也会剧烈波动。
   估计它的标准差，和 Lab 03 的 $\sqrt{(1+SR^2/2)/n}$ 对照。

## 拓展：用真实数据

```python
from quantlab import realdata

px = realdata.index_daily("sh000300")["close"]
r = px.pct_change().dropna()
lab.performance_summary(r)
```

对比沪深300、中证500（`sh000905`）的指标；再看 2015、2018、2022 年各自的最大回撤。
看看偏度和超额峰度，并比较历史 VaR 与正态 VaR 在 1% 水平上的差距。
