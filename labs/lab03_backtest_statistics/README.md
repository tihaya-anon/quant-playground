# Lab 03 · 回测的统计学

> 难度 ★★☆☆☆ · 预计 3 小时 · 前置：Lab 01、Lab 02，基础数理统计（delta 方法、中心极限定理）

回测 Sharpe 2.0——那又怎样？它是一个 **估计量**，有抽样误差；而且如果你试了 100 组参数才
挑出它，它还带着巨大的 **选择偏差**。这个 lab 回答两个问题：

1. 这个 Sharpe 的置信区间有多宽？
2. 考虑到我试了多少次，它还显著吗？

## 你要学会的

1. **Sharpe 的标准误**。iid 正态时（Lo, 2002；delta 方法）
   $$\operatorname{SE}(\widehat{SR}) = \sqrt{\frac{1 + SR^2/2}{n}}\qquad(\text{每期尺度}).$$
   直觉：年化 Sharpe 1.0，10 年日频数据，年化标准误约 $\sqrt{(1+0.002)/2520}\cdot\sqrt{252}\approx 0.32$。
   一年的数据？标准误约 1.0——一年的回测几乎什么都说明不了。
2. **非正态修正**（Mertens, 2002）：
   $$\operatorname{SE} = \sqrt{\frac{1 - \gamma_3 SR + \frac{\gamma_4 - 1}{4}SR^2}{n-1}},$$
   负偏度 $\gamma_3<0$ 与肥尾 $\gamma_4>3$ 都让 Sharpe 更不可信。卖波动率的策略要格外小心。
3. **Bootstrap**。不依赖分布假设地得到置信区间。iid bootstrap 会破坏自相关；
   **块 bootstrap** 保留块内的时间结构。收益正自相关时，前者给出的区间偏窄。
4. **Probabilistic Sharpe Ratio**：$PSR(SR^*) = \Phi\!\left(\frac{(\widehat{SR}-SR^*)\sqrt{n-1}}{\sqrt{1-\gamma_3\widehat{SR}+\frac{\gamma_4-1}{4}\widehat{SR}^2}}\right)$，
   “真实 Sharpe 超过 $SR^*$”的置信度。
5. **多重检验与 Deflated Sharpe Ratio**。$N$ 个真实 Sharpe 都为 0 的策略，最大的那个估计值期望为
   $$\mathbb E[\max_k \widehat{SR}_k] \approx \sqrt{V}\left((1-\gamma)\Phi^{-1}\!\left(1-\tfrac1N\right) + \gamma\,\Phi^{-1}\!\left(1-\tfrac{1}{Ne}\right)\right),$$
   $\gamma\approx0.5772$。DSR 就是以它为基准的 PSR。测试里会看到：200 个纯噪声策略中最好的那个，
   年化 Sharpe > 1.5、PSR > 95%，但 DSR < 60%。
6. **Walk-forward**：只用过去的数据选参数，在紧接着的未来上评估，滚动前进。
   全样本挑最优参数得到的 Sharpe 是乐观的；walk-forward 的样本外 Sharpe 才是你该期待的。

## 任务

实现 [`lab.py`](lab.py)，然后：

```bash
make test LAB=03
```

`momentum_strategy_returns` 是给定的，不用改。

## 思考题

1. 证明 Lo 的公式：对 $(\hat\mu, \hat\sigma^2)$ 用 delta 方法，正态下 $\operatorname{Var}(\hat\sigma^2) = 2\sigma^4/n$。
2. 块大小怎么选？在 AR(1) $\phi=0.3$ 的数据上，画出区间宽度随块大小 1、5、10、25、50、100 的变化。
   为什么太大的块也不好？
3. DSR 需要 $N$——“试了多少次”。如果你和同事都在同一个数据集上挖了一年，$N$ 是多少？
   如果 100 组参数高度相关（比如回看期 50 和 55），有效 $N$ 应该更小还是更大？
   （提示：López de Prado 建议对试验聚类后取簇数。）
4. Walk-forward 的训练窗口越长越好吗？在 `regime_switching_prices` 上试试 train_size = 126、504、1512。
5. 把 Lab 02 的双均线网格扫描结果代入 DSR，最优参数还显著吗？

## 拓展：用真实数据

```python
from quantlab import realdata

px = realdata.index_daily("sh000300")["close"]
wf = lab.walk_forward(px, range(5, 255, 10), train_size=756, test_size=63)
```

比较全样本最优回看期的 Sharpe、walk-forward 样本外 Sharpe、以及 DSR。
再看 `wf["lookback"]` 随时间怎么跳——参数稳定吗？参数不稳定本身就是一个危险信号。
