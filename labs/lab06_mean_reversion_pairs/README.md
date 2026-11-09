# Lab 06 · 均值回归与配对交易

> 难度 ★★★☆☆ · 预计 3–4 小时 · 前置：Lab 02（向量化回测），时间序列基础

单个股票价格基本是随机游走，没法预测。但两只股票的**某个线性组合**可能是平稳的——这就是协整，
也是配对交易（统计套利的鼻祖）的数学基础。这个 lab 从 AR(1)/OU 过程讲起，做协整检验、
构造交易规则、回测，最后用 Kalman 滤波处理“对冲比例会变”这个现实问题。

## 你要学会的

1. **AR(1) 与半衰期**：$x_t = c + \phi x_{t-1} + \varepsilon_t$，$|\phi|<1$ 时平稳，均值 $c/(1-\phi)$。
   偏离衰减一半所需时间 $h = -\ln 2/\ln\phi$。半衰期决定了持仓周期，也决定了 z-score 窗口该取多长。
2. **OU 过程**：$dX = \kappa(\theta - X)dt + \sigma dW$ 是 AR(1) 的连续时间版本，
   $\phi = e^{-\kappa\,dt}$，平稳方差 $\sigma^2/(2\kappa) = \sigma_\varepsilon^2/(1-\phi^2)$。
3. **单位根与 ADF 检验**：原假设是 $\phi = 1$（随机游走）。在原假设下 $\hat\phi$ 的分布不是 t 分布
   （Dickey–Fuller 分布），所以有自己的临界值。
4. **协整与 Engle–Granger**：$\log y_t$、$\log x_t$ 都是 I(1)，但 $\log y - \beta\log x$ 是 I(0)。
   第一步 OLS 估 $\beta$（超一致估计，收敛速度 $1/T$ 而非 $1/\sqrt T$），第二步检验残差平稳。
   **注意**：残差是“为了最小化方差而挑出来的”，天然更像平稳序列，因此要用 MacKinnon 协整临界值，
   而不是普通 ADF 的。两条独立随机游走做回归，常常能得到很高的 $R^2$——这就是**伪回归**。
5. **交易规则**：$z_t = (s_t - \bar s_t)/\hat\sigma_t$，$|z|$ 超过入场阈值反向开仓，回到出场阈值平仓。
   这是一个**有状态**的规则（和 Lab 02 的无状态信号不同），需要循环实现。
6. **对冲比例时变**：用 Kalman 滤波把 $(\beta_t, \alpha_t)$ 当成随机游走的隐状态，在线更新。
   它本质上是“带遗忘的递归最小二乘”，`delta` 控制遗忘速度。

## 任务

打开 [`lab.py`](lab.py)，实现每个 `raise NotImplementedError` 的函数。docstring 就是规格说明。

```bash
make test LAB=06
make test LAB=06 K=kalman
```

## 思考题

1. 本 lab 的回测在**全样本**上估计 hedge ratio，然后在同一段数据上交易。这有什么问题？
   改成“前 500 天估计、之后交易”，Sharpe 变多少？用 Kalman 或滚动 OLS 又如何？
2. 在 `pair_returns` 里把 `cost_bps` 从 0 调到 20，Sharpe 怎么变？入场阈值 entry 从 1.0 扫到 3.0，
   交易次数与单次收益如何权衡？最优阈值和半衰期有什么关系？
3. 为什么用对数价格回归而不是价格本身？两者对应的组合在现实中分别怎么持有（股数 vs 金额）？
4. 协整关系在现实中会“断裂”（公司被收购、行业格局变化）。你会设计什么样的止损或监控规则？

## 拓展：用真实数据

```python
from quantlab import realdata

a = realdata.a_share_daily("601398")["close"]  # 工商银行
b = realdata.a_share_daily("601939")["close"]  # 建设银行
```

在同行业里（银行、白酒、券商）找几对股票，用滚动窗口做 Engle–Granger 检验，看协整关系的 p 值
随时间如何变化；对其中最稳定的一对做样本外配对交易回测（含交易成本）。
