# Lab 07 · 波动率建模：EWMA、GARCH 与波动率目标

> 难度 ★★★☆☆ · 预计 3–4 小时 · 前置：Lab 00/01，极大似然估计，scipy.optimize

Lab 00 告诉你：期望收益几乎估不出来。这个 lab 讲它的反面——**波动率不仅好估，而且可预测**。
金融收益有两个稳健的“典型事实”：收益本身几乎不自相关，但**收益平方（波动率）强烈自相关**
（波动率聚集）。这让风险管理、仓位管理、期权定价都有了抓手。

## 你要学会的

1. **滚动波动率**：$\hat\sigma_t = \text{std}(r_{t-w+1..t})\sqrt{252}$。简单，但窗口内等权、窗口外权重骤降为 0，
   一个大冲击会在 $w$ 天后“突然消失”，造成伪跳变。
2. **EWMA（RiskMetrics）**：$\sigma^2_t = \lambda\sigma^2_{t-1} + (1-\lambda) r^2_{t-1}$，$\lambda=0.94$。
   等价于指数衰减权重，有效窗口约 $1/(1-\lambda) \approx 17$ 天。注意下标：$\sigma^2_t$ 只用到 $t-1$，
   它是**预测**，不是事后估计。
3. **GARCH(1,1)**：
   $$\varepsilon_t = \sigma_t z_t,\quad \sigma^2_t = \omega + \alpha\varepsilon^2_{t-1} + \beta\sigma^2_{t-1}$$
   EWMA 是 $\omega=0,\ \alpha+\beta=1$ 的特例（单位根，没有均值回归）。$\alpha+\beta<1$ 时方差回归到
   长期水平 $V_L = \omega/(1-\alpha-\beta)$，冲击半衰期 $h = \ln 0.5/\ln(\alpha+\beta)$。
4. **极大似然**：正态新息下
   $$-\ell(\theta) = \tfrac12\sum_t\left[\ln 2\pi + \ln\sigma^2_t + \varepsilon^2_t/\sigma^2_t\right]$$
   难点不在公式而在数值：$\omega\sim10^{-6}$ 与 $\alpha,\beta\sim10^{-1}$ 量级悬殊，还有 $\alpha+\beta<1$ 的约束。
   **重参数化**把有约束、病态的问题变成无约束、尺度良好的问题——这是实际做数值优化最常用的技巧。
5. **多步预测**：$\mathbb{E}_T[\sigma^2_{T+h}] = V_L + (\alpha+\beta)^{h-1}(\sigma^2_{T+1}-V_L)$，指数回归长期方差。
6. **波动率目标**：$w_t = \sigma^*/\hat\sigma_t$。高波动时降杠杆、低波动时加杠杆，组合波动率接近常数。
   因为波动率可预测而收益几乎不可预测，这通常能提升风险调整后收益（Moreira & Muir 2017）。
   **必须**只用 $t-1$ 的预测，否则是前视偏差——用含当天收益的波动率给当天定仓位，回测会好得离谱。

## 任务

实现 [`lab.py`](lab.py) 里的函数，然后：

```bash
make test LAB=07
```

测试会用已知参数 $(\omega,\alpha,\beta) = (2\times10^{-6}, 0.08, 0.90)$ 的模拟数据检验你的 MLE，
并和 `arch` 包的估计结果对照。

## 思考题

1. 把 `fit_garch` 换成直接对 $(\omega,\alpha,\beta)$ 用 Nelder-Mead、不做重参数化，会发生什么？试试不同初值。
2. 用 4000 天数据，$\hat\alpha,\hat\beta$ 各自的误差和 $\hat\alpha+\hat\beta$ 的误差，哪个更小？为什么？
   （提示：似然面在 $\alpha$-$\beta$ 方向上是一条细长的山谷。）
3. 正态 GARCH 的标准化残差 $\hat\varepsilon_t/\hat\sigma_t$ 在真实数据上峰度是多少？换成 Student-t 新息
   （`arch_model(..., dist="t")`）后似然提升多少？
4. 波动率目标策略在 GBM（常数波动）上和买入持有相比如何？在 GARCH 数据上呢？在“收益与波动负相关”
   （杠杆效应）的数据上呢？这说明它的收益来源是什么？
5. GJR-GARCH 加入 $\gamma\varepsilon^2_{t-1}\mathbb{1}[\varepsilon_{t-1}<0]$。为什么股票指数上 $\gamma$ 显著为正？

## 拓展：用真实数据

```python
from quantlab import realdata

r = realdata.index_daily("sh000300")["close"].pct_change().dropna()
```

在沪深300 上拟合 GARCH(1,1)，看 $\alpha+\beta$ 和半衰期；画出 2015 年股灾前后的条件波动率。
再做一个 10% 目标波动率的沪深300 策略，与买入持有比较 Sharpe 和最大回撤（`quantlab.perf`）。
