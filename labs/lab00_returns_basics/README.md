# Lab 00 · 收益的算术

> 难度 ★☆☆☆☆ · 预计 1–2 小时 · 前置：numpy / pandas 基础

量化里几乎所有 bug 都从“收益算错了”开始：简单收益和对数收益混用、组合收益用对数收益加权、
月度收益用日收益直接相加……这个 lab 把这些基本功一次练扎实，并且第一次碰到一个贯穿全课程的
事实：**波动率很好估，期望收益几乎估不出来。**

## 你要学会的

1. 简单收益 $r_t = P_t/P_{t-1} - 1$ 与对数收益 $\ell_t = \ln(P_t/P_{t-1})$，以及 $\ell = \ln(1+r)$。
2. **时间上**对数收益可加：$\ell_{0\to T} = \sum_t \ell_t$；简单收益要连乘：$1+R = \prod_t (1+r_t)$。
3. **横截面上**简单收益可加：组合收益 $r_p = \sum_i w_i r_i$；对数收益**不行**。
4. 每日再平衡 vs 买入持有：权重会漂移，$w_{i,t+1} \propto w_{i,t}(1+r_{i,t+1})$。
5. 几何布朗运动（GBM）：$S_T = S_0 \exp\big((\mu - \tfrac{1}{2}\sigma^2)T + \sigma W_T\big)$，
   所以 $\mathbb{E}[S_T] = S_0 e^{\mu T}$，但**中位数**是 $S_0 e^{(\mu-\sigma^2/2)T}$——波动拖累（volatility drag）。
6. 参数估计：从 $n$ 天对数收益估 $\hat\sigma$ 与 $\hat\mu$。$\hat\mu$ 的标准误是 $\sigma/\sqrt{T_{\text{years}}}$，
   和采样频率无关——10 年数据、20% 波动率，$\hat\mu$ 的标准误约 6.3%，比你想估的 8% 还大。

## 任务

打开 [`lab.py`](lab.py)，实现每个 `raise NotImplementedError` 的函数。docstring 就是规格说明。

```bash
make test LAB=00          # 只跑这个 lab 的测试
make test LAB=00 K=gbm    # 只跑名字里带 gbm 的测试
```

## 思考题（不计分，但值得写在 notebook 里）

1. 某基金第一年 +50%，第二年 -50%。算术平均收益是多少？实际净值是多少？用对数收益解释。
2. `annualize_log_stats` 在 10 年 GBM 上估出的 $\hat\mu$ 与真值差多少？把 `seed` 换 20 个看看分布。
   如果把数据从日频换成分钟频（同样 10 年），$\hat\mu$ 的精度会变好吗？$\hat\sigma$ 呢？为什么？
3. 一个“每日再平衡到 50/50”的两资产组合，和“期初 50/50 然后不动”的组合，长期谁更好？
   （提示：再平衡收益 / 波动拖累，在两资产相关性低的时候试试。）

## 拓展：用真实数据

```python
from quantlab import realdata

px = realdata.index_daily("sh000300")["close"]  # 沪深300
```

用 `annualize_log_stats` 估一下沪深300 的 $\mu, \sigma$，再算一下 $\hat\mu$ 的 95% 置信区间。
