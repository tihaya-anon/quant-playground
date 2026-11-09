# Lab 05 · 组合优化

> 难度 ★★★☆☆ · 预计 3–4 小时 · 前置：Lab 01，线性代数，拉格朗日乘子

Markowitz (1952) 的均值-方差优化是现代组合理论的起点，也是一个著名的“理论很美、实践很糟”的例子。
这个 lab 先推出所有闭式解，再用数值方法处理没有闭式解的风险平价和带约束问题，最后亲手证明：
**直接把样本估计量塞进优化器，得到的是“误差最大化器”。**

## 你要学会的

1. **最小方差组合**：$\min_w w'\Sigma w$ s.t. $\mathbf 1'w = 1$，解为 $w = \Sigma^{-1}\mathbf 1 / (\mathbf 1'\Sigma^{-1}\mathbf 1)$。
   一阶条件 $\Sigma w \propto \mathbf 1$：每个资产的**边际风险**相同。
2. **切点组合**：最大化 Sharpe，$w \propto \Sigma^{-1}(\mu - r_f)$。
3. **有效前沿与两基金分离**：记 $A=\mathbf 1'\Sigma^{-1}\mathbf 1,\ B=\mathbf 1'\Sigma^{-1}\mu,\ C=\mu'\Sigma^{-1}\mu,\ D=AC-B^2$，
   目标收益 $t$ 的最优组合是 $\Sigma^{-1}\mathbf 1$ 与 $\Sigma^{-1}\mu$ 的线性组合，前沿方程
   $\sigma^2(t) = (At^2 - 2Bt + C)/D$ 是一条双曲线。
4. **风险贡献**：$\sigma_p(w)$ 是一次齐次函数，欧拉定理给出 $\sigma_p = \sum_i w_i\,\partial\sigma_p/\partial w_i$，
   于是 $RC_i = w_i(\Sigma w)_i/\sigma_p$。**风险平价**要求所有 $RC_i$ 相等——它完全不用 $\mu$，
   这恰恰是它在实践中受欢迎的原因（$\mu$ 估不准，见 Lab 00）。
5. **协方差的估计误差与收缩**：$N$ 个资产有 $N(N+1)/2$ 个参数。$N$ 接近样本数 $T$ 时，样本协方差的
   小特征值被严重低估，而最小方差组合恰好押注在最小特征值的方向上。
   Ledoit-Wolf 把 $S$ 向结构化目标 $\mu I$ 收缩：$\hat\Sigma = (1-\delta)S + \delta\mu I$，
   $\delta$ 由最小化期望 Frobenius 损失解析地选出。
6. **约束优化**：现实中有只做多、单票上限、行业上限、换手约束……没有闭式解，交给凸优化（cvxpy）。
   均值-方差是一个 QP：$\max\ \mu'w - \tfrac{\gamma}{2}w'\Sigma w$。

## 任务

打开 [`lab.py`](lab.py)，实现每个 `raise NotImplementedError` 的函数。docstring 就是规格说明。

```bash
make test LAB=05
make test LAB=05 K=risk_parity
```

## 思考题

1. 推导 `test_mean_variance_unconstrained_matches_closed_form` 里的闭式解。$\gamma \to \infty$ 时它趋向哪个组合？
2. 把 `oos_min_variance` 的训练窗口从 120 天依次改成 60、250、500、900 天，画出 predicted/realized 比值。
   它和 $N/T$ 有什么关系？（随机矩阵理论里的 Marchenko–Pastur 分布给了答案。）
3. 风险平价组合在低波动资产（如债券）上权重很高。为什么实践中风险平价基金要加杠杆？
   在利率上行的环境里这会带来什么问题？
4. 给 `test_universe` 里的 $\mu$ 加一点点噪声（比如每个 ±1%），切点组合的权重变化多大？
   这就是为什么 Black–Litterman 和“对 $\mu$ 收缩”会被发明出来。

## 拓展：用真实数据

```python
from quantlab import realdata

codes = ["600519", "000858", "601318", "600036", "000333", "600276", "601012", "300750"]
px = pd.concat({c: realdata.a_share_daily(c)["close"] for c in codes}, axis=1).dropna()
```

做一个滚动回测：每月末用过去 250 天估计协方差，分别构建等权、最小方差（样本）、最小方差（LW）、
风险平价四个组合，持有一个月。比较它们的实现波动率、Sharpe 和换手率。
