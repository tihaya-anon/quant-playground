# Lab 04 · 因子与横截面

> 难度 ★★★☆☆ · 预计 3–4 小时 · 前置：Lab 00–01，线性回归

前面的 lab 都在看“一只资产的时间序列”。真正的股票量化大多是**横截面**问题：每天在几百上千只
股票里，谁明天相对更好？这个 lab 走一遍因子研究的标准流程——回归拆解收益、用 IC 衡量信号、
分组看单调性、做多空组合，最后把信号里“不想要的暴露”中性化掉。

## 你要学会的

1. **OLS 与它的标准误**：$\hat b = (X'X)^{-1}X'y$，$\widehat{\mathrm{Var}}(\hat b) = s^2 (X'X)^{-1}$，
   $s^2 = e'e/(n-k)$。手写一遍，再和 statsmodels 对上，以后看到回归输出就知道每个数字从哪来。
2. **CAPM**：$r_i - r_f = \alpha + \beta (r_m - r_f) + \varepsilon$。$\beta$ 是你承担的市场风险，
   $\alpha$ 才是“本事”。$\beta$ 的标准误约 $\sigma_\varepsilon / (\sigma_m\sqrt{n})$，很小；
   $\alpha$ 的标准误约 $\sigma_\varepsilon/\sqrt{n}$，年化后通常和 $\alpha$ 本身一个量级——又是 Lab 00 那个结论。
3. **滚动 beta**：$\beta_t = \widehat{\mathrm{Cov}}_t(r_i, r_m) / \widehat{\mathrm{Var}}_t(r_m)$，beta 会变。
4. **IC（信息系数）**：$IC_t = \mathrm{SpearmanCorr}(s_{\cdot,t},\ r_{\cdot,t+1})$。
   用秩相关是因为收益有厚尾，几只暴涨暴跌的股票不该主导结论。
   **主动管理基本定律**（Grinold）：$IR \approx IC\sqrt{\text{BR}}$，BR 是每年独立下注次数。
   IC = 0.03 看起来可怜，但 200 只股票 × 252 天的下注次数让它能变成可观的 IR。
5. **ICIR** = mean(IC)/std(IC)，衡量信号的稳定性；t 值 = ICIR·$\sqrt{T}$。
6. **分位数组合**：按信号排序分 5 组，看下一期收益是否**单调**。单调性比多空收益本身更能说明信号是真的。
7. **中性化**：信号常常“意外地”暴露在某些已知风险上（市值、行业、beta……）。
   对每天的横截面做回归 $s = Xg + e$，取残差 $e = (I - H)s$，$H = X(X'X)^{-1}X'$ 是帽子矩阵。
   这是一个**投影**：$e \perp \mathrm{span}(X)$，并且投影两次等于投影一次。

本 lab 的数据里，真实的 alpha 信号与特异收益的 IC ≈ 0.05，但观测到的信号混入了
$0.8 \times \text{size}$。中性化之后，IC 会回到接近真值——你会亲眼看到“脏”信号如何稀释了 IC。

## 任务

打开 [`lab.py`](lab.py)，实现每个 `raise NotImplementedError` 的函数。docstring 就是规格说明。

```bash
make test LAB=04
make test LAB=04 K=neutralize
```

## 思考题

1. `test_ic_no_lookahead_on_noise` 拿“当天收益”当信号。如果你的 IC 忘了错开一期，会得到多少？
   在真实研究里，这类 bug 通常不会这么明显——想想有哪些更隐蔽的方式会让 t 日信号用到 t+1 的信息
   （财报发布日 vs 报告期、复权因子、指数成分股的幸存者偏差……）。
2. 中性化之后 IC 升高了，多空组合的收益一定会升高吗？波动呢？（提示：size 本身在这份数据里有没有收益？）
3. 分 5 组和分 10 组，多空收益和它的 t 值分别怎么变？为什么极端分组的噪声更大？
4. 用 `capm` 对 `factor_panel` 里每只股票回归，把估出来的 beta 和真实的 `panel.beta` 画散点图。
   估计误差有多大？如果用这些估计的 beta 去做“beta 中性”，会发生什么？

## 拓展：用真实数据

```python
from quantlab import realdata

hs300 = realdata.index_daily("sh000300")["close"].pct_change()
maotai = realdata.a_share_daily("600519")["close"].pct_change()
```

1. 估计贵州茅台相对沪深300 的 CAPM alpha/beta，并画 250 日滚动 beta。
2. 选 30–50 只股票，用“过去 20 日收益”做信号（短期反转），算 IC 序列和分组收益。
   A 股的短期反转效应长期显著——看看它最近还在不在。
