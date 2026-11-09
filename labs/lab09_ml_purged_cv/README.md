# Lab 09 · 金融机器学习：标签、泄漏与 Purged K-Fold

> 难度 ★★★★☆ · 预计 4–6 小时 · 前置：Lab 02/03，scikit-learn 基本用法

把 sklearn 的套路原样搬到行情数据上，几乎一定会得到漂亮但虚假的交叉验证分数。原因不是模型，
而是**数据不满足 i.i.d.**：标签在时间上重叠，特征在时间上高度自相关。这个 lab 跟随
López de Prado《Advances in Financial Machine Learning》（AFML）第 3、4、7 章，
把“怎么打标签”和“怎么做交叉验证”这两件事做对。

## 你要学会的

1. **无泄漏特征**：$t$ 时刻的特征只能用 $\le t$ 的信息。检验方法很朴素也很有效——
   把 $t$ 之后的价格改掉，$t$ 及以前的特征必须一个字节都不变（测试里就是这么做的）。
2. **固定时间窗标签**：$y_t = \mathbb{1}[\ln(C_{t+h}/C_t) > 0]$。样本 $t$ 的“信息区间”是 $[t, t+h]$。
   相邻样本共享 $h-1$ 天收益——它们几乎是同一个样本。
3. **三重障碍标签**：上障碍 $+k_{pt}\sigma_t$（止盈）、下障碍 $-k_{sl}\sigma_t$（止损）、垂直障碍 $t+H$（最长持有）。
   谁先触及，标签就是谁。它比固定窗口更贴近真实交易（有止盈止损），障碍按当时的波动率 $\sigma_t$ 缩放，
   使不同波动环境下的标签可比。
4. **样本唯一性**：bar $s$ 的并发数 $c_s$ = 覆盖它的标签数；样本 $i$ 的平均唯一性
   $$\bar u_i = \frac{1}{|T_i|}\sum_{s\in T_i} \frac{1}{c_s}$$
   重叠 $h$ 天的固定窗口标签 $\bar u \approx 1/(h+1)$——**你的“有效样本量”只有表面的 $1/(h+1)$**。
   可以把 $\bar u_i$ 当作 `sample_weight` 传给模型。
5. **为什么打乱 K 折会泄漏**：测试样本 $t$ 的邻居 $t\pm1$ 在训练集里，它们的标签与 $t$ 高度相关，
   特征（如 60 日动量）也几乎相同。一个 k-NN 只需要“找到时间上的邻居”就能猜对——这是记忆，不是预测。
6. **Purged K-Fold + Embargo**：
   - 测试集取连续时间块，不打乱；
   - **Purge**：删除信息区间 $[t_{0,i}, t_{1,i}]$ 与测试区间重叠的训练样本；
   - **Embargo**：再删除测试块之后的一小段样本，因为慢变特征在测试期结束后仍“记得”测试期。
   测试会在**纯随机游走**上演示：打乱 K 折给出 ~0.60 的准确率，Purged K-Fold 回到 ~0.5。
7. **样本外**：最终评估永远是一段从没参与过任何决策（包括调参、选特征）的时间。

## 任务

实现 [`lab.py`](lab.py) 里的函数和 `PurgedKFold` 类：

```bash
make test LAB=09
```

`PurgedKFold` 与 sklearn 接口兼容，可以直接传给 `cross_val_score(..., cv=PurgedKFold(...))`。

## 思考题

1. 测试 `test_shuffled_kfold_leaks_purged_does_not` 里，把 KNN 换成 `LogisticRegression`，打乱 K 折还会虚高吗？
   为什么高容量、局部的模型（KNN、深树、GBDT）泄漏得特别厉害？
2. 不打乱的普通 `KFold`（`shuffle=False`）有泄漏吗？泄漏只发生在哪里？它和 Purged 版的差距随 $h$ 怎样变化？
3. Embargo 该取多长？它应该和哪个特征的回看窗口挂钩？
4. 用 `average_uniqueness` 作为 `sample_weight` 训练 `RandomForestClassifier`，同时设置
   `max_samples` ≈ 平均唯一性，样本外表现有变化吗？（AFML 第 4 章的 sequential bootstrap 是更彻底的做法。）
5. 三重障碍标签里 $k_{pt} \ne k_{sl}$ 时，标签分布会不平衡。这与“策略的盈亏比”是什么关系？
6. 准确率 55% 能赚钱吗？还需要知道什么？（提示：命中时赚多少、错时亏多少、交易成本、换手。Lab 03 的统计检验也该用上。）

## 拓展：用真实数据

```python
from quantlab import realdata

close = realdata.index_daily("sh000905")["close"]  # 中证500
```

在中证500 上：用三重障碍（$k=2$, $H=20$）打标签，`make_features` 做特征，`lightgbm.LGBMClassifier`
做模型，分别用打乱 K 折、Purged K-Fold 评估，最后留出最近 2 年做样本外。三个数字差多少？
再把预测概率转成仓位（例如 $2p-1$），用 `quantlab.perf.summary` 看看样本外的策略表现。
