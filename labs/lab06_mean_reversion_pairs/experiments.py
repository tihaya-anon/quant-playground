# %% [markdown]
# # Lab 06 实验：均值回归与配对交易
#
# 每个实验：先在 `预测：` 后面写下答案和一句理由，再运行，再对照。

# %%
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

from quantlab import data, pairs, perf

warnings.simplefilter("ignore")
DAYS = 252
rng = np.random.default_rng(0)

# %% [markdown]
# ## 实验 1：伪回归（讲义 §2）
#
# 两只**完全独立**的随机游走股票，各 3 年日数据。把一只的对数价格对另一只回归。重复 1000 次。
#
# - 斜率的 |t| > 2 的比例是多少？（如果回归是“正常”的，应该约 5%）
# - R² 的中位数大概多少？
# - 改成对**日收益**回归呢？
#
# 预测：

# %%
t_level, r2_level, t_ret = [], [], []
for _ in range(1000):
    y = np.cumsum(rng.normal(0, 0.015, 756))
    x = np.cumsum(rng.normal(0, 0.015, 756))
    for yy, xx, store_t, store_r2 in [
        (y, x, t_level, r2_level),
        (np.diff(y), np.diff(x), t_ret, None),
    ]:
        X = np.column_stack([np.ones(len(xx)), xx])
        beta, res, *_ = np.linalg.lstsq(X, yy, rcond=None)
        resid = yy - X @ beta
        s2 = resid @ resid / (len(yy) - 2)
        se = np.sqrt(s2 * np.linalg.inv(X.T @ X)[1, 1])
        store_t.append(beta[1] / se)
        if store_r2 is not None:
            store_r2.append(1 - resid @ resid / ((yy - yy.mean()) ** 2).sum())
share = np.mean(np.abs(t_level) > 2)
print(f"价格对价格：|t| > 2 的比例 {share:.0%}，R² 中位数 {np.median(r2_level):.2f}")
print(f"收益对收益：|t| > 2 的比例 {np.mean(np.abs(t_ret) > 2):.0%}")

# %% [markdown]
# ## 实验 2：在 20 只股票里找配对（讲义 §3）
#
# 20 只彼此独立的随机游走股票，两两做 Engle-Granger 协整检验（共 190 对）。
#
# - 有多少对 p < 0.05？
# - 最小的 p 值大概多少？Bonferroni 门槛 0.05 / 190 是多少？
#
# 预测：

# %%
prices = pairs.simulate_universe(seed=1)
log_p = np.log(prices.iloc[:750])
pvals = []
cols = list(log_p.columns)
for i in range(len(cols)):
    for j in range(i + 1, len(cols)):
        pvals.append(coint(log_p[cols[i]], log_p[cols[j]], maxlag=1, autolag=None)[1])
pvals = np.array(pvals)
print(f"p < 0.05：{(pvals < 0.05).sum()} 对（期望约 {0.05 * len(pvals):.0f}）")
print(f"最小 p = {pvals.min():.4f}，Bonferroni 门槛 = {0.05 / len(pvals):.5f}")

# %% [markdown]
# ## 实验 3：半衰期估得准吗（讲义 §1）
#
# 一个真实半衰期为 15 天的价差。分别用 1 年和 5 年的数据估计半衰期，重复 500 次。
#
# - 1 年数据：估计值的中位数和 90% 区间大概多少？是偏高还是偏低？
# - 5 年数据呢？
#
# 预测：

# %%
for years in [1, 5]:
    est = []
    for s in range(500):
        spread = data.cointegrated_pair(n_days=years * DAYS, half_life=15, seed=1000 + s)["spread"]
        _, phi, _ = pairs.fit_ar1(spread)
        est.append(pairs.half_life(phi))
    est = np.array(est)
    lo, med, hi = np.quantile(est[np.isfinite(est)], [0.05, 0.5, 0.95])
    print(
        f"{years} 年：中位数 {med:5.1f} 天，90% 区间 [{lo:5.1f}, {hi:6.1f}]，"
        f"估不出（phi ≥ 1）{np.mean(~np.isfinite(est)):.0%}"
    )

# %% [markdown]
# ## 实验 4：关系变了，以及自适应的代价（讲义 §5）
#
# 一对股票，价差半衰期 10 天。两种情形，各模拟 20 次取平均：
#
# - **没有断裂**：6 年里对冲比例一直是 1.0；
# - **断裂**：第 3 年末发生并购，之后对冲比例变成 1.5。
#
# 两种交易方式都在后 3 年交易：(a) 用前 3 年估计的**固定**对冲比例；
# (b) 用 Kalman 滤波**在线更新**对冲比例。
#
# - 断裂会让固定对冲的 Sharpe 下降多少？
# - Kalman 在两种情形下分别比固定对冲好还是差？
#
# 预测：

# %%


def simulate_break(seed: int, jump: float) -> tuple[pd.Series, pd.Series, np.ndarray]:
    n = 6 * DAYS
    g = np.random.default_rng(seed)
    log_x = np.log(50) + np.cumsum(g.normal(0.0002, 0.015, n))
    phi = np.exp(-np.log(2) / 10)
    spread = np.zeros(n)
    for t in range(1, n):
        spread[t] = phi * spread[t - 1] + g.normal(0, 0.01)
    beta = np.where(np.arange(n) < n // 2, 1.0, 1.0 + jump)
    log_y = 0.3 + beta * (log_x - log_x[n // 2]) + log_x[n // 2] + spread
    idx = data.business_index(n)
    return pd.Series(log_y, index=idx), pd.Series(log_x, index=idx), beta


def static_vs_kalman(ly: pd.Series, lx: pd.Series) -> tuple[float, float, pd.DataFrame]:
    half = len(ly) // 2
    py, px = np.exp(ly.iloc[half:]), np.exp(lx.iloc[half:])
    eg = pairs.engle_granger(ly.iloc[:half], lx.iloc[:half])
    s_static = ly - eg["hedge_ratio"] * lx - eg["intercept"]
    pos = pairs.pair_positions(pairs.rolling_zscore(s_static, 60).iloc[half:])
    static = pairs.pair_returns(py, px, eg["hedge_ratio"], pos)

    kf = pairs.kalman_hedge_ratio(ly, lx)
    s_kf = ly - kf["beta"].shift(1) * lx - kf["alpha"].shift(1)
    pos_k = pairs.pair_positions(pairs.rolling_zscore(s_kf, 60).iloc[half:])
    dynamic = pairs.pair_returns(py, px, kf["beta"].iloc[half:].to_numpy(), pos_k)
    return perf.sharpe(static), perf.sharpe(dynamic), kf


for label, jump in [("no break", 0.0), ("break", 0.5)]:
    res = np.array([static_vs_kalman(*simulate_break(s, jump)[:2])[:2] for s in range(20)])
    print(
        f"{label:9s} 固定对冲 Sharpe {res[:, 0].mean():.2f}   Kalman Sharpe {res[:, 1].mean():.2f}"
    )

ly, lx, beta_true = simulate_break(0, 0.5)
kf = static_vs_kalman(ly, lx)[2]
plt.plot(kf["beta"], label="Kalman beta")
plt.plot(ly.index, beta_true, "k--", label="true beta")
plt.title("Hedge ratio after a structural break")
plt.legend()
plt.show()

# %% [markdown]
# 追问：没有断裂时，Kalman 为什么反而更差？看看它的截距 `kf["alpha"]` 在做什么——
# 一个跟得很快的截距，和我们要交易的“价差偏离”是什么关系？把 `delta` 调小（适应得更慢）会怎样？
# 这和“用多长的窗口估计参数”是同一个权衡吗？

# %% [markdown]
# ## 实验 5：配对交易的收益长什么样（讲义 §4）
#
# 一对健康的协整股票（半衰期 10 天），z-score ±2 进场、±0.5 出场，交易 10 年。
# 按“每一笔交易”统计盈亏。
#
# - 胜率大概多少？
# - 平均盈利和平均亏损哪个大？每笔交易盈亏的偏度是正还是负？
#
# 预测：

# %%
pair = data.cointegrated_pair(n_days=10 * DAYS, half_life=10, seed=5)
ly, lx = np.log(pair["y"]), np.log(pair["x"])
eg = pairs.engle_granger(ly, lx)  # 这里只看收益形态，允许全样本估计
z = pairs.rolling_zscore(eg["spread"], 60)
pos = pairs.pair_positions(z)
ret = pairs.pair_returns(pair["y"], pair["x"], eg["hedge_ratio"], pos)
trade_id = (pos != pos.shift()).cumsum().loc[ret.index]
held = pos.shift().loc[ret.index] != 0
per_trade = ret[held].groupby(trade_id[held]).sum()
wins, losses = per_trade[per_trade > 0], per_trade[per_trade <= 0]
print(f"{len(per_trade)} 笔交易，胜率 {len(wins) / len(per_trade):.0%}")
print(f"平均盈利 {wins.mean():.2%}，平均亏损 {losses.mean():.2%}，偏度 {per_trade.skew():.2f}")

# %% [markdown]
# 追问：如果价差有很小的概率**永久性地**不再回归（比如其中一家公司出了问题），
# 这个收益分布的左尾会变成什么样？为什么配对交易常被称为“在压路机前捡硬币”？

# %% [markdown]
# ## 做完之后
#
# 去做 [`audit.py`](audit.py)。实验 1 和实验 2 的问题会出现在那里。
