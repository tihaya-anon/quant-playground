# %% [markdown]
# # Lab 04 实验：收益从哪来
#
# 每个实验：先在 `预测：` 后面写下答案和一句理由，再运行，再对照。
# 猜错的地方回到讲义对应小节。

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from quantlab import factors, perf

DAYS = 252


def fast_ic(signal: np.ndarray, returns: np.ndarray) -> np.ndarray:
    """每天的横截面 Spearman IC：t 日信号 vs t+1 日收益（numpy 版，比 pandas 快很多）。"""
    s = rankdata(signal[:-1], axis=1)
    r = rankdata(returns[1:], axis=1)
    s = s - s.mean(axis=1, keepdims=True)
    r = r - r.mean(axis=1, keepdims=True)
    return (s * r).sum(axis=1) / np.sqrt((s**2).sum(axis=1) * (r**2).sum(axis=1))


# %% [markdown]
# ## 实验 1：广度（讲义 §2）
#
# 同一个信号，每天的 IC 都约为 0.02（非常弱）。分别在 300 只和 30 只股票上做五分组多空。
#
# - 两者的多空年化 Sharpe 大概各是多少？
# - 比值接近多少？（提示：基本定律 IR ≈ IC × √breadth）
#
# 预测：

# %%
for n in [300, 30]:
    u = factors.simulate_universe(n_assets=n, ic=0.02, signal_persistence=0.0, seed=1)
    ls = factors.long_short_returns(u.signal, u.returns)
    ic = fast_ic(u.signal.to_numpy(), u.returns.to_numpy())
    print(f"{n:4d} 只股票：IC 均值 {ic.mean():.3f}，多空 Sharpe {perf.sharpe(ls):.2f}")

# %% [markdown]
# ## 实验 2：你赚的到底是什么（讲义 §3）
#
# 信号本身没有 alpha，但混入了大量小盘暴露，而小盘因子在这段样本里有 12% 的年化溢价。
#
# - 多空组合的 Sharpe 会是多少？看起来像不像一个好因子？
# - 只对市场做回归，alpha 的 t 值是多少？加上小盘因子之后呢？
# - 把信号对小盘暴露做中性化之后，多空 Sharpe 剩多少？
#
# 预测：

# %%
u = factors.simulate_universe(size_premium=0.12, size_tilt=3.0, seed=2)
ls = factors.long_short_returns(u.signal, u.returns)
df = pd.concat([ls, u.market, u.size_factor], axis=1, join="inner").dropna()
t_mkt = factors.ols(df.iloc[:, 0], df[["market"]])["tstats"][0]
t_both = factors.ols(df.iloc[:, 0], df[["market", "size"]])["tstats"][0]
print(f"多空 Sharpe {perf.sharpe(ls):.2f}")
print(f"alpha t：只控制市场 {t_mkt:.1f}，控制市场+小盘 {t_both:.1f}")

sector = pd.Series(0, index=u.smallness.index)  # 这里只有一个行业
clean = factors.neutralize(u.signal, u.smallness, sector)
print(f"中性化之后多空 Sharpe {perf.sharpe(factors.long_short_returns(clean, u.returns)):.2f}")

# %% [markdown]
# ## 实验 3：换手吃掉 alpha（讲义 §4）
#
# 三个 IC 相同（0.02）的信号，区别只在于信号每天变化有多快（自相关系数 0、0.95、0.99）。
# 单边费率 10bp。
#
# - 扣成本前，三者的多空 Sharpe 差不多吗？
# - 扣成本后，哪些还活着？
#
# 预测：

# %%
for rho in [0.0, 0.95, 0.99]:
    u = factors.simulate_universe(ic=0.02, signal_persistence=rho, seed=3)
    held = np.ceil(u.signal.rank(axis=1, pct=True) * 5).clip(1, 5).shift(1)
    long_ = (held == 5).astype(float)
    short = (held == 1).astype(float)
    w = long_.div(long_.sum(axis=1), axis=0) - short.div(short.sum(axis=1), axis=0)
    w = w.fillna(0.0)
    gross = (w * u.returns).sum(axis=1).iloc[2:]
    turnover = w.diff().abs().sum(axis=1).iloc[2:]
    net = gross - turnover * 10 / 1e4
    print(
        f"rho={rho:4.2f}  日均换手 {turnover.mean():5.2f}  "
        f"毛 Sharpe {perf.sharpe(gross):5.2f}  净 Sharpe {perf.sharpe(net):5.2f}"
    )

# %% [markdown]
# ## 实验 4：发表之后（讲义 §5）
#
# 一个因子在前 10 年 IC 为 0.04，被发表、被广泛使用之后，后 10 年 IC 降到 0.015。
#
# - 用全部 20 年数据回测，多空 Sharpe 是多少？
# - 只看后 10 年呢？你在第 10 年读到论文、开始交易，得到的是哪一个？
#
# 预测：

# %%
before = factors.simulate_universe(ic=0.04, seed=4)
after = factors.simulate_universe(ic=0.015, seed=5)
after_idx = pd.bdate_range(before.returns.index[-1] + pd.offsets.BDay(), periods=len(after.returns))
sig = pd.concat([before.signal, after.signal.set_axis(after_idx)])
ret = pd.concat([before.returns, after.returns.set_axis(after_idx)])
ls = factors.long_short_returns(sig, ret)
print(f"全样本 Sharpe {perf.sharpe(ls):.2f}")
print(f"前 10 年 {perf.sharpe(ls.loc[: before.returns.index[-1]]):.2f}")
print(f"后 10 年 {perf.sharpe(ls.loc[after_idx[0] :]):.2f}")

plt.plot((1 + ls).cumprod())
plt.axvline(after_idx[0], color="k", ls="--", label="publication")
plt.yscale("log")
plt.title("Long-short wealth: before and after publication")
plt.legend()
plt.show()

# %% [markdown]
# ## 实验 5：因子动物园（讲义 §6）
#
# 100 个**完全随机**的信号（缓慢变化，像真实的基本面因子），在没有任何 alpha 的股票池上
# 算 IC 的 t 值。
#
# - 有多少个 |t| > 2？有多少个 |t| > 3？
# - t 值最大的那个，五分组多空 Sharpe 是多少？它看起来像不像一个可以写成论文的因子？
#
# 预测：

# %%
u = factors.simulate_universe(n_assets=100, n_days=1260, seed=6)
rets = u.returns.to_numpy()
g = np.random.default_rng(7)
best_t, best_sig = 0.0, None
t_stats = []
for _ in range(100):
    z = np.empty_like(rets)
    z[0] = g.standard_normal(rets.shape[1])
    for t in range(1, len(z)):
        z[t] = 0.95 * z[t - 1] + np.sqrt(1 - 0.95**2) * g.standard_normal(rets.shape[1])
    ic = fast_ic(z, rets)
    t_stat = ic.mean() / ic.std(ddof=1) * np.sqrt(len(ic))
    t_stats.append(t_stat)
    if abs(t_stat) > abs(best_t):
        best_t, best_sig = t_stat, np.sign(t_stat) * z  # 负 t 的信号反过来用就是正 t
t_stats = np.abs(np.array(t_stats))
print(f"|t| > 2：{(t_stats > 2).sum()} 个；|t| > 3：{(t_stats > 3).sum()} 个")
best_df = pd.DataFrame(best_sig, index=u.returns.index, columns=u.returns.columns)
best_ls = factors.long_short_returns(best_df, u.returns)
print(f"|t| 最大的那个（按符号翻转后）：|t| = {abs(best_t):.2f}")
print(f"它的五分组多空 Sharpe {perf.sharpe(best_ls):.2f}")

# %% [markdown]
# 追问：如果你是期刊审稿人，看到一篇论文报告 t = 2.5 的新因子，你还需要知道什么？

# %% [markdown]
# ## 做完之后
#
# 去做 [`audit.py`](audit.py)。实验 2 和实验 3 的问题会以更隐蔽的形式出现在那里。
