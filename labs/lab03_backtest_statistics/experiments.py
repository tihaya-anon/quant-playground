# %% [markdown]
# # Lab 03 实验：信号还是噪声
#
# 每个实验的流程：
#
# 1. 读问题，**在 `预测：` 后面写下你的答案**（一个数字或一个范围，再加一句理由）。
# 2. 运行下面的代码单元格。
# 3. 对照结果。猜错了的话，回到讲义对应的小节，想清楚是哪一步直觉出了问题。
#
# 在 VS Code 里点单元格上方的 “Run Cell”，或者把光标放进去按 `Shift+Enter`。

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from quantlab import data, perf, stats

rng = np.random.default_rng(0)
DAYS = 252


def strategy_with_sharpe(sr_annual: float, years: float, seed: int) -> pd.Series:
    """真实年化 Sharpe 恰为 sr_annual 的 iid 正态日收益（年化波动 10%）。"""
    n = int(years * DAYS)
    vol = 0.10 / np.sqrt(DAYS)
    mu = sr_annual * vol / np.sqrt(DAYS)
    r = np.random.default_rng(seed).normal(mu, vol, n)
    return pd.Series(r, index=data.business_index(n))


# %% [markdown]
# ## 实验 1：样本 Sharpe 的离散程度（讲义 §1）
#
# 一个**真实**年化 Sharpe 为 0.5 的策略，跑 10 年回测。重复 2000 次（2000 个平行宇宙）。
#
# - 样本 Sharpe 的 95% 区间大约是多少？
# - 有百分之多少的宇宙里，样本 Sharpe 是负的？
#
# 预测：

# %%
srs = np.array([perf.sharpe(strategy_with_sharpe(0.5, 10, s)) for s in range(2000)])
lo, hi = np.quantile(srs, [0.025, 0.975])
print(f"样本 Sharpe 均值 {srs.mean():.2f}，95% 区间 [{lo:.2f}, {hi:.2f}]")
print(f"样本 Sharpe < 0 的比例：{(srs < 0).mean():.1%}")
print(f"公式 1/sqrt(T) 给出的标准误：{1 / np.sqrt(10):.3f}，模拟得到：{srs.std():.3f}")

plt.hist(srs, bins=60)
plt.axvline(0.5, color="k", ls="--", label="true Sharpe")
plt.title("Sample Sharpe: true Sharpe 0.5, 10-year backtests")
plt.xlabel("annualized sample Sharpe")
plt.legend()
plt.show()

# %% [markdown]
# 追问：把 10 年改成 3 年、30 年，区间怎么变？要多少年才能让 95% 区间不包含 0？

# %% [markdown]
# ## 实验 2：从 1000 个噪声策略里挑最好的（讲义 §3）
#
# 1000 个**真实 Sharpe 全为 0** 的策略，每个跑 10 年，挑出样本 Sharpe 最高的那个。
#
# - 它的年化 Sharpe 大概是多少？
# - 它的 PSR（只看它自己）会超过 0.95 吗？DSR（考虑 1000 次试验）呢？
#
# 预测：

# %%
n_trials = 1000
trials = [strategy_with_sharpe(0.0, 10, 10_000 + s) for s in range(n_trials)]
per_period = np.array([stats.per_period_sharpe(r) for r in trials])
best = int(np.argmax(per_period))

print(f"最好的那个：年化 Sharpe {per_period[best] * np.sqrt(DAYS):.2f}")
print(
    "理论期望最大值（年化）："
    f"{stats.expected_max_sharpe(n_trials, per_period.var()) * np.sqrt(DAYS):.2f}"
)
print(f"PSR = {stats.probabilistic_sharpe_ratio(trials[best]):.3f}")
print(f"DSR = {stats.deflated_sharpe_ratio(trials[best], per_period):.3f}")

# %% [markdown]
# 追问：只试 10 次、100 次呢？画一条“试验次数 → 最好的那个 Sharpe”的曲线，
# 它长得像 $\sqrt{2\ln N}$ 吗？

# %% [markdown]
# ## 实验 3：参数扫描的样本内与样本外（讲义 §4）
#
# 在**没有任何可预测性**的 GBM 价格上，对时间序列动量扫描 50 个回看期（5 到 250 天）。
#
# - 前 70% 上最好的那个回看期，样本内年化 Sharpe 大概多少？
# - 用它在后 30% 上交易，样本外 Sharpe 大概多少？
# - Walk-forward（每 3 年重新选一次，交易下半年）的 Sharpe 呢？
#
# 然后在**有真实趋势**的数据上做同样的事。哪个数字变化最大？
#
# 预测：

# %%
lookbacks = list(range(5, 255, 5))


def is_oos(prices: pd.Series) -> tuple[float, float, float]:
    split = int(len(prices) * 0.7)
    train = stats.sweep_returns(prices.iloc[:split], lookbacks)
    best_lb = train.mean().div(train.std()).idxmax()
    full = stats.momentum_strategy_returns(prices, best_lb)
    wf = stats.walk_forward(prices, lookbacks)
    return (
        perf.sharpe(full.iloc[: split - 1]),
        perf.sharpe(full.iloc[split - 1 :]),
        perf.sharpe(wf["ret"]),
    )


for name, make in [
    ("无信号 GBM", lambda s: data.gbm_prices(3780, mu=0.0, sigma=0.2, seed=s)),
    ("有趋势", lambda s: data.regime_switching_prices(3780, seed=s, trend_drift=0.4)),
]:
    res = np.array([is_oos(make(s)) for s in range(6)])
    print(
        f"{name:10s} 样本内 {res[:, 0].mean():5.2f}   "
        f"样本外 {res[:, 1].mean():5.2f}   walk-forward {res[:, 2].mean():5.2f}"
    )

# %% [markdown]
# ## 实验 4：卖保险的策略（讲义 §2）
#
# 下面这个策略每天赚一点点，平均每两年遇到一次大亏（像卖出深度虚值期权）。
#
# - 在一段**恰好没有发生大亏**的 3 年历史上，它的样本 Sharpe 会是多少？
# - 它的真实 Sharpe 是多少？
# - 在完整的 20 年数据上，iid 公式算出的标准误和 Mertens 公式（考虑偏度峰度）差多少？
#
# 预测：

# %%


def sell_insurance(years: float, seed: int) -> pd.Series:
    g = np.random.default_rng(seed)
    n = int(years * DAYS)
    r = g.normal(0.0004, 0.002, n)
    crash = g.random(n) < 1 / (2 * DAYS)
    r[crash] -= g.uniform(0.08, 0.15, crash.sum())
    return pd.Series(r, index=data.business_index(n))


long_run = sell_insurance(200, seed=1)
print(f"真实 Sharpe（200 年近似）：{perf.sharpe(long_run):.2f}")

calm = next(r for s in range(1000) if not ((r := sell_insurance(3, s)) < -0.05).any())
print(f"一段没有大亏的 3 年：样本 Sharpe {perf.sharpe(calm):.2f}")

r20 = sell_insurance(20, seed=7)
sr = stats.per_period_sharpe(r20)
skew, kurt = r20.skew(), r20.kurt() + 3
print(f"20 年：偏度 {skew:.1f}，峰度 {kurt:.0f}")
print(f"  iid 标准误     {stats.sharpe_se_iid(sr, len(r20)) * np.sqrt(DAYS):.3f}")
print(f"  Mertens 标准误 {stats.sharpe_se_mertens(sr, len(r20), skew, kurt) * np.sqrt(DAYS):.3f}")

# %% [markdown]
# ## 实验 5：自相关与块 bootstrap（讲义 §2）
#
# 收益有正自相关（AR(1)，系数 0.3）。普通 bootstrap 和块大小为 20 的块 bootstrap，
# 给出的 Sharpe 95% 置信区间，哪个更宽？宽多少？
#
# 预测：

# %%
n = 2520
eps = rng.normal(0, 0.01, n)
ar = np.empty(n)
ar[0] = eps[0]
for t in range(1, n):
    ar[t] = 0.3 * ar[t - 1] + eps[t]
ar_ret = pd.Series(ar + 0.0003, index=data.business_index(n))

iid_ci = stats.percentile_ci(stats.bootstrap_sharpe(ar_ret))
blk_ci = stats.percentile_ci(stats.block_bootstrap_sharpe(ar_ret, block_size=20))
print(
    f"普通 bootstrap 95% 区间：[{iid_ci[0]:.2f}, {iid_ci[1]:.2f}]，宽 {iid_ci[1] - iid_ci[0]:.2f}"
)
print(
    f"块 bootstrap   95% 区间：[{blk_ci[0]:.2f}, {blk_ci[1]:.2f}]，宽 {blk_ci[1] - blk_ci[0]:.2f}"
)

# %% [markdown]
# 追问：如果自相关系数是 -0.3（均值回复），哪个区间更宽？

# %% [markdown]
# ## 实验 6：重叠收益的 t 值（讲义 §2，审计题的预热）
#
# 1000 段**纯噪声**日收益（3 年），先聚合成每天滚动的 20 日收益，再按普通公式算 t 值。
# 有百分之多少的情况会出现 |t| > 1.96？（正确的 t 值应该是 5%。）
#
# 预测：

# %%
hits_overlap = hits_daily = 0
for s in range(1000):
    r = pd.Series(np.random.default_rng(50_000 + s).normal(0, 0.01, 756))
    agg = r.rolling(20).sum().dropna()
    t_overlap = agg.mean() / agg.std() * np.sqrt(len(agg))
    t_daily = r.mean() / r.std() * np.sqrt(len(r))
    hits_overlap += abs(t_overlap) > 1.96
    hits_daily += abs(t_daily) > 1.96
print(f"重叠 20 日收益：{hits_overlap / 1000:.1%} 显著")
print(f"日收益：        {hits_daily / 1000:.1%} 显著")

# %% [markdown]
# ## 做完之后
#
# 去做 [`audit.py`](audit.py)。实验 3 和实验 6 里出现的问题，会以更隐蔽的形式出现在那里。
