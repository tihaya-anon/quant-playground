# %% [markdown]
# # Lab 05 实验：组合构建与估计误差
#
# 每个实验：先在 `预测：` 后面写下答案和一句理由，再运行，再对照。

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from quantlab import perf, portfolio

DAYS = 252

# %% [markdown]
# ## 实验 1：用估计的 mu 算切点组合（讲义 §2）
#
# 10 个资产，真实 Sharpe 全部是 0.4。用 5 年日数据估计 mu 和协方差，算切点组合
# （按总敞口归一化，Σ|w| = 1）。
#
# - 真实的切点组合里，最大的单个权重绝对值大约多少？（提示：真实 Sharpe 相同时，权重和什么有关？）
# - 用估计值算出来的呢？有多少个资产是做空的？
#
# 预测：

# %%
r = portfolio.simulate_assets(n_assets=10, n_days=5 * DAYS, seed=0)
true_vol = r.std().to_numpy() * np.sqrt(DAYS)  # 用样本波动近似真实波动（波动率估得很准）
true_mu = 0.4 * true_vol
cov = r.cov().to_numpy() * DAYS
mu_hat = r.mean().to_numpy() * DAYS


def unit_gross(x):
    return x / np.abs(x).sum()


w_true = unit_gross(np.linalg.solve(cov, true_mu))
w_hat = unit_gross(np.linalg.solve(cov, mu_hat))
print("估计的年化 mu：", np.round(mu_hat, 2))
print("真实 mu：      ", np.round(true_mu, 2))
print(f"真实切点：最大 |w| {np.abs(w_true).max():.2f}，做空 {np.sum(w_true < 0)} 个")
print(f"估计切点：最大 |w| {np.abs(w_hat).max():.2f}，做空 {np.sum(w_hat < 0)} 个")

# %% [markdown]
# ## 实验 2：事前风险 vs 事后风险（讲义 §2）
#
# 50 个资产，用 T 天数据估计协方差，构建最小方差组合，然后看它在接下来一年的实际波动率。
# T = 60、250、1000，分别用样本协方差和 Ledoit-Wolf 收缩。
#
# - 样本协方差在 T = 60 时，事后波动率是事前的几倍？
# - 收缩能挽回多少？
#
# 预测：

# %%
r = portfolio.simulate_assets(n_assets=50, n_days=1000 + DAYS, seed=1)
test = r.iloc[-DAYS:]
for T in [60, 250, 1000]:
    train = r.iloc[-DAYS - T : -DAYS]
    for est in ["sample", "ledoit_wolf"]:
        pred, real = portfolio.oos_min_variance(train, test, estimator=est)
        print(f"T={T:5d} {est:12s} 事前 {pred:.3f}  事后 {real:.3f}  比值 {real / pred:.2f}")

# %% [markdown]
# ## 实验 3：样本外赛马（讲义 §3）
#
# 30 个资产，真实 Sharpe 全部相同，20 年，每月再平衡，用过去一年的数据估计。
# 五个选手：等权（1/N）、风险平价、最小方差（样本协方差）、最小方差（Ledoit-Wolf）、
# 切点组合（样本 mu 和协方差，总敞口归一化）。
#
# 排个序：样本外 Sharpe 从高到低。
#
# 预测：

# %%
r = portfolio.simulate_assets(n_assets=30, n_days=20 * DAYS, seed=2)
month_ends = r.groupby([r.index.year, r.index.month]).tail(1).index
dates = [d for d in month_ends if r.index.get_loc(d) >= DAYS]


def run(make_weights):
    rows = []
    for d in dates:
        window = r.loc[:d].iloc[-DAYS:]
        rows.append(make_weights(window))
    w = pd.DataFrame(rows, index=pd.DatetimeIndex(dates), columns=r.columns)
    held = w.reindex(r.index).ffill().shift(1).dropna()
    return (held * r.loc[held.index]).sum(axis=1)


n = r.shape[1]
contestants = {
    "1/N": lambda win: np.full(n, 1 / n),
    "risk parity": lambda win: portfolio.risk_parity_weights(win.cov().to_numpy()),
    "min-var sample": lambda win: portfolio.min_variance_weights(win.cov().to_numpy()),
    "min-var LW": lambda win: portfolio.min_variance_weights(portfolio.ledoit_wolf(win)[0]),
    "tangency": lambda win: unit_gross(
        np.linalg.solve(win.cov().to_numpy(), win.mean().to_numpy())
    ),
}
results = {name: run(f) for name, f in contestants.items()}
for name, ret in sorted(results.items(), key=lambda kv: -perf.sharpe(kv[1])):
    print(f"{name:15s} Sharpe {perf.sharpe(ret):5.2f}   年化波动 {perf.annualized_vol(ret):.3f}")

# %% [markdown]
# 追问：这里所有资产真实 Sharpe 相同，理论上的切点组合是 $w \propto \Sigma^{-1}\sigma$
# （$\sigma$ 是波动率向量）。它更接近 1/N、风险平价还是最小方差？
# 风险平价完全不用估计 mu，为什么能和 1/N 打平？切点组合输在哪里——协方差还是 mu？

# %% [markdown]
# ## 实验 4：危机中的分散化（讲义 §4）
#
# 10 个资产，每个年化波动 20%。平时两两相关 0.2，危机时升到 0.8。
#
# - 等权组合平时的年化波动率是多少？危机时呢？
# - 分散化“消失”了多少？
#
# 预测：

# %%
for rho in [0.2, 0.8]:
    r = portfolio.simulate_assets(
        n_assets=10, n_days=20 * DAYS, vol_range=(0.2, 0.2), corr=rho, seed=3
    )
    ew = r.mean(axis=1)
    theory = 0.2 * np.sqrt(rho + (1 - rho) / 10)
    print(f"相关 {rho}：等权波动 {perf.annualized_vol(ew):.3f}（公式 {theory:.3f}）")

rho = np.linspace(0, 1, 50)
for n_assets in [5, 10, 100]:
    plt.plot(rho, 0.2 * np.sqrt(rho + (1 - rho) / n_assets), label=f"N={n_assets}")
plt.xlabel("pairwise correlation")
plt.ylabel("equal-weight portfolio vol")
plt.title("Diversification disappears as correlation rises")
plt.legend()
plt.show()

# %% [markdown]
# ## 实验 5：约束就是收缩（讲义 §3）
#
# 和实验 3 相同的数据，比较无约束的均值方差和“只做多、单个权重不超过 10%”的均值方差
# （都用样本估计的 mu 和协方差，风险厌恶系数 5）。
#
# - 哪个样本外 Sharpe 更高？
# - 两者的平均月换手（Σ|Δw|）差多少？
#
# 预测：

# %%
r = portfolio.simulate_assets(n_assets=30, n_days=20 * DAYS, seed=2)


def mv(win, long_only, cap):
    mu = win.mean().to_numpy() * DAYS
    cov = win.cov().to_numpy() * DAYS
    return portfolio.mean_variance_cvxpy(mu, cov, 5.0, long_only=long_only, max_weight=cap)


for label, kw in [
    ("unconstrained", dict(long_only=False, cap=None)),
    ("long-only, cap 10%", dict(long_only=True, cap=0.10)),
]:
    rows = [mv(r.loc[:d].iloc[-DAYS:], **kw) for d in dates]
    w = pd.DataFrame(rows, index=pd.DatetimeIndex(dates), columns=r.columns)
    held = w.reindex(r.index).ffill().shift(1).dropna()
    ret = (held * r.loc[held.index]).sum(axis=1)
    turnover = w.diff().abs().sum(axis=1).mean()
    print(f"{label:20s} Sharpe {perf.sharpe(ret):5.2f}  月均换手 {turnover:.2f}")

# %% [markdown]
# ## 做完之后
#
# 去做 [`audit.py`](audit.py)。实验 1 和实验 2 的问题会出现在那里。
