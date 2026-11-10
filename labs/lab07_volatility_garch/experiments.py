# %% [markdown]
# # Lab 07 实验：波动率与仓位
#
# 每个实验：先在 `预测：` 后面写下答案，再运行，然后对照。猜错的地方回讲义对应小节。

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from quantlab import data, perf, vol

DAYS = 252
g = data.garch_returns(5000, seed=0)  # 已知参数：omega=2e-6, alpha=0.08, beta=0.90
r = g["ret"]

# %% [markdown]
# ## 实验 1：收益难预测，波动率好预测（讲义 §1）
#
# 在 GARCH 数据上：
# - 用昨天的收益预测今天的收益，R² 大概多少？
# - 用 EWMA 方差预测今天的**平方**收益，R² 大概多少？
#
# 预测：

# %%
x_ret = r.shift(1)
x_var = vol.ewma_variance(r)  # 第 t 个值只用到 t-1
df = pd.DataFrame({"r": r, "x_ret": x_ret, "r2": r**2, "x_var": x_var}).dropna()
print(f"收益对昨天收益的 R²：     {df['r'].corr(df['x_ret']) ** 2:.4f}")
print(f"平方收益对 EWMA 方差的 R²：{df['r2'].corr(df['x_var']) ** 2:.4f}")

# %% [markdown]
# 追问：平方收益是真实方差的一个**非常**嘈杂的代理（单个正态变量的平方，标准差是均值的 √2 倍）。
# 如果把它换成真实方差 `g["sigma"]**2`，R² 会变成多少？

# %% [markdown]
# ## 实验 2：哪种波动率预测更准？（讲义 §2）
#
# 21 日滚动标准差、EWMA（λ=0.94）、用 MLE 拟合的 GARCH(1,1)。和真实的条件波动率相比，
# 谁的误差最小？差距大吗？
#
# 预测：

# %%
fit = vol.fit_garch(r)
eps = (r - fit["mu"]).to_numpy()
garch_sigma = pd.Series(
    np.sqrt(vol.garch_variance((fit["omega"], fit["alpha"], fit["beta"]), eps)), index=r.index
)
candidates = {
    "rolling 21d": r.rolling(21).std().shift(1),
    "EWMA 0.94": np.sqrt(vol.ewma_variance(r)),
    "GARCH(1,1)": garch_sigma,
}
truth = g["sigma"]
for name, est in candidates.items():
    err = (est - truth).iloc[100:]
    print(f"{name:12s} RMSE {np.sqrt((err**2).mean()) * np.sqrt(DAYS):.4f}（年化）")
print(f"拟合参数：alpha={fit['alpha']:.3f}, beta={fit['beta']:.3f}（真值 0.08, 0.90）")
print(f"波动冲击的半衰期：{vol.half_life(fit['alpha'], fit['beta']):.0f} 天")

# %% [markdown]
# 注意：GARCH 是在同一段数据上拟合、又在同一段数据上评估的（样本内），
# 而且数据本来就是 GARCH 生成的。
# 追问：把前一半用来拟合、后一半用来评估，结论会变吗？
# 真实市场的数据并不是 GARCH 生成的，你会预期差距更大还是更小？

# %% [markdown]
# ## 实验 3：波动率目标什么时候有用？（讲义 §3）
#
# 目标年化波动 10%，杠杆上限 2 倍。在 GARCH 数据（有波动聚集）和 iid 正态数据（没有）上：
# - 波动率目标后的 Sharpe 比买入持有高还是低？
# - 滚动波动率的稳定性（vol of vol）变化多少？
#
# 预测：


# %%
def vol_of_vol(x: pd.Series) -> float:
    return float((x.rolling(63).std() * np.sqrt(DAYS)).std())


iid = pd.Series(np.random.default_rng(1).normal(0.0003, r.std(), len(r)), index=r.index, name="ret")
for name, x in [("GARCH", r), ("iid", iid)]:
    vt = vol.vol_target_returns(x, target_vol=0.10, max_leverage=2.0)
    print(
        f"{name:6s} Sharpe 买入持有 {perf.sharpe(x):.2f} → 波动率目标 {perf.sharpe(vt):.2f}   "
        f"vol-of-vol {vol_of_vol(x):.3f} → {vol_of_vol(vt):.3f}"
    )

# %% [markdown]
# ## 实验 4：杠杆与增长率（讲义 §4）
#
# 一个年化超额收益 6%、波动 20% 的资产（Sharpe 0.3）。Kelly 杠杆 f* = 0.06/0.04 = 1.5 倍。
# - 用 3 倍杠杆（2 倍 Kelly），长期复利增长率是多少？
# - 用 4 倍呢？
#
# 预测：

# %%
mu_ex, sigma = 0.06, 0.20
fs = np.linspace(0, 4, 81)
growth = fs * mu_ex - 0.5 * fs**2 * sigma**2
for f in [0.75, 1.5, 3.0, 4.0]:
    print(f"杠杆 {f:4.2f}：超额对数增长率 {f * mu_ex - 0.5 * f**2 * sigma**2:+.3f}")
plt.plot(fs, growth)
plt.axhline(0, color="k", lw=0.5)
plt.axvline(mu_ex / sigma**2, ls="--", label="Kelly")
plt.xlabel("leverage")
plt.ylabel("excess log growth per year")
plt.legend()
plt.show()

# %% [markdown]
# ## 实验 5：用估计出来的 mu 下注（讲义 §4）
#
# 同一个资产，但你不知道真实参数，只能用过去 5 年的数据估计 mu 和 sigma，再按估计值算 Kelly 杠杆
# （上限 5 倍）。在 2000 个平行宇宙里：
# - 完全 Kelly 的平均真实增长率是正是负？
# - 半 Kelly 呢？
#
# 预测：

# %%
rng = np.random.default_rng(2)
res = {1.0: [], 0.5: []}
for _ in range(2000):
    x = rng.normal(mu_ex / DAYS, sigma / np.sqrt(DAYS), 5 * DAYS)
    f_hat = (x.mean() * DAYS) / (x.var(ddof=1) * DAYS)
    for frac in res:
        f = float(np.clip(frac * f_hat, 0, 5))
        res[frac].append(f * mu_ex - 0.5 * f**2 * sigma**2)
for frac, gr in res.items():
    gr = np.array(gr)
    print(
        f"{frac:.1f} Kelly：平均真实超额增长 {gr.mean():+.4f}，"
        f"亏损（增长<0）的宇宙 {np.mean(gr < 0):.0%}"
    )
print(f"知道真实参数时的最优：{mu_ex**2 / (2 * sigma**2):+.4f}")

# %% [markdown]
# ## 实验 6：波动率目标挡不住的那一天（讲义 §5）
#
# 一段非常平静的行情之后，突然一天跌 20%。波动率目标策略（上限 3 倍）在那一天亏多少？
#
# 预测：

# %%
calm = pd.Series(
    np.random.default_rng(3).normal(0.0004, 0.004, 500), index=data.business_index(500)
)
crash = calm.copy()
crash.iloc[400] = -0.20
w = vol.vol_target_weights(np.sqrt(vol.ewma_variance(crash)), target_vol=0.10, max_leverage=3.0)
print(
    f"崩盘前一天的预测年化波动：{np.sqrt(vol.ewma_variance(crash)).iloc[400] * np.sqrt(DAYS):.1%}"
)
print(f"崩盘当天的仓位：{w.iloc[400]:.2f} 倍")
print(f"崩盘当天组合亏损：{w.iloc[400] * crash.iloc[400]:.1%}（买入持有：{crash.iloc[400]:.0%}）")
print(f"崩盘后第二天的仓位：{w.iloc[401]:.2f} 倍")

# %% [markdown]
# ## 做完之后
#
# 去做 [`audit.py`](audit.py)。实验 4、5、6 里的问题会以更隐蔽的形式出现在那里。
