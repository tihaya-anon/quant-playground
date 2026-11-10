# %% [markdown]
# # Lab 00 实验：收益与复利
#
# 每个实验的流程：
#
# 1. 读问题，**在 `预测：` 后面写下你的答案**（一个数字或一个范围，再加一句理由）。
# 2. 运行下面的代码单元格。
# 3. 对照结果。猜错了的话，回到讲义对应的小节，想清楚是哪一步直觉出了问题。

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from quantlab import data, returns

DAYS = 252

# %% [markdown]
# ## 实验 1：平均收益为 0，钱去哪了？（讲义 §2）
#
# 一个资产每天交替 +2% / -2%，算术平均日收益恰好为 0。持有一年（252 天）后净值是多少？
#
# 再把波动换成 ±5%，又是多少？
#
# 预测：

# %%
for step in [0.02, 0.05]:
    r = pd.Series([step, -step] * (DAYS // 2), index=data.business_index(DAYS))
    print(
        f"±{step:.0%}：算术平均 {r.mean():.4f}，一年后净值 {returns.cumulative_return(r) + 1:.3f}，"
        f"理论 (1-{step}^2)^126 = {(1 - step**2) ** (DAYS // 2):.3f}"
    )

# %% [markdown]
# ## 实验 2：3 倍杠杆 ETF 在震荡市里（讲义 §2、§5）
#
# 标的是零漂移、年化波动 30% 的 GBM。一个每日再平衡的 3 倍杠杆 ETF 持有 1 年、5 年。
#
# - 标的的中位数净值约为多少？杠杆 ETF 的中位数净值呢？
# - 杠杆 ETF 的**平均**净值呢？为什么平均和中位数差这么多？
#
# 预测：

# %%
n_paths = 20_000
rng = np.random.default_rng(0)
for years in [1, 5]:
    n = years * DAYS
    r = rng.normal(0.0, 0.30 / np.sqrt(DAYS), (n, n_paths))
    under = np.prod(1 + r, axis=0)
    lev3 = np.prod(1 + 3 * r, axis=0)
    print(
        f"{years} 年：标的中位数 {np.median(under):.3f}；3x ETF 中位数 {np.median(lev3):.3f}，"
        f"平均 {lev3.mean():.3f}，亏损比例 {(lev3 < 1).mean():.0%}"
    )
print(
    f"理论几何增长率：1x {returns.growth_rate(0, 0.3):.3f}/年，"
    f"3x {returns.growth_rate(0, 0.3, 3):.3f}/年"
)

# %% [markdown]
# ## 实验 3：期望很好，大多数人却亏钱（讲义 §3）
#
# 一个 μ = 8%、σ = 40% 的 GBM 资产，持有 20 年。2 万条路径里：
#
# - 期末净值的**平均值**大约是多少？**中位数**呢？
# - 有多少比例的路径期末亏钱？
#
# 预测：

# %%
paths = returns.simulate_gbm(1.0, 0.08, 0.40, n_days=20 * DAYS, n_paths=20_000, seed=1)
st = paths[-1]
print(f"平均 {st.mean():.2f}（理论 e^(0.08*20) = {np.exp(1.6):.2f}）")
print(f"中位数 {np.median(st):.2f}（理论 e^((0.08-0.08)*20) = 1.00）")
print(
    f"亏钱的比例 {(st < 1).mean():.0%}；"
    f"最好的 1% 路径贡献了总财富的 {np.sort(st)[-200:].sum() / st.sum():.0%}"
)

plt.hist(np.log10(st), bins=80)
plt.axvline(np.log10(st.mean()), color="r", label="mean")
plt.axvline(np.log10(np.median(st)), color="k", ls="--", label="median")
plt.xlabel("log10 terminal wealth")
plt.title("GBM mu=8%, sigma=40%, 20 years")
plt.legend()
plt.show()

# %% [markdown]
# ## 实验 4：杠杆多少最好？（讲义 §3）
#
# 资产 μ = 8%、σ = 20%，每日再平衡到固定杠杆 L。长期几何增长率在 L 等于多少时最高？
# 杠杆超过多少之后，长期增长率变成负的？
#
# 预测：

# %%
levs = np.linspace(0, 5, 51)
g = [returns.growth_rate(0.08, 0.20, L) for L in levs]
best = levs[int(np.argmax(g))]
print(
    f"最优杠杆 {best:.1f}（理论 mu/sigma^2 = {0.08 / 0.04:.1f}），"
    f"增长率归零的杠杆 {2 * 0.08 / 0.04:.1f}"
)
plt.plot(levs, g)
plt.axhline(0, color="k", lw=0.5)
plt.xlabel("leverage")
plt.ylabel("long-run growth rate")
plt.show()

# %% [markdown]
# ## 实验 5：μ 和 σ 哪个能估准？（讲义 §4）
#
# 真实 μ = 8%、σ = 20%。用 10 年的**日频**数据估计 100 次，再用 10 年的**分钟频**
# （每天 240 分钟）数据估计 100 次。
#
# - μ 估计值的标准差，日频是多少？分钟频呢？
# - σ 估计值的标准差呢？
#
# 预测：

# %%
for label, per_day in [("日频", 1), ("分钟频", 240)]:
    periods = DAYS * per_day
    n = 10 * periods
    mus, sigmas = [], []
    for s in range(100):
        g_ = np.random.default_rng(100 + s)
        lr = g_.normal((0.08 - 0.02) / periods, 0.20 / np.sqrt(periods), n)
        mu_hat, sigma_hat = returns.annualize_log_stats(pd.Series(lr), periods=periods)
        mus.append(mu_hat)
        sigmas.append(sigma_hat)
    print(f"{label:4s}：mu 估计的标准差 {np.std(mus):.4f}，sigma 估计的标准差 {np.std(sigmas):.5f}")
print(f"理论 mu 的标准误 = sigma / sqrt(10) = {returns.mu_standard_error(0.2, 2520):.4f}")

# %% [markdown]
# ## 实验 6：再平衡是免费的午餐吗？（讲义 §5）
#
# 两个资产，各自零漂移、年化波动 40%、互不相关。50/50 每日再平衡 vs 期初 50/50 买入持有，20 年。
# 然后换成两个**漂移差很大**的资产（一个 +15%，一个 -5%）。哪种情况下再平衡更好？
#
# 预测：

# %%
for label, mu in [("同为零漂移", [0.0, 0.0]), ("漂移 +15% / -5%", [0.15, -0.05])]:
    reb, bah = [], []
    for s in range(200):
        px = data.correlated_gbm(2, 20 * DAYS, mu=np.array(mu), sigma=0.40, corr=0.0, seed=s)
        r = returns.simple_returns(px)
        reb.append(returns.cumulative_return(returns.rebalanced_portfolio_returns(r, [0.5, 0.5])))
        bah.append(returns.cumulative_return(returns.buy_and_hold_portfolio_returns(r, [0.5, 0.5])))
    reb, bah = np.array(reb), np.array(bah)
    print(
        f"{label}：再平衡中位数 {np.median(1 + reb):.2f}，买入持有中位数 {np.median(1 + bah):.2f}，"
        f"再平衡胜出比例 {(reb > bah).mean():.0%}"
    )

# %% [markdown]
# ## 做完之后
#
# 去做 [`audit.py`](audit.py)。实验 1 和实验 5 里的道理，会以“月报里的数字”出现在那里。
