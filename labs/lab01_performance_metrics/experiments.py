# %% [markdown]
# # Lab 01 实验：风险指标藏住了什么
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

from quantlab import data, risk

DAYS = 252


def normal_strategy(years, seed, sharpe=1.0, vol=0.10):
    n = int(years * DAYS)
    g = np.random.default_rng(seed)
    r = g.normal(sharpe * vol / DAYS, vol / np.sqrt(DAYS), n)
    return pd.Series(r, index=data.business_index(n))


def sell_insurance(years, seed):
    """平时每天赚一点，平均每两年一次 8%~15% 的大亏。"""
    n = int(years * DAYS)
    g = np.random.default_rng(seed)
    r = g.normal(0.0004, 0.002, n)
    crash = g.random(n) < 1 / (2 * DAYS)
    r[crash] -= g.uniform(0.08, 0.15, crash.sum())
    return pd.Series(r, index=data.business_index(n))


# %% [markdown]
# ## 实验 1：同样的 Sharpe，不同的风险（讲义 §1、§4）
#
# 两个策略在同一段 5 年历史上的均值、波动、Sharpe 完全相同：一个收益是正态的，一个是“卖保险”。
# 先猜：哪一个最大回撤更大？哪一个 99% CVaR 更大？大多少倍？
#
# 预测：

# %%
calm_seed = next(s for s in range(1000) if (sell_insurance(5, s) < -0.05).sum() == 1)
ins = sell_insurance(5, calm_seed)
z = normal_strategy(5, seed=0)
norm = ins.mean() + ins.std() * (z - z.mean()) / z.std()  # 样本均值和波动与 ins 完全相同
summary = pd.DataFrame(
    {"normal": risk.performance_summary(norm), "sell insurance": risk.performance_summary(ins)}
)
print(summary.loc[["sharpe", "ann_vol", "max_drawdown", "cvar_95", "skew", "excess_kurt"]].round(3))
print(
    f"99% CVaR：正态 {risk.historical_cvar(norm, 0.01):.4f}，"
    f"卖保险 {risk.historical_cvar(ins, 0.01):.4f}"
)

# %% [markdown]
# ## 实验 2：最大回撤会随时间变长而变大（讲义 §3）
#
# 一个真实 Sharpe 1.0、年化波动 10% 的策略（已经非常好）。在 3 年、10 年、30 年的历史里，
# 最大回撤的**中位数**分别是多少？
#
# 预测：

# %%
for years in [3, 10, 30]:
    mdds = [risk.max_drawdown(normal_strategy(years, seed=s)) for s in range(300)]
    print(
        f"{years:2d} 年：最大回撤中位数 {np.median(mdds):.1%}，"
        f"90% 分位 {np.quantile(mdds, 0.9):.1%}"
    )

# %% [markdown]
# ## 实验 3：正态 VaR 在肥尾上（讲义 §4）
#
# 收益服从自由度为 3 的 t 分布（肥尾），标准差 1%。用正态公式和历史分位数分别估计
# 95%、99%、99.9% 的单日 VaR，然后在**另一段**数据上数一数实际突破的比例。
#
# 预测（正态 99.9% VaR 的实际突破率是多少？）：

# %%
g = np.random.default_rng(1)
t3 = g.standard_t(3, 400_000) / np.sqrt(3)
r = pd.Series(0.01 * t3)
train, test = r.iloc[:200_000], r.iloc[200_000:]
for alpha in [0.05, 0.01, 0.001]:
    pv, hv = risk.parametric_var(train, alpha), risk.historical_var(train, alpha)
    print(
        f"{1 - alpha:.1%}：正态 VaR {pv:.4f}（突破 {(test < -pv).mean():.3%}），"
        f"历史 VaR {hv:.4f}（突破 {(test < -hv).mean():.3%}）"
    )

# %% [markdown]
# ## 实验 4：平滑收益（讲义 §6）
#
# 一个私募产品的真实月收益：年化 10%、波动 15%。但它的资产没有市场价格，每月的估值是
# 0.4 × 本月真实变化 + 0.6 × 上月报告值（估值滞后）。
#
# 报告出来的年化波动和 Sharpe 是多少？去平滑（Geltner）之后呢？
#
# 预测：

# %%
months = 12 * 30
g = np.random.default_rng(2)
true = pd.Series(g.normal(0.10 / 12, 0.15 / np.sqrt(12), months))
reported = true.copy()
for t in range(1, months):
    reported.iloc[t] = 0.4 * true.iloc[t] + 0.6 * reported.iloc[t - 1]
recovered = risk.unsmooth(reported)
for label, s in [("真实", true), ("报告", reported), ("去平滑", recovered)]:
    print(
        f"{label}：年化波动 {s.std() * np.sqrt(12):.1%}，"
        f"Sharpe {s.mean() / s.std() * np.sqrt(12):.2f}，"
        f"一阶自相关 {s.autocorr(1):.2f}"
    )

# %% [markdown]
# ## 实验 5：VaR 的平方根法则（讲义 §4）
#
# 用 GARCH(1,1) 生成有波动聚集的收益。用过去 250 天估计单日 99% VaR，乘以 sqrt(10) 得到 10 日 VaR，
# 然后看未来 10 天的累计损失有多少比例超过它。
#
# - 整体突破率是多少（应为 1%）？
# - 突破是均匀分布在时间上，还是扎堆出现？
#
# 预测：

# %%
gr = data.garch_returns(n_days=12_000, omega=2e-6, alpha=0.10, beta=0.88, seed=3)["ret"]
breach = []
for t in range(250, len(gr) - 10, 5):
    var10 = risk.scale_var(risk.historical_var(gr.iloc[t - 250 : t], 0.01), 10)
    breach.append((gr.iloc[t : t + 10].sum() < -var10, t))
b = pd.Series([x for x, _ in breach], index=[t for _, t in breach])
print(f"整体突破率 {b.mean():.2%}")
rolling = b.rolling(100).mean()
print(f"按 100 个窗口滚动：最低 {rolling.min():.1%}，最高 {rolling.max():.1%}")
plt.plot(rolling.to_numpy())
plt.axhline(0.01, color="k", ls="--")
plt.title("10-day VaR breach rate, rolling")
plt.show()

# %% [markdown]
# ## 实验 6：偏度有多脆弱（讲义 §5）
#
# 10 年的肥尾日收益。算样本偏度，然后去掉**最差的那一天**再算一次。偏度会变多少？
#
# 预测：

# %%
g = np.random.default_rng(4)
r = pd.Series(0.01 * g.standard_t(3, 2520) / np.sqrt(3))
print(f"全部样本偏度 {r.skew():.2f}，峰度 {r.kurt():.1f}")
r2 = r.drop(r.idxmin())
print(f"去掉最差一天：偏度 {r2.skew():.2f}，峰度 {r2.kurt():.1f}")

# %% [markdown]
# ## 做完之后
#
# 去做 [`audit.py`](audit.py)。实验 3、4 里的道理会以“风险仪表盘”的形式出现在那里。
