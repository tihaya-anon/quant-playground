# %% [markdown]
# # Lab 11 实验：纸面 alpha 能剩下多少
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

from quantlab import execution

# %% [markdown]
# ## 实验 1：平方根冲击律（讲义 §1）
#
# 冲击成本（每交易 1 块钱付出的成本）≈ Y σ √(Q / V)，Y ≈ 0.8，日波动 σ = 2%。
#
# - 单子从当日成交额的 1% 放大到 4%，**每块钱**的成本变成几倍？**总成本**（金额）变成几倍？
# - 参与率 1% 时成本是多少个基点？参与率 25% 呢？
#
# 预测：

# %%
Y, sigma = 0.8, 0.02
for p in [0.01, 0.04, 0.16, 0.25]:
    bps = Y * sigma * np.sqrt(p) * 1e4
    print(f"参与率 {p:4.0%}：每块钱成本 {bps:5.1f} bps，总成本 ∝ {p * bps:6.2f}")

# %% [markdown]
# ## 实验 2：容量——Sharpe 最高的规模和赚钱最多的规模（讲义 §2）
#
# 一个策略：纸面年化收益 15%、波动 10%（Sharpe 1.5），每天双边换手 10%，交易 50 只股票，
# 每只日成交额 5000 万，半个价差 3 bps，冲击按实验 1。
#
# - 净 Sharpe 降到 1.0 时的规模大约是多少？
# - 每年**赚的钱**（净收益率 × 规模）在多大规模时最多？比上一个数大还是小？
#
# 预测：

# %%
alpha, vol, turnover, n_names, adv, half_spread = 0.15, 0.10, 0.10, 50, 5e7, 3e-4
aums = np.geomspace(1e7, 1e10, 200)
p = turnover * aums / n_names / adv
cost = turnover * 252 * (half_spread + Y * sigma * np.sqrt(p))
net = alpha - cost
sharpe = net / vol
profit = net * aums

print(f"净 Sharpe >= 1.0 的最大规模：{aums[sharpe >= 1.0].max() / 1e8:.1f} 亿")
best = profit.argmax()
print(f"年利润最大的规模：{aums[best] / 1e8:.1f} 亿，此时净 Sharpe {sharpe[best]:.2f}")
print(f"净收益归零的规模：{aums[net > 0].max() / 1e8:.1f} 亿")

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].semilogx(aums, sharpe)
ax[0].axhline(1.0, ls="--", color="k")
ax[0].set(xlabel="AUM", ylabel="net Sharpe", title="Net Sharpe vs size")
ax[1].semilogx(aums, profit / 1e6)
ax[1].set(xlabel="AUM", ylabel="net profit per year (millions)", title="Dollar profit vs size")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 实验 3：执行太慢，alpha 不等人（讲义 §3）
#
# 信号的 alpha 半衰期是 2 天。如果一笔交易要花 1、2、4、8 天才能执行完（每天执行相同份额），
# 平均能吃到多少比例的 alpha？
#
# 预测：

# %%
for days in [1, 2, 4, 8]:
    capture = np.mean(0.5 ** (np.arange(days) / 2.0))
    print(f"分 {days} 天执行：吃到 {capture:.0%} 的 alpha")

# %% [markdown]
# ## 实验 4：Almgren-Chriss——快慢之间的取舍（讲义 §4）
#
# 原文的例子：5 天内卖出 100 万股，股价 50，日波动 0.95 美元。
# 风险厌恶 λ 从接近 0（就是 TWAP）提高到 1e-5：
#
# - 期望成本会增加多少？成本的标准差会减少多少？
# - λ = 1e-6 时，第一天卖出的比例是多少？
#
# 预测：

# %%
P = dict(X=1e6, T=5.0, N=10, sigma=0.95, eta=2.5e-6, gamma=2.5e-7, epsilon=0.0625)
lams = [1e-9, 1e-7, 1e-6, 1e-5]
frontier = execution.efficient_frontier(**P, lams=lams)
frontier["std"] = np.sqrt(frontier["variance"])
print(frontier[["expected_cost", "std"]].round(0))

for lam in [1e-9, 1e-6, 1e-5]:
    x = execution.ac_trajectory(P["X"], P["T"], P["N"], P["sigma"], P["eta"], P["gamma"], lam)
    print(f"λ = {lam:.0e}：第一天（前两个区间）卖出 {(x[0] - x[2]) / x[0]:.0%}")

# %% [markdown]
# ## 实验 5：用 VWAP 衡量执行成本（讲义 §5，审计题的预热）
#
# 下面是 400 笔历史母单的执行记录。按参与率分组，比较两种成本衡量方式：
# 相对决策时价格（arrival）的成本，和相对当天全市场 VWAP 的滑点。
#
# - 参与率越高，哪一种成本涨得更快？
# - 参与率 20% 以上的单子，相对 VWAP 的滑点大约是相对 arrival 的几分之一？
#
# 预测：

# %%
log = execution.strategy_market(seed=0).exec_log
log["vs_arrival_bps"] = log["side"] * (log["avg_price"] / log["arrival"] - 1) * 1e4
log["vs_vwap_bps"] = log["side"] * (log["avg_price"] / log["market_vwap"] - 1) * 1e4
bucket = pd.cut(log["participation"], [0, 0.05, 0.1, 0.2, 0.25])
print(log.groupby(bucket, observed=True)[["vs_arrival_bps", "vs_vwap_bps"]].mean().round(1))

# %% [markdown]
# ## 实验 6：执行成本本身也是随机的（讲义 §4）
#
# 用实验 4 的参数，模拟 TWAP 和 λ = 1e-5 的前置轨迹各 20000 次。
#
# - 有多大比例的日子里，前置执行反而比 TWAP 更贵？
# - 两者最差 5% 的成本分别是多少？
#
# 预测：

# %%
tau = P["T"] / P["N"]
args = dict(S0=50.0, sigma=P["sigma"], eta=P["eta"], gamma=P["gamma"], epsilon=P["epsilon"])
x_twap = execution.ac_trajectory(P["X"], P["T"], P["N"], P["sigma"], P["eta"], P["gamma"], 1e-9)
x_fast = execution.ac_trajectory(P["X"], P["T"], P["N"], P["sigma"], P["eta"], P["gamma"], 1e-5)
c_twap = execution.simulate_shortfall(x_twap, **args, tau=tau, n_paths=20_000, seed=6)
c_fast = execution.simulate_shortfall(x_fast, **args, tau=tau, n_paths=20_000, seed=6)
print(f"前置比 TWAP 更贵的比例：{(c_fast > c_twap).mean():.0%}")
for name, c in [("TWAP", c_twap), ("front-loaded", c_fast)]:
    print(f"{name:13s} 均值 {c.mean():9,.0f}   最差 5% 均值 {np.sort(c)[-1000:].mean():9,.0f}")

# %% [markdown]
# ## 做完之后
#
# 去做 [`audit.py`](audit.py)。实验 2、3、5 里的东西，会以更隐蔽的形式出现在那里。
