# %% [markdown]
# # Lab 10 实验：卖期权，卖的是什么
#
# 每个实验的流程：
#
# 1. 读问题，**在 `预测：` 后面写下你的答案**（一个数字或一个范围，再加一句理由）。
# 2. 运行下面的代码单元格。
# 3. 对照结果。猜错了的话，回到讲义对应的小节，想清楚是哪一步直觉出了问题。

# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from quantlab import options

S0, K, T, R = 100.0, 100.0, 0.25, 0.0


def hedge_short_straddle(
    sigma_real: float,
    sigma_sold: float,
    n_steps: int,
    n_paths: int,
    seed: int,
    jump_per_year: float = 0.0,
    jump_mean: float = -0.06,
    jump_sd: float = 0.03,
    T: float = T,
) -> np.ndarray:
    """卖出平值跨式（按 sigma_sold 定价），用 sigma_sold 的 BS delta 离散对冲到期。

    标的为扩散（波动 sigma_real）加可选的跳跃。返回每条路径的 P&L（单位：标的初始价格的 %）。
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    S = np.full(n_paths, S0)
    call = options.bs_price(S0, K, T, R, sigma_sold, kind="call")
    put = options.bs_price(S0, K, T, R, sigma_sold, kind="put")
    cash = call + put
    d1, _ = options.d1_d2(S0, K, T, R, sigma_sold)
    delta = 2 * norm.cdf(d1) - 1
    cash -= delta * S
    for k in range(1, n_steps + 1):
        z = rng.standard_normal(n_paths)
        jump = np.where(
            rng.random(n_paths) < jump_per_year * dt, rng.normal(jump_mean, jump_sd, n_paths), 0.0
        )
        S = S * np.exp(-0.5 * sigma_real**2 * dt + sigma_real * np.sqrt(dt) * z + jump)
        if k < n_steps:
            d1, _ = options.d1_d2(S, K, T - k * dt, R, sigma_sold)
            new_delta = 2 * norm.cdf(d1) - 1
            cash -= (new_delta - delta) * S
            delta = new_delta
    return (cash + delta * S - np.abs(S - K)) / S0 * 100


# %% [markdown]
# ## 实验 1：期权价格不依赖于标的的期望收益（讲义 §1）
#
# 卖出一份 1 年期平值看涨，用真实波动率（20%）每日对冲。标的的年化漂移分别是 0%、+30%、-30%。
#
# - 三种情况下，对冲后 P&L 的均值分别是多少？
#
# 预测：

# %%
for mu in [0.0, 0.3, -0.3]:
    pnl = options.delta_hedge_pnl(100, 100, 1.0, 0.0, 0.2, 0.2, 252, 20_000, seed=1, mu=mu)
    print(f"漂移 {mu:+.0%}：对冲 P&L 均值 {pnl.mean():+.3f}，标准差 {pnl.std():.3f}")

# %% [markdown]
# ## 实验 2：离散对冲的误差（讲义 §2）
#
# 3 个月平值跨式，用正确的波动率对冲。每天对冲一次时，P&L 的标准差是多少（% of S）？
# 改成每周对冲一次，标准差变成几倍？每天对冲 10 次呢？
#
# 预测：

# %%
for steps, label in [(13, "weekly"), (63, "daily"), (630, "10x daily")]:
    p = hedge_short_straddle(0.2, 0.2, steps, 20_000, seed=2)
    print(f"{label:10s} 均值 {p.mean():+.3f}%   标准差 {p.std():.3f}%")

# %% [markdown]
# ## 实验 3：用错了波动率（讲义 §2）
#
# 你按 20% 的隐含波动率卖出跨式，每日对冲。实际实现的波动率是 15% 或 25%。
#
# - 两种情况下 P&L 的均值大约是多少？符号你一定能猜对，数量级呢？
# - 实现波动率 15% 的情况下，有多大比例的路径仍然亏钱？
#
# 预测：

# %%
for real in [0.15, 0.20, 0.25]:
    p = hedge_short_straddle(real, 0.20, 63, 20_000, seed=3)
    print(f"实现 {real:.0%}：均值 {p.mean():+.2f}%   亏钱路径 {(p < 0).mean():.0%}")

vega = 2 * options.greeks(S0, K, T, R, 0.2)["vega"] / S0 * 100
print(f"粗略估计：vega * Δσ = {vega:.1f} * 0.05 = {vega * 0.05:.2f}%")

# %% [markdown]
# 追问：既然 P&L ≈ ½ Γ S² (σ²_real − σ²_impl) 在时间上的积分，为什么同样实现了 15% 的波动，
# 有的路径赚得多、有的赚得少？（提示：大波动发生时，Γ 在哪里？）

# %% [markdown]
# ## 实验 4：跳跃让对冲失效（讲义 §3）
#
# 加入向下跳跃（平均每年 2 次，每次约 -6%），并按同样的跳跃风险把隐含波动率定高一点。
# 对冲频率从每周提高到每天 10 次，P&L 的标准差会像实验 2 那样一直下降吗？
#
# 预测：

# %%
for steps, label in [(13, "weekly"), (63, "daily"), (630, "10x daily")]:
    p = hedge_short_straddle(0.18, 0.22, steps, 20_000, seed=4, jump_per_year=2.0)
    print(
        f"{label:10s} 均值 {p.mean():+.2f}%   标准差 {p.std():.2f}%   "
        f"最差 1% 均值 {np.sort(p)[: len(p) // 100].mean():+.2f}%"
    )

# %% [markdown]
# ## 实验 5：波动率微笑从哪里来（讲义 §3）
#
# 在一个**有向下跳跃**的世界里（扩散波动 15%，每年 1 次 -10% 左右的跳跃），
# 用蒙特卡洛算出不同行权价的 3 个月看跌期权的“公平”价格，再用 BS 公式反解出隐含波动率。
#
# 如果 BS 是对的，所有行权价的隐含波动率应该一样。实际上：
#
# - 低行权价（深度虚值看跌）的隐含波动率比平值高还是低？
# - 高行权价呢？
#
# 预测：

# %%
rng = np.random.default_rng(5)
n = 400_000
lam_j, mu_j, sd_j, sig = 1.0, -0.10, 0.04, 0.15
n_jumps = rng.poisson(lam_j * T, n)
jump_sum = rng.normal(mu_j * n_jumps, sd_j * np.sqrt(n_jumps))
kappa = np.exp(mu_j + 0.5 * sd_j**2) - 1  # 跳跃补偿，使风险中性下 E[S_T] = S0
log_st = (-0.5 * sig**2 - lam_j * kappa) * T + sig * np.sqrt(T) * rng.standard_normal(n)
ST = S0 * np.exp(log_st + jump_sum)

strikes = np.arange(75, 121, 5)
ivs = []
for k in strikes:
    price = np.maximum(k - ST, 0).mean()
    ivs.append(options.implied_vol(price, S0, k, T, R, kind="put"))
for k, v in zip(strikes, ivs, strict=True):
    print(f"K = {k:3d}   implied vol {v:.1%}")

plt.plot(strikes, np.array(ivs) * 100, marker="o")
plt.xlabel("strike")
plt.ylabel("implied vol (%)")
plt.title("Implied vol smile in a world with downward jumps")
plt.show()

# %% [markdown]
# ## 实验 6：卖保险的 Sharpe（讲义 §4，回顾 lab03 §2）
#
# 每季度卖一次跨式，隐含波动率比扩散波动率高 4 个点（22% vs 18%），世界里有向下跳跃。
# 模拟 4000 个季度。
#
# - 有多大比例的季度是赚钱的？
# - 全样本年化 Sharpe 是多少？
# - 一段恰好没有跳跃的 3 年（12 个季度）里，年化 Sharpe 看起来是多少？
#
# 预测：

# %%
q = hedge_short_straddle(0.18, 0.22, 63, 4000, seed=6, jump_per_year=2.0)
print(f"赚钱的季度：{(q > 0).mean():.0%}")
print(f"偏度：{((q - q.mean()) ** 3).mean() / q.std() ** 3:.1f}")
print(f"全样本年化 Sharpe：{q.mean() / q.std() * 2:.2f}")

calm = q[q > np.quantile(q, 0.10)]  # 去掉最差的 10%：一段“运气好”的历史大概长这样
sharpe_calm = [
    (w := np.random.default_rng(i).choice(calm, 12)).mean() / w.std() * 2 for i in range(200)
]
print(f"没有大亏的 3 年，年化 Sharpe 中位数：{np.median(sharpe_calm):.2f}")

# %% [markdown]
# ## 做完之后
#
# 去做 [`audit.py`](audit.py)。实验 4 和实验 6 里的东西，会以更隐蔽的形式出现在那里。
