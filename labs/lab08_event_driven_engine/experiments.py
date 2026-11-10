# %% [markdown]
# # Lab 08 实验：成交假设
#
# 每个实验：先在 `预测：` 后面写下答案，再运行，然后对照。猜错的地方回讲义对应小节。

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from quantlab import engine, perf

DAYS = 252
d = engine.bounce_ohlcv(5000, sigma=0.15, spread_bps=80, volume=20_000, seed=0)

# %% [markdown]
# ## 实验 1：收盘价里的“反转 alpha”（讲义 §1）
#
# 真实价格（mid）是纯随机游走。成交价落在买价或卖价上，价差 80bp。
# 策略：今天跌幅超过 1.5%，就持有到明天收盘。
#
# - 用收盘价算，这个策略的 Sharpe 是多少？
# - 用真实的 mid 价格算呢？
#
# 预测：


# %%
def reversal_sharpe(px: pd.Series, entry: pd.Series | None = None) -> float:
    """今天跌超 1.5% → 明天持有一天。entry 给出实际买入价（默认就是 px）。"""
    entry = px if entry is None else entry
    signal = px.pct_change() < -0.015
    ret = px.shift(-1) / entry - 1
    return perf.sharpe(ret.where(signal, 0.0).dropna())


print(f"close-to-close 自相关：{d['close'].pct_change().autocorr():+.3f}")
print(f"mid-to-mid    自相关：{d['mid'].pct_change().autocorr():+.3f}")
print(f"反转策略 Sharpe（收盘价）：{reversal_sharpe(d['close']):.2f}")
print(f"反转策略 Sharpe（mid）：   {reversal_sharpe(d['mid']):.2f}")

# %% [markdown]
# ## 实验 2：同一个信号，换一个成交时点（讲义 §1）
#
# 信号还是“今天收盘跌超 1.5%”。但这次你只能在**明天开盘**买入（并付半个价差），持有到明天收盘。
# Sharpe 还剩多少？
#
# 预测：

# %%
half_spread = 0.008 / 2
next_open_buy = d["open"].shift(-1) * (1 + half_spread)
signal = d["close"].pct_change() < -0.015
ret = (d["close"].shift(-1) * (1 - half_spread)) / next_open_buy - 1
print(
    f"次日开盘买、次日收盘卖（付价差）：Sharpe {perf.sharpe(ret.where(signal, 0.0).dropna()):.2f}"
)
print(f"信号触发的天数占比：{signal.mean():.1%}")

# %% [markdown]
# ## 实验 3：限价单的逆向选择（讲义 §2）
#
# 每天在前一日收盘价下方 1% 挂一张买入限价单，成交后持有到当天收盘。比较两种成交假设：
#
# - A：盘中最低价**触及**限价就算成交；
# - B：最低价**穿过**限价至少 0.2% 才算成交（你排在队列后面）。
#
# 两种假设下，成交率分别是多少？每笔成交的平均收益（收盘价 / 限价 - 1）分别是多少？
#
# 预测：

# %%
mid_d = engine.bounce_ohlcv(5000, sigma=0.25, spread_bps=0, seed=1)
limit = mid_d["close"].shift(1) * 0.99
for name, through in [("A 触及", 0.0), ("B 穿过 0.2%", 0.002)]:
    filled = mid_d["low"] < limit * (1 - through) if through else mid_d["low"] <= limit
    fill_px = np.minimum(mid_d["open"], limit)
    pnl = (mid_d["close"] / fill_px - 1)[filled]
    print(f"{name:10s} 成交率 {filled.mean():5.1%}   每笔平均收益 {pnl.mean() * 1e4:+6.1f} bp")

# %% [markdown]
# 追问：B 的成交是 A 的一个子集。被 B 去掉的那些成交（只是碰了一下限价就反弹）平均收益是正还是负？
# 这就是“逆向选择”：你最容易拿到的成交，恰恰是对你最不利的那些。

# %% [markdown]
# ## 实验 4：规模与容量（讲义 §3）
#
# 一个毛收益 Sharpe 1.5、年化波动 10%、年换手 20 倍的策略。成本 = 半个价差（10bp）+ 冲击，
# 冲击按平方根模型：$k\,\sigma_{\text{daily}}\sqrt{Q/V}$，k=1，股票日波动 2%，日成交额 1 亿。
#
# 资金规模从 100 万到 10 亿，扣费后 Sharpe 什么时候降到 0.5 以下？
#
# 预测：

# %%
gross_ret, gross_vol, turnover = 0.15, 0.10, 20.0
adv, sigma_d, half_spread_bps = 1e8, 0.02, 10.0
aums = np.logspace(6, 9, 31)
net_sr = []
for aum in aums:
    trade = aum * turnover / DAYS  # 每天交易的金额
    impact = sigma_d * np.sqrt(trade / adv)
    cost_per_turn = half_spread_bps * 1e-4 + impact
    net_sr.append((gross_ret - turnover * cost_per_turn) / gross_vol)
net_sr = np.array(net_sr)
for aum, sr in zip(aums[::6], net_sr[::6], strict=True):
    print(f"资金 {aum / 1e6:8.0f} 百万：扣费后 Sharpe {sr:5.2f}")
plt.semilogx(aums, net_sr)
plt.axhline(0.5, ls="--", color="k", lw=0.5)
plt.xlabel("AUM")
plt.ylabel("net Sharpe")
plt.show()

# %% [markdown]
# ## 实验 5：成交量约束下的“成交”需要多久（讲义 §4）
#
# 你想买入相当于日均成交量 30% 的股票，但每天最多只能成交当日成交量的 5%。
# - 需要几天才能建完仓？
# - 如果建仓期间价格每天平均漂移 +0.3%（别人也在买），你的平均成本比第一天开盘价高多少？
#
# 预测：

# %%
vol_path = engine.bounce_ohlcv(30, volume=1_000_000, seed=2)
target, cap, done, cost, day = 300_000.0, 0.05, 0.0, 0.0, 0
p0 = vol_path["open"].iloc[0]
while done < target and day < len(vol_path):
    bar = vol_path.iloc[day]
    q = min(target - done, cap * bar["volume"])
    px = p0 * (1.003**day)
    cost += q * px
    done += q
    day += 1
print(f"用了 {day} 天建完仓；平均成本比第一天高 {(cost / done / p0 - 1) * 1e4:.0f} bp")

# %% [markdown]
# ## 做完之后
#
# 去做 [`audit.py`](audit.py)。那里的每一个成交假设，都在这几个实验里出现过。
