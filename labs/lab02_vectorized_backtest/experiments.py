# %% [markdown]
# # Lab 02 实验：回测的机制
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

from quantlab import backtest as bt
from quantlab import data, perf

DAYS = 252

# %% [markdown]
# ## 实验 1：少一个 shift（讲义 §1、§2）
#
# 在**完全不可预测**的 GBM 价格上跑 20 日动量。一个版本正确地把信号推迟一天，
# 另一个版本直接用当天的信号乘当天的收益。两者的年化 Sharpe 各是多少？
#
# 预测：

# %%
for s in range(3):
    px = data.gbm_prices(2520, mu=0.0, sigma=0.2, seed=s)
    r = px.pct_change().fillna(0.0)
    sig = bt.momentum_signal(px, 20)
    right = bt.backtest(px, sig)["net"]
    wrong = (sig * r).iloc[1:]
    print(f"seed {s}：正确 {perf.sharpe(right):5.2f}   不 shift {perf.sharpe(wrong):5.2f}")

# %% [markdown]
# ## 实验 2：成本吃掉多少（讲义 §3）
#
# 在有趋势的数据上跑三组均线交叉：(5, 20)、(20, 100)、(50, 200)。
#
# - 哪一组毛 Sharpe 最高？
# - 每组的年化换手是多少？盈亏平衡成本（bp）是多少？
# - 单边成本 10bp 时，哪一组净 Sharpe 最高？
#
# 预测：

# %%
rows = []
for fast, slow in [(5, 20), (20, 100), (50, 200)]:
    gross, net, to, be = [], [], [], []
    for s in range(5):
        px = data.regime_switching_prices(2520, seed=s)
        sig = bt.sma_crossover_signal(px, fast, slow)
        res = bt.backtest(px, sig, cost_bps=10)
        gross.append(perf.sharpe(res["gross"]))
        net.append(perf.sharpe(res["net"]))
        to.append(bt.annualized_turnover(res))
        be.append(bt.breakeven_cost_bps(res))
    rows.append(
        {
            "fast/slow": f"{fast}/{slow}",
            "gross sharpe": np.mean(gross),
            "net sharpe @10bp": np.mean(net),
            "ann turnover": np.mean(to),
            "breakeven bp": np.mean(be),
        }
    )
print(pd.DataFrame(rows).round(2))

# %% [markdown]
# ## 实验 3：幸存者偏差（讲义 §4）
#
# 30 只股票，漂移在 -15% 到 +15% 之间随机，跌破初始价的 25% 就退市。
# 等权买入持有：（a）期初的全部 30 只；（b）只买“到今天还在交易”的那些。
#
# 两者的年化收益差多少？
#
# 预测：

# %%
diffs = []
for s in range(20):
    u = bt.universe_with_delistings(seed=s)
    r = (u / u.shift(1) - 1).fillna(0.0)
    alive = u.columns[u.iloc[-1].notna()]
    all_names = (1 + r).cumprod().mean(axis=1)  # 期初等权、买入持有
    survivors = (1 + r[alive]).cumprod().mean(axis=1)
    diffs.append(
        (
            perf.annualized_return(all_names.pct_change().dropna()),
            perf.annualized_return(survivors.pct_change().dropna()),
            len(alive),
        )
    )
d = np.array(diffs)
print(
    f"全部股票：年化 {d[:, 0].mean():.1%}；只看幸存者：年化 {d[:, 1].mean():.1%}；"
    f"平均幸存 {d[:, 2].mean():.0f}/30 只"
)

# %% [markdown]
# ## 实验 4：“去噪”的居中窗口（讲义 §2）
#
# 有人觉得价格太噪，先用 21 日均线平滑，再按均线的斜率交易：斜率向上做多、向下做空。
# 仓位**正确地 shift 了一天**。两个版本的区别只有一个参数：均线用 `center=True`（以当天为中心，
# 前后各 10 天）还是默认的只用过去 21 天。
#
# 在**完全不可预测**的 GBM 上，两者的 Sharpe 各是多少？
#
# 预测：

# %%
res = {"trailing window": [], "centered window": []}
for s in range(5):
    px = data.gbm_prices(2520, mu=0.0, sigma=0.2, seed=s)
    for name, center in [("trailing window", False), ("centered window", True)]:
        sma = px.rolling(21, center=center).mean()
        sig = np.sign(sma.diff()).fillna(0.0)
        res[name].append(perf.sharpe(bt.backtest(px, sig)["net"]))
print({k: round(float(np.mean(v)), 2) for k, v in res.items()})

# %% [markdown]
# ## 实验 5：在哪个价格成交（讲义 §1）
#
# 用收盘价计算信号。三种成交假设：（a）当天收盘价成交（不可能做到：收盘价出来时已经收盘了）；
# （b）次日开盘价成交；（c）次日收盘价成交。在有趋势的数据上，三者的 Sharpe 差多少？
#
# 预测：

# %%
out = {"same close": [], "next open": [], "next close": []}
for s in range(5):
    close = data.regime_switching_prices(2520, seed=s)
    bar = data.ohlcv_from_close(close, seed=s)
    sig = bt.sma_crossover_signal(close, 10, 50)
    cc = close.pct_change().fillna(0.0)
    out["same close"].append(perf.sharpe((sig * cc).iloc[1:]))
    # 次日开盘建仓：t+1 的收益 = open->close 部分由新仓位赚，前一晚 close->open 由旧仓位赚
    pos_new, pos_old = sig.shift(1).fillna(0.0), sig.shift(2).fillna(0.0)
    overnight = (bar["open"] / close.shift(1) - 1).fillna(0.0)
    intraday = (close / bar["open"] - 1).fillna(0.0)
    out["next open"].append(perf.sharpe((pos_old * overnight + pos_new * intraday).iloc[2:]))
    out["next close"].append(perf.sharpe(bt.backtest(close, sig)["net"]))
print({k: round(float(np.mean(v)), 2) for k, v in out.items()})

# %% [markdown]
# ## 实验 6：波动率目标（讲义 §5）
#
# 同一个动量信号，一个版本固定仓位 ±1，一个版本把仓位缩放到目标波动 10%（用过去 60 天的波动）。
# 数据是有波动聚集的 GARCH 收益加上一段趋势。Sharpe 和最大回撤各变化多少？
#
# 预测：

# %%
fixed, targeted = [], []
for s in range(5):
    g = data.garch_returns(2520, omega=3e-6, alpha=0.10, beta=0.87, seed=s)["ret"]
    trend = data.regime_switching_prices(2520, seed=s, trend_vol=0.0).pct_change().fillna(0.0)
    px = 100 * (1 + g + trend.to_numpy()).cumprod()
    sig = bt.momentum_signal(px, 60)
    vt = bt.vol_target_signal(sig, px, target_vol=0.10, window=60)
    a, b = bt.backtest(px, sig)["net"], bt.backtest(px, vt)["net"]
    fixed.append((perf.sharpe(a), perf.max_drawdown(a)))
    targeted.append((perf.sharpe(b), perf.max_drawdown(b)))
fx, tg = np.array(fixed), np.array(targeted)
print(f"固定仓位：Sharpe {fx[:, 0].mean():.2f}，最大回撤 {fx[:, 1].mean():.1%}")
print(f"波动目标：Sharpe {tg[:, 0].mean():.2f}，最大回撤 {tg[:, 1].mean():.1%}")

plt.plot((1 + a).cumprod().to_numpy(), label="fixed")
plt.plot((1 + b).cumprod().to_numpy(), label="vol target 10%")
plt.yscale("log")
plt.legend()
plt.show()

# %% [markdown]
# ## 做完之后
#
# 去做 [`audit.py`](audit.py)。实验 1、3、4 里的问题会同时出现在那里。
