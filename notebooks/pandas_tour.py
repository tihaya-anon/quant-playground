# %% [markdown]
# # pandas 速览：这门课里用到的操作
#
# 每个单元格讲一类操作：先说它做什么、在哪个 lab 出现、最容易踩的坑，再给一个能直接运行的小例子。
# 在 VS Code 里把光标放进单元格按 `Shift+Enter` 逐格运行，结果显示在右侧的交互窗口里。
#
# 读这份文件的目的不是背 API——需要时随时可以查、也可以让 AI 写——而是在审计别人（或 AI）的代码时，
# 一眼看出某个操作**实际在算什么**。和它配套的是 `docs/pandas-pitfalls.md` 那份回测陷阱清单。
#
# 目录
#
# 1. Series、DataFrame 与索引
# 2. 取数：`[]`、`.loc`、`.iloc`、布尔筛选
# 3. 索引对齐：pandas 最重要的规则
# 4. 时间上的错位：`shift`、`diff`、`pct_change`
# 5. 缺失值：`isna`、`dropna`、`fillna`、`ffill`
# 6. 累计：`cumprod`、`cumsum`、`cummax`
# 7. 聚合与 `axis`：`mean`、`std`、`quantile`、`skew`、`kurt`
# 8. 滚动、扩展与指数加权：`rolling`、`expanding`、`ewm`
# 9. 重采样与按时间分组：`resample`、`groupby`、`day_of_week`
# 10. 横截面操作：`rank`、`corrwith`、`get_dummies`、`stack`
# 11. 条件与截断：`where`、`clip`、`abs`、`idxmax`
# 12. 拼接与重排：`concat`、`reindex`、`set_index`、`rename`
# 13. 和 numpy 互通：`to_numpy`、`@`、构造回 pandas
# 14. 小结：审计时的检查顺序

# %%
import numpy as np
import pandas as pd

from quantlab import data

pd.set_option("display.width", 110)
pd.set_option("display.precision", 4)

# %% [markdown]
# ## 1. Series、DataFrame 与索引
#
# - **Series**：一列带标签的数据。标签就是**索引（index）**。在这门课里，索引几乎总是日期。
# - **DataFrame**：多列共用同一个索引的表。在横截面问题里（lab04 起），行是日期、列是股票。
# - **DatetimeIndex**：以时间戳为元素的索引，支持按日期切片、按月重采样等时间操作。
#
# `pd.bdate_range` 生成工作日序列（周一到周五），`quantlab.data.business_index` 就是对它的封装。
# 注意：工作日不等于交易日——节假日也会被算进去（lab01 的审计题就和这个有关）。

# %%
idx = pd.bdate_range("2024-01-01", periods=6, name="date")
px = pd.Series([100.0, 101.0, 99.5, 100.5, 102.0, 101.0], index=idx, name="close")
print(px)
print("索引类型：", type(px.index).__name__, "| 长度：", len(px))

# DataFrame：每一列是一只股票
panel = data.correlated_gbm(n_assets=3, n_days=6, seed=0)
print(panel)
print("形状 (行, 列)：", panel.shape, "| 列名：", list(panel.columns))

# %% [markdown]
# ## 2. 取数：`[]`、`.loc`、`.iloc`、布尔筛选
#
# | 写法 | 按什么取 | 例子 |
# | --- | --- | --- |
# | `.iloc[...]` | **位置**（第几行、第几列，从 0 开始） | `px.iloc[0]`、`px.iloc[1:]`、`px.iloc[-1]` |
# | `.loc[...]` | **标签**（索引值、列名） | `px.loc["2024-01-03"]`、`df.loc[:, "A00"]` |
# | `df["col"]` | 取一列 | `panel["A00"]` |
# | `s[mask]` | 布尔筛选，保留 mask 为 True 的行 | `px[px > 100]` |
#
# **坑**：`.loc` 的切片**两端都包含**（因为是按标签），`.iloc` 的切片和 Python 一样**不含右端**。
# 建议：按位置就写 `.iloc`，按标签就写 `.loc`。`s[1:]` 这种写法在不同索引类型下含义会变，审计时不好判断。
#
# 出现的地方：`returns.iloc[1:]` 去掉第一行 NaN（lab00）；`prices.iloc[:split]` 切训练段（lab03）；
# `returns[returns < 0]` 取亏损日（lab01 Sortino 的原始错误写法）。

# %%
print("第一个：", px.iloc[0], "| 最后一个：", px.iloc[-1])
print("按位置切片 iloc[1:3]（不含第 3 个）：\n", px.iloc[1:3])
print(
    "按日期切片 loc['2024-01-02':'2024-01-04']（两端都含）：\n", px.loc["2024-01-02":"2024-01-04"]
)
print("布尔筛选 px > 100：\n", px[px > 100])
print("DataFrame 取一个格子 .loc[日期, 列名]：", panel.loc[panel.index[2], "A01"])

# %% [markdown]
# ## 3. 索引对齐：pandas 最重要的规则
#
# 两个 pandas 对象做 `+ - * /` 时，pandas **先按标签配对，再计算**：
#
# - 结果的索引是两边标签的**并集**；
# - 只在一边出现的标签，结果是 NaN；
# - **不会报错**——这正是它危险的地方。
#
# DataFrame 和 Series 运算时，默认把 Series 的索引对到 DataFrame 的**列名**上。
# 想按**行**（日期）对齐，必须用方法形式并写 `axis=0`：`df.sub(s, axis=0)`、`df.div(s, axis=0)`。
#
# 出现的地方：lab00 你自己踩过的“把索引加一天来错开一期”；lab02 横截面标准化
# `mom - mom.mean(axis=1)` 全变成 NaN；lab04 的中性化、分组收益。

# %%
a = pd.Series([1.0, 2.0, 3.0], index=["x", "y", "z"])
b = pd.Series([10.0, 20.0, 30.0], index=["y", "z", "w"])
print("a + b（按标签配对，并集，配不上的是 NaN）：\n", a + b)

# DataFrame 减“每天的均值”：
row_mean = panel.mean(axis=1)  # 索引是日期
wrong = panel - row_mean  # Series 的日期被对到列名上
right = panel.sub(row_mean, axis=0)  # 按行对齐
print("错误写法的列：", list(wrong.columns)[:5], "... 全是 NaN？", wrong.isna().all().all())
print("正确写法每天的均值（应为 0）：\n", right.mean(axis=1).round(12))

# %% [markdown]
# ## 4. 时间上的错位：`shift`、`diff`、`pct_change`
#
# - `s.shift(k)`：把**值**往后挪 k 行，**索引不动**。第 t 行得到的是 t-k 行的值；开头 k 行变成 NaN。
#   `shift(-k)` 是往前挪，即“看未来”，在回测里几乎总是 bug。
# - `s.diff()` = `s - s.shift(1)`：相邻两期的差。lab02 里用它算换手。
# - `s.pct_change()` = `s / s.shift(1) - 1`：简单收益。
#
# **三者开头都会产生 NaN。** 每次看到它们，都要问：开头那个 NaN 实际代表什么？
# lab02 的持仓 `weights.shift(1)` 开头的 NaN 代表“期初空仓”，应该是 0；
# 而 `pct_change()` 开头的 NaN 代表“没有前一天价格”，通常直接丢掉。
#
# 出现的地方：几乎每个 lab。`signal.shift(1) * returns` 是“t 的信号只能吃 t+1 的收益”的标准写法。

# %%
frame = pd.DataFrame(
    {
        "close": px,
        "shift(1)": px.shift(1),
        "shift(-1) 未来!": px.shift(-1),
        "diff()": px.diff(),
        "pct_change()": px.pct_change(),
    }
)
print(frame)

# 信号与收益的正确配对：今天收盘算出的信号，明天才持有
signal = np.sign(px.pct_change(2)).fillna(0.0)  # 过去 2 天涨了记 +1
strategy = signal.shift(1) * px.pct_change()
print("\n策略收益（注意第一个有效值出现在信号产生的后一天）：\n", strategy)

# %% [markdown]
# ## 5. 缺失值：`isna`、`dropna`、`fillna`、`ffill`
#
# - `s.isna()` / `s.notna()`：逐个判断是否为缺失值，返回布尔序列。
# - `s.dropna()`：删掉缺失值所在的行。DataFrame 上 `dropna(axis=1)` 删掉**含有**缺失值的**列**。
# - `s.fillna(v)`：用 v 填补。v 可以是一个数，也可以是**形状相同的另一个 pandas 对象**——按位置逐个填补。
# - `s.ffill()`：用前一个有效值往后填。
#
# **坑**：
#
# - `sum()`、`mean()` 默认**跳过** NaN，所以“漏掉的 NaN”常常不会报错，只会让结果悄悄偏掉。
# - `dropna(axis=1)` 删掉中途退市的股票，就是**幸存者偏差**（lab02、lab04）。
# - `ffill` 会把停牌期间的旧价格当成可成交的价格。
#
# 出现的地方：lab02 的 `pos.diff().fillna(pos)`（第 0 天的换手 = 第 0 天的持仓）。

# %%
pos = pd.Series([0.5, 0.5, 0.0, np.nan, 0.3], index=idx[:5])
print("isna：", pos.isna().tolist())
print("dropna 后长度：", len(pos.dropna()), "（原来", len(pos), "）")
print("sum 默认跳过 NaN：", pos.sum())
print("ffill：", pos.ffill().tolist())

turnover = pos.fillna(0.0).diff()
print("diff 后第一行是 NaN：", turnover.iloc[0])
print(
    "fillna(用另一个序列) 补上第 0 天的建仓：",
    pos.fillna(0.0).diff().fillna(pos.fillna(0.0)).tolist(),
)

# 幸存者偏差：B 中途退市（之后全是 NaN），dropna(axis=1) 会把它整列删掉
universe = pd.DataFrame({"A": [1.0, 1.1, 1.2, 1.3], "B": [1.0, 0.8, np.nan, np.nan]})
print("\ndropna(axis=1) 之后剩下的列：", list(universe.dropna(axis=1).columns))

# %% [markdown]
# ## 6. 累计：`cumprod`、`cumsum`、`cummax`
#
# - `(1 + r).cumprod()`：把简单收益连乘成**净值曲线**。
# - `log_r.cumsum()`：对数收益可以直接累加。
# - `wealth.cummax()`：截至每一天的**历史最高点**，用来算回撤。
#
# 回撤 = 净值 / 历史最高点 - 1。`quantlab.perf.drawdown` 还把期初资金 1.0 也算作一个高点，
# 否则第一天就亏的情况不会被记作回撤（lab01）。
#
# 出现的地方：lab00（总收益、净值）、lab01（回撤）、lab02、lab07。

# %%
r = px.pct_change().dropna()
wealth = (1 + r).cumprod()
peak = wealth.cummax().clip(lower=1.0)
dd = wealth / peak - 1
print(pd.DataFrame({"r": r, "wealth": wealth, "peak": peak, "drawdown": dd}))
print("最大回撤：", -dd.min())
print(
    "对数收益累加 == 累计对数收益：",
    np.isclose(np.log1p(r).cumsum().iloc[-1], np.log(wealth.iloc[-1])),
)

# %% [markdown]
# ## 7. 聚合与 `axis`：`mean`、`std`、`quantile`、`skew`、`kurt`
#
# DataFrame 上的聚合都有一个 `axis` 参数，这是第二个最常见的坑：
#
# | 写法 | 沿哪个方向 | 结果 | 在这门课里的含义 |
# | --- | --- | --- | --- |
# | `df.mean()` 即 `axis=0` | 沿着**行**往下 | 每**列**一个数 | 每只股票的**时间序列**均值 |
# | `df.mean(axis=1)` | 沿着**列**往右 | 每**行**一个数 | 每天的**横截面**均值 |
#
# 其他要记住的默认值：
#
# - `std()` 默认 `ddof=1`（样本标准差，分母 n-1），numpy 的 `np.std` 默认 `ddof=0`。两者混用会有细微差别。
# - `kurt()` 返回的是**超额峰度**（正态分布为 0）；Mertens 公式要的是非超额峰度，要加 3（lab03）。
# - `quantile(0.01)` 取 1% 分位数，历史 VaR 就是它的相反数（lab01）。
#
# 出现的地方：lab02 审计题里 `mom.mean()` 默认 axis=0，用了全样本（含未来）的均值。

# %%
rets = data.correlated_gbm(n_assets=3, n_days=500, seed=1).pct_change().dropna()
print("每只股票的时间序列均值 rets.mean()：\n", rets.mean())
print("每天的横截面均值 rets.mean(axis=1)（前 3 天）：\n", rets.mean(axis=1).head(3))
print(
    "pandas std(ddof=1) vs numpy std(ddof=0)：", rets["A00"].std(), np.std(rets["A00"].to_numpy())
)
print("超额峰度 kurt()：", rets["A00"].kurt(), "| 偏度 skew()：", rets["A00"].skew())
print("1% 分位数（历史 VaR 的相反数）：", rets["A00"].quantile(0.01))

# %% [markdown]
# ## 8. 滚动、扩展与指数加权：`rolling`、`expanding`、`ewm`
#
# - `s.rolling(n).mean()`：最近 n 期（**包含当期 t**）的均值。前 n-1 个是 NaN，可以用 `min_periods` 放宽。
# - `s.expanding().mean()`：从开头到 t 的全部历史。
# - `s.ewm(span=n).mean()`：指数加权，越近的权重越大。RiskMetrics 的 λ=0.94 对应 `alpha=0.06`。
#
# **坑**：
#
# - 窗口**包含当期**。用它在 t 收盘做决策没问题，但要用来预测 t 本身的收益，就必须再 `shift(1)`。
# - `rolling(n, center=True)` 是**居中**窗口，会用到未来 n/2 期的数据——lab02、lab09 审计题里的“去噪”。
# - 在每天滚动的 n 日收益上算 t 值，相邻观测共享 n-1 天，显著性被放大约 √n 倍（lab03）。
#
# 出现的地方：lab02 均线、lab06 z-score、lab07 波动率、lab09 特征。

# %%
s = pd.Series(np.arange(1.0, 9.0), index=pd.bdate_range("2024-01-01", periods=8))
print(
    pd.DataFrame(
        {
            "s": s,
            "rolling(3)": s.rolling(3).mean(),
            "rolling(3,min_periods=1)": s.rolling(3, min_periods=1).mean(),
            "center=True 未来!": s.rolling(3, center=True).mean(),
            "expanding()": s.expanding().mean(),
            "ewm(alpha=0.5)": s.ewm(alpha=0.5).mean(),
        }
    )
)

# %% [markdown]
# ## 9. 重采样与按时间分组：`resample`、`groupby`、`day_of_week`
#
# - `s.resample("ME")`：按自然月分组，标签取月末（`ME` = month end）。后面必须接一个聚合：
#   `.last()` 取月末值，`.prod()` 连乘，`.sum()` 相加，`.mean()` 平均。
#   常用频率：`"W-FRI"` 周五、`"ME"` 月末、`"QE"` 季末、`"YE"` 年末。
# - `s.groupby(key).mean()`：按任意标签分组聚合，`resample` 就是按时间分组的特例。
# - `idx.day_of_week`：周一是 0，周日是 6。`s[s.index.day_of_week < 5]` 只保留工作日。
#
# **坑**：月收益要用 `(1 + r).resample("ME").prod() - 1`（复利），不能 `.sum()`（lab00 审计题）。
# 回撤也不能只看 `resample("ME").last()` 的月末净值——月中的最低点会被抹掉（lab01 审计题）。

# %%
daily = data.gbm_prices(n_days=300, seed=2).pct_change().dropna()
monthly_right = (1 + daily).resample("ME").prod() - 1
monthly_wrong = daily.resample("ME").sum()
print(pd.DataFrame({"复利 prod": monthly_right, "相加 sum（错）": monthly_wrong}).head(4))
print("月收益连乘回去 == 总收益？", np.isclose((1 + monthly_right).prod(), (1 + daily).prod()))

# 日历日文件只保留工作日
cal = pd.Series(1.0, index=pd.date_range("2024-01-01", periods=10, freq="D"))
print("\n10 个日历日里的工作日数：", (cal.index.day_of_week < 5).sum())

# 按任意标签分组：每个月有几个交易日
print("\ngroupby(月份).count()：\n", daily.groupby(daily.index.month).count().head(3))

# %% [markdown]
# ## 10. 横截面操作：`rank`、`corrwith`、`get_dummies`、`stack`
#
# 这一组从 lab04 开始频繁出现：每天在一群股票之间比较。
#
# - `df.rank(axis=1)`：每天对股票排名；`pct=True` 得到 0~1 之间的分位。
# - `df.corrwith(other, axis=1, method="spearman")`：每天算一次横截面秩相关，这就是 **IC**。
#   注意信号要和**下一天**的收益配对：`signal.corrwith(returns.shift(-1), ...)`，
#   这里的 `shift(-1)` 是把未来收益拿来**评估**，不是用来交易，所以是对的。
# - `pd.get_dummies(sector)`：把行业标签变成 0/1 哑变量（lab04 中性化）。
# - `df.stack()`：把“日期 × 股票”的宽表压成一列，索引变成 (日期, 股票) 两层。适合做全样本统计——
#   但要警惕：在全样本上算分位数再拿来逐日分组，就用到了未来（lab04 审计题）。

# %%
panel_info = data.factor_panel(n_assets=8, n_days=30, seed=3)
sig, ret = panel_info.signal, panel_info.returns
print("每天的排名分位（前 2 天）：\n", sig.rank(axis=1, pct=True).head(2).round(2))
ic = sig.corrwith(ret.shift(-1), axis=1, method="spearman").dropna()
print("\n每日 IC（前 5 天）：\n", ic.head())
print("\n行业哑变量：\n", pd.get_dummies(panel_info.sector).astype(int).head(4))
print("\nstack 后的形状：", sig.stack().shape, "| 索引层：", sig.stack().index.names)

# %% [markdown]
# ## 11. 条件与截断：`where`、`clip`、`abs`、`idxmax`
#
# - `s.where(cond, other)`：cond 为 True 的位置保留原值，否则替换成 other（默认 NaN）。
#   `np.where(cond, a, b)` 是 numpy 版本，返回的是数组，要自己套回索引。
# - `s.clip(lower=, upper=)`：截断到区间内。`score.clip(lower=0)` = 只保留正分（lab02 只做多）。
#   `pos.diff().clip(lower=0)` = 只看买入，这正是 lab02 审计题“只对买入收费”的错误写法。
# - `s.abs()`：绝对值，`diff().abs()` 就是买卖双边的换手。
# - `s.idxmax()`：最大值所在的**标签**（不是位置）。lab03 用它选最优回看期。

# %%
score = pd.Series([0.8, -0.3, 1.2, -1.0], index=["A", "B", "C", "D"])
print("where(score > 0, 0)：", score.where(score > 0, 0.0).tolist())
print("clip(lower=0)：", score.clip(lower=0).tolist())
print("abs()：", score.abs().tolist())
print("idxmax()（标签）：", score.idxmax(), "| argmax（位置）：", int(np.argmax(score.to_numpy())))

# %% [markdown]
# ## 12. 拼接与重排：`concat`、`reindex`、`set_index`、`rename`
#
# - `pd.concat([a, b])`：上下拼接（比如把 walk-forward 每一段的样本外收益接起来，lab03）；
#   `pd.concat([a, b], axis=1)`：左右拼接，按索引对齐。
# - `s.reindex(new_index)`：按新的索引重新排列，新索引里原来没有的标签填 NaN。
#   lab02 用它把信号对齐到价格的日期上。
# - `df.set_index("date")`：把某一列变成索引（读进来的行情数据通常要这一步，见 `quantlab.realdata`）。
# - `df.rename(columns={...})`：改列名。

# %%
seg1 = pd.Series([0.01, -0.02], index=pd.bdate_range("2024-01-01", periods=2))
seg2 = pd.Series([0.03], index=pd.bdate_range("2024-01-03", periods=1))
print("concat 上下拼接：\n", pd.concat([seg1, seg2]))
print("concat 左右拼接（按索引对齐）：\n", pd.concat([seg1.rename("a"), seg2.rename("b")], axis=1))
print("reindex 到更长的日期：\n", seg1.reindex(pd.bdate_range("2024-01-01", periods=4)))

raw = pd.DataFrame({"日期": ["2024-01-02", "2024-01-03"], "收盘": [10.0, 10.2]})
clean = raw.rename(columns={"日期": "date", "收盘": "close"}).assign(
    date=lambda d: pd.to_datetime(d["date"])
)
print("rename + set_index：\n", clean.set_index("date"))

# %% [markdown]
# ## 13. 和 numpy 互通：`to_numpy`、`@`、构造回 pandas
#
# 经验法则：**需要按日期对齐的地方用 pandas，纯数值计算（矩阵、优化、循环）转成 numpy。**
#
# - `df.to_numpy()`：取出数组，**丢掉索引**。之后的运算按位置进行，不会再对齐。
# - `X @ w`：矩阵乘法。组合收益 = 资产收益矩阵 @ 权重（lab00、lab05）。
# - `pd.Series(arr, index=df.index)`：算完再把索引套回去。忘了这一步，后面再和别的 Series 运算会对不上。
# - numpy 的 ufunc（`np.log1p`、`np.sign`、`np.exp`）作用在 Series 上会**保留索引**。
#
# **坑**：转成 numpy 之后，“按位置”是你自己保证的——两个数组的行如果对应的不是同一天，numpy 不会提醒你。

# %%
R = rets.iloc[:5]
w = np.array([0.5, 0.3, 0.2])
port_np = R.to_numpy() @ w  # numpy 数组，没有索引
port = pd.Series(port_np, index=R.index, name="portfolio")  # 套回索引
print(port)
print("np.log1p 作用在 Series 上仍是 Series：", type(np.log1p(port)).__name__)

# %% [markdown]
# ## 14. 小结：审计时的检查顺序
#
# 读一段 pandas 回测代码时，按这个顺序问：
#
# 1. **形状和索引**：每一步的输出是什么形状？索引是什么（日期？股票？）？是不是意外地全是 NaN？
# 2. **对齐**：两个对象运算时，是按标签对齐的吗？DataFrame 和 Series 之间有没有写 `axis=0`？
# 3. **时间**：有没有 `shift(-k)`、`center=True`、全样本的 `mean()`/`std()`/`quantile()`？
#    t 的信号是不是只吃 t+1 的收益？
# 4. **开头的 NaN**：`shift`/`diff`/`pct_change`/`rolling` 产生的 NaN 代表什么？被 `fillna`、`dropna`、
#    `sum` 怎样处理了？
# 5. **聚合口径**：`axis` 对不对？`ddof`、年化因子（252）、超额峰度这些默认值是不是你以为的那个？
# 6. **样本**：`dropna` 有没有悄悄删掉退市股票、截短样本？
#
# 两个通用的验证手段，适用于任何回测：在**没有信号的数据**上跑，结果应该接近 0；
# **改动未来的数据**，过去的结果必须完全不变。

# %%
