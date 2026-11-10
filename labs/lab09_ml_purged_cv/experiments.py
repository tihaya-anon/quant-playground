# %% [markdown]
# # Lab 09 实验：机器学习与泄漏
#
# 每个实验：先在 `预测：` 后面写下答案，再运行，然后对照。猜错的地方回讲义对应小节。

# %%
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier

from quantlab import data, ml

# %% [markdown]
# ## 实验 1：重叠标签 + 打乱的交叉验证（讲义 §2）
#
# 纯随机游走，没有任何可预测性。标签是“未来 20 天涨还是跌”，特征是过去 1/5/20/60 天的收益，
# 模型是 3 近邻。用 5 折交叉验证：
#
# - 打乱顺序的普通 KFold，准确率大概多少？
# - Purged K-Fold（剔除标签区间重叠的训练样本 + embargo）呢？
#
# 预测：


# %%
def dataset(close: pd.Series, horizon: int):
    feats = ml.make_features(close)[["ret_1", "ret_5", "ret_20", "ret_60"]]
    lab = ml.fixed_horizon_labels(close, horizon)
    d = feats.join(lab, how="inner").dropna()
    return d[feats.columns], d["label"], d["t1"]


accs = {"shuffled KFold": [], "purged KFold": []}
for s in range(4):
    X, y, t1 = dataset(data.gbm_prices(2000, mu=0.0, sigma=0.2, seed=s), 20)
    knn = lambda: KNeighborsClassifier(n_neighbors=3)  # noqa: E731
    shuffled = KFold(5, shuffle=True, random_state=0).split(X)
    purged = ml.PurgedKFold(5, t1, embargo_pct=0.01).split()
    accs["shuffled KFold"].append(ml.cv_accuracy(knn, X, y, shuffled).mean())
    accs["purged KFold"].append(ml.cv_accuracy(knn, X, y, purged).mean())
for k, v in accs.items():
    print(f"{k:15s} 平均准确率 {np.mean(v):.3f}")

# %% [markdown]
# 追问：为什么是近邻模型泄漏得最厉害？换成逻辑回归，差距会变大还是变小？

# %% [markdown]
# ## 实验 2：一个看起来无害的特征（讲义 §2）
#
# 两个“趋势”特征：价格相对 21 日均线的偏离。一个用普通（滞后的）均线，一个用**居中**均线
# （`rolling(21, center=True)`，画图时常用，因为它“不滞后”）。
# 它们和未来 10 天收益的相关系数分别是多少？
#
# 预测：

# %%
px = data.gbm_prices(5000, mu=0.0, sigma=0.2, seed=7)
fwd = px.shift(-10) / px - 1
trailing = px / px.rolling(21).mean() - 1
centered = px / px.rolling(21, center=True).mean() - 1
print(f"滞后均线特征 与 未来 10 天收益的相关：{trailing.corr(fwd):+.3f}")
print(f"居中均线特征 与 未来 10 天收益的相关：{centered.corr(fwd):+.3f}")

# %% [markdown]
# ## 实验 3：准确率要和什么比（讲义 §3）
#
# 一个年化漂移 15% 的随机游走（没有可预测性）。标签是未来 10 天涨跌。
# - 一个“永远猜涨”的模型，准确率是多少？
# - 一个用过去收益训练的 KNN 模型，样本外准确率是多少？它打败“永远猜涨”了吗？
#
# 预测：

# %%
X, y, t1 = dataset(data.gbm_prices(5000, mu=0.15, sigma=0.2, seed=3), 10)
split = int(len(X) * 0.7)
tr = ml.purged_train_test_split(t1, 0.3)[0]
model = KNeighborsClassifier(n_neighbors=31).fit(X.iloc[tr], y.iloc[tr])
test_acc = float(np.mean(model.predict(X.iloc[split:]) == y.iloc[split:]))
print(f"测试段里涨的比例（= 永远猜涨的准确率）：{y.iloc[split:].mean():.3f}")
print(f"KNN 样本外准确率：{test_acc:.3f}")

# %% [markdown]
# ## 实验 4：在噪声上搜索模型（讲义 §2，回顾 lab03 的多重检验）
#
# 在纯噪声数据上（只有 4 年），随机组合特征子集和近邻数，试 80 个模型，
# 按（正确的）purged CV 准确率挑出最好的。
# - 最好的那个 CV 准确率是多少？
# - 它在从没碰过的测试段上的准确率是多少？
#
# 预测：

# %%
rng = np.random.default_rng(0)
X, y, t1 = dataset(data.gbm_prices(1000, mu=0.0, sigma=0.2, seed=11), 10)
split = int(len(X) * 0.7)
Xtr, ytr, t1tr = X.iloc[: split - 10], y.iloc[: split - 10], t1.iloc[: split - 10]
trials = []
for _ in range(80):
    cols = list(rng.choice(X.columns, size=rng.integers(1, 5), replace=False))
    k = int(rng.choice([1, 3, 5, 15, 31]))
    folds = ml.PurgedKFold(5, t1tr, embargo_pct=0.01).split()
    cv = ml.cv_accuracy(lambda k=k: KNeighborsClassifier(n_neighbors=k), Xtr[cols], ytr, folds)
    trials.append((cv.mean(), cols, k))
best_cv, cols, k = max(trials, key=lambda t: t[0])
m = KNeighborsClassifier(n_neighbors=k).fit(Xtr[cols], ytr)
print(f"80 个模型中最好的 CV 准确率：{best_cv:.3f}（特征 {cols}，k={k}）")
print(f"它在测试段上的准确率：{np.mean(m.predict(X.iloc[split:][cols]) == y.iloc[split:]):.3f}")
base = max(y.iloc[split:].mean(), 1 - y.iloc[split:].mean())
print(f"测试段上“永远猜多数类”的准确率：{base:.3f}")

# %% [markdown]
# ## 实验 5：重叠标签让样本变少了多少（讲义 §2）
#
# 2000 个交易日，标签期限分别是 1、10、50 天。用平均唯一性（average uniqueness）估计，
# “等效的独立样本数”大约是多少？
#
# 预测：

# %%
close = data.gbm_prices(2000, seed=5)
for h in [1, 10, 50]:
    lab = ml.fixed_horizon_labels(close, h)
    u = ml.average_uniqueness(lab["t1"], close.index)
    print(f"期限 {h:2d} 天：{len(lab)} 个样本，平均唯一性 {u.mean():.3f}，等效样本约 {u.sum():.0f}")

# %% [markdown]
# 注：期限 1 天时唯一性是 0.5，因为这里的标签区间 [t, t1] 包含两个端点，相邻标签共享一天。
#
# 追问：如果你用 10 天期限的标签训练模型，你的“2000 个样本”实际上值多少个独立样本？
# 这对模型复杂度（比如树的深度、近邻数）意味着什么？

# %% [markdown]
# ## 做完之后
#
# 去做 [`audit.py`](audit.py)。实验 1、2、3 的问题都藏在那里。
