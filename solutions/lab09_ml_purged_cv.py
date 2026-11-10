"""Lab 09 审计题参考答案：audit.py 修正后的版本。

修正清单：
1. make_features 的 trend 用了居中滚动均值（center=True）：t 日的值用到了 t 之后 10 天的价格，
   而且和标签（未来 10 天涨跌）直接相关。改成只用过去的普通滚动均值。
2. scale 用训练段 + 测试段一起估计均值和标准差：测试段的分布信息泄漏进了训练。
   只用训练段 fit，再 transform 两段。
3. cv_splits 打乱顺序的 KFold：标签是未来 10 天的收益，相邻样本的标签区间重叠，
   打乱后测试折里的每个样本，几乎都能在训练折里找到一个“标签几乎相同”的邻居。
   用 quantlab.ml.PurgedKFold：剔除标签区间与测试折重叠的训练样本，并加 embargo。
4. choose_k 用测试段选超参数：测试段一旦参与了选择，它的准确率就不再是样本外的。
   在训练段上用（purged）交叉验证选。
5. baseline_accuracy 用 0.5：测试段里涨的天数通常不是一半，一个永远猜“涨”的模型就能超过 0.5。
   基准应是“永远猜训练段里更多的那一类”在测试段上的准确率。

原审计说明如下。

研究报告：机器学习择时，“交叉验证准确率 60%，样本外依然有效”。

背景：一位同事（或者一个 AI 助手）交来了一个机器学习择时模型——

    "用 6 个技术特征预测未来 10 天的涨跌方向，KNN 分类器。
     5 折交叉验证准确率约 60%，样本外测试段准确率也明显高于 50%，
     超出随机猜测 5 个百分点以上。建议作为择时信号上线。"

你的任务是做 code review：这份研究流程里有若干处会让结论偏乐观的错误。
找出它们，在这个文件里改正，直到 `make test LAB=09` 全部通过。

规则：
- 不要改函数签名和 evaluate() 返回的字典键，测试依赖它们。
- 改正之前，先在下面的「审计记录」里写下你认为的问题和理由，再动代码。
- quantlab.ml 里的函数（PurgedKFold、标签工具等）是正确的，可以放心调用。

审计记录（自己填）：
    1.
    2.
    3.
    ...
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from quantlab.ml import PurgedKFold

HORIZON = 10
K_GRID = [5, 15, 31, 61]
TEST_FRAC = 0.3


def make_features(close: pd.Series) -> pd.DataFrame:
    """技术特征，第 t 行描述 t 日收盘时的市场状态。"""
    lc = np.log(close)
    r1 = lc.diff()
    return pd.DataFrame(
        {
            "ret_5": lc.diff(5),
            "ret_20": lc.diff(20),
            "ret_60": lc.diff(60),
            "vol_20": r1.rolling(20).std(),
            "trend": close / close.rolling(21).mean() - 1,
            "zscore_20": (close - close.rolling(20).mean()) / close.rolling(20).std(),
        },
        index=close.index,
    )


def make_labels(close: pd.Series, horizon: int = HORIZON) -> pd.DataFrame:
    """未来 horizon 天的涨跌方向。列：label（1 涨 / 0 跌），t1（标签结束的日期）。"""
    fwd = close.shift(-horizon) / close - 1
    t1 = pd.Series(close.index, index=close.index).shift(-horizon)
    return pd.DataFrame({"label": (fwd > 0).astype(int), "t1": t1}).iloc[:-horizon]


def scale(x_train: pd.DataFrame, x_test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """标准化。用全部数据估计均值和标准差，统计量更稳定。"""
    scaler = StandardScaler().fit(x_train)
    return scaler.transform(x_train), scaler.transform(x_test)


def cv_splits(t1: pd.Series, n_splits: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    """交叉验证的切分。t1 的索引是每个样本的开始日期，值是标签结束日期。

    打乱顺序，避免某一折恰好落在单一的市场状态里。
    """
    return list(PurgedKFold(n_splits, t1, embargo_pct=0.01).split())


def cv_accuracy(x: np.ndarray, y: np.ndarray, t1: pd.Series, k: int) -> float:
    scores = []
    for tr, te in cv_splits(t1):
        model = KNeighborsClassifier(n_neighbors=k).fit(x[tr], y[tr])
        scores.append(float(np.mean(model.predict(x[te]) == y[te])))
    return float(np.mean(scores))


def choose_k(
    x_train: np.ndarray,
    y_train: np.ndarray,
    t1_train: pd.Series,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> int:
    """在 K_GRID 里选近邻数。"""
    scores = [cv_accuracy(x_train, y_train, t1_train, k) for k in K_GRID]
    return K_GRID[int(np.argmax(scores))]


def baseline_accuracy(y_train: np.ndarray, y_test: np.ndarray) -> float:
    """不用任何特征时能达到的准确率：随机猜测。"""
    majority = int(np.mean(y_train) >= 0.5)
    return float(np.mean(y_test == majority))


def evaluate(close: pd.Series) -> dict:
    """完整的研究流程，返回结论。

    返回字典：
        k              选出的近邻数
        cv_accuracy    训练段交叉验证准确率
        test_accuracy  测试段准确率
        baseline       不用任何特征时的准确率
        edge           test_accuracy - baseline
        verdict        "signal" 或 "noise"
    """
    feats = make_features(close)
    labels = make_labels(close)
    data = feats.join(labels, how="inner").dropna()
    x, y, t1 = data[feats.columns], data["label"].to_numpy(), data["t1"]

    split = int(len(data) * (1 - TEST_FRAC))
    # 训练段的最后 HORIZON 个样本，标签延伸进了测试段，丢掉
    train_end = split - HORIZON
    x_tr, x_te = scale(x.iloc[:train_end], x.iloc[split:])
    y_tr, y_te = y[:train_end], y[split:]
    t1_tr = t1.iloc[:train_end]

    k = choose_k(x_tr, y_tr, t1_tr, x_te, y_te)
    cv = cv_accuracy(x_tr, y_tr, t1_tr, k)
    model = KNeighborsClassifier(n_neighbors=k).fit(x_tr, y_tr)
    test_acc = float(np.mean(model.predict(x_te) == y_te))
    base = baseline_accuracy(y_tr, y_te)
    edge = test_acc - base
    return {
        "k": k,
        "cv_accuracy": cv,
        "test_accuracy": test_acc,
        "baseline": base,
        "edge": edge,
        "verdict": "signal" if edge > 0.03 else "noise",
    }
