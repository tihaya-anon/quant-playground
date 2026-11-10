"""研究报告：机器学习择时，“交叉验证准确率 60%，样本外依然有效”。

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
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

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
            # 居中均线比普通均线更平滑、没有滞后，更能反映当前的趋势位置
            "trend": close / close.rolling(21, center=True).mean() - 1,
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
    scaler = StandardScaler().fit(pd.concat([x_train, x_test]))
    return scaler.transform(x_train), scaler.transform(x_test)


def cv_splits(t1: pd.Series, n_splits: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    """交叉验证的切分。t1 的索引是每个样本的开始日期，值是标签结束日期。

    打乱顺序，避免某一折恰好落在单一的市场状态里。
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    return list(kf.split(np.arange(len(t1))))


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
    best_k, best = K_GRID[0], -np.inf
    for k in K_GRID:
        model = KNeighborsClassifier(n_neighbors=k).fit(x_train, y_train)
        score = float(np.mean(model.predict(x_test) == y_test))
        if score > best:
            best_k, best = k, score
    return best_k


def baseline_accuracy(y_train: np.ndarray, y_test: np.ndarray) -> float:
    """不用任何特征时能达到的准确率：随机猜测。"""
    return 0.5


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
