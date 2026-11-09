"""Lab 09 · 金融机器学习：无泄漏特征、三重障碍标签、Purged K-Fold。

把每个 `raise NotImplementedError` 换成你的实现，然后 `make test LAB=09`。

核心问题：金融样本**不是 i.i.d.**。用 t 时刻起未来 h 天的收益做标签，相邻样本的标签共享
h-1 天的收益，于是 sklearn 默认的打乱 K 折会把“几乎同一个样本”同时放进训练集和测试集。
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import numpy as np
import pandas as pd


def make_features(close: pd.Series) -> pd.DataFrame:
    """只用 t 及以前信息的特征（与 close 同索引，前面不足窗口的地方为 NaN）。

    列（顺序固定）：
      ret_1     ln(C_t / C_{t-1})
      ret_5     ln(C_t / C_{t-5})
      ret_20    ln(C_t / C_{t-20})
      ret_60    ln(C_t / C_{t-60})
      vol_20    ret_1 的 20 日滚动标准差（ddof=1）
      zscore_20 (C_t - MA20_t) / STD20_t，MA/STD 为 close 的 20 日滚动均值/标准差
    """
    raise NotImplementedError


def fixed_horizon_labels(close: pd.Series, horizon: int) -> pd.DataFrame:
    """固定时间窗标签。索引为事件起点 t0（与 close 同索引，去掉最后 horizon 行）。

    列：
      ret    ln(C_{t+h} / C_t)
      label  1 若 ret > 0 否则 0
      t1     标签“结束”的时间戳，即 close.index[t + h]
    """
    raise NotImplementedError


def daily_vol(close: pd.Series, span: int = 50) -> pd.Series:
    """对数收益的指数加权标准差：np.log(close).diff().ewm(span=span).std()。用于缩放障碍宽度。"""
    raise NotImplementedError


def triple_barrier_labels(
    close: pd.Series,
    vol: pd.Series,
    pt_mult: float = 2.0,
    sl_mult: float = 2.0,
    max_holding: int = 20,
) -> pd.DataFrame:
    """三重障碍法（López de Prado, AFML 第 3 章）。

    对每个位置 i（要求 vol_i 有限且 > 0，且 i + max_holding < len(close)），看后续路径
        path_k = C_{i+k} / C_i - 1,  k = 1..max_holding
    上障碍 +pt_mult * vol_i，下障碍 -sl_mult * vol_i：
      - 先触及上障碍（path >= 上）：label = +1
      - 先触及下障碍（path <= 下）：label = -1
      - 都没触及：在 k = max_holding（垂直障碍）结束，label = 0
      （同一个 k 同时触及两个不可能，因为上 > 0 > 下。）
    返回 DataFrame，索引为事件起点时间戳 t0，列：
      t1     结束时间戳 close.index[i + k*]
      ret    path_{k*}
      label  {-1, 0, +1}
    """
    raise NotImplementedError


def average_uniqueness(t1: pd.Series, bar_index: pd.DatetimeIndex) -> pd.Series:
    """每个标签的平均唯一性。

    t1: 索引为事件起点 t0，值为结束时间 t1；事件占据 bar_index 上 [t0, t1] 闭区间的所有 bar。
    并发数 c_s = 覆盖 bar s 的事件个数。事件 i 的平均唯一性 = 它覆盖的 bar 上 1/c_s 的平均值。
    完全不重叠的事件唯一性为 1；每个都和另外 h 个重叠的事件约为 1/(h+1)。
    返回与 t1 同索引的 Series。
    提示：差分数组 + 前缀和可以 O(n) 算出并发数，再用 1/c 的前缀和求区间均值。
    """
    raise NotImplementedError


class PurgedKFold:
    """Purged K-Fold + embargo。

    - 测试集：把 n 个样本按时间顺序用 np.array_split 切成 n_splits 个连续块，
      每块轮流做测试集（不打乱）。
    - Purge：训练集中删除所有与测试集时间区间重叠的样本，即满足
          t0_i <= max(测试集 t1)  且  t1_i >= min(测试集 t0)
      的样本 i。
    - Embargo：再删除紧跟在测试块之后的 ceil(n * embargo_pct) 个样本（按位置），
      因为特征（例如 60 日动量）在测试块结束后一段时间内仍然“记得”测试期的信息。
    split 产出 (train_positions, test_positions)，都是 np.ndarray 的整数位置。
    t1 的索引为 t0，假设已按时间排序。接口与 sklearn 的 CV splitter 兼容。
    """

    def __init__(self, n_splits: int, t1: pd.Series, embargo_pct: float = 0.0):
        raise NotImplementedError

    def split(self, X=None, y=None, groups=None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        raise NotImplementedError

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        raise NotImplementedError


def cv_accuracy(model_factory: Callable, X: pd.DataFrame, y: pd.Series, splits) -> np.ndarray:
    """对每个 (train, test) 位置对：model = model_factory()，fit 训练集，测试集上算准确率。

    splits 是任何产出位置索引对的可迭代对象（sklearn 的 KFold(...).split(X) 或你的 PurgedKFold）。
    返回每折准确率组成的 ndarray。
    """
    raise NotImplementedError


def purged_train_test_split(t1: pd.Series, test_frac: float = 0.3) -> tuple[np.ndarray, np.ndarray]:
    """时间序列的样本外切分。

    最后 test_frac 比例（位置 split = int(n * (1 - test_frac)) 起）为测试集；
    训练集为 split 之前、且 t1 严格早于测试集第一个 t0 的样本（即标签窗口不伸进测试期）。
    返回 (train_positions, test_positions)。
    """
    raise NotImplementedError
