"""Lab 09 参考答案。"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import numpy as np
import pandas as pd


def make_features(close: pd.Series) -> pd.DataFrame:
    lc = np.log(close)
    r1 = lc.diff()
    feats = pd.DataFrame(
        {
            "ret_1": r1,
            "ret_5": lc.diff(5),
            "ret_20": lc.diff(20),
            "ret_60": lc.diff(60),
            "vol_20": r1.rolling(20).std(),
            "zscore_20": (close - close.rolling(20).mean()) / close.rolling(20).std(),
        },
        index=close.index,
    )
    return feats


def fixed_horizon_labels(close: pd.Series, horizon: int) -> pd.DataFrame:
    fwd = np.log(close.shift(-horizon) / close)
    t1 = pd.Series(close.index, index=close.index).shift(-horizon)
    out = pd.DataFrame({"ret": fwd, "label": (fwd > 0).astype(int), "t1": t1})
    return out.iloc[:-horizon]


def daily_vol(close: pd.Series, span: int = 50) -> pd.Series:
    return np.log(close).diff().ewm(span=span).std()


def triple_barrier_labels(
    close: pd.Series,
    vol: pd.Series,
    pt_mult: float = 2.0,
    sl_mult: float = 2.0,
    max_holding: int = 20,
) -> pd.DataFrame:
    c = close.to_numpy()
    v = vol.reindex(close.index).to_numpy()
    idx = close.index
    rows = []
    for i in range(len(c) - max_holding):
        if not np.isfinite(v[i]) or v[i] <= 0:
            continue
        up, lo = pt_mult * v[i], -sl_mult * v[i]
        path = c[i + 1 : i + 1 + max_holding] / c[i] - 1
        hit_up = np.flatnonzero(path >= up)
        hit_lo = np.flatnonzero(path <= lo)
        first_up = hit_up[0] if hit_up.size else max_holding
        first_lo = hit_lo[0] if hit_lo.size else max_holding
        if first_up == max_holding and first_lo == max_holding:
            j, label = max_holding - 1, 0
        elif first_up < first_lo:
            j, label = first_up, 1
        else:
            j, label = first_lo, -1
        rows.append((idx[i], idx[i + 1 + j], path[j], label))
    out = pd.DataFrame(rows, columns=["t0", "t1", "ret", "label"]).set_index("t0")
    out.index.name = None
    return out


def average_uniqueness(t1: pd.Series, bar_index: pd.DatetimeIndex) -> pd.Series:
    start = bar_index.get_indexer(t1.index)
    end = bar_index.get_indexer(pd.DatetimeIndex(t1.to_numpy()))
    # 差分数组：在 start 处 +1、end+1 处 -1，前缀和就是每根 bar 的并发标签数
    diff = np.zeros(len(bar_index) + 1)
    np.add.at(diff, start, 1)
    np.add.at(diff, end + 1, -1)
    conc = np.cumsum(diff)[:-1]
    inv = np.concatenate([[0.0], np.cumsum(1.0 / np.where(conc > 0, conc, np.inf))])
    u = (inv[end + 1] - inv[start]) / (end - start + 1)
    return pd.Series(u, index=t1.index, name="uniqueness")


class PurgedKFold:
    def __init__(self, n_splits: int, t1: pd.Series, embargo_pct: float = 0.0):
        self.n_splits = n_splits
        self.t1 = t1
        self.embargo_pct = embargo_pct

    def split(self, X=None, y=None, groups=None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        n = len(self.t1)
        t0 = self.t1.index.to_numpy()
        t1 = self.t1.to_numpy()
        embargo = int(np.ceil(n * self.embargo_pct))
        for test in np.array_split(np.arange(n), self.n_splits):
            lo, hi = test[0], test[-1]
            test_start, test_end = t0[lo], t1[test].max()
            overlap = (t0 <= test_end) & (t1 >= test_start)
            emb = np.zeros(n, dtype=bool)
            emb[hi + 1 : hi + 1 + embargo] = True
            keep = ~overlap & ~emb
            keep[test] = False
            yield np.flatnonzero(keep), test

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits


def cv_accuracy(model_factory: Callable, X: pd.DataFrame, y: pd.Series, splits) -> np.ndarray:
    scores = []
    Xv, yv = X.to_numpy(), y.to_numpy()
    for tr, te in splits:
        m = model_factory()
        m.fit(Xv[tr], yv[tr])
        scores.append(float(np.mean(m.predict(Xv[te]) == yv[te])))
    return np.array(scores)


def purged_train_test_split(t1: pd.Series, test_frac: float = 0.3) -> tuple[np.ndarray, np.ndarray]:
    n = len(t1)
    split = int(n * (1 - test_frac))
    test = np.arange(split, n)
    test_start = t1.index[split]
    train = np.flatnonzero(t1.to_numpy()[:split] < test_start)
    return train, test
