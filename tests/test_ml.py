import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier

from quantlab import data
from quantlab import ml as lab

FEATURES = ["ret_1", "ret_5", "ret_20", "ret_60", "vol_20", "zscore_20"]


@pytest.fixture(scope="module")
def close():
    return data.gbm_prices(n_days=800, mu=0.0, sigma=0.2, seed=3)


def dataset(close, horizon, cols=None):
    X = lab.make_features(close)
    cols = cols or list(X.columns)
    df = X[cols].join(lab.fixed_horizon_labels(close, horizon), how="inner").dropna()
    return df[cols], df["label"], df["t1"]


# ---------------------------------------------------------------- features & labels


def test_features_values(close):
    X = lab.make_features(close)
    assert list(X.columns) == FEATURES
    assert X.index.equals(close.index)
    t = 100
    assert X["ret_5"].iloc[t] == pytest.approx(np.log(close.iloc[t] / close.iloc[t - 5]))
    assert X["ret_60"].iloc[t] == pytest.approx(np.log(close.iloc[t] / close.iloc[t - 60]))
    lr = np.log(close).diff()
    assert X["vol_20"].iloc[t] == pytest.approx(lr.iloc[t - 19 : t + 1].std())
    win = close.iloc[t - 19 : t + 1]
    assert X["zscore_20"].iloc[t] == pytest.approx((close.iloc[t] - win.mean()) / win.std())
    assert X["ret_60"].iloc[:60].isna().all()


def test_features_no_lookahead(close):
    base = lab.make_features(close)
    bumped_close = close.copy()
    bumped_close.iloc[400:] *= 1.5
    bumped = lab.make_features(bumped_close)
    pd.testing.assert_frame_equal(base.iloc[:400], bumped.iloc[:400])


def test_fixed_horizon_labels(close):
    lab_df = lab.fixed_horizon_labels(close, 10)
    assert len(lab_df) == len(close) - 10
    assert lab_df.index.equals(close.index[:-10])
    assert lab_df["t1"].iloc[0] == close.index[10]
    assert lab_df["ret"].iloc[5] == pytest.approx(np.log(close.iloc[15] / close.iloc[5]))
    assert set(lab_df["label"].unique()) <= {0, 1}
    assert (lab_df["label"] == (lab_df["ret"] > 0)).all()


def test_daily_vol(close):
    v = lab.daily_vol(close, span=30)
    expected = np.log(close).diff().ewm(span=30).std()
    pd.testing.assert_series_equal(v, expected, check_names=False)


def test_triple_barrier_handmade():
    idx = data.business_index(8)
    c = pd.Series([100, 101, 103, 99, 100, 100, 100, 100], index=idx, dtype=float)
    vol = pd.Series(0.01, index=idx)
    tb = lab.triple_barrier_labels(c, vol, pt_mult=2, sl_mult=2, max_holding=3)
    # i=0: 路径 +1%, +3%(触上) -> +1 在 idx[2]
    assert tb.loc[idx[0], "label"] == 1 and tb.loc[idx[0], "t1"] == idx[2]
    assert tb.loc[idx[0], "ret"] == pytest.approx(0.03)
    # i=2: 103 -> 99 = -3.9% 触下
    assert tb.loc[idx[2], "label"] == -1 and tb.loc[idx[2], "t1"] == idx[3]
    # i=4: 一直 100，垂直障碍
    assert tb.loc[idx[4], "label"] == 0 and tb.loc[idx[4], "t1"] == idx[7]
    assert len(tb) == 8 - 3


def test_triple_barrier_properties(close):
    vol = lab.daily_vol(close)
    tb = lab.triple_barrier_labels(close, vol, 1.0, 1.0, max_holding=10)
    assert set(tb["label"].unique()) == {-1, 0, 1}
    pos = close.index.get_indexer(tb.index)
    end = close.index.get_indexer(pd.DatetimeIndex(tb["t1"]))
    assert np.all(end > pos) and np.all(end - pos <= 10)
    v = vol.reindex(tb.index)
    assert np.all(tb.loc[tb.label == 1, "ret"] >= v[tb.label == 1] - 1e-12)
    assert np.all(tb.loc[tb.label == -1, "ret"] <= -v[tb.label == -1] + 1e-12)
    # 障碍越宽，触及越少
    wide = lab.triple_barrier_labels(close, vol, 3.0, 3.0, max_holding=10)
    assert (wide.label == 0).mean() > (tb.label == 0).mean()


# ---------------------------------------------------------------- uniqueness


def test_uniqueness_non_overlapping():
    idx = data.business_index(20)
    t1 = pd.Series(idx[[1, 4, 9]], index=idx[[0, 2, 5]])
    u = lab.average_uniqueness(t1, idx)
    assert np.allclose(u.to_numpy(), 1.0)


def test_uniqueness_handmade():
    idx = data.business_index(5)
    # A 覆盖 bar 0..2，B 覆盖 bar 2..3 -> bar2 并发 2
    t1 = pd.Series([idx[2], idx[3]], index=[idx[0], idx[2]])
    u = lab.average_uniqueness(t1, idx)
    assert u.iloc[0] == pytest.approx((1 + 1 + 0.5) / 3)
    assert u.iloc[1] == pytest.approx((0.5 + 1) / 2)


def test_uniqueness_fixed_horizon(close):
    h = 10
    t1 = lab.fixed_horizon_labels(close, h)["t1"]
    u = lab.average_uniqueness(t1, close.index)
    assert u.index.equals(t1.index)
    assert u.iloc[100:-100].to_numpy() == pytest.approx(1 / (h + 1))
    assert u.iloc[0] > u.iloc[100]


# ---------------------------------------------------------------- purged k-fold


def test_purged_kfold_structure(close):
    _, _, t1 = dataset(close, 20)
    n = len(t1)
    cv = lab.PurgedKFold(5, t1, embargo_pct=0.02)
    assert cv.get_n_splits() == 5
    splits = list(cv.split())
    assert len(splits) == 5
    tests = np.concatenate([te for _, te in splits])
    assert np.array_equal(tests, np.arange(n))  # 不打乱、覆盖全部
    t0 = t1.index.to_numpy()
    t1v = t1.to_numpy()
    embargo = int(np.ceil(n * 0.02))
    for tr, te in splits:
        assert np.all(np.diff(te) == 1)
        assert len(np.intersect1d(tr, te)) == 0
        # 训练样本的标签窗口不得与测试区间重叠
        assert not np.any((t0[tr] <= t1v[te].max()) & (t1v[tr] >= t0[te].min()))
        # embargo
        after = np.arange(te[-1] + 1, min(te[-1] + 1 + embargo, n))
        assert len(np.intersect1d(tr, after)) == 0
        # 没有删过头：离测试集足够远的样本都在训练集里
        far = np.flatnonzero((t1v < t0[te].min()) | (np.arange(n) > te[-1] + 20 + embargo))
        assert np.all(np.isin(far, tr))


def test_purged_kfold_sklearn_compatible(close):
    from sklearn.model_selection import cross_val_score

    X, y, t1 = dataset(close, 10)
    scores = cross_val_score(LogisticRegression(), X, y, cv=lab.PurgedKFold(4, t1, 0.01))
    assert len(scores) == 4


def test_cv_accuracy_matches_manual(close):
    X, y, _ = dataset(close, 10)
    splits = list(KFold(3).split(X))
    got = lab.cv_accuracy(lambda: LogisticRegression(), X, y, splits)
    tr, te = splits[1]
    m = LogisticRegression().fit(X.iloc[tr], y.iloc[tr])
    assert got[1] == pytest.approx(np.mean(m.predict(X.iloc[te]) == y.iloc[te]))
    assert len(got) == 3


def test_shuffled_kfold_leaks_purged_does_not():
    """随机游走上本无可预测性。打乱 K 折靠“邻居标签重叠”得到虚高分数，purged 不会。"""
    cols = ["ret_60", "ret_20", "vol_20"]
    shuffled, purged = [], []
    for seed in range(4):
        c = data.gbm_prices(n_days=2000, mu=0.0, sigma=0.2, seed=seed)
        X, y, t1 = dataset(c, 20, cols)
        knn = lambda: KNeighborsClassifier(3)  # noqa: E731
        shuffled.append(
            lab.cv_accuracy(knn, X, y, KFold(5, shuffle=True, random_state=0).split(X)).mean()
        )
        purged.append(lab.cv_accuracy(knn, X, y, lab.PurgedKFold(5, t1, 0.01).split(X)).mean())
    assert np.mean(shuffled) > 0.57
    assert np.mean(purged) < 0.56
    assert np.mean(shuffled) - np.mean(purged) > 0.04


# ---------------------------------------------------------------- out of sample


def test_purged_train_test_split(close):
    _, _, t1 = dataset(close, 10)
    tr, te = lab.purged_train_test_split(t1, 0.3)
    n = len(t1)
    split = int(n * 0.7)
    assert np.array_equal(te, np.arange(split, n))
    assert t1.iloc[tr].max() < t1.index[split]
    assert len(tr) == split - 10  # 最后 10 个训练样本的标签伸进了测试期


def test_out_of_sample_on_trending_data():
    """在有趋势状态的数据上，动量特征样本外应有一点（但不多的）预测力。"""
    accs, base = [], []
    for seed in range(8):
        c = data.regime_switching_prices(
            n_days=3000, seed=seed, trend_vol=0.12, trend_drift=0.4, regime_len=200
        )
        X, y, t1 = dataset(c, 10)
        tr, te = lab.purged_train_test_split(t1, 0.3)
        m = LogisticRegression(max_iter=1000).fit(X.iloc[tr], y.iloc[tr])
        accs.append(np.mean(m.predict(X.iloc[te]) == y.iloc[te]))
        base.append(np.mean(y.iloc[te] == int(y.iloc[tr].mean() > 0.5)))
    assert np.mean(accs) > 0.55
    assert np.mean(accs) > np.mean(base) + 0.02
