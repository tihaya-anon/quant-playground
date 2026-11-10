"""审计 audit.py 的自动检查。

建议先自己读 audit.py、写下审计记录，再跑这些测试——测试的名字本身就是提示。
每个测试都是审查任何金融机器学习流程时值得做的一类检查。
"""

import numpy as np
import pandas as pd
import pytest

from labs._harness import load_impl
from quantlab import data

audit = load_impl(__file__, module="audit")

N = 5000


def no_signal(seed):
    """有正漂移、但完全不可预测的价格：涨的天数多于跌的天数。"""
    return data.gbm_prices(N, mu=0.15, sigma=0.2, seed=seed)


def trending(seed):
    """有持续的趋势状态：动量类特征确实能预测未来 10 天的方向。"""
    return data.regime_switching_prices(
        N, seed=seed, trend_drift=0.4, trend_vol=0.12, regime_len=200
    )


@pytest.fixture(scope="module")
def null_reports():
    return [audit.evaluate(no_signal(s)) for s in range(6)]


@pytest.fixture(scope="module")
def signal_reports():
    return [audit.evaluate(trending(s)) for s in range(8)]


# ------------------------------------------------------------ 检查一：特征只用过去


def test_features_do_not_depend_on_future_prices():
    """把 t 之后的价格全部换掉，t 及之前的特征必须一模一样。"""
    px = no_signal(0)
    t = 2000
    other = px.copy()
    other.iloc[t + 1 :] = no_signal(9).iloc[t + 1 :].to_numpy()
    a = audit.make_features(px).iloc[: t + 1]
    b = audit.make_features(other).iloc[: t + 1]
    pd.testing.assert_frame_equal(a, b)


# ------------------------------------------------------------ 检查二：测试段不能影响训练


def test_training_transform_ignores_test_data():
    """预处理的参数只能从训练数据里学：换掉测试段，训练段变换后的结果不能变。"""
    feats = audit.make_features(no_signal(1)).dropna()
    tr, te = feats.iloc[:2000], feats.iloc[2500:]
    a, _ = audit.scale(tr, te)
    b, _ = audit.scale(tr, te * 3 + 1)
    assert np.allclose(a, b)


def test_hyperparameters_are_chosen_without_the_test_set():
    """超参数不能用测试段选：打乱甚至反转测试段的标签，选出来的超参数必须不变。"""
    feats = audit.make_features(trending(2))
    labels = audit.make_labels(trending(2))
    d = feats.join(labels, how="inner").dropna()
    x, y, t1 = d[feats.columns].to_numpy(), d["label"].to_numpy(), d["t1"]
    tr, te = slice(0, 3000), slice(3200, None)
    rng = np.random.default_rng(0)
    variants = [y[te], 1 - y[te]] + [rng.permutation(y[te]) for _ in range(4)]
    ks = {audit.choose_k(x[tr], y[tr], t1.iloc[tr], x[te], v) for v in variants}
    assert len(ks) == 1


# ------------------------------------------------------------ 检查三：交叉验证的折之间不能重叠


def test_cv_training_labels_never_overlap_test_labels():
    """每个样本的标签覆盖 [开始日期, t1]。训练折里任何一个样本的标签区间，
    都不能和测试折里任何一个样本的标签区间重叠——否则训练时就“见过”了测试的答案。"""
    labels = audit.make_labels(no_signal(3)).iloc[:1500]
    t1 = labels["t1"]
    t0 = t1.index.to_numpy()
    end = t1.to_numpy()
    for train, test in audit.cv_splits(t1):
        assert len(train) and len(test)
        test_start, test_end = t0[test].min(), end[test].max()
        overlap = (t0[train] <= test_end) & (end[train] >= test_start)
        assert not overlap.any()


# ------------------------------------------------------------ 检查四：和什么比


def test_baseline_is_at_least_the_trivial_constant_prediction():
    """“不用任何特征”能达到的准确率，至少是“永远猜训练段里更多的那一类”在测试段上的准确率。"""
    rng = np.random.default_rng(4)
    y_tr = (rng.random(2000) < 0.6).astype(int)
    y_te = (rng.random(800) < 0.6).astype(int)
    constant = float(np.mean(y_te == 1))
    assert audit.baseline_accuracy(y_tr, y_te) >= constant - 1e-12


# ------------------------------------------------------------ 检查五：结论


def test_verdict_on_unpredictable_prices_is_noise(null_reports):
    assert sum(r["verdict"] == "signal" for r in null_reports) <= 1
    assert np.mean([r["edge"] for r in null_reports]) < 0.03


def test_cv_accuracy_is_honest_on_unpredictable_prices(null_reports):
    """没有可预测性时，交叉验证准确率不应明显高于“永远猜多数类”。"""
    gaps = [r["cv_accuracy"] - max(r["baseline"], 0.5) for r in null_reports]
    assert np.mean(gaps) < 0.05


def test_real_signal_is_still_found(signal_reports):
    """修正不能矫枉过正：真实存在的趋势，大多数情况下仍应被识别出来。"""
    assert sum(r["verdict"] == "signal" for r in signal_reports) >= 5
