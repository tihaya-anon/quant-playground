"""审计 audit.py 的自动检查。

建议先自己读 audit.py、写下审计记录，再跑这些测试——测试的名字本身就是提示。
每个测试都是审查任何回测时都值得做的一类检查，可以带到以后的项目里。
"""

import numpy as np
import pandas as pd
import pytest

from labs._harness import load_impl
from quantlab import data, perf

audit = load_impl(__file__, module="audit")

N_DAYS = 3780  # 15 年
NULL_SEEDS = range(8)
SIGNAL_SEEDS = range(8)


def null_prices(seed):
    """没有任何可预测性的价格（零漂移 GBM）。"""
    return data.gbm_prices(N_DAYS, mu=0.0, sigma=0.2, seed=seed)


def trending_prices(seed):
    """有真实趋势状态的价格：动量策略在这里确实有效。"""
    return data.regime_switching_prices(N_DAYS, seed=seed, trend_drift=0.4, trend_vol=0.12)


@pytest.fixture(scope="module")
def null_reports():
    return [audit.evaluate(null_prices(s)) for s in NULL_SEEDS]


@pytest.fixture(scope="module")
def signal_reports():
    return [audit.evaluate(trending_prices(s)) for s in SIGNAL_SEEDS]


# ------------------------------------------------------------ 检查一：改动未来，过去不应改变


def test_todays_position_is_decided_before_todays_close():
    """t 当天的仓位只能用 t-1 收盘时已知的信息决定。

    只改动第 t 天的收盘价（让过去 20 天的涨跌方向反转），t 当天策略持有的仓位
    （= 策略收益 / 标的收益）必须不变。
    """
    px = null_prices(0)
    mom = px / px.shift(20) - 1
    t = int(np.flatnonzero((mom > 0.02).to_numpy())[50])  # 一个 20 日动量为正的日子
    other = px.copy()
    other.iloc[t] = px.iloc[t - 20] * 0.8  # 当天暴跌，20 日动量变为负

    def position_on(prices):
        strat = audit.strategy_returns(prices, 20)
        asset = prices.pct_change()
        return strat.iloc[t - 1] / asset.iloc[t]  # strategy_returns 从第 1 天开始

    assert position_on(other) == pytest.approx(position_on(px))


def test_strategy_has_no_edge_on_unpredictable_prices():
    """在不可预测的价格上，任何不偷看未来的策略长期 Sharpe 都应接近 0。"""
    srs = [perf.sharpe(audit.strategy_returns(null_prices(s), 20)) for s in NULL_SEEDS]
    assert abs(np.mean(srs)) < 0.3


# ------------------------------------------------------------ 检查二：数字的口径


def test_annualization_matches_data_frequency():
    """年化口径要和数据频率一致：交易日数据，一年 252 期。"""
    r = null_prices(1).pct_change().dropna()
    assert audit.annualized_sharpe(r) == pytest.approx(perf.sharpe(r), rel=1e-9)


# ------------------------------------------------------------ 检查三：样本外是不是真的“外”


def test_parameter_choice_ignores_out_of_sample_data(monkeypatch):
    """选参数时不能看到样本外：evaluate 交给 choose_lookback 的数据不能越过训练段末尾。"""
    px = null_prices(2)
    split = int(len(px) * audit.TRAIN_FRAC)
    seen = []
    original = audit.choose_lookback

    def spy(prices, *args, **kwargs):
        seen.append(prices.index.max())
        return original(prices, *args, **kwargs)

    monkeypatch.setattr(audit, "choose_lookback", spy)
    audit.evaluate(px)
    assert seen, "evaluate 应该调用 choose_lookback 选参数"
    assert max(seen) <= px.index[split - 1]


# ------------------------------------------------------------ 检查四：统计量是否校准


def test_tstat_false_positive_rate_is_about_five_percent():
    """t 统计量的意义在于：没有信号时，|t| > 1.96 只应出现约 5% 的次数。"""
    rng = np.random.default_rng(0)
    idx = data.business_index(756)
    hits = 0
    n = 400
    for _ in range(n):
        r = pd.Series(rng.normal(0.0, 0.01, len(idx)), index=idx)
        hits += abs(audit.oos_tstat(r)) > 1.96
    assert 0.02 < hits / n < 0.09


def test_reported_significance_accounts_for_the_search(null_reports):
    """从很多个回看期里挑最好的，“显著”的门槛必须随之提高。

    在没有信号的数据上，报告的 dsr 极少应超过 0.95。
    """
    exceed = sum(r["dsr"] > 0.95 for r in null_reports)
    assert exceed <= 1


# ------------------------------------------------------------ 检查五：结论


def test_verdict_on_noise_is_noise(null_reports):
    assert sum(r["verdict"] == "signal" for r in null_reports) <= 1


def test_verdict_still_finds_a_real_signal(signal_reports):
    """修正不能矫枉过正：真实存在的趋势，大多数情况下仍应被识别出来。"""
    assert sum(r["verdict"] == "signal" for r in signal_reports) >= 5


def test_oos_sharpe_is_honest_on_noise(null_reports):
    assert abs(np.mean([r["oos_sharpe"] for r in null_reports])) < 0.4
