import numpy as np
import pandas as pd
import pytest

from labs._harness import load_impl
from quantlab import data, perf

lab = load_impl(__file__)

COLUMNS = ["position", "asset_ret", "gross", "turnover", "cost", "net"]


def series(values):
    return pd.Series(values, index=data.business_index(len(values)), dtype=float)


@pytest.fixture(scope="module")
def gbm():
    return data.gbm_prices(n_days=2520, mu=0.05, sigma=0.2, seed=21)


# ---------------------------------------------------------------- 基本机制


def test_signal_to_position():
    s = series([1, 1, -1, 0, 1])
    p = lab.signal_to_position(s)
    assert p.index.equals(s.index)
    assert p.tolist() == [0, 1, 1, -1, 0]


def test_turnover():
    p = series([0, 1, 1, -1, -1, 0.5])
    assert lab.turnover(p).tolist() == [0, 1, 0, 2, 0, 1.5]
    # 第一期之前仓位为 0
    assert lab.turnover(series([1, 1])).tolist() == [1, 0]


def test_backtest_hand():
    prices = series([100, 110, 99, 99, 108.9])
    signal = series([1, -1, -1, 0, 1])
    res = lab.backtest(prices, signal, cost_bps=10)
    assert list(res.columns) == COLUMNS
    assert res.index.equals(prices.index[1:])
    assert res["position"].tolist() == [1, -1, -1, 0]
    assert np.allclose(res["asset_ret"], [0.1, -0.1, 0.0, 0.1])
    assert np.allclose(res["gross"], [0.1, 0.1, 0.0, 0.0])
    assert np.allclose(res["turnover"], [1, 2, 0, 1])
    assert np.allclose(res["cost"], [0.001, 0.002, 0.0, 0.001])
    assert np.allclose(res["net"], res["gross"] - res["cost"])


def test_backtest_reindexes_signal(gbm):
    sig = pd.Series(1.0, index=gbm.index[100:200])  # 只在部分日期有信号
    res = lab.backtest(gbm, sig)
    assert res.index.equals(gbm.index[1:])
    assert res["position"].iloc[:100].eq(0).all()
    assert res["position"].iloc[100:200].eq(1).all()


def test_buy_and_hold_equals_asset(gbm):
    res = lab.backtest(gbm, pd.Series(1.0, index=gbm.index))
    # 第 0 天收盘建仓，从第 1 天起与资产本身完全一致；建仓那一下产生 1 倍换手
    assert np.allclose(res["gross"], gbm.pct_change().iloc[1:])
    assert res["turnover"].iloc[0] == 1
    assert res["turnover"].iloc[1:].eq(0).all()


# ---------------------------------------------------------------- 前视偏差


def test_lookahead_is_not_profitable(gbm):
    """“偷看”今天收益决定今天仓位的信号：正确回测下它没有任何优势。"""
    peek = np.sign(gbm.pct_change()).fillna(0.0)
    res = lab.backtest(gbm, peek)
    assert abs(perf.sharpe(res["gross"])) < 1.0
    # 而没 shift 的“回测”会给出荒谬的 Sharpe
    wrong = peek * gbm.pct_change().fillna(0.0)
    assert perf.sharpe(wrong) > 10


def test_backtest_uses_only_past(gbm):
    """把未来的价格改掉，过去的回测结果不能变。"""
    sig = lab.sma_crossover_signal(gbm, 10, 50)
    res_full = lab.backtest(gbm, sig)
    tampered = gbm.copy()
    tampered.iloc[1500:] *= np.linspace(1, 3, len(gbm) - 1500)
    sig2 = lab.sma_crossover_signal(tampered, 10, 50)
    res_tampered = lab.backtest(tampered, sig2)
    assert np.allclose(res_full["net"].iloc[:1498], res_tampered["net"].iloc[:1498])


# ---------------------------------------------------------------- 信号


def test_sma_crossover_signal(gbm):
    sig = lab.sma_crossover_signal(gbm, 20, 100)
    assert sig.index.equals(gbm.index)
    assert (sig.iloc[:99] == 0).all()
    assert set(sig.iloc[99:].unique()) <= {-1.0, 1.0}
    f, s = gbm.rolling(20).mean(), gbm.rolling(100).mean()
    t = 500
    assert sig.iloc[t] == (1.0 if f.iloc[t] > s.iloc[t] else -1.0)
    lo = lab.sma_crossover_signal(gbm, 20, 100, long_only=True)
    assert set(lo.unique()) <= {0.0, 1.0}
    assert ((lo == 1) == (sig == 1)).all()


def test_momentum_signal(gbm):
    sig = lab.momentum_signal(gbm, 60)
    assert (sig.iloc[:60] == 0).all()
    t = 300
    assert sig.iloc[t] == np.sign(gbm.iloc[t] / gbm.iloc[t - 60] - 1)


def _avg_sharpe(make_prices, make_signal, seeds):
    out = []
    for seed in seeds:
        px = make_prices(seed)
        out.append(perf.sharpe(lab.backtest(px, make_signal(px))["gross"]))
    return float(np.mean(out))


def test_trend_following_works_on_trending_data():
    def gen(s):
        return data.regime_switching_prices(seed=s)

    sma = _avg_sharpe(gen, lambda px: lab.sma_crossover_signal(px, 20, 100), range(8))
    mom = _avg_sharpe(gen, lambda px: lab.momentum_signal(px, 60), range(8))
    assert sma > 0.3
    assert mom > 0.3


def test_trend_following_is_noise_on_gbm():
    def gen(s):
        return data.gbm_prices(mu=0.0, sigma=0.2, seed=100 + s)

    sma = _avg_sharpe(gen, lambda px: lab.sma_crossover_signal(px, 20, 100), range(12))
    assert abs(sma) < 0.25


# ---------------------------------------------------------------- 成本与换手


def test_costs_and_breakeven():
    px = data.regime_switching_prices(seed=1)
    res0 = lab.backtest(px, lab.momentum_signal(px, 20), cost_bps=0)
    res = lab.backtest(px, lab.momentum_signal(px, 20), cost_bps=5)
    assert np.allclose(res0["gross"], res["gross"])
    assert res["net"].sum() < res0["net"].sum()
    be = lab.breakeven_cost_bps(res0)
    assert be > 0
    at_be = lab.backtest(px, lab.momentum_signal(px, 20), cost_bps=be)
    assert at_be["net"].mean() == pytest.approx(0, abs=1e-12)


def test_annualized_turnover():
    px = series(np.linspace(100, 120, 253))
    # 每天在 +1 / -1 之间翻：除第一天外每天换手 2
    sig = series([(-1) ** i for i in range(253)])
    res = lab.backtest(px, sig)
    assert lab.annualized_turnover(res) == pytest.approx((1 + 2 * 251) / 252 * 252)


def test_faster_signal_trades_more():
    px = data.regime_switching_prices(seed=2)
    fast = lab.backtest(px, lab.momentum_signal(px, 5))
    slow = lab.backtest(px, lab.momentum_signal(px, 120))
    assert lab.annualized_turnover(fast) > 3 * lab.annualized_turnover(slow)


# ---------------------------------------------------------------- 波动率目标


def test_vol_target_signal():
    px = data.garch_returns(n_days=3000, seed=4)["ret"].add(1).cumprod().mul(100)
    raw = pd.Series(1.0, index=px.index)
    sig = lab.vol_target_signal(raw, px, target_vol=0.10, window=60)
    assert (sig.iloc[:60] == 0).all()
    assert (sig <= 3.0 + 1e-12).all()
    res = lab.backtest(px, sig)
    realized = perf.annualized_vol(res["gross"].iloc[100:])
    assert realized == pytest.approx(0.10, rel=0.25)
    # 波动目标化后，滚动波动应该比买入持有稳定得多
    bh = px.pct_change().iloc[100:]
    vt = res["gross"].iloc[100:]
    assert vt.rolling(63).std().std() / vt.std() < bh.rolling(63).std().std() / bh.std()
