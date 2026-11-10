"""审计 audit.py 的自动检查。

建议先自己读 audit.py、写下审计记录，再跑这些测试——测试的名字本身就是提示。
这几类检查适用于任何回测，包括 AI 写的：它们不关心策略是什么，只关心回测有没有偷看未来、
成本有没有算全。
"""

import numpy as np
import pandas as pd
import pytest

from labs._harness import load_impl
from quantlab import backtest as bt
from quantlab import data, perf

audit = load_impl(__file__, module="audit")


def universe(seed):
    """30 只股票，部分会在中途跌破初始价 25% 后退市（之后为 NaN）。"""
    return bt.universe_with_delistings(n_assets=30, n_days=2520, seed=seed)


def null_universe(seed):
    """30 只零漂移、互相独立的股票，没有退市：任何策略都不该有超额收益。"""
    return data.correlated_gbm(n_assets=30, n_days=2520, mu=0.0, sigma=0.35, corr=0.0, seed=seed)


# ------------------------------------------------------------ 检查一：任何时点只能用当时的数据


def test_todays_positions_ignore_todays_and_future_prices():
    """把第 t 天及以后的价格全部换掉，第 t 天及之前的持仓必须一模一样。"""
    px = null_universe(0)
    t = 1200
    other = px.copy()
    other.iloc[t:] = null_universe(1).iloc[t:].to_numpy() * (px.iloc[t - 1] / 100.0).to_numpy()
    a = audit.strategy_positions(px).iloc[: t + 1]
    b = audit.strategy_positions(other).iloc[: t + 1]
    assert np.allclose(a.to_numpy(), b.to_numpy())


def test_results_up_to_a_date_depend_only_on_data_up_to_that_date():
    """用截至某天的数据回测，结果应和完整回测在那天之前的部分完全一致。

    如果不一致，说明回测在某个时点用到了那时还不知道的信息。
    """
    for seed in [0, 1, 2]:
        px = universe(seed)
        cut = px.index[1500]
        full = audit.backtest(px).loc[:cut]
        partial = audit.backtest(px.loc[:cut])
        assert np.allclose(full["net"].to_numpy(), partial["net"].to_numpy()), f"seed={seed}"


# ------------------------------------------------------------ 检查二：没有信号的数据上没有收益


def test_no_edge_on_unpredictable_prices():
    srs = [perf.sharpe(audit.backtest(null_universe(s), cost_bps=0.0)["gross"]) for s in range(6)]
    assert abs(np.mean(srs)) < 0.3


# ------------------------------------------------------------ 检查三：成本算全


def test_round_trip_costs_both_legs():
    """空仓 -> 满仓 A -> 换成满仓 B -> 空仓：换手 1 + 2 + 1 = 4 个单位，每个单位都要付成本。"""
    idx = data.business_index(4)
    pos = pd.DataFrame({"A": [0.0, 1.0, 0.0, 0.0], "B": [0.0, 0.0, 1.0, 0.0]}, index=idx)
    costs = audit.transaction_costs(pos, cost_bps=10.0)
    assert costs.tolist() == pytest.approx([0.0, 1e-3, 2e-3, 1e-3])


def test_entering_from_cash_is_charged():
    idx = data.business_index(3)
    pos = pd.DataFrame({"A": [0.5, 0.5, 0.0]}, index=idx)
    costs = audit.transaction_costs(pos, cost_bps=20.0)
    assert costs.iloc[0] == pytest.approx(0.5 * 20.0 / 1e4)
    assert costs.iloc[2] == pytest.approx(0.5 * 20.0 / 1e4)
