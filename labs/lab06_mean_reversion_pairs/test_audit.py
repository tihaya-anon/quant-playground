"""审计 audit.py 的自动检查。

建议先自己读 audit.py、写下审计记录，再跑这些测试——测试的名字本身就是提示。
每个测试都是审查任何配对交易研究时都值得做的一类检查。
"""

import numpy as np
import pandas as pd
import pytest

from labs._harness import load_impl
from quantlab import pairs

audit = load_impl(__file__, module="audit")


def no_pairs(seed):
    """20 只彼此独立的随机游走股票：没有任何一对是协整的。"""
    return pairs.simulate_universe(seed=seed)


def one_real_pair(seed):
    """同样的股票池，但 P00 与 P01 是一对真正协整、均值回复很快（半衰期 5 天）的股票。"""
    return pairs.simulate_universe(cointegrated_pair=True, half_life=5.0, seed=seed)


# ------------------------------------------------------------ 检查一：改动未来，过去不应改变


def test_past_returns_do_not_depend_on_future_prices():
    """把 t 之后的价格全部换掉，t 及之前的样本外收益必须一模一样。"""
    prices = one_real_pair(0)
    t = 1100  # 样本外区间内的某一天
    other = prices.copy()
    scale = np.random.default_rng(1).uniform(0.5, 1.5, prices.shape[1])
    other.iloc[t + 1 :] = other.iloc[t + 1 :] * scale  # 每只股票未来的价格水平各自跳一下
    a = audit.evaluate(prices)["daily_returns"].loc[: prices.index[t]]
    b = audit.evaluate(other)["daily_returns"].loc[: prices.index[t]]
    assert np.allclose(a.to_numpy(), b.to_numpy())


# ------------------------------------------------------------ 检查二：检验本身是否校准


def test_cointegration_test_false_positive_rate_is_about_five_percent():
    """两只独立的随机游走，“p < 0.05”只应出现约 5% 的次数。"""
    g = np.random.default_rng(0)
    n, hits = 300, 0
    for _ in range(n):
        y = pd.Series(np.cumsum(g.normal(0, 0.015, 750)))
        x = pd.Series(np.cumsum(g.normal(0, 0.015, 750)))
        hits += audit.cointegration_pvalue(y, x) < 0.05
    assert 0.02 < hits / n < 0.09


# ------------------------------------------------------------ 检查三：搜索的代价


def test_no_pair_is_claimed_in_a_universe_without_pairs():
    """一对协整的股票都没有时，在 190 对里“找到”的最好那对不应被宣布为协整。"""
    verdicts = [audit.evaluate(no_pairs(s))["verdict"] for s in range(6)]
    assert verdicts.count("not cointegrated") >= 5


# ------------------------------------------------------------ 检查四：成本


def test_costs_reduce_performance():
    """费率越高，样本外表现必须越差。"""
    prices = one_real_pair(1)
    free = audit.evaluate(prices, cost_bps=0.0)["oos_sharpe"]
    costly = audit.evaluate(prices, cost_bps=30.0)["oos_sharpe"]
    assert costly < free - 0.2


# ------------------------------------------------------------ 检查五：结论


@pytest.mark.parametrize("seed", range(4))
def test_real_pair_is_still_found(seed):
    """修正不能矫枉过正：一对真正协整的股票应被找到，并且样本外仍然赚钱。"""
    r = audit.evaluate(one_real_pair(seed))
    assert set(r["pair"]) == {"P00", "P01"}
    assert r["verdict"] == "cointegrated"
