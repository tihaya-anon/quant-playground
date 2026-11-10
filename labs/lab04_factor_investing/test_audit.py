"""审计 audit.py 的自动检查。

建议先自己读 audit.py、写下审计记录，再跑这些测试——测试的名字本身就是提示。
每个测试都是审查任何因子研究时都值得做的一类检查。
"""

import numpy as np
import pytest

from labs._harness import load_impl
from quantlab import factors

audit = load_impl(__file__, module="audit")


def no_alpha(seed=0):
    """信号与未来收益毫无关系，也没有任何风险溢价。"""
    return factors.simulate_universe(seed=seed)


def size_only(seed):
    """信号里混入了大量小盘暴露，小盘因子有正溢价，但信号本身没有 alpha。"""
    return factors.simulate_universe(seed=seed, size_premium=0.12, size_tilt=3.0)


def with_delistings(seed):
    """信号越高的股票越容易退市（比如“便宜”的股票里有更多濒临破产的公司），信号本身没有 alpha。"""
    return factors.simulate_universe(seed=seed, delist_rate=0.05, delist_signal_link=1.0)


def real_alpha(seed):
    return factors.simulate_universe(seed=seed, ic=0.04)


# ------------------------------------------------------------ 检查一：信息是否在收益之前可得


def test_unpredictive_signal_has_zero_ic():
    """信号对未来毫无预测力时，IC 应接近 0。"""
    r = audit.evaluate(no_alpha())
    assert abs(r["ic_mean"]) < 0.01
    assert abs(r["ic_t"]) < 3


def test_todays_buckets_do_not_depend_on_future_factor_values():
    """改动 t 之后的因子值，t 及之前的分组必须完全不变。"""
    u = no_alpha()
    f = audit.build_factor(u.returns, u.signal)
    t = 1200
    other = f.copy()
    other.iloc[t + 1 :] = other.iloc[t + 1 :] * 3.0 + 2.0
    a = audit.quantile_buckets(f).iloc[: t + 1]
    b = audit.quantile_buckets(other).iloc[: t + 1]
    assert a.equals(b)


# ------------------------------------------------------------ 检查二：回测宇宙


def test_universe_contains_every_stock_alive_that_day():
    """回测每天参与排序的股票，应该是当天所有还在交易的股票，而不只是活到最后的那些。"""
    u = with_delistings(0)
    r = audit.evaluate(u)
    assert r["universe_size"] == pytest.approx(u.alive.sum(axis=1).mean(), rel=0.02)


@pytest.mark.parametrize("seed", [0, 1])
def test_delisting_losses_show_up_in_the_result(seed):
    """信号偏向的股票更容易以暴跌退市时，多空组合理应亏钱。"""
    r = audit.evaluate(with_delistings(seed))
    assert r["ls_sharpe_gross"] < 0
    assert r["verdict"] == "no alpha"


# ------------------------------------------------------------ 检查三：成本


def test_costs_reduce_net_performance():
    """费率越高，扣成本后的表现必须越差；零费率时净值等于毛收益。"""
    u = real_alpha(0)
    free = audit.evaluate(u, cost_bps=0.0)
    costly = audit.evaluate(u, cost_bps=20.0)
    assert free["ls_sharpe_net"] == pytest.approx(free["ls_sharpe_gross"])
    assert costly["ls_sharpe_net"] < free["ls_sharpe_net"] - 0.5


# ------------------------------------------------------------ 检查四：alpha 还是已知溢价


@pytest.mark.parametrize("seed", range(4))
def test_known_premium_is_not_reported_as_alpha(seed):
    """信号只是在暴露小盘因子时，控制已知因子之后的 alpha 不应显著。"""
    r = audit.evaluate(size_only(seed), cost_bps=0.0)
    assert abs(r["alpha_t"]) < 2.5


# ------------------------------------------------------------ 检查五：结论


def test_real_alpha_is_still_found():
    """修正不能矫枉过正：真实存在、扣成本后仍然可观的 alpha，应被识别出来。"""
    hits = sum(audit.evaluate(real_alpha(s))["verdict"] == "alpha" for s in range(4))
    assert hits >= 3


def test_no_alpha_verdict_on_noise():
    assert audit.evaluate(no_alpha(1))["verdict"] == "no alpha"
    assert np.isfinite(audit.evaluate(no_alpha(1))["alpha_t"])
