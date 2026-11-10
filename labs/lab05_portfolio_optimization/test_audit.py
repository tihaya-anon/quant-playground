"""审计 audit.py 的自动检查。

建议先自己读 audit.py、写下审计记录，再跑这些测试——测试的名字本身就是提示。
每个测试都是审查任何组合优化回测时都值得做的一类检查。
"""

import numpy as np
import pytest

from labs._harness import load_impl
from quantlab import portfolio

audit = load_impl(__file__, module="audit")

N_ASSETS = 30


def equal_sharpe(seed):
    """30 个资产真实 Sharpe 全部相同：历史均值里没有任何可利用的信息。"""
    return portfolio.simulate_assets(n_assets=N_ASSETS, seed=seed)


# ------------------------------------------------------------ 检查一：改动未来，过去不应改变


def test_past_weights_do_not_depend_on_future_returns():
    """把 t 之后的收益全部换掉，t 及之前的目标权重和策略收益必须一模一样。"""
    r = equal_sharpe(0)
    t = r.index[2500]
    other = r.copy()
    other.loc[t:] = equal_sharpe(99).loc[t:].to_numpy()
    a, b = audit.evaluate(r), audit.evaluate(other)
    before = a["weights"].index < t
    assert np.allclose(a["weights"].loc[before].to_numpy(), b["weights"].loc[before].to_numpy())
    ra = a["daily_returns"].loc[: r.index[2499]]
    rb = b["daily_returns"].loc[: r.index[2499]]
    assert np.allclose(ra.to_numpy(), rb.to_numpy())


# ------------------------------------------------------------ 检查二：组合本身是否合理


@pytest.mark.parametrize(
    "mu",
    [
        np.array([-0.05, -0.03, -0.02, -0.04]),  # 估计出来的收益全为负
        np.array([0.10, -0.12, 0.02, -0.08]),  # 有正有负，多空抵消
    ],
)
def test_optimal_portfolio_has_positive_expected_return(mu):
    """“最优”组合按它自己的输入，预期收益必须为正——否则反过来持有更好。"""
    cov = np.diag([0.04, 0.02, 0.03, 0.05]) + 0.005
    w = audit.tangency(mu, cov)
    assert w @ mu > 0


# ------------------------------------------------------------ 检查三：报告的风险是否诚实


def test_reported_vol_matches_realized_vol():
    """报告给投资人的波动率，应该和策略实际实现的波动率一致。"""
    e = audit.evaluate(equal_sharpe(1))
    realized = e["daily_returns"].std(ddof=1) * np.sqrt(252)
    assert e["vol"] == pytest.approx(realized, rel=0.10)


# ------------------------------------------------------------ 检查四：成本


def test_costs_reduce_performance():
    """费率越高，表现必须越差。"""
    r = equal_sharpe(2)
    assert (
        audit.evaluate(r, cost_bps=30.0)["sharpe"] < audit.evaluate(r, cost_bps=0.0)["sharpe"] - 0.1
    )


# ------------------------------------------------------------ 检查五：结论


def test_no_edge_when_history_has_no_information():
    """真实 Sharpe 全部相同时，基于历史均值的优化不应宣称跑赢等权。"""
    verdicts = [audit.evaluate(equal_sharpe(s))["verdict"] for s in range(6)]
    assert verdicts.count("no better than 1/N") >= 5
