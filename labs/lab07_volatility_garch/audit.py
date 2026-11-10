"""研究报告：波动率目标 + Kelly 杠杆，“年化 15% 以上，回撤可控”。

背景：一位同事交来了这份杠杆方案——

    "先用 EWMA 波动率把组合波动率稳定在 10%，再按 Kelly 公式加杠杆。
     训练段估出的 Kelly 杠杆约 3 倍，对应预期年化增长 15% 以上；
     波动率目标后的 Sharpe 明显高于买入持有。建议按这个杠杆上线。"

你的任务是做 code review：这份方案里有若干处会让结论偏乐观的错误。
找出它们，在这个文件里改正，直到 `make test LAB=07` 全部通过。

规则：
- 不要改函数签名和 evaluate() 返回的字典键，测试依赖它们。
- 改正之前，先在下面的「审计记录」里写下你认为的问题和理由，再动代码。
- quantlab.vol 和 quantlab.perf 里的函数是正确的，可以放心调用。

审计记录（自己填）：
    1.
    2.
    3.
    ...
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab import perf

TRADING_DAYS = 252
RF = 0.02  # 年化无风险利率，也是借钱加杠杆的成本
MAX_LEVERAGE = 5.0


def vol_forecast(returns: pd.Series, lam: float = 0.94) -> pd.Series:
    """EWMA（RiskMetrics）日波动率预测，用来决定每天的仓位。"""
    r = returns.to_numpy(dtype=float)
    var = np.empty(len(r))
    var[0] = float(np.mean(r[:20] ** 2))
    for t in range(1, len(r)):
        # 用最新的信息更新方差：越新的收益越能反映当前的波动状态
        var[t] = lam * var[t - 1] + (1 - lam) * r[t] ** 2
    return pd.Series(np.sqrt(var), index=returns.index, name="sigma")


def vol_target_weights(
    returns: pd.Series, target_vol: float = 0.10, max_leverage: float = MAX_LEVERAGE
) -> pd.Series:
    """把年化波动率稳定在 target_vol 所需的仓位。"""
    sigma_annual = vol_forecast(returns) * np.sqrt(TRADING_DAYS)
    return (target_vol / sigma_annual).clip(upper=max_leverage).rename("weight")


def levered_returns(returns: pd.Series, weights: pd.Series, rf: float = RF) -> pd.Series:
    """按权重持有资产的日收益（权重 > 1 即加杠杆）。"""
    return (weights * returns).rename("levered")


def kelly_leverage(
    train_returns: pd.Series, rf: float = RF, max_leverage: float = MAX_LEVERAGE
) -> float:
    """Kelly 最优杠杆 f* = (mu - rf) / sigma^2，用训练段估计。

    Kelly 杠杆使长期复利增长率最大，是理论上最优的下注比例。
    """
    mu = train_returns.mean() * TRADING_DAYS
    var = train_returns.var(ddof=1) * TRADING_DAYS
    return float(np.clip((mu - rf) / var, 0.0, max_leverage))


def expected_growth(leverage: float, mu: float, sigma: float, rf: float = RF) -> float:
    """以 leverage 倍杠杆持有一个年化期望收益 mu、波动 sigma 的资产，预期的年化对数增长率。"""
    return rf + leverage * (mu - rf)


def evaluate(returns: pd.Series, train_frac: float = 0.5, rf: float = RF) -> dict:
    """完整的评估流程，返回研究结论。

    返回字典：
        kelly_leverage   训练段估出的杠杆
        expected_cagr    按该杠杆预期的年化复利增长率
        sharpe_buy_hold  测试段买入持有的 Sharpe（超额收益）
        sharpe_vol_target 测试段波动率目标的 Sharpe（超额收益）
        vol_of_vol_buy_hold / vol_of_vol_target  测试段 63 日滚动年化波动率的标准差
    """
    split = int(len(returns) * train_frac)
    train, test = returns.iloc[:split], returns.iloc[split:]

    f = kelly_leverage(train, rf)
    mu_hat = train.mean() * TRADING_DAYS
    sigma_hat = train.std(ddof=1) * np.sqrt(TRADING_DAYS)
    growth = expected_growth(f, mu_hat, sigma_hat, rf)

    w = vol_target_weights(returns)
    vt = levered_returns(returns, w, rf).iloc[split:]

    def vol_of_vol(r: pd.Series) -> float:
        return float((r.rolling(63).std() * np.sqrt(TRADING_DAYS)).std())

    return {
        "kelly_leverage": f,
        "expected_cagr": float(np.expm1(growth)),
        "sharpe_buy_hold": perf.sharpe(test, rf=rf),
        "sharpe_vol_target": perf.sharpe(vt, rf=rf),
        "vol_of_vol_buy_hold": vol_of_vol(test),
        "vol_of_vol_target": vol_of_vol(vt),
    }
