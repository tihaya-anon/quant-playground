"""Lab 07 审计题参考答案：audit.py 修正后的版本。

修正清单：
1. vol_forecast 前视：决定第 t 天仓位的方差只能用到 t-1 的收益（r[t-1]，不是 r[t]）。
   用当天收益更新，等于大波动的那天“提前”降了仓，人为削掉了尾部，Sharpe 和波动稳定性都被高估。
2. levered_returns 漏了融资成本：权重 w 的组合收益是 w*r - (w-1)*rf（w>1 借钱要付利息，
   w<1 闲置资金赚无风险利率）。漏掉之后，对无风险资产加杠杆都能凭空赚钱。
3. expected_growth 漏了波动拖累：对数增长率是 rf + f(mu-rf) - f^2 sigma^2 / 2。
   杠杆越高，拖累按 f^2 增长；2 倍 Kelly 时增长率回到 rf。
4. kelly_leverage 忽略了 mu 的估计误差：mu_hat 的标准误约 sigma/sqrt(年数)，
   代入点估计的完全 Kelly 平均而言会过度下注，在估计误差下期望增长甚至为负。
   用分数 Kelly（这里取一半）；也可以对 mu 做收缩。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab import perf

TRADING_DAYS = 252
RF = 0.02
MAX_LEVERAGE = 5.0
KELLY_FRACTION = 0.5


def vol_forecast(returns: pd.Series, lam: float = 0.94) -> pd.Series:
    r = returns.to_numpy(dtype=float)
    var = np.empty(len(r))
    var[0] = float(np.mean(r[:20] ** 2))
    for t in range(1, len(r)):
        var[t] = lam * var[t - 1] + (1 - lam) * r[t - 1] ** 2
    return pd.Series(np.sqrt(var), index=returns.index, name="sigma")


def vol_target_weights(
    returns: pd.Series, target_vol: float = 0.10, max_leverage: float = MAX_LEVERAGE
) -> pd.Series:
    sigma_annual = vol_forecast(returns) * np.sqrt(TRADING_DAYS)
    return (target_vol / sigma_annual).clip(upper=max_leverage).rename("weight")


def levered_returns(returns: pd.Series, weights: pd.Series, rf: float = RF) -> pd.Series:
    return (weights * returns - (weights - 1) * rf / TRADING_DAYS).rename("levered")


def kelly_leverage(
    train_returns: pd.Series, rf: float = RF, max_leverage: float = MAX_LEVERAGE
) -> float:
    mu = train_returns.mean() * TRADING_DAYS
    var = train_returns.var(ddof=1) * TRADING_DAYS
    return float(np.clip(KELLY_FRACTION * (mu - rf) / var, 0.0, max_leverage))


def expected_growth(leverage: float, mu: float, sigma: float, rf: float = RF) -> float:
    return rf + leverage * (mu - rf) - 0.5 * leverage**2 * sigma**2


def evaluate(returns: pd.Series, train_frac: float = 0.5, rf: float = RF) -> dict:
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
