"""Lab 03 审计题参考答案：audit.py 修正后的版本。

修正清单：
1. strategy_returns 前视偏差：t 收盘的信号只能用于 t+1 的收益，信号要 shift(1)。
2. annualized_sharpe 年化因子：日频交易日数据一年 252 期，不是 365。
3. choose_lookback 选参数用了全样本：样本外那 30% 参与了“选最优”，不再是样本外。
   只能在训练段价格上扫描。
4. oos_tstat 用了重叠的 20 日滚动收益：相邻观测共享 19 天，几乎不独立，
   却按 len(agg) 个独立样本算标准误，t 值被放大约 sqrt(20) 倍。直接用日收益算即可
   （或取不重叠的 20 日收益 agg.iloc[::20]）。
5. 多重检验：从 50 个回看期里挑出最好的再算 PSR，等于把“试了 50 次”藏起来了。
   要用 DSR，以 50 个尝试的 Sharpe 分布为基准。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab import stats

TRADING_DAYS = 252
LOOKBACKS = list(range(5, 255, 5))
TRAIN_FRAC = 0.7


def strategy_returns(prices: pd.Series, lookback: int) -> pd.Series:
    signal = np.sign(prices / prices.shift(lookback) - 1).fillna(0.0)
    daily = prices.pct_change().fillna(0.0)
    return (signal.shift(1).fillna(0.0) * daily).iloc[1:]


def annualized_sharpe(returns: pd.Series) -> float:
    r = returns.dropna()
    return float(r.mean() / r.std(ddof=1) * np.sqrt(TRADING_DAYS))


def choose_lookback(prices: pd.Series, lookbacks=LOOKBACKS) -> tuple[int, list[float]]:
    trial_sharpes = [stats.per_period_sharpe(strategy_returns(prices, lb)) for lb in lookbacks]
    best = lookbacks[int(np.argmax(trial_sharpes))]
    return best, trial_sharpes


def oos_tstat(returns: pd.Series, horizon: int = 20) -> float:
    r = returns.dropna()
    return float(r.mean() / r.std(ddof=1) * np.sqrt(len(r)))


def evaluate(prices: pd.Series, lookbacks=LOOKBACKS, train_frac: float = TRAIN_FRAC) -> dict:
    split = int(len(prices) * train_frac)
    best, trial_sharpes = choose_lookback(prices.iloc[:split], lookbacks)

    rets = strategy_returns(prices, best)
    train, test = rets.iloc[: split - 1], rets.iloc[split - 1 :]

    dsr = stats.deflated_sharpe_ratio(train, trial_sharpes)
    t = oos_tstat(test)

    return {
        "lookback": best,
        "is_sharpe": annualized_sharpe(train),
        "oos_sharpe": annualized_sharpe(test),
        "oos_tstat": t,
        "dsr": dsr,
        "verdict": "signal" if (dsr > 0.95 and t > 2.0) else "noise",
    }
