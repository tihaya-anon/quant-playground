"""研究报告：时间序列动量在某指数上“显著有效”。

背景：一位同事（或者一个 AI 助手）交来了这份研究代码和结论——

    "在 15 年日频数据上，用前 70% 选出最优回看期，样本外 Sharpe 很高，
     PSR > 0.95，样本外 t 值 > 2，建议上线。"

你的任务是做 code review：这份评估流程里有若干处会让结论偏乐观的错误。
找出它们，在这个文件里改正，直到 `make test LAB=03` 全部通过。

规则：
- 不要改函数签名和 evaluate() 返回的字典键，测试依赖它们。
- 改正之前，先在下面的「审计记录」里写下你认为的问题和理由，再动代码。
- quantlab.stats 里的函数是正确的，可以放心调用（讲义里有它们的含义）。

审计记录（自己填）：
    1. signal * daily 前视偏差：t 收盘的信号乘 t 当天的收益，等于用收盘价决定当天的仓位。
       修正：signal.shift(1)，t 的信号只吃 t+1 的收益。
    2. 年化收益应使用交易日：日频交易日数据一年 252 期，用 sqrt(365) 把 Sharpe 高估约 20%。
    3. choose_lookback 导致测试集泄露，oos 数据参与了 lb 的选择：把全部价格交给它扫描，
       “样本外”在选参数时已经被看过。修正：只传训练段 prices.iloc[:split]。
       选定之后仍用全部价格算策略收益（每天的信号只用当天以前的价格，没有问题）。
       原则：样本外只能在最后看一次，只用于评估，不参与任何选择。
    4. oos_tstat 用重叠的 20 日滚动收益：相邻观测共享 19 天，却按 len(agg) 个独立样本
       算标准误，t 值被放大约 sqrt(20) ≈ 4.5 倍；纯噪声下约 68% 的情况“显著”（应为 5%）。
       “聚合后更稳健”是借口：聚合不创造信息，只是把同一份信息重复计算。
       修正：直接用日收益算 t 值（或只取不重叠的 20 日收益）。
    5. 名为 dsr 的数字其实是 PSR：PSR 假设这是唯一试过的策略，而这里从 50 个回看期里挑了
       最好的。纯噪声下 8 组里有 4 组 PSR > 0.95。修正：deflated_sharpe_ratio(train,
       trial_sharpes)，以“试了 50 次后最好的那个在噪声下本该有多好”为基准。
       线索：choose_lookback 返回的 trial_sharpes 原本没有被使用——算出来却没人用的值，
       往往说明某一步被跳过了。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab import stats

TRADING_DAYS = 252
LOOKBACKS = list(range(5, 255, 5))
TRAIN_FRAC = 0.7


def strategy_returns(prices: pd.Series, lookback: int) -> pd.Series:
    """时间序列动量：过去 lookback 天涨了就做多，跌了就做空。返回日收益。"""
    signal = np.sign(prices / prices.shift(lookback) - 1).fillna(0.0)
    daily = prices.pct_change().fillna(0.0)
    return (signal.shift(1) * daily).iloc[1:]


def annualized_sharpe(returns: pd.Series) -> float:
    """年化 Sharpe。"""
    r = returns.dropna()
    return float(r.mean() / r.std(ddof=1) * np.sqrt(TRADING_DAYS))


def choose_lookback(prices: pd.Series, lookbacks=LOOKBACKS) -> tuple[int, list[float]]:
    """在给定价格上扫描回看期，返回 (最优回看期, 每个回看期的每期 Sharpe)。"""
    trial_sharpes = [stats.per_period_sharpe(strategy_returns(prices, lb)) for lb in lookbacks]
    best = lookbacks[int(np.argmax(trial_sharpes))]
    return best, trial_sharpes


def oos_tstat(returns: pd.Series, horizon: int = 20) -> float:
    """样本外均值收益的 t 统计量。"""
    agg = returns.dropna()
    return float(agg.mean() / agg.std(ddof=1) * np.sqrt(len(agg)))


def evaluate(prices: pd.Series, lookbacks=LOOKBACKS, train_frac: float = TRAIN_FRAC) -> dict:
    """完整的评估流程，返回研究结论。

    返回字典：
        lookback     选出的回看期
        is_sharpe    训练段年化 Sharpe
        oos_sharpe   样本外年化 Sharpe
        oos_tstat    样本外 t 统计量
        dsr          “选出的策略真实 Sharpe > 0”的概率（已考虑多重检验）
        verdict      "signal" 或 "noise"
    """
    split = int(len(prices) * train_frac)
    best, trial_sharpes = choose_lookback(prices.iloc[:split], lookbacks)

    rets = strategy_returns(prices, best)
    train, test = rets.iloc[: split - 1], rets.iloc[split - 1 :]

    # DSR > 0.95：考虑了试过的全部回看期之后，真实 Sharpe 仍大于 0 的概率超过 95%
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
