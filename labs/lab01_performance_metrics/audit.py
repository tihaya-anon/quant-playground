"""风险仪表盘：给投委会看的一页风险指标。

背景：风控同事（或者一个 AI 助手）写了下面这段代码，输入是基金管理人每天发来的净值文件，
输出年化波动、Sharpe、Sortino、最大回撤、99% VaR 和 CVaR。投委会看了结论很满意：
“波动低、回撤小、尾部风险可控”。

你的任务是做 code review：这里有几处会让风险看起来比实际小的错误。
找出它们，在这个文件里改正，直到 `make test LAB=01` 全部通过。

关于输入：管理人的净值文件是**按日历日**给的——周末和节假日没有交易，那几天沿用前一个交易日的净值。

规则：
- 不要改函数签名和 dashboard() 返回的字典键，测试依赖它们。
- 改正之前，先在下面的「审计记录」里写下你认为的问题和理由，再动代码。
- quantlab.risk / quantlab.perf 里的函数是正确的，可以放心调用（讲义里有它们的含义）。

审计记录（自己填）：
    1. 日历日和交易日对不上，应该drop掉周末和节假日。
       周末沿用周五净值，产生 2/7 的 0 收益：日标准差被摊薄约 sqrt(5/7)，再乘 sqrt(252)
       年化，波动被低估约 15%，Sharpe 相应被高估。修正：按 day_of_week < 5 只保留工作日。
       局限：节假日仍会留下少量 0 收益，严格做法是用交易所日历过滤；
       不能用“净值不变”来判断非交易日，那会误删真实的平盘日。
    2. Sortino 的分母用了“亏损日收益的标准差”：只对亏损日求平均，且围绕亏损日自己的均值
       计算离散度，两者都让分母偏小、Sortino 偏高。正确的下行偏差是
       sqrt(mean(min(r - MAR, 0)^2))：对全部 n 天求平均（上涨日贡献 0），偏离基准是 MAR=0。
       实现时注意口径：公式里的 n 是观测天数 len(returns)，不是 252；结果是日频的，
       要乘 sqrt(252) 年化，才能和面板上的年化波动、Sharpe 放在一起比较。
    3. 回撤只用月末净值，借口是“避免日内噪声”。回撤是路径的性质，月中的最低点被抹掉，
       回撤只会被低估。展示频率和风险计量是两回事：用全部观测计算。
    4. VaR/CVaR 用正态参数化（均值 + 标准差），借口是“更平滑、更稳定”。公式本身没错，
       错在正态假设：日收益肥尾，正态 VaR 的样本外突破率约 1.45%（应为 1%），
       CVaR 低估约 35%（2.55% vs 实际 3.93%），因为 CVaR 正好落在正态分布描述最差的尾部。
       这类错误读代码看不出来，只能拿实际数据回测。修正：历史分位数作 VaR，
       低于分位数的平均损失作 CVaR。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252
ALPHA = 0.01  # 99% VaR / CVaR


def daily_returns(nav: pd.Series) -> pd.Series:
    """每日收益。"""
    # 只保留交易日：周末沿用周五净值，会产生一串 0 收益
    trading = nav[nav.index.day_of_week < 5]
    return trading.pct_change().dropna()


def ann_vol(returns: pd.Series) -> float:
    return float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS))


def sharpe(returns: pd.Series) -> float:
    return float(returns.mean() / returns.std(ddof=1) * np.sqrt(TRADING_DAYS))


def sortino(returns: pd.Series) -> float:
    """Sortino：只把下跌算作风险。"""
    MAR = 0
    n = len(returns)
    downside = returns[returns < MAR]
    dd = np.sqrt(downside.pow(2).sum() / n)
    sortino = (returns.mean() - MAR) / dd
    return float(sortino) * np.sqrt(TRADING_DAYS)


def max_drawdown(nav: pd.Series) -> float:
    """最大回撤（正数）。"""
    return float(-(nav / nav.cummax() - 1).min())


def var_cvar(returns: pd.Series, alpha: float = ALPHA) -> tuple[float, float]:
    """单日 VaR 与 CVaR（正数表示损失）。"""
    q = np.quantile(returns.dropna().to_numpy(), alpha)
    return float(-q), float(-returns[returns <= q].mean())


def dashboard(nav: pd.Series) -> dict:
    """返回字典：ann_vol, sharpe, sortino, max_drawdown, var_99, cvar_99。"""
    r = daily_returns(nav)
    var, cvar = var_cvar(r)
    return {
        "ann_vol": ann_vol(r),
        "sharpe": sharpe(r),
        "sortino": sortino(r),
        "max_drawdown": max_drawdown(nav),
        "var_99": var,
        "cvar_99": cvar,
    }
