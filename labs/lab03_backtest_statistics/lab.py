"""Lab 03 · 回测的统计学。

把每个 `raise NotImplementedError` 换成你的实现，然后 `make test LAB=03`。

约定：
- 除非函数名或参数说明是“年化”，Sharpe 一律指 **每期**（per-period）Sharpe：
      SR = mean(r) / std(r, ddof=1)
  年化 = SR * sqrt(252)。下面的标准误公式都写在每期尺度上。
- returns 可以是 pd.Series 或 ndarray；计算前去掉 NaN。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats  # noqa: F401

TRADING_DAYS = 252
EULER_GAMMA = 0.5772156649015329


# ---------------------------------------------------------------- 给定的辅助函数（不用改）


def momentum_strategy_returns(prices: pd.Series, lookback: int) -> pd.Series:
    """时间序列动量策略的日收益（已正确 shift，无前视），索引为 prices.index[1:]。"""
    sig = np.sign(prices / prices.shift(lookback) - 1).fillna(0.0)
    return (sig.shift(1).fillna(0.0) * prices.pct_change().fillna(0.0)).iloc[1:]


# ---------------------------------------------------------------- Sharpe 的抽样分布


def per_period_sharpe(returns) -> float:
    """每期 Sharpe：mean / std(ddof=1)，忽略 NaN。"""
    raise NotImplementedError


def sharpe_se_iid(sr: float, n: int) -> float:
    """iid 正态收益下 Sharpe 估计量的渐近标准误（Lo, 2002）：

        SE(SR_hat) = sqrt( (1 + SR^2 / 2) / n )

    推导思路：delta 方法。SR = mu / sigma 是 (mu_hat, sigma_hat^2) 的函数，正态下两者渐近独立，
    Var(mu_hat) = sigma^2/n，Var(sigma_hat^2) = 2 sigma^4 / n。
    """
    raise NotImplementedError


def sharpe_se_mertens(sr: float, n: int, skew: float, kurt: float) -> float:
    """非正态（但 iid）收益下的标准误（Mertens, 2002；Bailey & López de Prado 用的就是它）：

        SE(SR_hat) = sqrt( (1 - skew * SR + (kurt - 1) / 4 * SR^2) / (n - 1) )

    kurt 是 **非超额** 峰度（正态为 3）。正态时化简为 sqrt((1 + SR^2/2) / (n-1))。
    负偏度（skew < 0）会 *放大* 标准误——卖期权式的策略，Sharpe 比看上去更不可信。
    """
    raise NotImplementedError


def bootstrap_sharpe(
    returns, n_boot: int = 2000, seed: int | None = 0, periods: int = TRADING_DAYS
) -> np.ndarray:
    """iid bootstrap：有放回地重抽 n 个收益，计算 **年化** Sharpe，重复 n_boot 次。

    返回长度 n_boot 的 ndarray。随机数用 np.random.default_rng(seed)。
    提示：一次性生成 (n_boot, n) 的索引矩阵，向量化计算，别写 Python 循环。
    """
    raise NotImplementedError


def block_bootstrap_sharpe(
    returns,
    block_size: int = 20,
    n_boot: int = 2000,
    seed: int | None = 0,
    periods: int = TRADING_DAYS,
) -> np.ndarray:
    """循环移动块 bootstrap（circular moving block bootstrap）。

    每个重抽样本由 ceil(n / block_size) 个长度为 block_size 的连续块拼成：块起点在 [0, n) 上
    均匀抽取，越过末尾时绕回开头（下标对 n 取模），拼好后截断到长度 n。返回年化 Sharpe 数组。

    为什么要块？iid bootstrap 打乱了时间顺序，把收益的自相关抹掉了。若收益正自相关
    （趋势策略很常见），真实的 Sharpe 波动比 iid 公式大，iid bootstrap 会给出过窄的置信区间。
    """
    raise NotImplementedError


def percentile_ci(samples, alpha: float = 0.05) -> tuple[float, float]:
    """百分位置信区间：(alpha/2 分位数, 1 - alpha/2 分位数)。"""
    raise NotImplementedError


# ---------------------------------------------------------------- PSR / DSR


def probabilistic_sharpe_ratio(returns, sr_benchmark: float = 0.0) -> float:
    """Probabilistic Sharpe Ratio（Bailey & López de Prado, 2012）。

        PSR(SR*) = Phi( (SR_hat - SR*) * sqrt(n - 1) / D )
        D        = sqrt(1 - g3 * SR_hat + (g4 - 1)/4 * SR_hat^2)

    SR_hat、SR* 都是每期尺度；g3 = scipy.stats.skew(r)，g4 = scipy.stats.kurtosis(r, fisher=False)
    （都用默认的有偏估计）。含义：真实 Sharpe 超过 SR* 的“置信度”。
    """
    raise NotImplementedError


def expected_max_sharpe(n_trials: int, sr_variance: float, sr_mean: float = 0.0) -> float:
    """N 个独立试验中，最大 Sharpe 估计值的期望（真实 Sharpe 全为 sr_mean 时）：

        E[max] ≈ sr_mean + sqrt(V) * ( (1 - γ) Φ^{-1}(1 - 1/N) + γ Φ^{-1}(1 - 1/(N e)) )

    γ 是 Euler–Mascheroni 常数（EULER_GAMMA），V 是各试验 Sharpe 的方差。
    这就是“试了 N 次之后，纯靠运气能达到的 Sharpe”。
    """
    raise NotImplementedError


def deflated_sharpe_ratio(returns, trial_sharpes) -> float:
    """Deflated Sharpe Ratio（Bailey & López de Prado, 2014）。

    把基准 SR* 换成 expected_max_sharpe(N, V) 的 PSR，其中 N = len(trial_sharpes)，
    V = trial_sharpes 的样本方差（ddof=1）。returns 是被选中的那个策略的收益，
    trial_sharpes 是 **所有** 试过的配置的每期 Sharpe（包括被选中的那个）。
    """
    raise NotImplementedError


# ---------------------------------------------------------------- 参数扫描与 walk-forward


def sweep_returns(prices: pd.Series, lookbacks) -> pd.DataFrame:
    """对每个回看期运行 momentum_strategy_returns。

    返回 DataFrame：列名为 lookback，行与策略收益同索引。
    """
    raise NotImplementedError


def walk_forward(
    prices: pd.Series, lookbacks, train_size: int = 756, test_size: int = 126
) -> pd.DataFrame:
    """滚动样本外评估。

    令 R = sweep_returns(prices, lookbacks)（共 m 行）。
    对 start = train_size, train_size + test_size, ...（start < m）：
      1. 训练窗口 R.iloc[start - train_size : start]，选出每期 Sharpe 最高的 lookback；
      2. 在测试窗口 R.iloc[start : start + test_size] 上使用这个 lookback 的收益。
    把所有测试窗口拼起来，返回 DataFrame：列 "ret"（样本外收益）、"lookback"（当期所选参数），
    索引为测试日期。

    注意：这里允许先在全样本上算好 R，因为动量策略在 t 日的收益只依赖 t 日之前的价格；
    真正可能造成前视的是 *参数选择*，而我们只用训练窗口做选择。
    """
    raise NotImplementedError
