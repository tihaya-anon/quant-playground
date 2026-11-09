"""Lab 00 · 收益的算术。

把每个 `raise NotImplementedError` 换成你的实现，然后 `make test LAB=00`。
约定：一年 252 个交易日；“收益”不加说明时指简单收益。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def simple_returns(prices: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """简单收益 r_t = P_t / P_{t-1} - 1。

    去掉第一行（没有前一天价格的那个 NaN），输出比输入短 1 行，索引对齐到 t。
    Series 进 Series 出，DataFrame 进 DataFrame 出（逐列计算）。
    """
    raise NotImplementedError


def log_returns(prices: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """对数收益 l_t = ln(P_t / P_{t-1})。形状约定同 simple_returns。"""
    raise NotImplementedError


def simple_to_log(r):
    """简单收益 -> 对数收益。对标量、ndarray、Series 都要能用。

    提示：数值上 np.log1p 比 np.log(1 + r) 更准（r 很小时）。
    """
    raise NotImplementedError


def log_to_simple(lr):
    """对数收益 -> 简单收益。simple_to_log 的逆。"""
    raise NotImplementedError


def cumulative_return(returns: pd.Series) -> float:
    """一串简单收益的累计收益（期末净值 / 期初净值 - 1）。"""
    raise NotImplementedError


def wealth_index(returns: pd.Series | pd.DataFrame, start: float = 1.0):
    """净值曲线：W_t = start * prod_{s<=t} (1 + r_s)。与 returns 同索引（不额外补期初那一点）。"""
    raise NotImplementedError


def resample_returns(returns: pd.Series | pd.DataFrame, rule: str = "ME"):
    """把日收益聚合到更低频率（"ME" 月末、"QE" 季末、"YE" 年末、"W-FRI" 周五……）。

    每个区间的收益 = 区间内日收益复利连乘 - 1。索引用 pandas resample 给出的区间标签。
    """
    raise NotImplementedError


def rebalanced_portfolio_returns(asset_returns: pd.DataFrame, weights) -> pd.Series:
    """每日再平衡到固定权重的组合收益。

    asset_returns: T x N 的简单收益；weights: 长度 N。
    权重不要求和为 1：和小于 1 的部分是现金，现金收益为 0。
    返回与 asset_returns 同索引的 Series。
    """
    raise NotImplementedError


def buy_and_hold_portfolio_returns(asset_returns: pd.DataFrame, weights) -> pd.Series:
    """期初按 weights 买入、之后再也不交易的组合，每日的简单收益。

    权重会随着各资产涨跌漂移。返回与 asset_returns 同索引的 Series。
    提示：先算每个资产的净值，组合净值是它们按初始权重的加权和。
    """
    raise NotImplementedError


def simulate_gbm(
    s0: float, mu: float, sigma: float, n_days: int, n_paths: int, seed: int | None = None
) -> np.ndarray:
    """模拟 GBM 价格路径（用精确解，不要用欧拉离散）。

    mu、sigma 为年化参数，dt = 1/252。
    返回形状 (n_days + 1, n_paths) 的数组，第 0 行全是 s0。
    用 np.random.default_rng(seed) 生成随机数。
    """
    raise NotImplementedError


def annualize_log_stats(log_rets: pd.Series, periods: int = TRADING_DAYS) -> tuple[float, float]:
    """由日对数收益估计 GBM 的年化 (mu, sigma)。

    注意 GBM 下对数收益的均值是 (mu - sigma^2 / 2) dt，不是 mu dt。
    sigma 用样本标准差（ddof=1）。
    """
    raise NotImplementedError


def mu_standard_error(sigma: float, n_days: int, periods: int = TRADING_DAYS) -> float:
    """mu 估计量的标准误（年化）。

    推导：日均值估计的标准误是 sigma_d / sqrt(n)，乘以 periods 年化，
    其中 sigma_d = sigma / sqrt(periods)。
    化简后你会发现它只和“数据覆盖了多少年”有关。
    """
    raise NotImplementedError
