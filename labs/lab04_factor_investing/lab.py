"""Lab 04 · 因子与横截面。

把每个 `raise NotImplementedError` 换成你的实现，然后 `make test LAB=04`。

数据约定（quantlab.data.factor_panel）：
- returns.loc[t]：t-1 收盘到 t 收盘的简单收益，T x N。
- signal.loc[t]：t 收盘时已知的信号，用来预测 returns.loc[t+1]。
  横截面分析里最常见的 bug 就是把 signal.loc[t] 和 returns.loc[t] 放在一起比——那是在偷看答案。
"""

from __future__ import annotations

import pandas as pd

# ---------------------------------------------------------------------------
# 1. 时间序列回归：OLS 与 CAPM
# ---------------------------------------------------------------------------


def ols(y, X, add_const: bool = True) -> dict:
    """普通最小二乘，用 numpy 手写（不要调 statsmodels / sklearn）。

    参数
        y: 长度 n 的一维数组。
        X: n x k 的二维数组，或长度 n 的一维数组（视为 k=1）。
        add_const: True 时在 X 最左边加一列 1（截距排在 params[0]）。

    返回 dict：
        "params": 系数向量 b = (X'X)^{-1} X'y
        "se":     经典（同方差）标准误 sqrt(diag((X'X)^{-1}) * s^2)，s^2 = e'e / (n - k)
                  这里 k 是 *包含* 截距在内的回归元个数。
        "tstats": params / se
        "resid":  残差 e = y - Xb
        "r2":     1 - e'e / sum((y - mean(y))^2)

    结果应与 statsmodels.OLS(y, sm.add_constant(X)).fit() 的
    params / bse / tvalues / resid / rsquared 一致。
    """
    raise NotImplementedError


def capm(asset: pd.Series, market: pd.Series, rf: float = 0.0, periods: int = 252) -> dict:
    """CAPM 回归：(r_i - rf_d) = alpha + beta (r_m - rf_d) + e，rf_d = rf / periods。

    rf 是年化无风险利率。两个 Series 先按索引对齐（取交集）并去掉 NaN。

    返回 dict：alpha（日频）、beta、t_alpha、t_beta、r2、alpha_annual（= alpha * periods）。
    """
    raise NotImplementedError


def rolling_beta(asset: pd.Series, market: pd.Series, window: int = 60) -> pd.Series:
    """滚动 beta = rolling Cov(r_i, r_m) / rolling Var(r_m)（样本协方差/方差，ddof=1）。

    与输入同索引；前 window-1 个值为 NaN；第 t 个值只用 [t-window+1, t] 这段数据。
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# 2. 信号评估：IC
# ---------------------------------------------------------------------------


def information_coefficient(signal: pd.DataFrame, returns: pd.DataFrame) -> pd.Series:
    """逐日横截面 Rank IC（Spearman 相关）。

    ic[t] = SpearmanCorr( signal.loc[t, :], returns.loc[t+1, :] )

    返回 Series，索引为 signal.index[:-1]（最后一天没有“下一天”）。
    提示：Spearman 相关 = 秩的 Pearson 相关；
    DataFrame.rank(axis=1) 与 corrwith(axis=1) 可以完全向量化。
    """
    raise NotImplementedError


def ic_summary(ic: pd.Series, periods: int = 252) -> dict:
    """IC 序列的汇总统计（先 dropna）。

    返回 dict：
        "mean":        IC 均值
        "std":         IC 标准差（ddof=1）
        "icir":        mean / std
        "icir_annual": icir * sqrt(periods)
        "t_stat":      mean / std * sqrt(n)    —— 把每天的 IC 当作独立样本
        "hit_rate":    IC > 0 的比例
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# 3. 分位数组合
# ---------------------------------------------------------------------------


def quantile_returns(
    signal: pd.DataFrame, returns: pd.DataFrame, n_quantiles: int = 5
) -> pd.DataFrame:
    """按 t 日信号把股票分成 n_quantiles 组，计算每组在 t+1 日的等权平均收益。

    分组方式：每天横截面百分位秩 p = rank / N（signal.rank(axis=1, pct=True)），
    组号 q = ceil(p * n_quantiles)，即 1 = 信号最低的一组，n_quantiles = 最高。
    N 能被 n_quantiles 整除时每组恰好 N / n_quantiles 只股票。

    返回 DataFrame：列为 1..n_quantiles（整数），索引为 returns.index[1:]
    （第 k 行是“用 index[k-1] 的信号分组、在 index[k] 实现的收益”）。
    """
    raise NotImplementedError


def long_short_returns(
    signal: pd.DataFrame, returns: pd.DataFrame, n_quantiles: int = 5
) -> pd.Series:
    """多空组合收益：最高组减最低组（quantile_returns 的最后一列减第一列）。"""
    raise NotImplementedError


# ---------------------------------------------------------------------------
# 4. 因子中性化
# ---------------------------------------------------------------------------


def cross_sectional_zscore(signal: pd.DataFrame) -> pd.DataFrame:
    """逐日横截面标准化：(x - 当日均值) / 当日标准差（ddof=1）。"""
    raise NotImplementedError


def neutralize(signal: pd.DataFrame, size: pd.Series, sector: pd.Series) -> pd.DataFrame:
    """对 size 和行业做横截面中性化：每天把信号对 [size, 行业哑变量] 回归，取残差。

        s_t = X g_t + e_t,   返回 e_t = (I - X (X'X)^{-1} X') s_t

    X 的列：size，以及每个行业一列 0/1 哑变量（全部行业都放进去，它们加起来就是常数列，
    所以不要再额外加截距，否则 X 不满秩）。size/sector 是按股票代码索引的静态属性，
    用 signal.columns 去对齐。

    性质（测试会检查）：
        - 每一天残差与 size 正交，每个行业内残差均值为 0；
        - 幂等：neutralize(neutralize(s)) == neutralize(s)。

    提示：X 不随时间变化，所以帽子矩阵 H 只需算一次，然后一个矩阵乘法处理所有日期。
    """
    raise NotImplementedError
