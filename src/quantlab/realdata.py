"""真实行情（可选，用于 lab 的“拓展练习”）。

需要 `data` 这个 extra（镜像里已经装好）：
    uv sync --extra data

两个数据源：
- akshare：A 股 / 指数 / 基金，国内网络直连即可。
- yfinance：美股等，需要能访问 Yahoo（通常要代理）。

下载结果缓存成 parquet 放在 data/cache/ 下（已 gitignore），同样的请求第二次直接读盘。
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pandas as pd

CACHE_DIR = Path(
    os.environ.get("QUANT_DATA_CACHE", Path(__file__).resolve().parents[2] / "data" / "cache")
)


def _cache_path(*parts: str) -> Path:
    key = hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]
    return CACHE_DIR / f"{parts[0]}-{key}.parquet"


def _cached(path: Path, fetch) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)
    df = fetch()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return df


def a_share_daily(
    symbol: str, start: str = "20150101", end: str = "20251231", adjust: str = "hfq"
) -> pd.DataFrame:
    """A 股日线（默认后复权，回测应使用复权价）。symbol 形如 "600519"。

    返回列：open, high, low, close, volume，索引为 date。
    """

    def fetch() -> pd.DataFrame:
        import akshare as ak

        raw = ak.stock_zh_a_hist(
            symbol=symbol, period="daily", start_date=start, end_date=end, adjust=adjust
        )
        df = raw.rename(
            columns={
                "日期": "date",
                "开盘": "open",
                "最高": "high",
                "最低": "low",
                "收盘": "close",
                "成交量": "volume",
            }
        )[["date", "open", "high", "low", "close", "volume"]]
        df["date"] = pd.to_datetime(df["date"])
        return df.set_index("date").astype(float)

    return _cached(_cache_path("ashare", symbol, start, end, adjust), fetch)


def index_daily(symbol: str = "sh000300") -> pd.DataFrame:
    """A 股指数日线，例如沪深300 "sh000300"、中证500 "sh000905"。"""

    def fetch() -> pd.DataFrame:
        import akshare as ak

        df = ak.stock_zh_index_daily(symbol=symbol)
        df["date"] = pd.to_datetime(df["date"])
        return df.set_index("date")[["open", "high", "low", "close", "volume"]].astype(float)

    return _cached(_cache_path("index", symbol), fetch)


def yahoo_daily(
    tickers: str | list[str], start: str = "2010-01-01", end: str | None = None
) -> pd.DataFrame:
    """Yahoo Finance 复权收盘价（auto_adjust=True），返回宽表：列为 ticker。"""
    tickers = [tickers] if isinstance(tickers, str) else list(tickers)

    def fetch() -> pd.DataFrame:
        import yfinance as yf

        df = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]
        if isinstance(df, pd.Series):
            df = df.to_frame(tickers[0])
        df.index.name = "date"
        return df

    return _cached(_cache_path("yahoo", ",".join(tickers), start, str(end)), fetch)
