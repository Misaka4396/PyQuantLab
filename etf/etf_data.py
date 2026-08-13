"""B1 ETF 套利数据主模块：行情 / IOPV / NAV / PCF 对齐 + 折溢价序列。

数据口径：
- ETF 分钟行情：优先 akshare（``fund_etf_hist_min_em``）；失败回退合成数据（注明 source）。
- IOPV：交易所约每 15s 发布。akshare 免费源对盘中 IOPV 覆盖有限，故标准接口为
  "行情表带 iopv 列"；缺失时用 ``etf.iopv_estimator.IOPVEstimator`` 由成分股实时合成。
- NAV：基金净值（收盘后公布），日频。
- 折溢价：
    盘中折溢价 = (ETF价 - IOPV) / IOPV
    收盘折溢价 = (ETF收盘价 - NAV) / NAV
- 停牌/涨跌停成分股标记：复用 A1 ``clean_ohlcv`` 的停牌与涨跌停状态。

所有函数在网络不可用时回退合成数据，并在返回中注明 ``source``。
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from data import schemas as sc
from data.data_loader import clean_ohlcv
from etf.pcf_parser import PCFBasket

DateLike = Union[str, pd.Timestamp]

# 折溢价列名
COL_PREM_INTRADAY = "premium_intraday"
COL_PREM_CLOSE = "premium_close"


# ---------------------------------------------------------------------------
# akshare 抓取（失败返回 None，由调用方回退合成）
# ---------------------------------------------------------------------------
def fetch_etf_minute_akshare(
    code: str,
    start: str,
    end: str,
    period: str = "1",
    retries: int = 2,
) -> Optional[pd.DataFrame]:
    """用 akshare 拉取 ETF 分钟行情（无 IOPV 列，IOPV 需另行合成/接入）。"""
    import akshare as ak

    for attempt in range(retries):
        try:
            raw = ak.fund_etf_hist_min_em(symbol=code, period=period,
                                          start_date=start, end_date=end, adjust="")
            if raw is None or raw.empty:
                return None
            df = pd.DataFrame({
                sc.COL_OPEN: pd.to_numeric(raw["开盘"], errors="coerce"),
                sc.COL_HIGH: pd.to_numeric(raw["最高"], errors="coerce"),
                sc.COL_LOW: pd.to_numeric(raw["最低"], errors="coerce"),
                sc.COL_CLOSE: pd.to_numeric(raw["收盘"], errors="coerce"),
                sc.COL_VOLUME: pd.to_numeric(raw["成交量"], errors="coerce"),
                sc.COL_AMOUNT: pd.to_numeric(raw["成交额"], errors="coerce"),
            })
            df.index = pd.to_datetime(raw["时间"])
            df.index.name = sc.COL_DATETIME
            df = df[~df.index.duplicated(keep="last")].sort_index()
            return df
        except Exception:
            if attempt < retries - 1:
                time.sleep(1)
    return None


# ---------------------------------------------------------------------------
# 合成数据（网络不可用/测试时的确定性别名数据）
# ---------------------------------------------------------------------------
def synthetic_etf_minute(
    code: str,
    start: DateLike,
    end: DateLike,
    seed: int = 1,
    freq: str = "1min",
    base_price: float = 3.0,
) -> pd.DataFrame:
    """合成 ETF 分钟行情（含 iopv 列，围绕 ETF 价小幅波动）。

    简化交易时段：保留 9:30-15:00 分钟，剔除 11:30-13:00 午休。
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    # 纯日期（00:00）输入时，扩展为当日交易时段 09:30-15:00
    if start.hour == 0 and start.minute == 0:
        start = start.replace(hour=9, minute=30)
    if end.hour == 0 and end.minute == 0:
        end = end.replace(hour=15, minute=0)
    idx = pd.date_range(start, end, freq=freq)
    mask = (idx.hour >= 9) & (idx.hour <= 15)
    mask &= ~((idx.hour == 11) & (idx.minute > 30))
    mask &= idx.hour != 12
    idx = idx[mask]

    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0, 0.0005, len(idx))
    close = base_price * np.exp(np.cumsum(rets))
    open_ = close * (1 + rng.normal(0.0, 0.0002, len(idx)))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0.0, 0.0002, len(idx))))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0.0, 0.0002, len(idx))))
    volume = rng.integers(100_000, 2_000_000, len(idx)).astype(float)
    # 合成 IOPV：围绕 ETF 价，模拟折溢价（约 ±10bp 内随机）
    iopv = close * (1 + rng.normal(0.0, 0.0003, len(idx)))

    df = pd.DataFrame({
        sc.COL_OPEN: open_,
        sc.COL_HIGH: high,
        sc.COL_LOW: low,
        sc.COL_CLOSE: close,
        sc.COL_VOLUME: volume,
        sc.COL_AMOUNT: volume * close,
        sc.COL_IOPV: iopv,
    }, index=idx)
    df.index.name = sc.COL_DATETIME
    return df


def synthetic_nav(code: str, dates: List[DateLike], seed: int = 2, base: float = 3.0) -> pd.DataFrame:
    """合成 NAV 日频序列（index=日期，列 nav/iopv 收盘参考）。"""
    rng = np.random.default_rng(seed)
    idx = pd.DatetimeIndex([pd.Timestamp(d) for d in dates])
    rets = rng.normal(0.0, 0.001, len(idx))
    nav = base * np.exp(np.cumsum(rets))
    return pd.DataFrame({
        sc.COL_NAV: nav,
        sc.COL_IOPV: nav * (1 + rng.normal(0.0, 0.0002, len(idx))),
    }, index=idx)


# ---------------------------------------------------------------------------
# 折溢价计算
# ---------------------------------------------------------------------------
def premium_intraday(etf_price: pd.Series, iopv: pd.Series) -> pd.Series:
    """盘中折溢价 = (ETF价 - IOPV) / IOPV，仅两者同时有效处计算。"""
    aligned = pd.concat([etf_price, iopv], axis=1).dropna()
    if aligned.empty:
        return pd.Series(dtype=float)
    return (aligned.iloc[:, 0] - aligned.iloc[:, 1]) / aligned.iloc[:, 1]


def premium_close(etf_close: pd.Series, nav: pd.Series) -> pd.Series:
    """收盘折溢价 = (ETF收盘价 - NAV) / NAV，仅两者同时有效处计算。"""
    aligned = pd.concat([etf_close, nav], axis=1).dropna()
    if aligned.empty:
        return pd.Series(dtype=float)
    return (aligned.iloc[:, 0] - aligned.iloc[:, 1]) / aligned.iloc[:, 1]


# ---------------------------------------------------------------------------
# 停牌/涨跌停成分股标记
# ---------------------------------------------------------------------------
def mark_untradeable_constituents(
    basket: PCFBasket,
    quotes: Dict[str, pd.DataFrame],
    limit_pct: float = 0.10,
) -> pd.DataFrame:
    """标记篮子成分股的可交易状态（停牌 / 涨跌停 / 无数据）。

    quotes：{symbol: OHLCV DataFrame（index=datetime，含 open/high/low/close/volume）}。
    返回 DataFrame：symbol / is_suspended / limit_status / tradeable。
    """
    rows = []
    for c in basket.constituents:
        df = quotes.get(c.symbol)
        if df is None or len(df) == 0:
            rows.append({"symbol": c.symbol, "is_suspended": True,
                         "limit_status": "no_data", "tradeable": False})
            continue
        cleaned = clean_ohlcv(df, limit_pct=limit_pct)
        last = cleaned.iloc[-1]
        is_susp = bool(last[sc.COL_IS_SUSPENDED])
        limit = str(last[sc.COL_LIMIT_STATUS])
        tradeable = bool((not is_susp) and (limit == sc.LIMIT_NORMAL))
        rows.append({"symbol": c.symbol, "is_suspended": is_susp,
                     "limit_status": limit, "tradeable": tradeable})
    return pd.DataFrame(rows, columns=["symbol", "is_suspended", "limit_status", "tradeable"])


# ---------------------------------------------------------------------------
# 主服务
# ---------------------------------------------------------------------------
@dataclass
class ETFDataService:
    """ETF 套利数据服务：行情/NAV 加载、折溢价序列构建。"""

    use_akshare: bool = True
    seed: int = 1

    def load_etf_quotes(
        self,
        code: str,
        start: DateLike,
        end: DateLike,
        freq: str = "1min",
    ) -> pd.DataFrame:
        """加载 ETF 分钟行情（含 iopv 列）。

        akshare 成功时无 iopv（置 NaN，注明 source='akshare'）；失败/禁用时合成
        （含合成 iopv，source='synthetic'）。
        """
        if self.use_akshare:
            raw = fetch_etf_minute_akshare(
                code, pd.Timestamp(start).strftime("%Y%m%d"),
                pd.Timestamp(end).strftime("%Y%m%d"),
            )
            if raw is not None and not raw.empty:
                raw[sc.COL_IOPV] = np.nan
                raw.attrs["source"] = "akshare"
                return raw
        df = synthetic_etf_minute(code, start, end, seed=self.seed, freq=freq)
        df.attrs["source"] = "synthetic"
        return df

    def load_nav(self, code: str, dates: List[DateLike]) -> pd.DataFrame:
        """加载 NAV 日频序列（合成；akshare 免费源对 NAV 覆盖有限，注明合成）。"""
        df = synthetic_nav(code, dates, seed=self.seed + 1)
        df.attrs["source"] = "synthetic"
        return df

    def build_premium_discount(
        self,
        quotes: pd.DataFrame,
        nav: pd.DataFrame,
    ) -> pd.DataFrame:
        """由分钟行情 + NAV 构建折溢价序列 DataFrame。

        返回 DataFrame（index=日期），列：
        - premium_intraday：当日盘中折溢价的均值（(ETF-IOPV)/IOPV 按分钟）
        - premium_close：收盘折溢价（(ETF收盘-NAV)/NAV）
        """
        intraday = premium_intraday(quotes[sc.COL_CLOSE], quotes[sc.COL_IOPV])
        intraday_daily = intraday.resample("D").mean().rename(COL_PREM_INTRADAY) if not intraday.empty else pd.Series(dtype=float)

        daily_close = quotes[sc.COL_CLOSE].resample("D").last()
        close_p = premium_close(daily_close, nav[sc.COL_NAV]).rename(COL_PREM_CLOSE)

        out = pd.concat([intraday_daily, close_p], axis=1)
        out.index = pd.DatetimeIndex(out.index).normalize()
        return out


__all__ = [
    "ETFDataService",
    "fetch_etf_minute_akshare",
    "synthetic_etf_minute",
    "synthetic_nav",
    "premium_intraday",
    "premium_close",
    "mark_untradeable_constituents",
    "COL_PREM_INTRADAY",
    "COL_PREM_CLOSE",
]
