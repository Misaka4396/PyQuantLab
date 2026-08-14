"""统一数据加载入口：PIT 对齐 + 清洗 + 增量更新 + 各 parquet 表读写。

存储目录结构（data_root 默认 ./data_cache/pit）：
    ohlcv/{symbol}.parquet            行情（长表，index=datetime）
    adj_factor/{symbol}.parquet       复权因子
    constituents/{index_code}.parquet 成分股会员记录
    pcf/{etf_code}_{trade_date}.parquet  PCF 篮子文件
    nav/{etf_code}.parquet            NAV/IOPV

设计约定：
- 所有行情数据落盘时都带 ``as_of`` 时间戳（point-in-time）。
- 落盘的是"原始已归一化"数据；清洗（缺失/异常/停牌/涨跌停）在读取时进行，
  这样质量报告仍能基于原始数据统计缺失率与异常率。
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from core.exceptions import DataError
from data import schemas as sc
from data.pit import (
    compute_adj_factor,
    normalize_timestamps,
    slice_as_of,
)
from data.survivorship import build_tradeable_universe

DateLike = str | datetime | pd.Timestamp

DEFAULT_DATA_ROOT = Path("./data_cache/pit")

SUBDIR_OHLCV = "ohlcv"
SUBDIR_ADJ_FACTOR = "adj_factor"
SUBDIR_CONSTITUENTS = "constituents"
SUBDIR_PCF = "pcf"
SUBDIR_NAV = "nav"


# ---------------------------------------------------------------------------
# 清洗
# ---------------------------------------------------------------------------
def detect_limit_status(df: pd.DataFrame, limit_pct: float = 0.10) -> pd.Series:
    """按上一交易日收盘价标记涨跌停状态（近似：真实涨跌停价会四舍五入到分）。"""
    close = df[sc.COL_CLOSE]
    prev_close = close.shift(1)
    status = pd.Series(sc.LIMIT_NORMAL, index=df.index, dtype=object)
    with np.errstate(invalid="ignore"):
        up = prev_close * (1.0 + limit_pct)
        down = prev_close * (1.0 - limit_pct)
        is_up = (close >= up - 1e-9).fillna(False)
        is_down = (close <= down + 1e-9).fillna(False)
    status[is_up] = sc.LIMIT_UP
    status[is_down] = sc.LIMIT_DOWN
    return status


def clean_ohlcv(
    df: pd.DataFrame, limit_pct: float = 0.10, ffill_close: bool = True
) -> pd.DataFrame:
    """清洗 OHLCV 长表：去重排序、异常价、缺失值、停牌、涨跌停。

    处理步骤：
    1. 时间戳统一为 tz-naive，去重（keep=last）并排序；
    2. 非正价格（<=0）置为 NaN；
    3. 收盘价前向填充（停牌日沿用最近有效价），仍缺失则删除该行；
    4. open/high/low 缺失时用收盘价兜底；保证 high>=max(open,close)、low<=min(open,close)；
    5. volume/amount 缺失补 0；
    6. 标记停牌（volume==0）与涨跌停状态。
    """
    df = df.copy()
    if sc.COL_DATETIME in df.columns:
        df = df.set_index(sc.COL_DATETIME)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df.index = normalize_timestamps(df.index)
    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()

    for c in sc.OHLCV_REQUIRED:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # 非正价格视为异常，置 NaN
    for c in sc.OHLCV_REQUIRED:
        df.loc[df[c] <= 0, c] = np.nan

    if ffill_close:
        df[sc.COL_CLOSE] = df[sc.COL_CLOSE].ffill()
    df = df.dropna(subset=[sc.COL_CLOSE])

    for c in (sc.COL_OPEN, sc.COL_HIGH, sc.COL_LOW):
        df[c] = df[c].fillna(df[sc.COL_CLOSE])
    df[sc.COL_HIGH] = df[[sc.COL_HIGH, sc.COL_OPEN, sc.COL_CLOSE]].max(axis=1)
    df[sc.COL_LOW] = df[[sc.COL_LOW, sc.COL_OPEN, sc.COL_CLOSE]].min(axis=1)

    if sc.COL_VOLUME not in df.columns:
        df[sc.COL_VOLUME] = 0.0
    df[sc.COL_VOLUME] = pd.to_numeric(df[sc.COL_VOLUME], errors="coerce").fillna(0.0)
    if sc.COL_AMOUNT not in df.columns:
        df[sc.COL_AMOUNT] = 0.0
    df[sc.COL_AMOUNT] = pd.to_numeric(df[sc.COL_AMOUNT], errors="coerce").fillna(0.0)

    df[sc.COL_IS_SUSPENDED] = df[sc.COL_VOLUME] <= 0
    df[sc.COL_LIMIT_STATUS] = detect_limit_status(df, limit_pct)
    df.loc[df[sc.COL_IS_SUSPENDED], sc.COL_LIMIT_STATUS] = sc.SUSPENDED
    return df


def pivot_field(df: pd.DataFrame, field: str) -> pd.DataFrame:
    """把长表透视成宽表（行=datetime，列=symbol），供回测/组合模块使用。"""
    if field not in df.columns:
        raise DataError(f"透视字段不存在: {field}")
    return df.pivot(columns=sc.COL_SYMBOL, values=field)


def pivot_close(df: pd.DataFrame) -> pd.DataFrame:
    """宽表收盘价（行=datetime，列=symbol）。"""
    return pivot_field(df, sc.COL_CLOSE)


# ---------------------------------------------------------------------------
# 数据加载器
# ---------------------------------------------------------------------------
class DataLoader:
    """统一数据层入口：负责各 parquet 表的读写、PIT 过滤与增量更新。"""

    def __init__(self, data_root: str | Path = DEFAULT_DATA_ROOT):
        self.data_root = Path(data_root)
        self.data_root.mkdir(parents=True, exist_ok=True)

    # ------------------------------ 路径 ------------------------------
    def _subdir(self, name: str) -> Path:
        p = self.data_root / name
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _ohlcv_path(self, symbol: str) -> Path:
        return self._subdir(SUBDIR_OHLCV) / f"{symbol}.parquet"

    def _adj_path(self, symbol: str) -> Path:
        return self._subdir(SUBDIR_ADJ_FACTOR) / f"{symbol}.parquet"

    def _constituents_path(self, index_code: str) -> Path:
        return self._subdir(SUBDIR_CONSTITUENTS) / f"{index_code}.parquet"

    def _pcf_path(self, etf_code: str, trade_date: str) -> Path:
        return self._subdir(SUBDIR_PCF) / f"{etf_code}_{trade_date}.parquet"

    def _nav_path(self, etf_code: str) -> Path:
        return self._subdir(SUBDIR_NAV) / f"{etf_code}.parquet"

    # ------------------------------ OHLCV ------------------------------
    def _ensure_as_of(self, df: pd.DataFrame, default_lag: str = "0D") -> pd.DataFrame:
        """缺省时自动生成 as_of = 索引 + lag（数据须带 as_of 时间戳）。"""
        df = df.copy()
        if sc.COL_AS_OF not in df.columns:
            idx = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df.index)
            df[sc.COL_AS_OF] = idx + pd.Timedelta(default_lag)
        return df

    def _prepare_ohlcv(
        self, symbol: str, df: pd.DataFrame, default_as_of_lag: str = "0D"
    ) -> pd.DataFrame:
        """归一化 OHLCV（不落盘）：时间索引、必填列校验、补可选列、去重排序。"""
        df = df.copy()
        if sc.COL_DATETIME in df.columns:
            df = df.set_index(sc.COL_DATETIME)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df.index = normalize_timestamps(df.index)
        df.index.name = sc.COL_DATETIME

        for c in sc.OHLCV_REQUIRED:
            if c not in df.columns:
                raise DataError(f"OHLCV 缺少必需列: {c}")
        for c in sc.OHLCV_OPTIONAL:
            if c not in df.columns:
                df[c] = np.nan

        df = self._ensure_as_of(df, default_as_of_lag)
        df[sc.COL_SYMBOL] = symbol
        df = df[sc.OHLCV_STORED_COLUMNS]
        df = df[~df.index.duplicated(keep="last")].sort_index()
        return df

    def save_ohlcv(
        self, symbol: str, df: pd.DataFrame, default_as_of_lag: str = "0D"
    ) -> pd.DataFrame:
        """落盘 OHLCV（覆盖写）。返回归一化后的 DataFrame。"""
        prepared = self._prepare_ohlcv(symbol, df, default_as_of_lag)
        prepared.to_parquet(self._ohlcv_path(symbol), index=True)
        return prepared

    def load_ohlcv(
        self,
        symbol: str,
        start: DateLike | None = None,
        end: DateLike | None = None,
        as_of: DateLike | None = None,
        clean: bool = True,
    ) -> pd.DataFrame:
        """读取 OHLCV，支持时间区间与 as_of 过滤，按需清洗。

        - as_of：只返回 ``as_of <= as_of`` 的行（防未来函数）。
        - clean=True：应用 clean_ohlcv（补停牌/涨跌停标记）。
        """
        path = self._ohlcv_path(symbol)
        if not path.exists():
            return self._empty_ohlcv()
        df = pd.read_parquet(path)
        if start is not None:
            df = df[df.index >= pd.Timestamp(start)]
        if end is not None:
            df = df[df.index <= pd.Timestamp(end)]
        if as_of is not None:
            df = slice_as_of(df, as_of)
        if clean and not df.empty:
            df = clean_ohlcv(df)
        return df

    def load_ohlcv_many(
        self,
        symbols: Iterable[str],
        start: DateLike | None = None,
        end: DateLike | None = None,
        as_of: DateLike | None = None,
        clean: bool = True,
    ) -> pd.DataFrame:
        """批量读取多只证券，纵向拼接为长表。"""
        frames = [
            self.load_ohlcv(s, start=start, end=end, as_of=as_of, clean=clean) for s in symbols
        ]
        frames = [f for f in frames if not f.empty]
        if not frames:
            return self._empty_ohlcv()
        return pd.concat(frames).sort_index()

    def incremental_update_ohlcv(
        self,
        symbol: str,
        new_df: pd.DataFrame,
        default_as_of_lag: str = "0D",
    ) -> pd.DataFrame:
        """增量更新：合并新 bar，按时间戳去重（保留最新），覆盖写回。"""
        prepared = self._prepare_ohlcv(symbol, new_df, default_as_of_lag)
        existing = self.load_ohlcv(symbol, clean=False)
        combined = prepared if existing.empty else pd.concat([existing, prepared])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
        combined.to_parquet(self._ohlcv_path(symbol), index=True)
        return combined

    def list_ohlcv_symbols(self) -> list[str]:
        """返回已落盘的全部行情代码。"""
        d = self.data_root / SUBDIR_OHLCV
        if not d.exists():
            return []
        return sorted(p.stem for p in d.glob("*.parquet"))

    def _empty_ohlcv(self) -> pd.DataFrame:
        idx = pd.DatetimeIndex([], name=sc.COL_DATETIME)
        return pd.DataFrame(columns=sc.OHLCV_STORED_COLUMNS, index=idx)

    # ------------------------------ 复权因子 ------------------------------
    def save_adj_factor(self, symbol: str, events: pd.DataFrame) -> pd.DataFrame:
        """落盘复权因子：由除权事件（ex_date、ratio）计算累计因子。"""
        factor = compute_adj_factor(events)
        factor[sc.COL_SYMBOL] = symbol
        factor.to_parquet(self._adj_path(symbol), index=True)
        return factor

    def load_adj_factor(self, symbol: str, as_of: DateLike | None = None) -> pd.DataFrame:
        """读取复权因子，可选 as_of 过滤（仅含当时已知的除权事件）。"""
        path = self._adj_path(symbol)
        if not path.exists():
            return pd.DataFrame(
                columns=[sc.COL_RATIO, sc.COL_ADJ_FACTOR, sc.COL_AS_OF, sc.COL_SYMBOL],
                index=pd.DatetimeIndex([], name=sc.COL_EX_DATE),
            )
        df = pd.read_parquet(path)
        if as_of is not None:
            df = slice_as_of(df, as_of)
        return df

    # ------------------------------ 成分股 ------------------------------
    def save_constituents(self, index_code: str, df: pd.DataFrame) -> pd.DataFrame:
        """落盘成分股会员记录（幸存者偏差数据）。"""
        df = df.copy()
        for c in (sc.COL_SYMBOL, sc.COL_ENTRY_DATE):
            if c not in df.columns:
                raise DataError(f"成分股记录缺少必需列: {c}")
        if sc.COL_EXIT_DATE not in df.columns:
            df[sc.COL_EXIT_DATE] = pd.NaT
        if sc.COL_AS_OF not in df.columns:
            df[sc.COL_AS_OF] = df[sc.COL_ENTRY_DATE]
        if sc.COL_REASON not in df.columns:
            df[sc.COL_REASON] = sc.REASON_ADD
        df[sc.COL_INDEX_CODE] = index_code
        df[sc.COL_ENTRY_DATE] = pd.to_datetime(df[sc.COL_ENTRY_DATE])
        df[sc.COL_EXIT_DATE] = pd.to_datetime(df[sc.COL_EXIT_DATE])
        df[sc.COL_AS_OF] = pd.to_datetime(df[sc.COL_AS_OF])
        out = df[sc.CONSTITUENTS_STORED_COLUMNS].reset_index(drop=True)
        out.to_parquet(self._constituents_path(index_code), index=False)
        return out

    def load_constituents(self, index_code: str, as_of: DateLike | None = None) -> pd.DataFrame:
        """读取成分股会员记录，可选 as_of 过滤。"""
        path = self._constituents_path(index_code)
        if not path.exists():
            return pd.DataFrame(columns=sc.CONSTITUENTS_STORED_COLUMNS)
        df = pd.read_parquet(path)
        if as_of is not None:
            df = slice_as_of(df, as_of)
        return df

    def get_tradeable_universe(self, index_code: str, as_of: DateLike) -> list[str]:
        """返回指数在 as_of 时点的"当时可交易全集"。"""
        membership = self.load_constituents(index_code)
        return build_tradeable_universe(membership, as_of)

    # ------------------------------ PCF（预留 schema） ------------------------------
    def save_pcf(self, etf_code: str, trade_date: str, df: pd.DataFrame) -> pd.DataFrame:
        """落盘 PCF 篮子文件（本批次仅预留 schema，抓取在 B1 完成）。"""
        df = df.copy()
        for c in (sc.COL_SYMBOL, sc.COL_QUANTITY):
            if c not in df.columns:
                raise DataError(f"PCF 缺少必需列: {c}")
        df[sc.COL_ETF_CODE] = etf_code
        df[sc.COL_TRADE_DATE] = pd.Timestamp(trade_date)
        for c in (sc.COL_CASH_COMPONENT, sc.COL_CREATION_UNIT):
            if c not in df.columns:
                df[c] = 0.0
        if sc.COL_AS_OF not in df.columns:
            df[sc.COL_AS_OF] = pd.Timestamp(trade_date)
        out = df[sc.PCF_STORED_COLUMNS].reset_index(drop=True)
        out.to_parquet(self._pcf_path(etf_code, trade_date), index=False)
        return out

    def load_pcf(
        self, etf_code: str, trade_date: str | None = None, as_of: DateLike | None = None
    ) -> pd.DataFrame:
        """读取 PCF 篮子文件；trade_date 为空时返回该 ETF 全部篮子文件。"""
        d = self._subdir(SUBDIR_PCF)
        if trade_date is not None:
            path = self._pcf_path(etf_code, trade_date)
            if not path.exists():
                return pd.DataFrame(columns=sc.PCF_STORED_COLUMNS)
            df = pd.read_parquet(path)
        else:
            paths = sorted(d.glob(f"{etf_code}_*.parquet"))
            if not paths:
                return pd.DataFrame(columns=sc.PCF_STORED_COLUMNS)
            df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
        if as_of is not None:
            df = slice_as_of(df, as_of)
        return df

    # ------------------------------ NAV / IOPV ------------------------------
    def save_nav(self, etf_code: str, df: pd.DataFrame) -> pd.DataFrame:
        """落盘 NAV/IOPV（增量合并，按时间戳去重）。"""
        df = df.copy()
        if sc.COL_DATETIME in df.columns:
            df = df.set_index(sc.COL_DATETIME)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df.index = normalize_timestamps(df.index)
        df.index.name = sc.COL_DATETIME
        for c in (sc.COL_NAV, sc.COL_IOPV):
            if c not in df.columns:
                raise DataError(f"NAV 表缺少必需列: {c}")
        df = self._ensure_as_of(df, "1D")
        df[sc.COL_ETF_CODE] = etf_code
        df = df[sc.NAV_STORED_COLUMNS]
        df = df[~df.index.duplicated(keep="last")].sort_index()

        path = self._nav_path(etf_code)
        if path.exists():
            old = pd.read_parquet(path)
            df = pd.concat([old, df])
            df = df[~df.index.duplicated(keep="last")].sort_index()
        df.to_parquet(path, index=True)
        return df

    def load_nav(
        self,
        etf_code: str,
        start: DateLike | None = None,
        end: DateLike | None = None,
        as_of: DateLike | None = None,
    ) -> pd.DataFrame:
        """读取 NAV/IOPV，支持时间区间与 as_of 过滤。"""
        path = self._nav_path(etf_code)
        if not path.exists():
            return pd.DataFrame(
                columns=sc.NAV_STORED_COLUMNS,
                index=pd.DatetimeIndex([], name=sc.COL_DATETIME),
            )
        df = pd.read_parquet(path)
        if start is not None:
            df = df[df.index >= pd.Timestamp(start)]
        if end is not None:
            df = df[df.index <= pd.Timestamp(end)]
        if as_of is not None:
            df = slice_as_of(df, as_of)
        return df
