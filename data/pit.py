"""point-in-time 对齐与防前视工具。

核心思想：
- 每一条数据都带 ``as_of`` 时间戳，表示该数据"可被观察到"的时间点。
- 在任意时点 T 做决策/回测时，只能使用 ``as_of <= T`` 的数据（slice_as_of）。
- 复权因子只在除权除息生效日（ex_date）之后才生效；查询时点之前的未来因子不得参与计算。
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional, Union

import pandas as pd

from core.exceptions import DataError
from data import schemas as sc

DateLike = Union[str, datetime, pd.Timestamp]

__all__ = [
    "normalize_timestamps",
    "as_of_mask",
    "slice_as_of",
    "as_of_barrier",
    "compute_adj_factor",
    "forward_adjust",
    "backward_adjust",
]


def _to_timestamp(value: DateLike) -> pd.Timestamp:
    """统一转换为 tz-naive 的 Timestamp。"""
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(None)
    return ts


def normalize_timestamps(values) -> pd.DatetimeIndex:
    """把任意时间序列转换为 tz-naive 的 DatetimeIndex（不排序、不去重）。"""
    idx = pd.DatetimeIndex(pd.to_datetime(values))
    if idx.tz is not None:
        idx = idx.tz_convert(None)
    return idx


def as_of_mask(df: pd.DataFrame, as_of: DateLike, as_of_col: str = sc.COL_AS_OF) -> pd.Series:
    """返回布尔掩码：仅保留 ``as_of_col <= as_of`` 的行。

    这是防未来函数的最基本原语——同一时点只能看见当时已发布的数据。
    """
    if as_of_col not in df.columns:
        raise DataError(f"数据缺少 point-in-time 列: {as_of_col}")
    barrier = _to_timestamp(as_of)
    return pd.to_datetime(df[as_of_col]) <= barrier


def slice_as_of(df: pd.DataFrame, as_of: DateLike, as_of_col: str = sc.COL_AS_OF) -> pd.DataFrame:
    """返回 ``as_of`` 过滤后的数据副本（只含当时已发布数据）。"""
    return df.loc[as_of_mask(df, as_of, as_of_col)].copy()


def as_of_barrier(df: pd.DataFrame, as_of_col: str = sc.COL_AS_OF) -> pd.Timestamp:
    """返回数据中最早的 as_of，作为"可用数据墙"（早于此的记录视为未发布）。"""
    if as_of_col not in df.columns:
        raise DataError(f"数据缺少 point-in-time 列: {as_of_col}")
    return pd.to_datetime(df[as_of_col]).min()


def compute_adj_factor(
    events: pd.DataFrame,
    ex_date_col: str = sc.COL_EX_DATE,
    ratio_col: str = sc.COL_RATIO,
    as_of_col: str = sc.COL_AS_OF,
) -> pd.DataFrame:
    """由除权除息事件计算累计复权因子。

    参数 events 至少包含 ex_date（生效日）与 ratio（新价/旧价）两列；
    若提供 as_of 列则保留（公告日/生效日），否则默认 as_of = ex_date。

    返回以 ex_date 为索引的 DataFrame，列：ratio、adj_factor、as_of。
    adj_factor = ratio 的累计乘积（除权前因子 = 1）。
    """
    if events is None or len(events) == 0:
        return pd.DataFrame(
            columns=[sc.COL_RATIO, sc.COL_ADJ_FACTOR, sc.COL_AS_OF],
            index=pd.DatetimeIndex([], name=sc.COL_EX_DATE),
        )
    df = events.copy()
    if ex_date_col not in df.columns or ratio_col not in df.columns:
        raise DataError(f"复权事件缺少必需列: {ex_date_col} / {ratio_col}")
    df[ex_date_col] = pd.to_datetime(df[ex_date_col])
    df = df.sort_values(ex_date_col).reset_index(drop=True)
    if as_of_col not in df.columns:
        df[as_of_col] = df[ex_date_col]
    df[sc.COL_ADJ_FACTOR] = df[ratio_col].astype(float).cumprod()
    out = df.set_index(ex_date_col)[[sc.COL_RATIO, sc.COL_ADJ_FACTOR, sc.COL_AS_OF]]
    out.index.name = sc.COL_EX_DATE
    return out


def _align_factor(adj_factor: pd.Series, target_index: pd.DatetimeIndex) -> pd.Series:
    """把以 ex_date 为索引的累计因子向前填充对齐到目标时间索引。

    除权事件之前的日期因子填 1.0（尚未发生任何除权）。
    """
    target = pd.DatetimeIndex(target_index)
    combined = adj_factor.index.union(target).sort_values()
    return adj_factor.reindex(combined).ffill().reindex(target).fillna(1.0)


def _coerce_factor(adj_factor: Union[pd.Series, pd.DataFrame]) -> pd.Series:
    """接受 Series 或 DataFrame（取 adj_factor 列），返回排序后的累计因子 Series。"""
    if isinstance(adj_factor, pd.DataFrame):
        if sc.COL_ADJ_FACTOR not in adj_factor.columns:
            raise DataError(f"复权因子表缺少列: {sc.COL_ADJ_FACTOR}")
        adj_factor = adj_factor[sc.COL_ADJ_FACTOR]
    return pd.Series(adj_factor).astype(float).sort_index()


def forward_adjust(
    prices: pd.Series,
    adj_factor: Union[pd.Series, pd.DataFrame],
    as_of: Optional[DateLike] = None,
) -> pd.Series:
    """前复权：qfq_t = price_t * f_end / f_t。

    - f_t 为截至 t 的累计复权因子；f_end 为数据中最后一个因子的值。
    - 前复权保证最新价等于原始价，且除权当日不产生虚假跳空。
    - 传入 as_of 时，只使用 ex_date <= as_of 的因子（防止用未来除权信息调整历史价格）。
    """
    prices = pd.Series(prices)
    factor = _coerce_factor(adj_factor)
    if as_of is not None:
        factor = factor[factor.index <= _to_timestamp(as_of)]
    if factor.empty:
        return prices.copy()
    aligned = _align_factor(factor, prices.index)
    f_end = float(factor.iloc[-1])
    return prices * f_end / aligned


def backward_adjust(
    prices: pd.Series,
    adj_factor: Union[pd.Series, pd.DataFrame],
    as_of: Optional[DateLike] = None,
) -> pd.Series:
    """后复权：hfq_t = price_t / f_t（约定除权前因子 f_0 = 1）。"""
    prices = pd.Series(prices)
    factor = _coerce_factor(adj_factor)
    if as_of is not None:
        factor = factor[factor.index <= _to_timestamp(as_of)]
    if factor.empty:
        return prices.copy()
    aligned = _align_factor(factor, prices.index)
    return prices / aligned
