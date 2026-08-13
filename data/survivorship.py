"""幸存者偏差处理：成分股历史调进调出、退市股记录，构建"当时可交易全集"。

动机：若只用当前成分股快照做回测，会漏掉已退市/被调出的股票，导致收益被高估。
本模块记录每只股票完整的会员生命周期（调入生效日、调出/退市生效日、公告日），
使任意历史时点都能还原"当时真正可交易的股票集合"。
"""
from __future__ import annotations

from datetime import datetime
from typing import Iterable, List, Optional, Union

import pandas as pd

from core.exceptions import DataError
from data import schemas as sc

DateLike = Union[str, datetime, pd.Timestamp]

__all__ = [
    "UniverseTracker",
    "build_tradeable_universe",
    "filter_universe",
    "detect_survivorship_bias",
]


def _empty_membership() -> pd.DataFrame:
    return pd.DataFrame(columns=sc.CONSTITUENTS_STORED_COLUMNS)


class UniverseTracker:
    """追踪某指数/ETF 成分股的生命周期，支持任意时点还原可交易全集。"""

    def __init__(self, index_code: str = ""):
        self.index_code = index_code
        self._records = _empty_membership()

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------
    def add_member(
        self,
        symbol: str,
        entry_date: DateLike,
        exit_date: Optional[DateLike] = None,
        reason: str = sc.REASON_ADD,
        as_of: Optional[DateLike] = None,
    ) -> None:
        """登记一条会员记录。

        - entry_date：调入生效日（该日及之后属于指数）。
        - exit_date：调出/退市生效日（该日之后不再属于指数，None = 仍有效）。
        - as_of：该记录可被得知的时间（公告日）；默认等于 entry_date（生效即已知）。
        """
        entry = pd.Timestamp(entry_date)
        exit_ts = pd.NaT if exit_date is None else pd.Timestamp(exit_date)
        if exit_ts is not pd.NaT and exit_ts < entry:
            raise DataError(f"{symbol} 调出日早于调入日: {exit_ts} < {entry}")
        known = entry if as_of is None else pd.Timestamp(as_of)
        row = pd.DataFrame(
            [{
                sc.COL_INDEX_CODE: self.index_code,
                sc.COL_SYMBOL: symbol,
                sc.COL_ENTRY_DATE: entry,
                sc.COL_EXIT_DATE: exit_ts,
                sc.COL_AS_OF: known,
                sc.COL_REASON: reason,
            }]
        )
        if self._records.empty:
            self._records = row
        else:
            self._records = pd.concat([self._records, row], ignore_index=True)

    def add_records(self, df: pd.DataFrame) -> None:
        """批量导入会员记录（列名需符合成分股表 schema）。"""
        if df is None or len(df) == 0:
            return
        if self._records.empty:
            self._records = df.copy()
        else:
            self._records = pd.concat([self._records, df.copy()], ignore_index=True)

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------
    def to_frame(self) -> pd.DataFrame:
        """返回全部会员记录 DataFrame。"""
        return self._records.copy()

    def get_all_symbols(self) -> List[str]:
        """返回历史上出现过的全部证券代码（含已退市/已调出）。"""
        if self._records.empty:
            return []
        return sorted(self._records[sc.COL_SYMBOL].unique().tolist())

    def get_membership(self, symbol: str) -> pd.DataFrame:
        """返回某证券的全部会员记录。"""
        if self._records.empty:
            return _empty_membership()
        return self._records[self._records[sc.COL_SYMBOL] == symbol].copy()

    def get_delisted(self) -> List[str]:
        """返回已退出（调出或退市）的证券代码列表。"""
        if self._records.empty:
            return []
        mask = self._records[sc.COL_EXIT_DATE].notna()
        return sorted(self._records.loc[mask, sc.COL_SYMBOL].unique().tolist())

    def get_universe(self, as_of: DateLike) -> List[str]:
        """返回 as_of 时点可交易的股票集合（生效日已到、尚未退出、记录已公告）。"""
        return build_tradeable_universe(self._records, as_of)


def filter_universe(membership: pd.DataFrame, as_of: DateLike) -> pd.DataFrame:
    """返回 as_of 时点处于有效状态的会员记录行。"""
    if membership is None or len(membership) == 0:
        return _empty_membership()
    ts = pd.Timestamp(as_of)
    df = membership.copy()
    entry_ok = pd.to_datetime(df[sc.COL_ENTRY_DATE]) <= ts
    exit_active = df[sc.COL_EXIT_DATE].isna() | (pd.to_datetime(df[sc.COL_EXIT_DATE]) > ts)
    known_ok = pd.to_datetime(df[sc.COL_AS_OF].fillna(df[sc.COL_ENTRY_DATE])) <= ts
    return df.loc[entry_ok & exit_active & known_ok].copy()


def build_tradeable_universe(membership: pd.DataFrame, as_of: DateLike) -> List[str]:
    """返回 as_of 时点"当时可交易全集"（证券代码列表，已排序去重）。

    三重过滤：
    1. 生效日已到（entry_date <= as_of）
    2. 尚未退出（exit_date 为空或 > as_of）
    3. 记录可得知（as_of <= 查询时点，防用公告日之后的信息）
    """
    active = filter_universe(membership, as_of)
    if active.empty:
        return []
    return sorted(active[sc.COL_SYMBOL].unique().tolist())


def detect_survivorship_bias(
    all_historical: Iterable[str],
    current_snapshot: Iterable[str],
) -> set:
    """检测幸存者偏差：返回"当前快照"遗漏的历史证券（已退市/已调出）。"""
    return set(all_historical) - set(current_snapshot)
