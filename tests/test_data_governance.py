"""A1 数据层与数据治理单元测试。

覆盖验收标准：
- as_of 时间戳过滤生效（防未来函数）
- 退市/调出股票在组合中正确消失（幸存者偏差）
- 前复权与已知基准一致
- 质量报告统计缺失/异常率并输出 markdown
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data import schemas as sc
from data.data_loader import DataLoader, clean_ohlcv
from data.pit import (
    as_of_barrier,
    as_of_mask,
    compute_adj_factor,
    forward_adjust,
    backward_adjust,
    normalize_timestamps,
    slice_as_of,
)
from data.quality_report import QualityReporter
from data.survivorship import (
    UniverseTracker,
    build_tradeable_universe,
    detect_survivorship_bias,
)


# ---------------------------------------------------------------------------
# 测试辅助
# ---------------------------------------------------------------------------
def make_ohlcv(index, close, volume=1000.0, as_of=None) -> pd.DataFrame:
    """构造标准 OHLCV 长表（index=datetime）。"""
    idx = pd.to_datetime(index)
    close = pd.Series(close, index=idx, dtype=float)
    df = pd.DataFrame({
        sc.COL_OPEN: close,
        sc.COL_HIGH: close * 1.01,
        sc.COL_LOW: close * 0.99,
        sc.COL_CLOSE: close,
        sc.COL_VOLUME: volume,
        sc.COL_AMOUNT: close * volume,
    }, index=idx)
    df.index.name = sc.COL_DATETIME
    df[sc.COL_AS_OF] = idx if as_of is None else pd.to_datetime(as_of)
    return df


# ---------------------------------------------------------------------------
# point-in-time
# ---------------------------------------------------------------------------
def test_normalize_timestamps_strips_tz():
    idx = normalize_timestamps(pd.to_datetime(["2024-01-01 09:30:00+08:00"]))
    assert idx.tz is None


def test_as_of_filter_blocks_future_data():
    idx = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
    df = pd.DataFrame({"value": [1, 2, 3], sc.COL_AS_OF: idx})
    mask = as_of_mask(df, "2024-01-02")
    assert mask.tolist() == [True, True, False]
    sliced = slice_as_of(df, "2024-01-02")
    assert sliced["value"].tolist() == [1, 2]


def test_as_of_barrier_returns_min_as_of():
    idx = pd.to_datetime(["2024-01-03", "2024-01-01", "2024-01-02"])
    df = pd.DataFrame({"value": [1, 2, 3], sc.COL_AS_OF: idx})
    assert as_of_barrier(df) == pd.Timestamp("2024-01-01")


def test_compute_adj_factor():
    events = pd.DataFrame({
        sc.COL_EX_DATE: pd.to_datetime(["2024-01-02", "2024-01-03"]),
        sc.COL_RATIO: [0.5, 0.8],
    })
    factor = compute_adj_factor(events)
    assert factor[sc.COL_ADJ_FACTOR].tolist() == [0.5, 0.4]
    assert factor.index.name == sc.COL_EX_DATE
    assert (factor[sc.COL_AS_OF] == factor.index).all()


def test_forward_adjust_matches_baseline():
    idx = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    prices = pd.Series([10.0, 5.0, 5.5], index=idx)
    events = pd.DataFrame({
        sc.COL_EX_DATE: pd.to_datetime(["2024-01-03"]),
        sc.COL_RATIO: [0.5],
    })
    factor = compute_adj_factor(events)
    qfq = forward_adjust(prices, factor)
    # 10 送 10：除权前 10 元前复权为 5 元，除权后不变，最新价等于原始价
    pd.testing.assert_series_equal(qfq, pd.Series([5.0, 5.0, 5.5], index=idx), check_names=False)
    assert qfq.iloc[-1] == prices.iloc[-1]


def test_backward_adjust():
    idx = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    prices = pd.Series([10.0, 5.0, 5.5], index=idx)
    events = pd.DataFrame({
        sc.COL_EX_DATE: pd.to_datetime(["2024-01-03"]),
        sc.COL_RATIO: [0.5],
    })
    factor = compute_adj_factor(events)
    hfq = backward_adjust(prices, factor)
    pd.testing.assert_series_equal(hfq, pd.Series([10.0, 10.0, 11.0], index=idx), check_names=False)


def test_forward_adjust_as_of_blocks_future_factor():
    """除权生效日之前不得使用未来的复权因子（防前视）。"""
    idx = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    prices = pd.Series([10.0, 5.0, 5.5], index=idx)
    events = pd.DataFrame({
        sc.COL_EX_DATE: pd.to_datetime(["2024-01-03"]),
        sc.COL_RATIO: [0.5],
    })
    factor = compute_adj_factor(events)
    qfq = forward_adjust(prices, factor, as_of="2024-01-02")
    pd.testing.assert_series_equal(qfq, prices, check_names=False)


# ---------------------------------------------------------------------------
# 幸存者偏差
# ---------------------------------------------------------------------------
def test_universe_tracker_add_and_delist():
    tracker = UniverseTracker("000300")
    tracker.add_member("600000", "2024-01-01")
    tracker.add_member("000002", "2024-01-01", exit_date="2024-06-01", reason=sc.REASON_DELIST)
    assert "000002" in tracker.get_universe("2024-05-01")
    assert "000002" not in tracker.get_universe("2024-07-01")
    assert "600000" in tracker.get_universe("2024-07-01")
    assert tracker.get_delisted() == ["000002"]


def test_universe_tracker_added_stock_appears_after_effective_date():
    tracker = UniverseTracker("000300")
    tracker.add_member("600000", "2024-01-01")
    tracker.add_member("601318", "2024-07-01", reason=sc.REASON_ADD)
    assert "601318" not in tracker.get_universe("2024-06-30")
    assert "601318" in tracker.get_universe("2024-07-01")


def test_universe_tracker_announcement_vs_effective():
    """生效日已到但公告日未到：公告前不可见（公告日 vs 生效日防未来函数）。"""
    tracker = UniverseTracker("000300")
    tracker.add_member("600519", "2024-01-01", as_of="2024-02-01")
    assert "600519" not in tracker.get_universe("2024-01-15")
    assert "600519" in tracker.get_universe("2024-02-01")


def test_build_tradeable_universe():
    memb = pd.DataFrame({
        sc.COL_SYMBOL: ["A", "B", "C"],
        sc.COL_ENTRY_DATE: pd.to_datetime(["2024-01-01", "2024-01-01", "2024-06-01"]),
        sc.COL_EXIT_DATE: [pd.NaT, pd.Timestamp("2024-06-01"), pd.NaT],
        sc.COL_AS_OF: pd.to_datetime(["2024-01-01", "2024-01-01", "2024-06-01"]),
    })
    assert build_tradeable_universe(memb, "2024-03-01") == ["A", "B"]
    assert build_tradeable_universe(memb, "2024-07-01") == ["A", "C"]


def test_detect_survivorship_bias():
    missing = detect_survivorship_bias(["A", "B", "C"], ["A", "B"])
    assert missing == {"C"}


# ---------------------------------------------------------------------------
# 清洗
# ---------------------------------------------------------------------------
def test_clean_ohlcv_dedupe_and_nonpositive():
    idx = pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-03"])
    df = pd.DataFrame({
        sc.COL_OPEN: [10, 10, 10, 0],
        sc.COL_HIGH: [11, 11, 11, 9],
        sc.COL_LOW: [9, 9, 9, 8],
        sc.COL_CLOSE: [10, 10, 10, -5],
        sc.COL_VOLUME: [100, 100, 100, 0],
        sc.COL_AMOUNT: [1000, 1000, 1000, 0],
    }, index=idx)
    df.index.name = sc.COL_DATETIME
    cleaned = clean_ohlcv(df)
    assert not cleaned.index.duplicated().any()
    assert len(cleaned) == 3
    assert (cleaned[sc.COL_CLOSE] > 0).all()


def test_clean_ohlcv_flags_limit_up():
    idx = pd.to_datetime(["2024-01-01", "2024-01-02"])
    df = pd.DataFrame({
        sc.COL_OPEN: [10, 11],
        sc.COL_HIGH: [10, 11],
        sc.COL_LOW: [10, 11],
        sc.COL_CLOSE: [10, 11],
        sc.COL_VOLUME: [1000, 1000],
        sc.COL_AMOUNT: [10000, 11000],
    }, index=idx)
    df.index.name = sc.COL_DATETIME
    cleaned = clean_ohlcv(df, limit_pct=0.10)
    assert cleaned[sc.COL_LIMIT_STATUS].iloc[0] == sc.LIMIT_NORMAL
    assert cleaned[sc.COL_LIMIT_STATUS].iloc[1] == sc.LIMIT_UP


def test_clean_ohlcv_flags_suspension():
    idx = pd.to_datetime(["2024-01-01", "2024-01-02"])
    df = pd.DataFrame({
        sc.COL_OPEN: [10, 10],
        sc.COL_HIGH: [10, 10],
        sc.COL_LOW: [10, 10],
        sc.COL_CLOSE: [10, 10],
        sc.COL_VOLUME: [1000, 0],
        sc.COL_AMOUNT: [10000, 0],
    }, index=idx)
    df.index.name = sc.COL_DATETIME
    cleaned = clean_ohlcv(df)
    assert cleaned[sc.COL_IS_SUSPENDED].iloc[1]
    assert cleaned[sc.COL_LIMIT_STATUS].iloc[1] == sc.SUSPENDED


# ---------------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------------
def test_data_loader_ohlcv_roundtrip(tmp_path):
    loader = DataLoader(tmp_path)
    df = make_ohlcv(["2024-01-01", "2024-01-02", "2024-01-03"], [10.0, 10.5, 10.8])
    loader.save_ohlcv("510300", df)
    loaded = loader.load_ohlcv("510300")
    assert loaded.index.tolist() == pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]).tolist()
    assert loaded[sc.COL_CLOSE].tolist() == [10.0, 10.5, 10.8]
    assert (loaded[sc.COL_SYMBOL] == "510300").all()
    assert sc.COL_AS_OF in loaded.columns
    assert sc.COL_IS_SUSPENDED in loaded.columns


def test_data_loader_as_of_filter_on_load(tmp_path):
    loader = DataLoader(tmp_path)
    idx = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
    df = make_ohlcv(idx, [10.0, 11.0, 12.0], as_of=idx + pd.Timedelta(days=2))
    loader.save_ohlcv("X", df)
    loaded = loader.load_ohlcv("X", as_of="2024-01-04")
    assert loaded.index.tolist() == idx[:2].tolist()


def test_data_loader_incremental_update(tmp_path):
    loader = DataLoader(tmp_path)
    idx1 = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
    loader.save_ohlcv("X", make_ohlcv(idx1, [10.0] * 5))
    idx2 = pd.to_datetime(["2024-01-04", "2024-01-05", "2024-01-06", "2024-01-07"])
    loader.incremental_update_ohlcv("X", make_ohlcv(idx2, [20.0] * 4))
    out = loader.load_ohlcv("X", clean=False)
    assert out.index.tolist() == pd.to_datetime(
        ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-06", "2024-01-07"]
    ).tolist()
    assert not out.index.duplicated().any()
    assert out.loc[pd.Timestamp("2024-01-04"), sc.COL_CLOSE] == 20.0
    assert out.loc[pd.Timestamp("2024-01-03"), sc.COL_CLOSE] == 10.0


def test_data_loader_constituents_and_universe(tmp_path):
    loader = DataLoader(tmp_path)
    memb = pd.DataFrame({
        sc.COL_SYMBOL: ["600000", "000002", "601318"],
        sc.COL_ENTRY_DATE: pd.to_datetime(["2024-01-01", "2024-01-01", "2024-06-01"]),
        sc.COL_EXIT_DATE: [pd.NaT, pd.Timestamp("2024-06-01"), pd.NaT],
        sc.COL_REASON: [sc.REASON_INITIAL, sc.REASON_DELIST, sc.REASON_ADD],
    })
    loader.save_constituents("000300", memb)
    before = loader.get_tradeable_universe("000300", "2024-03-01")
    after = loader.get_tradeable_universe("000300", "2024-07-01")
    assert before == ["000002", "600000"]
    assert after == ["600000", "601318"]


def test_data_loader_adj_factor_nav_pcf_roundtrip(tmp_path):
    loader = DataLoader(tmp_path)
    # 复权因子
    loader.save_adj_factor("510300", pd.DataFrame({
        sc.COL_EX_DATE: [pd.Timestamp("2024-06-01")],
        sc.COL_RATIO: [0.8],
    }))
    assert loader.load_adj_factor("510300")[sc.COL_ADJ_FACTOR].iloc[-1] == 0.8
    # NAV
    nav = pd.DataFrame({
        sc.COL_NAV: [3.0, 3.1],
        sc.COL_IOPV: [3.0, 3.11],
    }, index=pd.to_datetime(["2024-01-01", "2024-01-02"]))
    nav.index.name = sc.COL_DATETIME
    loader.save_nav("510300", nav)
    assert loader.load_nav("510300")[sc.COL_NAV].tolist() == [3.0, 3.1]
    # PCF
    pcf = pd.DataFrame({sc.COL_SYMBOL: ["600000", "000001"], sc.COL_QUANTITY: [100, 200]})
    loader.save_pcf("510300", "20240601", pcf)
    assert len(loader.load_pcf("510300", "20240601")) == 2


# ---------------------------------------------------------------------------
# 质量报告
# ---------------------------------------------------------------------------
def test_quality_report_computes_rates():
    reporter = QualityReporter(limit_pct=0.10)
    idx = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"])
    df = pd.DataFrame({
        sc.COL_OPEN: [10, 10, 10, 10],
        sc.COL_HIGH: [11, 11, 11, 9],
        sc.COL_LOW: [9, 9, 9, 10],
        sc.COL_CLOSE: [10, np.nan, 10, 10],
        sc.COL_VOLUME: [1000, 1000, 1000, 1000],
        sc.COL_AMOUNT: [10000, 10000, 10000, 10000],
    }, index=idx)
    df.index.name = sc.COL_DATETIME
    df.loc[idx[0], sc.COL_OPEN] = -1.0  # 制造一个非正价异常
    stats = reporter.inspect_ohlcv(df)
    assert stats["rows"] == 4
    assert stats["missing_rate"] == pytest.approx(1 / 24)
    assert stats["missing"][sc.COL_CLOSE] == pytest.approx(0.25)
    assert stats["anomaly_rate"] == pytest.approx(0.5)


def test_quality_report_markdown():
    reporter = QualityReporter(limit_pct=0.10)
    df = make_ohlcv(["2024-01-01", "2024-01-02"], [10.0, 10.5])
    report = {"510300": reporter.inspect_ohlcv(df)}
    md = reporter.to_markdown(report)
    assert "# 数据质量报告" in md
    assert "510300" in md
    assert "缺失率" in md
    assert "异常率" in md


def test_quality_report_write_file(tmp_path):
    loader = DataLoader(tmp_path)
    loader.save_ohlcv("510300", make_ohlcv(["2024-01-01", "2024-01-02"], [10.0, 10.5]))
    reporter = QualityReporter(tmp_path)
    out = reporter.write_report(path=tmp_path / "report.md", symbols=["510300"])
    assert Path(out).exists()
    content = Path(out).read_text(encoding="utf-8")
    assert "510300" in content
