"""B1 ETF 套利数据层单元测试。

覆盖验收标准：
- PCF 解析（标准格式 + 示例文件）
- 篮子权重之和 = 1（±0.5% 容差）
- PCF 生效日不串期（T 日 PCF 次日生效，周末/日历跳过）
- IOPV 合成估算（含缺失成分股兜底）
- 折溢价计算（盘中/收盘，手算对照）
- 停牌/涨跌停成分股识别
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from etf.etf_data import (
    ETFDataService,
    mark_untradeable_constituents,
    premium_close,
    premium_intraday,
    synthetic_etf_minute,
)
from etf.iopv_estimator import IOPVEstimator
from etf.pcf_parser import (
    PCFBasket,
    PCFConstituent,
    next_trading_day,
    parse_pcf,
    parse_pcf_file,
    pcf_from_dataframe,
)

ROOT = Path(__file__).resolve().parents[1]

PCF_TEXT = """# 沪深300ETF 申赎清单示例
ETF代码=510300
清单日期=20240102
生效日期=20240103
最小申赎单位=900000
现金差额=1234.56
[成分股]
证券代码,证券名称,数量,现金替代标志,前收盘价
600000,浦发银行,1000,允许,7.20
600036,招商银行,500,允许,33.00
601318,中国平安,800,禁止,42.50
600519,贵州茅台,100,禁止,1700.00
000858,五粮液,300,允许,130.00
"""


def make_ohlcv(index, opens, closes, volumes):
    """构造单标的 OHLCV DataFrame（index=datetime）。"""
    idx = pd.to_datetime(index)
    opens, closes = list(opens), list(closes)
    high = [max(o, c) for o, c in zip(opens, closes, strict=False)]
    low = [min(o, c) for o, c in zip(opens, closes, strict=False)]
    return pd.DataFrame(
        {
            "open": opens,
            "high": high,
            "low": low,
            "close": closes,
            "volume": list(volumes),
        },
        index=idx,
    )


# ---------------------------------------------------------------------------
# PCF 解析
# ---------------------------------------------------------------------------
def test_parse_pcf_basic():
    basket = parse_pcf(PCF_TEXT)
    assert basket.etf_code == "510300"
    assert basket.trade_date == pd.Timestamp("2024-01-02")
    assert basket.effective_date == pd.Timestamp("2024-01-03")
    assert basket.creation_unit == pytest.approx(900000.0)
    assert basket.cash_component == pytest.approx(1234.56)
    assert len(basket.constituents) == 5
    assert basket.constituents[0].symbol == "600000"
    assert basket.constituents[0].quantity == pytest.approx(1000.0)


def test_parse_pcf_file():
    path = ROOT / "etf" / "demo_data" / "pcf_510300_20240102.txt"
    basket = parse_pcf_file(path)
    assert basket.etf_code == "510300"
    assert len(basket.constituents) == 5


def test_pcf_weight_sum_equals_one():
    """篮子权重（价值加权）之和 = 1，±0.5% 容差内。"""
    basket = parse_pcf(PCF_TEXT)
    weights = basket.weights()
    assert abs(sum(weights.values()) - 1.0) <= 0.005
    assert basket.validate_weight_sum(tolerance=0.005)


def test_pcf_effective_next_trading_day():
    """T 日 PCF 次日生效：无生效日期时自动取下一交易日；周五顺延到周一。"""
    # 未给生效日期：2024-01-02（周二）→ 2024-01-03
    text_no_eff = PCF_TEXT.replace("生效日期=20240103\n", "")
    b1 = parse_pcf(text_no_eff)
    assert b1.effective_date == pd.Timestamp("2024-01-03")

    # 周五 2024-01-05 → 下周一 2024-01-08
    b2 = parse_pcf(
        PCF_TEXT.replace("清单日期=20240102", "清单日期=20240105").replace(
            "生效日期=20240103\n", ""
        )
    )
    assert b2.effective_date == pd.Timestamp("2024-01-08")


def test_next_trading_day_with_calendar():
    """传入交易日历可跳过周末/节假日。"""
    cal = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-05", "2024-01-08"])
    assert next_trading_day("2024-01-03", calendar=cal) == pd.Timestamp("2024-01-05")


def test_pcf_from_dataframe():
    df = pd.DataFrame({"symbol": ["600000", "600036"], "quantity": [1000.0, 500.0]})
    basket = pcf_from_dataframe(
        df, etf_code="510300", trade_date="2024-01-02", creation_unit=900000.0, cash_component=0.0
    )
    assert basket.effective_date == pd.Timestamp("2024-01-03")
    assert len(basket.constituents) == 2


# ---------------------------------------------------------------------------
# IOPV 合成估算
# ---------------------------------------------------------------------------
def _simple_basket() -> PCFBasket:
    return PCFBasket(
        etf_code="510300",
        trade_date=pd.Timestamp("2024-01-02"),
        effective_date=pd.Timestamp("2024-01-03"),
        creation_unit=1000.0,
        cash_component=0.0,
        constituents=[
            PCFConstituent(symbol="A", quantity=100.0, price=2.0),
            PCFConstituent(symbol="B", quantity=100.0, price=3.0),
        ],
    )


def test_iopv_estimate_basic():
    est = IOPVEstimator()
    basket = _simple_basket()
    # IOPV = (100*2 + 100*3) / 1000 = 0.5
    res = est.estimate(basket, {"A": 2.0, "B": 3.0})
    assert res.iopv == pytest.approx(0.5)
    assert res.used_constituents == 2
    assert res.missing_symbols == []


def test_iopv_estimate_with_fallback_and_missing():
    est = IOPVEstimator()
    basket = _simple_basket()
    # A 实时价缺失，用兜底价 2.0
    res_fb = est.estimate(basket, {"B": 3.0}, fallback_prices={"A": 2.0})
    assert res_fb.iopv == pytest.approx(0.5)
    # A 无兜底：缺失，IOPV = 300/1000 = 0.3
    res_missing = est.estimate(basket, {"B": 3.0})
    assert res_missing.iopv == pytest.approx(0.3)
    assert res_missing.missing_symbols == ["A"]


# ---------------------------------------------------------------------------
# 折溢价
# ---------------------------------------------------------------------------
def test_premium_intraday_and_close():
    idx = pd.date_range("2024-01-02", periods=3, freq="B")
    etf_price = pd.Series([3.006, 2.994, 3.000], index=idx)
    iopv = pd.Series([3.000, 3.000, 3.000], index=idx)
    prem = premium_intraday(etf_price, iopv)
    assert prem.iloc[0] == pytest.approx(0.006 / 3.0)
    assert prem.iloc[1] == pytest.approx(-0.006 / 3.0)

    nav = pd.Series([3.000, 3.000, 3.000], index=idx)
    cprem = premium_close(etf_price, nav)
    assert cprem.iloc[0] == pytest.approx(0.006 / 3.0)


# ---------------------------------------------------------------------------
# 停牌/涨跌停成分股识别
# ---------------------------------------------------------------------------
def test_mark_untradeable_constituents():
    basket = PCFBasket(
        etf_code="510300",
        trade_date=pd.Timestamp("2024-01-02"),
        effective_date=pd.Timestamp("2024-01-03"),
        creation_unit=1000.0,
        cash_component=0.0,
        constituents=[
            PCFConstituent(symbol="A", quantity=100.0),  # 正常
            PCFConstituent(symbol="B", quantity=100.0),  # 停牌
            PCFConstituent(symbol="C", quantity=100.0),  # 涨停
        ],
    )
    quotes = {
        "A": make_ohlcv(["2024-01-02", "2024-01-03"], [10.0, 10.1], [10.0, 10.1], [1000.0, 1000.0]),
        "B": make_ohlcv(["2024-01-02", "2024-01-03"], [5.0, 5.0], [5.0, 5.0], [0.0, 0.0]),  # 停牌
        "C": make_ohlcv(
            ["2024-01-02", "2024-01-03"], [10.0, 11.0], [10.0, 11.0], [1000.0, 1000.0]
        ),  # 涨停
    }
    df = mark_untradeable_constituents(basket, quotes)
    by_sym = df.set_index("symbol")
    assert bool(by_sym.loc["A", "tradeable"]) is True
    assert bool(by_sym.loc["B", "is_suspended"]) is True
    assert bool(by_sym.loc["B", "tradeable"]) is False
    assert by_sym.loc["C", "limit_status"] == "limit_up"
    assert bool(by_sym.loc["C", "tradeable"]) is False


# ---------------------------------------------------------------------------
# 合成数据 + 服务
# ---------------------------------------------------------------------------
def test_synthetic_etf_minute_has_iopv():
    df = synthetic_etf_minute("510300", "2024-01-02", "2024-01-02", seed=1, freq="5min")
    assert not df.empty
    assert "iopv" in df.columns
    assert df["close"].notna().all()
    assert df["iopv"].notna().all()


def test_service_build_premium_discount():
    svc = ETFDataService(use_akshare=False, seed=1)
    quotes = svc.load_etf_quotes("510300", "2024-01-02", "2024-01-03", freq="5min")
    nav = svc.load_nav("510300", pd.date_range("2024-01-02", "2024-01-03", freq="B"))
    prem = svc.build_premium_discount(quotes, nav)
    assert set(prem.columns) == {"premium_intraday", "premium_close"}
    # 收盘折溢价有限（合成 NAV 与 ETF 收盘接近）
    assert prem["premium_close"].notna().all()
