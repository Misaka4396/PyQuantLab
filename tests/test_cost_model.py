"""A3 交易成本与滑点模型单元测试。

覆盖验收标准：
- 逐笔成本可追溯（每笔有明细条目）
- 固定成本（佣金/印花税/过户费）、买卖价差、冲击成本正确
- ETF 免印花税/过户费
- 篮子成本逐只计费 + 篮子整体滑点加成
- 成本敏感性测试（参数变化对总成本的敏感度）
"""
from __future__ import annotations

import math

import pytest

from cost_config import CostConfig
from cost_model import CostModel, sensitivity_table


def cfg(**overrides) -> CostConfig:
    """默认零滑点、零最低佣金的测试配置，便于精确断言。"""
    base = dict(
        commission_rate=0.00025,
        min_commission=0.0,
        stamp_tax_rate=0.0005,
        transfer_fee_rate=0.00001,
        spread_rate=0.0,
        impact_coef=0.0,
        basket_slippage_bps=1.0,
    )
    base.update(overrides)
    return CostConfig(**base)


# ---------------------------------------------------------------------------
# 固定成本
# ---------------------------------------------------------------------------
def test_commission_rate():
    model = CostModel(cfg(commission_rate=0.0003))
    b = model.compute("BUY", 100, 10.0, symbol="600000", is_etf=False)
    assert b.commission == pytest.approx(100 * 10.0 * 0.0003)


def test_min_commission_floor():
    model = CostModel(cfg(commission_rate=0.0003, min_commission=5.0))
    b = model.compute("BUY", 100, 10.0, symbol="600000", is_etf=False)
    # 1000 * 0.0003 = 0.3 < 5 → 按最低 5 元
    assert b.commission == pytest.approx(5.0)


def test_stamp_tax_sell_only_and_etf_exempt():
    model = CostModel(cfg(stamp_tax_rate=0.0005))
    buy = model.compute("BUY", 100, 10.0, symbol="600000", is_etf=False)
    sell = model.compute("SELL", 100, 10.0, symbol="600000", is_etf=False)
    sell_etf = model.compute("SELL", 100, 10.0, symbol="510300", is_etf=True)
    assert buy.stamp_tax == 0.0
    assert sell.stamp_tax == pytest.approx(100 * 10.0 * 0.0005)
    assert sell_etf.stamp_tax == 0.0


def test_transfer_fee_and_etf_exempt():
    model = CostModel(cfg(transfer_fee_rate=0.00001))
    stock = model.compute("BUY", 100, 10.0, symbol="600000", is_etf=False)
    etf = model.compute("BUY", 100, 10.0, symbol="510300", is_etf=True)
    assert stock.transfer_fee == pytest.approx(100 * 10.0 * 0.00001)
    assert etf.transfer_fee == 0.0


# ---------------------------------------------------------------------------
# 买卖价差（半价差）
# ---------------------------------------------------------------------------
def test_half_spread():
    model = CostModel(cfg(spread_rate=0.002))  # 全价差 0.2%，半价差 0.1%
    buy = model.compute("BUY", 100, 10.0, symbol="510300", is_etf=True, volume=0)
    sell = model.compute("SELL", 100, 10.0, symbol="510300", is_etf=True, volume=0)
    assert buy.exec_price == pytest.approx(10.0 * 1.001)
    assert sell.exec_price == pytest.approx(10.0 * 0.999)
    assert buy.spread_cost == pytest.approx(10.0 * 0.001 * 100)


# ---------------------------------------------------------------------------
# 冲击成本
# ---------------------------------------------------------------------------
def test_impact_sqrt():
    model = CostModel(cfg(impact_model="sqrt", impact_coef=0.1))
    b = model.compute("BUY", 1000, 10.0, symbol="510300", is_etf=True, volume=10000)
    # 参与率 0.1，冲击率 = 0.1 * sqrt(0.1)
    assert b.impact_cost == pytest.approx(10.0 * (0.1 * math.sqrt(0.1)) * 1000)


def test_impact_linear_and_monotonic_in_participation():
    model = CostModel(cfg(impact_model="linear", impact_coef=0.1))
    b_small = model.compute("BUY", 1000, 10.0, symbol="510300", is_etf=True, volume=10000)
    b_large = model.compute("BUY", 5000, 10.0, symbol="510300", is_etf=True, volume=10000)
    # 线性：参与率 0.1 → 0.01；参与率 0.5 → 0.05
    assert b_small.impact_cost == pytest.approx(10.0 * 0.01 * 1000)
    assert b_large.impact_cost == pytest.approx(10.0 * 0.05 * 5000)
    assert b_large.impact_cost > b_small.impact_cost


def test_zero_volume_no_impact():
    model = CostModel(cfg(impact_model="sqrt", impact_coef=0.1))
    b = model.compute("BUY", 1000, 10.0, symbol="510300", is_etf=True, volume=0)
    assert b.impact_cost == 0.0


# ---------------------------------------------------------------------------
# 篮子成本
# ---------------------------------------------------------------------------
def test_basket_cost_per_leg_plus_surcharge():
    model = CostModel(cfg(basket_slippage_bps=2.0))
    legs = [
        {"symbol": "600000", "side": "BUY", "quantity": 100, "price": 10.0, "is_etf": False, "volume": 10000},
        {"symbol": "000001", "side": "BUY", "quantity": 200, "price": 5.0, "is_etf": False, "volume": 20000},
    ]
    bc = model.compute_basket(legs)
    # 基准成交额 = 100*10 + 200*5 = 2000；篮子滑点 = 2000 * 2bp
    assert bc.total_gross_value == pytest.approx(2000.0)
    assert bc.basket_slippage == pytest.approx(2000.0 * 2.0 / 10000)
    assert len(bc.legs) == 2
    assert bc.total_fee == pytest.approx(sum(leg.total_fee for leg in bc.legs))


# ---------------------------------------------------------------------------
# 逐笔可追溯
# ---------------------------------------------------------------------------
def test_per_trade_detail_auditable():
    model = CostModel(CostConfig())  # 默认参数，含价差与冲击
    b = model.compute("BUY", 1000, 10.0, symbol="510300", is_etf=True, volume=100000)
    names = [d.name for d in b.details]
    for expected in ("佣金", "印花税", "过户费", "价差成本(半价差)", "冲击成本"):
        assert expected in names
    d = b.to_dict()
    assert d["details"][0]["name"] == "佣金"
    assert "total_fee" in d and "slippage_cost" in d


# ---------------------------------------------------------------------------
# 成本敏感性
# ---------------------------------------------------------------------------
def test_sensitivity_commission_rate():
    rows = sensitivity_table(
        CostConfig(), "commission_rate", [0.0001, 0.0003, 0.0005],
        side="BUY", quantity=10000, price=3.0, volume=1_000_000, is_etf=True,
    )
    totals = [r["total_cost"] for r in rows]
    assert totals[0] < totals[1] < totals[2]
    assert rows[0]["impact_cost"] == pytest.approx(rows[2]["impact_cost"])  # 只变佣金，冲击不变


def test_sensitivity_impact_coef_and_spread():
    rows = sensitivity_table(
        CostConfig(), "impact_coef", [0.0, 0.05, 0.10],
        side="BUY", quantity=10000, price=3.0, volume=100_000, is_etf=True,
    )
    totals = [r["total_cost"] for r in rows]
    assert totals[0] < totals[1] < totals[2]

    rows2 = sensitivity_table(
        CostConfig(), "spread_rate", [0.0, 0.0002, 0.0004],
        side="BUY", quantity=10000, price=3.0, volume=1_000_000, is_etf=True,
    )
    totals2 = [r["total_cost"] for r in rows2]
    assert totals2[0] < totals2[1] < totals2[2]


def test_sensitivity_output_conclusion():
    """敏感性测试输出：打印一张表，证明"成本敏感"结论可跑出来。"""
    rows = sensitivity_table(
        CostConfig(), "impact_coef", [0.0, 0.03, 0.05, 0.10],
        side="BUY", quantity=10000, price=3.0, volume=100_000, is_etf=True,
    )
    print("\n=== 成本敏感性：impact_coef 对总成本的影响 ===")
    print("impact_coef | total_fee | slippage_cost | total_cost")
    for r in rows:
        print(f"{r['impact_coef']:<11} | {r['total_fee']:.4f} | {r['slippage_cost']:.4f} | {r['total_cost']:.4f}")
    assert rows[-1]["total_cost"] > rows[0]["total_cost"]
