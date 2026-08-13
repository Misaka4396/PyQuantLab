"""B3 篮子同步执行与申赎模拟单元测试。

覆盖验收标准：
- 能模拟部分成交与延迟（设置部分成交概率后输出有未成交部分）
- 执行成本与 A3 一致（调用 cost_model.compute_basket 计费）
- 申赎开关不影响二级对冲模式（关闭时走二级市场成交路径）
- 输出敞口（tracking error）供风险评估
"""
from __future__ import annotations

import pandas as pd
import pytest

from cost_config import CostConfig
from cost_model import CostModel
from etf.execution_config import ExecutionConfig
from etf.basket_execution import BasketExecutor, make_basket_legs
from etf.pcf_parser import PCFBasket, PCFConstituent


def _basket() -> PCFBasket:
    return PCFBasket(
        etf_code="510300",
        trade_date=pd.Timestamp("2024-01-02"),
        effective_date=pd.Timestamp("2024-01-03"),
        creation_unit=1000.0,
        cash_component=50.0,
        constituents=[
            PCFConstituent(symbol="A", quantity=100.0, price=2.0),
            PCFConstituent(symbol="B", quantity=100.0, price=3.0),
        ],
    )


def _zero_cost_model() -> CostModel:
    """零滑点、零佣金成本模型，便于精确断言。"""
    return CostModel(CostConfig(
        commission_rate=0.0, min_commission=0.0, stamp_tax_rate=0.0,
        transfer_fee_rate=0.0, spread_rate=0.0, impact_coef=0.0, basket_slippage_bps=0.0,
    ))


# ---------------------------------------------------------------------------
# 部分成交 + 延迟
# ---------------------------------------------------------------------------
def test_partial_fill_produces_unfilled():
    cfg = ExecutionConfig(
        partial_fill_prob=1.0, partial_fill_ratio_min=0.5, partial_fill_ratio_max=0.6,
        delay_minutes_min=0.0, delay_minutes_max=0.0, seed=1,
    )
    executor = BasketExecutor(cfg, cost_model=_zero_cost_model())
    res = executor.execute(
        [{"symbol": "A", "side": "BUY", "quantity": 1000},
         {"symbol": "B", "side": "BUY", "quantity": 1000}],
        prices={"A": 10.0, "B": 5.0},
    )
    assert all(le.is_partial for le in res.legs)
    assert all(le.unfilled_quantity > 0 for le in res.legs)
    assert res.total_unfilled_value > 0
    assert res.exposure_ratio > 0
    assert res.tracking_error >= 0


def test_delay_configurable():
    cfg = ExecutionConfig(delay_minutes_min=3.0, delay_minutes_max=3.0,
                          partial_fill_prob=0.0, seed=0)
    executor = BasketExecutor(cfg, cost_model=_zero_cost_model())
    res = executor.execute([{"symbol": "A", "side": "BUY", "quantity": 10}], prices={"A": 1.0})
    assert res.legs[0].delay_minutes == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# 执行成本与 A3 一致
# ---------------------------------------------------------------------------
def test_execution_cost_consistent_with_a3():
    cfg = ExecutionConfig(partial_fill_prob=0.0, delay_minutes_min=0.0, delay_minutes_max=0.0, seed=0)
    cost_cfg = CostConfig(
        commission_rate=0.001, min_commission=0.0, stamp_tax_rate=0.0,
        transfer_fee_rate=0.0, spread_rate=0.002, impact_coef=0.0, basket_slippage_bps=2.0,
    )
    model = CostModel(cost_cfg)
    executor = BasketExecutor(cfg, cost_model=model)
    res = executor.execute(
        [{"symbol": "A", "side": "BUY", "quantity": 100},
         {"symbol": "B", "side": "SELL", "quantity": 200}],
        prices={"A": 10.0, "B": 5.0},
    )
    manual = model.compute_basket([
        {"symbol": "A", "side": "BUY", "quantity": 100, "price": 10.0, "is_etf": False, "volume": 0.0},
        {"symbol": "B", "side": "SELL", "quantity": 200, "price": 5.0, "is_etf": False, "volume": 0.0},
    ])
    assert res.basket_cost.total_fee == pytest.approx(manual.total_fee)
    assert res.basket_cost.total_slippage == pytest.approx(manual.total_slippage)
    assert res.basket_cost.basket_slippage == pytest.approx(manual.basket_slippage)
    # 单腿成交价与 cost_model.compute 一致
    b = model.compute("BUY", 100, 10.0, symbol="A", is_etf=False, volume=0.0)
    assert res.legs[0].exec_price == pytest.approx(b.exec_price)


# ---------------------------------------------------------------------------
# 申赎开关
# ---------------------------------------------------------------------------
def test_creation_redemption_off_goes_secondary():
    cfg = ExecutionConfig(enable_creation_redemption=False, partial_fill_prob=0.0,
                          delay_minutes_min=0.0, delay_minutes_max=0.0)
    executor = BasketExecutor(cfg, cost_model=_zero_cost_model())
    res = executor.execute(
        [{"symbol": "510300", "side": "BUY", "quantity": 1000}], prices={"510300": 3.0},
    )
    assert res.creation_redemption is None       # 开关关闭 → 无申赎结果
    assert res.basket_cost is not None           # 走二级市场成交路径
    assert res.basket_cost.total_gross_value > 0


def test_creation_redemption_on_produces_cr():
    basket = _basket()
    cfg = ExecutionConfig(
        enable_creation_redemption=True, creation_unit=1000.0, creation_fee=10.0,
        partial_fill_prob=0.0, delay_minutes_min=0.0, delay_minutes_max=0.0,
    )
    executor = BasketExecutor(cfg, cost_model=_zero_cost_model())
    res = executor.execute(
        [{"symbol": "510300", "side": "SELL", "quantity": 2500}],
        prices={"510300": 3.0}, basket=basket, etf_symbol="510300",
    )
    cr = res.creation_redemption
    assert cr is not None
    assert cr["type"] == "redeem"                 # 卖 ETF → 赎回
    assert cr["units"] == 2                       # 2500 // 1000
    assert cr["fee"] == pytest.approx(10.0)
    assert cr["confirm_day"] == "T"


def test_make_basket_legs_direction():
    basket = _basket()
    legs = make_basket_legs(basket, etf_quantity=1000, etf_price=3.0,
                            stock_prices={"A": 2.0, "B": 3.0}, direction="long")
    # long：ETF 腿 BUY，成分股腿 SELL
    etf_leg = next(l for l in legs if l["symbol"] == "510300")
    assert etf_leg["side"] == "BUY"
    stock_legs = [l for l in legs if l["symbol"] != "510300"]
    assert all(l["side"] == "SELL" for l in stock_legs)
    # 篮子名义 1000*3=3000；A 权重 200/(200+300)=0.4 → qty = 3000*0.4/2 = 600
    a = next(l for l in stock_legs if l["symbol"] == "A")
    assert a["quantity"] == pytest.approx(600.0)
