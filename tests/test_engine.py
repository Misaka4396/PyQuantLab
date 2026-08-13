"""A2 事件驱动回测引擎单元测试。

覆盖验收标准：
- 买入持有策略权益曲线与手算一致（含手算对照）
- 撮合无前视（next_bar 撮合下，当 bar 信号不成交于当 bar open）
- 多标的核算正确
- 可复现（同种子两次运行结果一致）
- 回调机制可挂载自定义策略与成本模型（解耦验证）
"""
from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from cost_config import CostConfig
from cost_model import CostModel
from engine import (
    EngineConfig,
    EventEngine,
    FillEvent,
    MarketDataEvent,
    PortfolioEvent,
    SignalEvent,
    Strategy,
)
from engine.order import OrderStatus


# ---------------------------------------------------------------------------
# 测试辅助
# ---------------------------------------------------------------------------
def make_df(index, opens, closes, volumes=None) -> pd.DataFrame:
    """构造单标的 OHLCV DataFrame（index=datetime）。"""
    idx = pd.to_datetime(index)
    opens = list(opens)
    closes = list(closes)
    high = [max(o, c) for o, c in zip(opens, closes)]
    low = [min(o, c) for o, c in zip(opens, closes)]
    vol = list(volumes) if volumes else [1000.0] * len(idx)
    df = pd.DataFrame({
        "open": opens, "high": high, "low": low, "close": closes, "volume": vol,
    }, index=idx)
    df.index.name = "datetime"
    return df


def simple_cost() -> CostModel:
    """只收佣金（费率 0.1%）的简单成本模型，便于手算对照。"""
    cfg = CostConfig(
        commission_rate=0.001,
        min_commission=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        spread_rate=0.0,
        impact_coef=0.0,
    )
    return CostModel(cfg)


class BuyAndHold(Strategy):
    """首个 bar 买入固定数量后持有。"""

    def __init__(self, symbol: str, quantity: float):
        self.symbol = symbol
        self.quantity = quantity
        self._bought = False

    def on_bar(self, timestamp, bars, portfolio):
        if self._bought:
            return []
        if self.symbol not in bars:
            return []
        self._bought = True
        return [SignalEvent(
            timestamp=timestamp, symbol=self.symbol, side="BUY",
            quantity=self.quantity, order_type="market", reason="首次建仓",
        )]


class MultiBuy(Strategy):
    """首个 bar 一次性买入多标的（模拟 ETF 篮子下单）。"""

    def __init__(self, orders: dict):
        self.orders = orders
        self._done = False

    def on_bar(self, timestamp, bars, portfolio):
        if self._done:
            return []
        if not all(s in bars for s in self.orders):
            return []
        self._done = True
        return [
            SignalEvent(timestamp=timestamp, symbol=s, side="BUY",
                        quantity=q, order_type="market", reason="篮子买入")
            for s, q in self.orders.items()
        ]


class LimitBuy(Strategy):
    """首个 bar 下限价买单。"""

    def __init__(self, symbol: str, quantity: float, limit_price: float):
        self.symbol = symbol
        self.quantity = quantity
        self.limit_price = limit_price
        self._done = False

    def on_bar(self, timestamp, bars, portfolio):
        if self._done or self.symbol not in bars:
            return []
        self._done = True
        return [SignalEvent(
            timestamp=timestamp, symbol=self.symbol, side="BUY", quantity=self.quantity,
            order_type="limit", limit_price=self.limit_price, reason="限价买入",
        )]


class DummyCostModel:
    """极简成本模型（固定 1 元费用），验证 engine 通过接口注入、不硬编码。"""

    def compute(self, side, quantity, price, symbol="", is_etf=False, volume=0.0):
        return SimpleNamespace(
            exec_price=price, commission=1.0, stamp_tax=0.0,
            transfer_fee=0.0, total_fee=1.0,
        )


# ---------------------------------------------------------------------------
# 买入持有 + 手算对照（next_bar 无前视）
# ---------------------------------------------------------------------------
def test_buy_and_hold_equity_matches_manual():
    data = {
        "510300": make_df(
            ["2024-01-02", "2024-01-03", "2024-01-04"],
            opens=[10.0, 10.5, 11.0],
            closes=[10.0, 11.0, 12.0],
        ),
    }
    engine = EventEngine(
        config=EngineConfig(initial_cash=1_000_000.0, seed=42, fill_mode="next_bar"),
        data=data,
        cost_model=simple_cost(),
        strategy=BuyAndHold("510300", 100.0),
    )
    result = engine.run()

    # 手算：t0 无成交；t1 以开盘价 10.5 成交 100 股，佣金 10.5*100*0.001=1.05；
    # t1 权益 = 1000000 - 1050 - 1.05 + 100*11 = 1000048.95；
    # t2 权益 = 998948.95 + 100*12 = 1000148.95
    expected = [1_000_000.0, 1_000_048.95, 1_000_148.95]
    equity = result.equity_curve["equity"].tolist()
    assert equity == pytest.approx(expected, rel=1e-9)

    assert len(result.fills) == 1
    fill = result.fills.iloc[0]
    assert fill["symbol"] == "510300"
    assert fill["fill_price"] == pytest.approx(10.5)
    assert fill["quantity"] == pytest.approx(100.0)
    assert fill["commission"] == pytest.approx(1.05)


def test_next_bar_no_lookahead_fill_at_next_open():
    """next_bar 撮合：当 bar 信号不成交于当 bar open，而是下一 bar open。"""
    data = {
        "510300": make_df(
            ["2024-01-02", "2024-01-03"],
            opens=[10.0, 99.0],   # 下一 bar 开盘价与当 bar 开盘价差异极大
            closes=[10.0, 99.0],
        ),
    }
    engine = EventEngine(
        config=EngineConfig(initial_cash=1_000_000.0, fill_mode="next_bar"),
        data=data,
        cost_model=simple_cost(),
        strategy=BuyAndHold("510300", 100.0),
    )
    result = engine.run()

    assert len(result.fills) == 1
    fill = result.fills.iloc[0]
    # 成交价必须是下一 bar 开盘 99，而非当 bar 开盘 10
    assert fill["fill_price"] == pytest.approx(99.0)
    assert pd.Timestamp(fill["timestamp"]) == pd.Timestamp("2024-01-03")
    # 当 bar（首个 bar）期末尚未持仓，权益仍为初始资金
    assert result.equity_curve["equity"].iloc[0] == pytest.approx(1_000_000.0)


def test_signal_on_last_bar_not_filled_next_bar():
    """next_bar 撮合下，最后一根 bar 的信号无下一 bar，不成交。"""
    data = {"510300": make_df(["2024-01-02"], opens=[10.0], closes=[10.0])}
    engine = EventEngine(
        config=EngineConfig(initial_cash=1_000_000.0, fill_mode="next_bar"),
        data=data,
        cost_model=simple_cost(),
        strategy=BuyAndHold("510300", 100.0),
    )
    result = engine.run()
    assert len(result.fills) == 0
    assert result.portfolio.position("510300") == 0.0
    assert not result.orders.empty
    assert result.orders.iloc[0]["status"] == OrderStatus.REJECTED.value


def test_open_mode_fills_at_current_bar_open():
    """open 撮合：信号当 bar 即按当 bar 开盘价成交。"""
    data = {
        "510300": make_df(
            ["2024-01-02", "2024-01-03"],
            opens=[10.0, 10.5],
            closes=[10.0, 11.0],
        ),
    }
    engine = EventEngine(
        config=EngineConfig(initial_cash=1_000_000.0, fill_mode="open"),
        data=data,
        cost_model=simple_cost(),
        strategy=BuyAndHold("510300", 100.0),
    )
    result = engine.run()
    assert len(result.fills) == 1
    assert result.fills.iloc[0]["fill_price"] == pytest.approx(10.0)
    assert pd.Timestamp(result.fills.iloc[0]["timestamp"]) == pd.Timestamp("2024-01-02")


# ---------------------------------------------------------------------------
# 多标的核算
# ---------------------------------------------------------------------------
def test_multi_symbol_accounting():
    index = ["2024-01-02", "2024-01-03", "2024-01-04"]
    data = {
        "510300": make_df(index, opens=[10.0, 10.5, 11.0], closes=[10.0, 11.0, 12.0]),
        "510500": make_df(index, opens=[5.0, 5.0, 5.5], closes=[5.0, 5.5, 6.0]),
    }
    engine = EventEngine(
        config=EngineConfig(initial_cash=1_000_000.0, fill_mode="next_bar"),
        data=data,
        cost_model=simple_cost(),
        strategy=MultiBuy({"510300": 100.0, "510500": 100.0}),
    )
    result = engine.run()

    # 手算 t1 成交：510300 支出 100*10.5*1.001=1051.05；510500 支出 100*5.0*1.001=500.5
    cash_after = 1_000_000.0 - 1051.05 - 500.5
    assert result.portfolio.cash == pytest.approx(cash_after)
    assert result.portfolio.position("510300") == pytest.approx(100.0)
    assert result.portfolio.position("510500") == pytest.approx(100.0)

    # t2 权益 = 现金 + 100*12 + 100*6
    expected_final = cash_after + 1200.0 + 600.0
    assert result.equity_curve["equity"].iloc[-1] == pytest.approx(expected_final, rel=1e-9)


# ---------------------------------------------------------------------------
# 复现
# ---------------------------------------------------------------------------
def test_reproducible_same_seed():
    index = ["2024-01-02", "2024-01-03", "2024-01-04"]
    data = {
        "510300": make_df(index, opens=[10.0, 10.5, 11.0], closes=[10.0, 11.0, 12.0]),
        "510500": make_df(index, opens=[5.0, 5.0, 5.5], closes=[5.0, 5.5, 6.0]),
    }

    def run_once():
        engine = EventEngine(
            config=EngineConfig(initial_cash=1_000_000.0, seed=42, fill_mode="next_bar"),
            data=data,
            cost_model=simple_cost(),
            strategy=MultiBuy({"510300": 100.0, "510500": 200.0}),
        )
        return engine.run()

    r1, r2 = run_once(), run_once()
    pd.testing.assert_frame_equal(r1.equity_curve, r2.equity_curve)
    pd.testing.assert_frame_equal(r1.fills, r2.fills)


# ---------------------------------------------------------------------------
# 回调 / 插件机制
# ---------------------------------------------------------------------------
def test_callback_mechanism():
    data = {"510300": make_df(
        ["2024-01-02", "2024-01-03"],
        opens=[10.0, 10.5], closes=[10.0, 11.0],
    )}
    engine = EventEngine(
        config=EngineConfig(initial_cash=1_000_000.0, fill_mode="next_bar"),
        data=data,
        cost_model=simple_cost(),
    )
    engine.register_strategy(BuyAndHold("510300", 100.0))

    fills, portfolios = [], []
    engine.register_callback(FillEvent, lambda ev, eng: fills.append(ev))
    engine.register_callback(PortfolioEvent, lambda ev, eng: portfolios.append(ev))

    engine.run()

    assert len(fills) == 1
    assert isinstance(fills[0], FillEvent)
    assert len(portfolios) == 2  # 两个 bar 各一次盯市
    assert portfolios[-1].equity > 0


def test_cost_model_decoupled_via_interface():
    """注入自定义成本模型（非 CostModel），engine 应原样使用其输出（不硬编码）。"""
    data = {"510300": make_df(
        ["2024-01-02", "2024-01-03"],
        opens=[10.0, 10.5], closes=[10.0, 11.0],
    )}
    engine = EventEngine(
        config=EngineConfig(initial_cash=1_000_000.0, fill_mode="next_bar"),
        data=data,
        cost_model=DummyCostModel(),
        strategy=BuyAndHold("510300", 100.0),
    )
    result = engine.run()
    assert len(result.fills) == 1
    assert result.fills.iloc[0]["total_fee"] == pytest.approx(1.0)
    assert result.fills.iloc[0]["commission"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 限价单 / 资金不足
# ---------------------------------------------------------------------------
def test_limit_buy_above_open_fills_at_open():
    data = {"510300": make_df(
        ["2024-01-02", "2024-01-03"],
        opens=[10.0, 10.5], closes=[10.0, 11.0],
    )}
    engine = EventEngine(
        config=EngineConfig(initial_cash=1_000_000.0, fill_mode="next_bar"),
        data=data,
        cost_model=simple_cost(),
        strategy=LimitBuy("510300", 100.0, limit_price=10.8),
    )
    result = engine.run()
    assert len(result.fills) == 1
    # 开盘价 10.5 <= 限价 10.8，按开盘价成交
    assert result.fills.iloc[0]["fill_price"] == pytest.approx(10.5)


def test_limit_buy_below_open_not_filled():
    data = {"510300": make_df(
        ["2024-01-02", "2024-01-03"],
        opens=[10.0, 10.5], closes=[10.0, 11.0],
    )}
    engine = EventEngine(
        config=EngineConfig(initial_cash=1_000_000.0, fill_mode="next_bar"),
        data=data,
        cost_model=simple_cost(),
        strategy=LimitBuy("510300", 100.0, limit_price=10.2),
    )
    result = engine.run()
    # 下一 bar 开盘 10.5 > 限价 10.2，无法成交
    assert len(result.fills) == 0
    assert result.orders.iloc[0]["status"] == OrderStatus.REJECTED.value


def test_insufficient_funds_rejected():
    data = {"510300": make_df(
        ["2024-01-02", "2024-01-03"],
        opens=[10.0, 10.5], closes=[10.0, 11.0],
    )}
    # 初始资金不足以买 100 股（含手续费）
    engine = EventEngine(
        config=EngineConfig(initial_cash=100.0, fill_mode="next_bar"),
        data=data,
        cost_model=simple_cost(),
        strategy=BuyAndHold("510300", 100.0),
    )
    result = engine.run()
    assert len(result.fills) == 0
    assert result.portfolio.position("510300") == 0.0
    assert result.portfolio.cash == pytest.approx(100.0)
    assert result.orders.iloc[0]["status"] == OrderStatus.REJECTED.value


# ---------------------------------------------------------------------------
# 长表输入（A1 DataLoader 风格）
# ---------------------------------------------------------------------------
def test_long_format_data_input():
    idx = pd.to_datetime(["2024-01-02", "2024-01-03"])
    long_df = pd.DataFrame({
        "open": [10.0, 10.5, 5.0, 5.5],
        "high": [10.0, 11.0, 5.0, 5.5],
        "low": [10.0, 10.5, 5.0, 5.5],
        "close": [10.0, 11.0, 5.0, 5.5],
        "volume": [1000.0] * 4,
        "symbol": ["510300", "510300", "510500", "510500"],
        "datetime": [idx[0], idx[1], idx[0], idx[1]],
    })
    engine = EventEngine(
        config=EngineConfig(initial_cash=1_000_000.0, fill_mode="next_bar"),
        data=long_df,
        cost_model=simple_cost(),
        strategy=MultiBuy({"510300": 100.0, "510500": 100.0}),
    )
    result = engine.run()
    assert len(result.fills) == 2
    assert set(result.fills["symbol"]) == {"510300", "510500"}
