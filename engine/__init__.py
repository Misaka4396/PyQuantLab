"""事件驱动回测引擎（A2）。

对外暴露：引擎配置、事件、订单、撮合、组合、核算与主引擎。
"""
from engine.config import EngineConfig, FILL_NEXT_BAR, FILL_OPEN, FILL_MODES
from engine.events import (
    Event,
    MarketDataEvent,
    SignalEvent,
    OrderEvent,
    FillEvent,
    PortfolioEvent,
    BUY,
    SELL,
)
from engine.order import (
    Order,
    OrderSide,
    OrderType,
    OrderStatus,
    coerce_side,
    create_market_order,
    create_limit_order,
)
from engine.matching import MatchingEngine, CostModelLike
from engine.portfolio import Portfolio
from engine.accounting import Accounting
from engine.engine import EventEngine, Strategy, EngineResult, is_etf_symbol

__all__ = [
    "EngineConfig", "FILL_NEXT_BAR", "FILL_OPEN", "FILL_MODES",
    "Event", "MarketDataEvent", "SignalEvent", "OrderEvent", "FillEvent", "PortfolioEvent",
    "BUY", "SELL",
    "Order", "OrderSide", "OrderType", "OrderStatus", "coerce_side",
    "create_market_order", "create_limit_order",
    "MatchingEngine", "CostModelLike",
    "Portfolio", "Accounting",
    "EventEngine", "Strategy", "EngineResult", "is_etf_symbol",
]
