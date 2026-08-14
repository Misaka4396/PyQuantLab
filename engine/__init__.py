"""事件驱动回测引擎（A2）。

对外暴露：引擎配置、事件、订单、撮合、组合、核算与主引擎。
"""

from engine.accounting import Accounting
from engine.config import FILL_MODES, FILL_NEXT_BAR, FILL_OPEN, EngineConfig
from engine.engine import EngineResult, EventEngine, Strategy, is_etf_symbol
from engine.events import (
    BUY,
    SELL,
    Event,
    FillEvent,
    MarketDataEvent,
    OrderEvent,
    PortfolioEvent,
    SignalEvent,
)
from engine.matching import CostModelLike, MatchingEngine
from engine.order import (
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    coerce_side,
    create_limit_order,
    create_market_order,
)
from engine.portfolio import Portfolio

__all__ = [
    "BUY",
    "FILL_MODES",
    "FILL_NEXT_BAR",
    "FILL_OPEN",
    "SELL",
    "Accounting",
    "CostModelLike",
    "EngineConfig",
    "EngineResult",
    "Event",
    "EventEngine",
    "FillEvent",
    "MarketDataEvent",
    "MatchingEngine",
    "Order",
    "OrderEvent",
    "OrderSide",
    "OrderStatus",
    "OrderType",
    "Portfolio",
    "PortfolioEvent",
    "SignalEvent",
    "Strategy",
    "coerce_side",
    "create_limit_order",
    "create_market_order",
    "is_etf_symbol",
]
