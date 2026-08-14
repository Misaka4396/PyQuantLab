"""事件定义：事件循环中传递的数据载体。

事件流：MarketDataEvent → SignalEvent → OrderEvent → FillEvent → PortfolioEvent
每个事件都带 ``timestamp``（bar 时间戳），保证可追溯与可审计。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

# 方向常量
BUY = "BUY"
SELL = "SELL"


@dataclass
class Event:
    """事件基类。"""

    timestamp: pd.Timestamp


@dataclass
class MarketDataEvent(Event):
    """行情事件：某一证券在某 bar 的 OHLCV。"""

    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    amount: float = 0.0


@dataclass
class SignalEvent(Event):
    """信号事件：策略在某一时点发出的交易意图。"""

    symbol: str
    side: str  # BUY / SELL
    quantity: float
    order_type: str = "market"  # market / limit
    limit_price: float | None = None
    reason: str = ""


@dataclass
class OrderEvent(Event):
    """订单事件：信号转化为订单后的载体。"""

    order_id: str
    symbol: str
    side: str
    quantity: float
    order_type: str = "market"
    limit_price: float | None = None
    status: str = "created"  # created / pending / filled / rejected / cancelled
    fill_timestamp: pd.Timestamp | None = None  # 计划撮合时点（next_bar 模式）


@dataclass
class FillEvent(Event):
    """成交事件：撮合 + 成本模型计算后的成交结果。"""

    order_id: str
    symbol: str
    side: str
    quantity: float
    fill_price: float  # 撮合参考价（如开盘价）
    exec_price: float  # 滑点调整后成交价
    commission: float = 0.0
    stamp_tax: float = 0.0
    transfer_fee: float = 0.0
    total_fee: float = 0.0
    cost_breakdown: Any = None  # 成本明细对象（由成本模型返回，审计用）

    @property
    def cash_flow(self) -> float:
        """成交净现金流：买入为负（含费流出）、卖出为正（扣费后流入）。"""
        if self.side == SELL:
            return self.exec_price * self.quantity - self.total_fee
        return -(self.exec_price * self.quantity + self.total_fee)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "fill_price": self.fill_price,
            "exec_price": self.exec_price,
            "commission": self.commission,
            "stamp_tax": self.stamp_tax,
            "transfer_fee": self.transfer_fee,
            "total_fee": self.total_fee,
            "cash_flow": self.cash_flow,
        }


@dataclass
class PortfolioEvent(Event):
    """组合事件：某一 bar 期末的持仓与资金快照。"""

    cash: float
    positions: dict[str, float]
    market_value: float
    equity: float
    available_cash: float = 0.0
    total_fees: float = 0.0
