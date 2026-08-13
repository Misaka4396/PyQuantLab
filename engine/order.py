"""订单类型：market/limit，bar 内撮合 open/next-bar。

订单本身只描述意图与状态，不包含撮合与成本逻辑（撮合在 matching.py）。
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum
from typing import Optional

import pandas as pd

from engine.events import BUY, SELL


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"


class OrderStatus(str, Enum):
    CREATED = "created"
    PENDING = "pending"
    FILLED = "filled"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


def coerce_side(side) -> OrderSide:
    """把 BUY/SELL/buy/sell/OrderSide 归一化为 OrderSide。"""
    if isinstance(side, OrderSide):
        return side
    s = str(side).upper()
    if s not in (BUY, SELL):
        raise ValueError(f"非法方向: {side!r}")
    return OrderSide(s)


@dataclass
class Order:
    """订单对象。"""

    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    timestamp: Optional[pd.Timestamp] = None        # 下单时点（信号所在 bar）
    status: OrderStatus = OrderStatus.CREATED
    fill_timestamp: Optional[pd.Timestamp] = None   # 计划撮合时点（next_bar 模式）

    def to_dict(self) -> dict:
        d = asdict(self)
        d["side"] = self.side.value
        d["order_type"] = self.order_type.value
        d["status"] = self.status.value
        return d


def create_market_order(
    order_id: str,
    symbol: str,
    side,
    quantity: float,
    timestamp: Optional[pd.Timestamp] = None,
) -> Order:
    """创建市价单。"""
    return Order(
        order_id=order_id,
        symbol=symbol,
        side=coerce_side(side),
        quantity=float(quantity),
        order_type=OrderType.MARKET,
        timestamp=timestamp,
    )


def create_limit_order(
    order_id: str,
    symbol: str,
    side,
    quantity: float,
    limit_price: float,
    timestamp: Optional[pd.Timestamp] = None,
) -> Order:
    """创建限价单。"""
    return Order(
        order_id=order_id,
        symbol=symbol,
        side=coerce_side(side),
        quantity=float(quantity),
        order_type=OrderType.LIMIT,
        limit_price=float(limit_price),
        timestamp=timestamp,
    )
