"""撮合逻辑：按 bar 撮合（open 参考价），支持 market/limit 单。

与成本模型解耦：撮合器只负责"以什么价成交、能否成交"，随后调用注入的成本模型
接口（``CostModelLike``）计算滑点后的成交价与费用，生成 FillEvent。撮合器不
import cost_model.py，只依赖接口约定。
"""

from __future__ import annotations

from typing import Any, Protocol

from engine.events import FillEvent, MarketDataEvent
from engine.order import Order, OrderSide, OrderType


class CostModelLike(Protocol):
    """成本模型接口约定（engine 与 A3 成本模型的解耦点）。

    实现类只需提供 ``compute`` 方法，返回对象需包含以下属性：
    exec_price（滑点后成交价）、commission、stamp_tax、transfer_fee、total_fee。
    """

    def compute(
        self,
        side: str,
        quantity: float,
        price: float,
        symbol: str = "",
        is_etf: bool = False,
        volume: float = 0.0,
    ) -> Any:
        """计算单笔成交成本明细。"""
        ...


class MatchingEngine:
    """按 bar 撮合器。

    撮合规则（确定性、可复现）：
    - 参考价：撮合 bar 的 **开盘价**（open）。
    - market 单：以开盘价无条件成交。
    - limit 买单：开盘价 <= 限价则成交（成交价=开盘价），否则不成交（返回 None）。
    - limit 卖单：开盘价 >= 限价则成交（成交价=开盘价），否则不成交（返回 None）。
    - 成交后调用成本模型，得到滑点后成交价（exec_price）与费用明细。
    """

    def __init__(self, cost_model: CostModelLike):
        if cost_model is None:
            raise ValueError("MatchingEngine 需要注入成本模型（cost_model 不能为 None）")
        self.cost_model = cost_model

    def match(
        self,
        order: Order,
        bar: MarketDataEvent,
        is_etf: bool = False,
    ) -> FillEvent | None:
        """对订单在给定 bar 上撮合，返回 FillEvent 或 None（限价未成交）。"""
        ref_price = float(bar.open)

        # 限价单可成交性判断
        if order.order_type == OrderType.LIMIT:
            if order.limit_price is None:
                return None
            if order.side == OrderSide.BUY and ref_price > order.limit_price + 1e-12:
                return None
            if order.side == OrderSide.SELL and ref_price < order.limit_price - 1e-12:
                return None

        # 调用注入的成本模型
        breakdown = self.cost_model.compute(
            side=order.side.value,
            quantity=order.quantity,
            price=ref_price,
            symbol=order.symbol,
            is_etf=is_etf,
            volume=bar.volume,
        )

        return FillEvent(
            timestamp=bar.timestamp,
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side.value,
            quantity=order.quantity,
            fill_price=ref_price,
            exec_price=breakdown.exec_price,
            commission=breakdown.commission,
            stamp_tax=breakdown.stamp_tax,
            transfer_fee=breakdown.transfer_fee,
            total_fee=breakdown.total_fee,
            cost_breakdown=breakdown,
        )
