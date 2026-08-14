"""组合核算：现金、持仓、市值、权益、可用资金、手续费预留。

手续费预留：买入下单时按"成交额 + 预估手续费"核算资金占用，避免可用资金
不足以覆盖手续费导致超买。本实现中，撮合器在生成 FillEvent 时已含 total_fee，
引擎在买入前用 ``can_afford(cash_out)`` 校验（cash_out 已含手续费），
因此"可用资金"天然覆盖手续费预留。
"""

from __future__ import annotations

from engine.events import BUY, FillEvent


class Portfolio:
    """组合账户：现金 + 多标的持仓。"""

    def __init__(self, initial_cash: float):
        self.initial_cash = float(initial_cash)
        self.cash = float(initial_cash)
        self.positions: dict[str, float] = {}  # symbol -> 持仓数量
        self.total_fees: float = 0.0  # 累计已支付手续费

    # ------------------------------------------------------------------
    # 资金与持仓变更
    # ------------------------------------------------------------------
    def apply_fill(self, fill: FillEvent) -> None:
        """按成交更新现金与持仓（买入资金流出、卖出资金流入，均含手续费）。"""
        self.cash += fill.cash_flow
        if fill.side == BUY:
            self.positions[fill.symbol] = self.positions.get(fill.symbol, 0.0) + fill.quantity
        else:
            self.positions[fill.symbol] = self.positions.get(fill.symbol, 0.0) - fill.quantity
            if abs(self.positions[fill.symbol]) < 1e-12:
                del self.positions[fill.symbol]
        self.total_fees += fill.total_fee

    def position(self, symbol: str) -> float:
        """返回某标的持仓数量（无持仓返回 0）。"""
        return self.positions.get(symbol, 0.0)

    # ------------------------------------------------------------------
    # 估值
    # ------------------------------------------------------------------
    def market_value(self, prices: dict[str, float]) -> float:
        """按给定价格表计算持仓市值。"""
        return sum(
            qty * prices[sym] for sym, qty in self.positions.items() if sym in prices and qty != 0
        )

    def equity(self, prices: dict[str, float]) -> float:
        """总权益 = 现金 + 持仓市值。"""
        return self.cash + self.market_value(prices)

    # ------------------------------------------------------------------
    # 可用资金与手续费预留
    # ------------------------------------------------------------------
    @property
    def available_cash(self) -> float:
        """可用资金（现金余额，手续费已在成交时从现金中扣除/预留）。"""
        return self.cash

    def can_afford(self, cash_out: float) -> bool:
        """买入资金校验：现金是否足以覆盖"成交额 + 手续费"（cash_out 为正数）。"""
        return self.cash >= float(cash_out) - 1e-12

    # ------------------------------------------------------------------
    # 快照
    # ------------------------------------------------------------------
    def snapshot(self, prices: dict[str, float]) -> dict:
        """生成组合快照（供 PortfolioEvent 与盯市记录使用）。"""
        return {
            "cash": self.cash,
            "positions": dict(self.positions),
            "market_value": self.market_value(prices),
            "equity": self.equity(prices),
            "available_cash": self.available_cash,
            "total_fees": self.total_fees,
        }
