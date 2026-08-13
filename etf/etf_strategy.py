"""B4 ETF 套利策略：消费 B2 信号流、生成 A2 订单、挂载 B3 执行。

把 B1/B2/B3 接到 A2 事件驱动引擎的桥接层：

- 构造时用 B2 ``ETFSignalGenerator`` 由折溢价序列**预生成**信号事件流（滚动判据，
  无前视），并按时间戳建索引，``on_bar`` 在信号时点触发。
- ``on_bar`` 把信号转为 A2 ``SignalEvent`` 订单流，交给引擎撮合（成本由引擎注入的
  A3 成本模型统一计费，保证扣成本口径单一、可复现）。
- 两种执行模式（凸显执行风险影响）：
  - ``ideal``：理想执行，全部腿满量成交，无同步风险。
  - ``sync_risk``：真实执行，挂载 B3 ``BasketExecutor`` 模拟逐笔时延 + 部分成交，
    只把**已成交数量**转为订单，未成交部分记为敞口/跟踪误差。

方向约定与 B2 一致：
- 溢价（premium > 0，direction=short）：卖 ETF、买篮子。
- 折价（premium < 0，direction=long）：买 ETF、卖篮子。
- 平仓与开仓相反。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from engine import SignalEvent, Strategy
from etf.basket_execution import BasketExecutor
from etf.etf_signal import (
    ACTION_CLOSE,
    ACTION_OPEN,
    DIR_LONG,
    DIR_SHORT,
    ETFSignalGenerator,
)
from etf.threshold_config import ThresholdConfig

# 执行模式
IDEAL = "ideal"              # 理想执行（满量成交）
SYNC_RISK = "sync_risk"      # 含同步风险（时延 + 部分成交）
EXECUTION_MODES = (IDEAL, SYNC_RISK)


@dataclass
class ArbitrageExecution:
    """单次套利信号的执行结果（理想满量 或 真实部分成交）。"""

    ts: pd.Timestamp
    action: str
    direction: str
    target_value: float       # 目标名义（全部腿满量）
    filled_value: float       # 实际成交名义
    unfilled_value: float     # 未成交名义（敞口）
    exposure_ratio: float     # 敞口比例 = 未成交 / 目标
    tracking_error: float     # 执行缺口加权标准差
    n_legs: int               # 腿数
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "ts": self.ts,
            "action": self.action,
            "direction": self.direction,
            "target_value": self.target_value,
            "filled_value": self.filled_value,
            "unfilled_value": self.unfilled_value,
            "exposure_ratio": self.exposure_ratio,
            "tracking_error": self.tracking_error,
            "n_legs": self.n_legs,
            "reason": self.reason,
        }


def _leg_notional(legs: List[dict], prices: Dict[str, float]) -> float:
    """篮子腿的名义价值合计（|价格 × 数量|，缺价跳过）。"""
    total = 0.0
    for leg in legs:
        price = prices.get(str(leg["symbol"]), np.nan)
        qty = float(leg["quantity"])
        if np.isfinite(price) and qty > 0:
            total += abs(float(price) * qty)
    return total


class ETFArbitrageStrategy(Strategy):
    """ETF 折溢价套利策略（A2 插件）。"""

    def __init__(
        self,
        premium: pd.Series,
        etf_symbol: str,
        etf_quantity: float = 100_000.0,
        *,
        basket: Optional[Dict[str, float]] = None,
        execution_mode: str = IDEAL,
        executor: Optional[BasketExecutor] = None,
        threshold_config: Optional[ThresholdConfig] = None,
        unit_cost_rate: Optional[float] = None,
        seed: int = 42,
    ):
        if execution_mode not in EXECUTION_MODES:
            raise ValueError(f"execution_mode 必须为 {EXECUTION_MODES} 之一，得到 {execution_mode!r}")
        self.etf_symbol = str(etf_symbol)
        self.etf_quantity = float(etf_quantity)
        self.basket = dict(basket) if basket else None
        self.execution_mode = execution_mode
        self.seed = int(seed)

        # 预生成 B2 信号事件流（无前视）
        gen = ETFSignalGenerator(
            config=threshold_config, unit_cost_rate=unit_cost_rate
        )
        self.signals: pd.DataFrame = gen.generate(premium)
        self._signal_map: Dict[pd.Timestamp, pd.Series] = {}
        for _, row in self.signals.iterrows():
            self._signal_map[pd.Timestamp(row["ts"])] = row

        # B3 执行器（真实模式挂载；理想模式不使用）
        if execution_mode == SYNC_RISK:
            self.executor: Optional[BasketExecutor] = executor or BasketExecutor(seed=seed)
        else:
            self.executor = None

        # 最近已知价格/成交量（供执行器与名义估算）
        self._last_prices: Dict[str, float] = {}
        self._last_volumes: Dict[str, float] = {}

        # 执行记录（风险/容量/报告用）
        self.executions: List[ArbitrageExecution] = []

    # ------------------------------------------------------------------
    # A2 插件入口
    # ------------------------------------------------------------------
    def on_bar(
        self,
        timestamp: pd.Timestamp,
        bars,
        portfolio,
    ) -> List[SignalEvent]:
        # 维护最近价格/成交量（执行器与名义估算）
        for sym, bar in bars.items():
            self._last_prices[str(sym)] = float(bar.close)
            self._last_volumes[str(sym)] = float(getattr(bar, "volume", 0.0) or 0.0)

        row = self._signal_map.get(timestamp)
        if row is None:
            return []
        return self._orders_for_signal(timestamp, row)

    # ------------------------------------------------------------------
    # 信号 → 订单
    # ------------------------------------------------------------------
    def _orders_for_signal(self, ts: pd.Timestamp, row: pd.Series) -> List[SignalEvent]:
        action = str(row["action"])
        direction = str(row["direction"])
        reason = str(row.get("threshold_basis", ""))

        legs = self._build_legs(action, direction)
        if not legs:
            return []

        if self.execution_mode == IDEAL:
            filled_legs = legs
            target_value = _leg_notional(legs, self._last_prices)
            filled_value = target_value
            unfilled_value = 0.0
            exposure_ratio = 0.0
            tracking_error = 0.0
        else:
            result = self.executor.execute(
                legs,
                prices=self._last_prices,
                volumes=self._last_volumes,
                basket=None,
                etf_symbol=self.etf_symbol,
            )
            filled_legs = [
                {"symbol": le.symbol, "side": le.side, "quantity": le.filled_quantity}
                for le in result.legs if le.filled_quantity > 0
            ]
            target_value = result.total_target_value
            filled_value = result.total_filled_value
            unfilled_value = result.total_unfilled_value
            exposure_ratio = result.exposure_ratio
            tracking_error = result.tracking_error

        self.executions.append(ArbitrageExecution(
            ts=ts, action=action, direction=direction,
            target_value=target_value, filled_value=filled_value,
            unfilled_value=unfilled_value, exposure_ratio=exposure_ratio,
            tracking_error=tracking_error, n_legs=len(filled_legs), reason=reason,
        ))

        events: List[SignalEvent] = []
        for leg in filled_legs:
            events.append(SignalEvent(
                timestamp=ts,
                symbol=str(leg["symbol"]),
                side=str(leg["side"]).upper(),
                quantity=float(leg["quantity"]),
                order_type="market",
                reason=reason,
            ))
        return events

    def _build_legs(self, action: str, direction: str) -> List[dict]:
        """按方向构造套利腿（ETF 腿 + 可选篮子成分腿）。"""
        if action == ACTION_OPEN:
            etf_side = "SELL" if direction == DIR_SHORT else "BUY"
            basket_side = "BUY" if direction == DIR_SHORT else "SELL"
        else:  # close：与开仓相反
            etf_side = "BUY" if direction == DIR_SHORT else "SELL"
            basket_side = "SELL" if direction == DIR_SHORT else "BUY"

        legs = [{"symbol": self.etf_symbol, "side": etf_side, "quantity": self.etf_quantity}]
        if self.basket:
            for sym, qty in self.basket.items():
                legs.append({"symbol": str(sym), "side": basket_side, "quantity": float(qty)})
        return legs

    # ------------------------------------------------------------------
    # 汇总（供容量/压力/报告）
    # ------------------------------------------------------------------
    def exposure_frame(self) -> pd.DataFrame:
        """逐笔执行记录 DataFrame。"""
        if not self.executions:
            return pd.DataFrame(columns=[
                "ts", "action", "direction", "target_value", "filled_value",
                "unfilled_value", "exposure_ratio", "tracking_error", "n_legs", "reason",
            ])
        return pd.DataFrame([e.to_dict() for e in self.executions])


__all__ = [
    "ETFArbitrageStrategy",
    "ArbitrageExecution",
    "IDEAL",
    "SYNC_RISK",
    "EXECUTION_MODES",
]
