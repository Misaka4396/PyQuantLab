"""B3 篮子同步执行与申赎模拟。

ETF 套利最大风险是"篮子同步成交困难"。本模块模拟真实执行不确定性：
- 逐笔时延（可配 1-5 分钟）与部分成交（概率/比例可配）。
- 滑点：复用 A3 ``cost_model.compute_basket`` 逐只计费 + 篮子整体滑点。
- 敞口：未成交部分产生 tracking error（执行缺口加权标准差）。
- 申赎（可选，开关默认关闭）：AP 实物/现金申赎、最小申赎单位、费用、T 日确认。
  关闭时走二级市场成交路径（compute_basket）；开启时额外输出申赎结果。

执行成本与 A3 一致：本模块不自行发明费率，全部经 CostModel 计费。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from cost_model import BasketCost, CostModel
from etf.execution_config import ExecutionConfig
from etf.pcf_parser import PCFBasket

BUY = "BUY"
SELL = "SELL"


@dataclass
class LegExecution:
    """单腿执行结果。"""

    symbol: str
    side: str
    quantity: float            # 目标数量
    filled_quantity: float     # 实际成交数量
    unfilled_quantity: float   # 未成交数量
    ref_price: float           # 基准价（撮合参考价）
    exec_price: float          # 滑点后成交价（由成本模型）
    delay_minutes: float       # 成交时延（分钟）
    is_partial: bool           # 是否部分成交
    total_fee: float = 0.0
    slippage_cost: float = 0.0

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "filled_quantity": self.filled_quantity,
            "unfilled_quantity": self.unfilled_quantity,
            "ref_price": self.ref_price,
            "exec_price": self.exec_price,
            "delay_minutes": self.delay_minutes,
            "is_partial": self.is_partial,
            "total_fee": self.total_fee,
            "slippage_cost": self.slippage_cost,
        }


@dataclass
class ExecutionResult:
    """一次篮子执行的完整结果。"""

    legs: List[LegExecution] = field(default_factory=list)
    basket_cost: Optional[BasketCost] = None        # 二级市场成本（A3）
    creation_redemption: Optional[dict] = None      # 申赎结果（开关开启时有）
    total_target_value: float = 0.0                 # 目标名义
    total_filled_value: float = 0.0                 # 实际成交名义
    total_unfilled_value: float = 0.0               # 未成交名义（敞口）
    exposure_ratio: float = 0.0                     # 敞口比例 = 未成交 / 目标
    tracking_error: float = 0.0                     # 跟踪误差（执行缺口加权 std）

    def to_dataframe(self) -> pd.DataFrame:
        """逐腿执行明细 DataFrame。"""
        return pd.DataFrame([leg.to_dict() for leg in self.legs])

    def summary(self) -> dict:
        """执行结果汇总（供风险评估 / 报告）。"""
        bc = self.basket_cost
        return {
            "total_target_value": self.total_target_value,
            "total_filled_value": self.total_filled_value,
            "total_unfilled_value": self.total_unfilled_value,
            "exposure_ratio": self.exposure_ratio,
            "tracking_error": self.tracking_error,
            "total_fee": bc.total_fee if bc else 0.0,
            "total_slippage": bc.total_slippage if bc else 0.0,
            "basket_slippage": bc.basket_slippage if bc else 0.0,
            "creation_redemption": self.creation_redemption,
        }


def _weighted_std(values: Sequence[float], weights: Sequence[float]) -> float:
    """加权标准差（执行缺口口径）。"""
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    if v.size == 0 or w.sum() <= 0:
        return 0.0
    mean = float(np.average(v, weights=w))
    var = float(np.average((v - mean) ** 2, weights=w))
    return float(np.sqrt(var))


def make_basket_legs(
    basket: PCFBasket,
    etf_quantity: float,
    etf_price: float,
    stock_prices: Dict[str, float],
    direction: str = "long",
) -> List[dict]:
    """由 PCF 篮子构造套利腿（供 BasketExecutor.execute 使用）。

    direction="long"（折价套利）：买 ETF、卖成分股。
    direction="short"（溢价套利）：卖 ETF、买成分股。
    成分股数量 = 篮子名义 × 权重 / 价格。
    """
    if direction == "long":
        etf_side, stock_side = BUY, SELL
    elif direction == "short":
        etf_side, stock_side = SELL, BUY
    else:
        raise ValueError(f"direction 需为 long/short，得到 {direction!r}")

    legs = [{"symbol": basket.etf_code, "side": etf_side, "quantity": float(etf_quantity)}]
    weights = basket.weights()
    basket_notional = float(etf_quantity) * float(etf_price)
    for c in basket.constituents:
        w = weights.get(c.symbol, 0.0)
        px = float(stock_prices.get(c.symbol, np.nan)) if c.symbol in stock_prices else np.nan
        if not np.isfinite(px) or px <= 0 or w <= 0:
            continue
        legs.append({"symbol": c.symbol, "side": stock_side,
                     "quantity": basket_notional * w / px})
    return legs


class BasketExecutor:
    """篮子同步执行模拟器。"""

    def __init__(
        self,
        config: Optional[ExecutionConfig] = None,
        cost_model: Optional[CostModel] = None,
        seed: Optional[int] = None,
    ):
        self.config = config or ExecutionConfig()
        self.config.validate()
        self.cost_model = cost_model or CostModel()
        self._rng = np.random.default_rng(seed if seed is not None else self.config.seed)

    # ------------------------------------------------------------------
    def execute(
        self,
        legs: Sequence[dict],
        prices: Dict[str, float],
        volumes: Optional[Dict[str, float]] = None,
        basket: Optional[PCFBasket] = None,
        etf_symbol: Optional[str] = None,
    ) -> ExecutionResult:
        """模拟一次篮子同步执行。

        - legs：[{symbol, side, quantity}]（ETF 腿 + 成分股腿）。
        - prices：{symbol: 基准价}。
        - volumes：{symbol: 成交量}（冲击成本参与率用，缺省 0 → 无冲击）。
        - basket：PCF 篮子（启用申赎时必需）。
        - etf_symbol：ETF 腿代码（用于 is_etf 判断与申赎定位）。
        """
        cfg = self.config
        volumes = volumes or {}

        # 1) 逐笔时延 + 部分成交
        leg_execs: List[LegExecution] = []
        for leg in legs:
            sym = str(leg["symbol"])
            side = str(leg["side"]).upper()
            qty = float(leg["quantity"])
            price = float(prices.get(sym, np.nan))
            delay = float(self._rng.uniform(cfg.delay_minutes_min, cfg.delay_minutes_max))
            is_partial = bool(self._rng.random() < cfg.partial_fill_prob)
            fill_ratio = 1.0
            if is_partial:
                fill_ratio = float(self._rng.uniform(
                    cfg.partial_fill_ratio_min, cfg.partial_fill_ratio_max))
            filled = qty * fill_ratio
            leg_execs.append(LegExecution(
                symbol=sym, side=side, quantity=qty,
                filled_quantity=filled, unfilled_quantity=qty - filled,
                ref_price=price, exec_price=price,
                delay_minutes=delay, is_partial=is_partial,
            ))

        # 2) 申赎（可选）：开启且有篮子时输出 AP 申赎结果
        cr = None
        if cfg.enable_creation_redemption and basket is not None:
            cr = self._creation_redemption(leg_execs, basket, etf_symbol)

        # 3) 二级市场成本（A3 compute_basket，只对已成交数量计费）
        etf_code = etf_symbol or (basket.etf_code if basket is not None else "")
        cost_legs = [{
            "symbol": le.symbol, "side": le.side, "quantity": le.filled_quantity,
            "price": le.ref_price, "is_etf": le.symbol == etf_code,
            "volume": float(volumes.get(le.symbol, 0.0)),
        } for le in leg_execs if le.filled_quantity > 0]
        basket_cost = self.cost_model.compute_basket(cost_legs) if cost_legs else None

        # 回填成交价 / 费用到各腿
        if basket_cost is not None:
            by_sym = {b.symbol: b for b in basket_cost.legs}
            for le in leg_execs:
                b = by_sym.get(le.symbol)
                if b is not None:
                    le.exec_price = b.exec_price
                    le.total_fee = b.total_fee
                    le.slippage_cost = b.slippage_cost

        # 4) 敞口 / 跟踪误差
        total_target = sum(abs(le.ref_price * le.quantity)
                           for le in leg_execs if np.isfinite(le.ref_price))
        total_filled = sum(abs(le.ref_price * le.filled_quantity)
                           for le in leg_execs if np.isfinite(le.ref_price))
        total_unfilled = sum(abs(le.ref_price * le.unfilled_quantity)
                             for le in leg_execs if np.isfinite(le.ref_price))
        exposure = total_unfilled / total_target if total_target > 0 else 0.0

        gaps, weights = [], []
        for le in leg_execs:
            if le.quantity > 0 and np.isfinite(le.ref_price):
                gaps.append(le.unfilled_quantity / le.quantity)
                weights.append(abs(le.ref_price * le.quantity))
        tracking_error = _weighted_std(gaps, weights)

        return ExecutionResult(
            legs=leg_execs,
            basket_cost=basket_cost,
            creation_redemption=cr,
            total_target_value=total_target,
            total_filled_value=total_filled,
            total_unfilled_value=total_unfilled,
            exposure_ratio=exposure,
            tracking_error=tracking_error,
        )

    # ------------------------------------------------------------------
    def _creation_redemption(
        self,
        leg_execs: List[LegExecution],
        basket: PCFBasket,
        etf_symbol: Optional[str],
    ) -> Optional[dict]:
        """AP 实物/现金申赎模拟（ETF 腿）。"""
        cfg = self.config
        etf_leg = None
        for le in leg_execs:
            if le.symbol == (etf_symbol or "") or le.symbol == basket.etf_code:
                etf_leg = le
                break
        if etf_leg is None:
            return None

        units = int(etf_leg.quantity // cfg.creation_unit) if cfg.creation_unit > 0 else 0
        if units <= 0:
            return {
                "type": "insufficient_units", "units": 0, "etf_quantity": 0.0,
                "fee": 0.0, "cash_component": 0.0, "confirm_day": cfg.confirm_day,
            }
        typ = "redeem" if etf_leg.side == SELL else "create"
        cash_component = basket.cash_component * units
        fee = cfg.creation_fee + cash_component * cfg.cash_substitute_fee_bp / 10000.0
        return {
            "type": typ,
            "units": units,
            "etf_quantity": units * cfg.creation_unit,
            "fee": fee,
            "cash_component": cash_component,
            "confirm_day": cfg.confirm_day,
        }


__all__ = [
    "BasketExecutor",
    "LegExecution",
    "ExecutionResult",
    "make_basket_legs",
]
