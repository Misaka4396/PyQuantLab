"""A3 交易成本与滑点模型（固定成本 + 买卖价差 + 冲击成本 + 篮子成本）。

设计目标：
- 逐笔成本可追溯：每笔成交返回 CostBreakdown，内含逐项 CostDetailItem 明细。
- 与 engine 解耦：engine 通过 ``CostModel.compute`` 接口注入调用，不 import 本模块。
- 参数集中配置：所有费率来自 CostConfig（cost_config.py），默认值注明"以券商为准需确认"。

成本构成（成交方向不同，符号方向相反）：
1. 固定成本：佣金（按成交额，有最低佣金）、印花税（仅卖出，ETF 免）、过户费（ETF 免）。
2. 买卖价差：半价差模型，买入在基准价上加半价差，卖出减半价差。
3. 冲击成本：按成交量参与率（订单量/成交量）的冲击函数（sqrt 或 linear）。
4. 篮子成本：逐只计费 + 篮子整体滑点加成。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from cost_config import IMPACT_LINEAR, IMPACT_SQRT, CostConfig

BUY = "BUY"
SELL = "SELL"
SIDES = (BUY, SELL)


def normalize_side(side: str) -> str:
    """把 BUY/SELL/buy/sell 归一化为大写方向。"""
    s = str(side).upper()
    if s not in SIDES:
        raise ValueError(f"side 必须为 {SIDES} 之一，得到 {side!r}")
    return s


@dataclass
class CostDetailItem:
    """单条成本明细（审计用）。"""

    name: str  # 费用/成本名称，如 "佣金"、"价差成本"
    amount: float  # 金额（元，>=0）
    note: str = ""  # 说明（费率/模型/参与率等）


@dataclass
class CostBreakdown:
    """单笔成交的成本明细分解。"""

    symbol: str
    side: str
    quantity: float
    base_price: float  # 基准价（撮合参考价，如开盘价）
    exec_price: float  # 价差/冲击调整后的成交价
    commission: float
    stamp_tax: float
    transfer_fee: float
    total_fee: float  # 固定费用合计 = 佣金 + 印花税 + 过户费
    spread_cost: float  # 半价差成本（金额）
    impact_cost: float  # 冲击成本（金额）
    slippage_cost: float  # 滑点成本合计 = 价差 + 冲击
    gross_value: float  # 基准价口径成交额
    net_cash: float  # 成交净现金流（买为负、卖为正，含固定费用）
    details: list[CostDetailItem] = field(default_factory=list)

    @property
    def exec_value(self) -> float:
        """成交额（按滑点后成交价）。"""
        return self.exec_price * self.quantity

    def to_dict(self) -> dict:
        """转为 dict（细节列表也展开，便于落盘/审计）。"""
        d = asdict(self)
        d["details"] = [asdict(x) for x in self.details]
        return d


@dataclass
class BasketCost:
    """ETF 篮子整体成本汇总。"""

    legs: list[CostBreakdown] = field(default_factory=list)
    basket_slippage: float = 0.0  # 篮子整体滑点加成（金额）
    total_fee: float = 0.0  # 固定费用合计（逐只累加）
    total_slippage: float = 0.0  # 滑点合计（逐只价差+冲击 + 篮子加成）
    total_gross_value: float = 0.0  # 篮子基准价口径成交额

    def to_dict(self) -> dict:
        return {
            "legs": [leg.to_dict() for leg in self.legs],
            "basket_slippage": self.basket_slippage,
            "total_fee": self.total_fee,
            "total_slippage": self.total_slippage,
            "total_gross_value": self.total_gross_value,
        }


class CostModel:
    """交易成本模型主类（可注入回测引擎）。"""

    def __init__(self, config: CostConfig | None = None):
        self.config = config or CostConfig()
        self.config.validate()

    # ------------------------------------------------------------------
    # 单笔成本
    # ------------------------------------------------------------------
    def compute(
        self,
        side: str,
        quantity: float,
        price: float,
        symbol: str = "",
        is_etf: bool = False,
        volume: float = 0.0,
    ) -> CostBreakdown:
        """计算单笔成交的成本明细。

        参数：
        - side: 方向 BUY/SELL
        - quantity: 成交数量
        - price: 基准价（撮合参考价，如开盘价）
        - symbol: 证券代码（仅用于明细标识）
        - is_etf: 是否 ETF（决定印花税/过户费豁免）
        - volume: 该 bar 成交量（用于计算冲击参与率，0 则无冲击）
        """
        side = normalize_side(side)
        cfg = self.config
        qty = float(quantity)
        base = float(price)

        # 半价差与冲击（以"率"表示，相对基准价）
        half_spread_rate = cfg.spread_rate / 2.0
        impact_rate = self._impact_rate(qty, volume)

        slip_rate = half_spread_rate + impact_rate
        exec_price = base * (1.0 + slip_rate) if side == BUY else base * (1.0 - slip_rate)

        exec_value = exec_price * qty

        # 固定费用
        commission = self._commission(exec_value)
        stamp_tax = (
            0.0
            if (is_etf and cfg.etf_exempt_stamp_tax) or side == BUY
            else exec_value * cfg.stamp_tax_rate
        )
        transfer_fee = (
            0.0 if (is_etf and cfg.etf_exempt_transfer_fee) else exec_value * cfg.transfer_fee_rate
        )
        total_fee = commission + stamp_tax + transfer_fee

        # 滑点金额（相对基准价的口径）
        spread_cost = base * half_spread_rate * qty
        impact_cost = base * impact_rate * qty
        slippage_cost = spread_cost + impact_cost

        gross_value = base * qty
        net_cash = -(exec_value + total_fee) if side == BUY else (exec_value - total_fee)

        details = [
            CostDetailItem(
                "佣金",
                commission,
                f"费率 {cfg.commission_rate:.4%}，最低 {cfg.min_commission:.2f} 元",
            ),
            CostDetailItem(
                "印花税",
                stamp_tax,
                f"卖出 {cfg.stamp_tax_rate:.4%}"
                + ("（ETF 免）" if is_etf and cfg.etf_exempt_stamp_tax else ""),
            ),
            CostDetailItem(
                "过户费",
                transfer_fee,
                f"双向 {cfg.transfer_fee_rate:.4%}"
                + ("（ETF 免）" if is_etf and cfg.etf_exempt_transfer_fee else ""),
            ),
            CostDetailItem("价差成本(半价差)", spread_cost, f"半价差 {half_spread_rate:.4%}"),
            CostDetailItem(
                "冲击成本",
                impact_cost,
                f"{cfg.impact_model} 模型，参与率 {self._participation(qty, volume):.4%}",
            ),
        ]

        return CostBreakdown(
            symbol=symbol,
            side=side,
            quantity=qty,
            base_price=base,
            exec_price=exec_price,
            commission=commission,
            stamp_tax=stamp_tax,
            transfer_fee=transfer_fee,
            total_fee=total_fee,
            spread_cost=spread_cost,
            impact_cost=impact_cost,
            slippage_cost=slippage_cost,
            gross_value=gross_value,
            net_cash=net_cash,
            details=details,
        )

    # ------------------------------------------------------------------
    # 篮子成本（逐只计费 + 篮子整体滑点加成）
    # ------------------------------------------------------------------
    def compute_basket(self, legs: list[dict]) -> BasketCost:
        """计算 ETF 篮子同步执行的整体成本。

        legs 每项为 dict，键同 compute 参数：
            {symbol, side, quantity, price, is_etf, volume}
        返回逐只 CostBreakdown + 篮子整体滑点加成。
        """
        if not legs:
            return BasketCost()

        breakdowns = [self.compute(**leg) for leg in legs]
        gross = sum(b.gross_value for b in breakdowns)
        # 篮子整体滑点加成：对整体成交额按 bps 计
        basket_slippage = gross * self.config.basket_slippage_bps / 10000.0
        total_fee = sum(b.total_fee for b in breakdowns)
        total_slippage = sum(b.slippage_cost for b in breakdowns) + basket_slippage

        return BasketCost(
            legs=breakdowns,
            basket_slippage=basket_slippage,
            total_fee=total_fee,
            total_slippage=total_slippage,
            total_gross_value=gross,
        )

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    def _commission(self, exec_value: float) -> float:
        """佣金（含最低佣金约束）。"""
        cfg = self.config
        if exec_value <= 0:
            return 0.0
        return max(exec_value * cfg.commission_rate, cfg.min_commission)

    def _participation(self, quantity: float, volume: float) -> float:
        """成交参与率 = 订单量/成交量，封顶于 participation_cap。"""
        if volume is None or volume <= 0:
            return 0.0
        return min(float(quantity) / float(volume), self.config.participation_cap)

    def _impact_rate(self, quantity: float, volume: float) -> float:
        """冲击成本率（相对基准价的占比）。"""
        cfg = self.config
        participation = self._participation(quantity, volume)
        if participation <= 0:
            return 0.0
        if cfg.impact_model == IMPACT_SQRT:
            return cfg.impact_coef * np.sqrt(participation)
        if cfg.impact_model == IMPACT_LINEAR:
            return cfg.impact_coef * participation
        raise ValueError(f"未知冲击模型: {cfg.impact_model}")


def sensitivity_table(
    base: CostConfig,
    varied_param: str,
    values: list[float],
    side: str = BUY,
    quantity: float = 10000,
    price: float = 3.0,
    volume: float = 1_000_000,
    is_etf: bool = True,
) -> list[dict]:
    """成本敏感性分析：固定其他参数，仅变动一个参数，输出总成本变化。

    返回 [{param_value, commission, stamp_tax, transfer_fee, spread_cost,
            impact_cost, slippage_cost, total_fee, total_cost}]，
    其中 total_cost = 固定费用 + 滑点成本（买方口径的总成本）。
    用于输出"成本敏感"结论：参数对总成本的敏感度。
    """
    rows = []
    for v in values:
        cfg = CostConfig(**{**base.__dict__, varied_param: v})
        model = CostModel(cfg)
        b = model.compute(
            side=side, quantity=quantity, price=price, symbol="510300", is_etf=is_etf, volume=volume
        )
        total_cost = b.total_fee + b.slippage_cost
        rows.append(
            {
                varied_param: v,
                "commission": b.commission,
                "stamp_tax": b.stamp_tax,
                "transfer_fee": b.transfer_fee,
                "spread_cost": b.spread_cost,
                "impact_cost": b.impact_cost,
                "slippage_cost": b.slippage_cost,
                "total_fee": b.total_fee,
                "total_cost": total_cost,
            }
        )
    return rows
