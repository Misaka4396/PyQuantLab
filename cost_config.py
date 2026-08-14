"""A3 交易成本参数集中配置。

所有默认值均注明来源口径，并标注"以券商为准需确认"：
- 实际费率随券商/资金量/品种不同而不同，上线前必须与开户券商对账确认。
- 印花税/过户费按当前 A 股政策口径填写，政策调整时需同步更新。

本模块只定义参数，不包含任何计算逻辑（计算在 cost_model.py）。
"""

from __future__ import annotations

from dataclasses import dataclass

# 冲击成本模型可选类型
IMPACT_SQRT = "sqrt"  # 平方根冲击：impact = coef * sqrt(参与率)
IMPACT_LINEAR = "linear"  # 线性冲击：impact = coef * 参与率
IMPACT_MODELS = (IMPACT_SQRT, IMPACT_LINEAR)


@dataclass
class CostConfig:
    """交易成本参数（A 股口径，默认值以券商为准需确认）。"""

    # ------------------------------------------------------------------
    # 固定成本
    # ------------------------------------------------------------------
    commission_rate: float = 0.00025  # 佣金 万2.5（以券商为准需确认）
    min_commission: float = 5.0  # 单笔最低佣金（元，多数券商 5 元，需确认）
    stamp_tax_rate: float = 0.0005  # 印花税 卖出 0.05%（2023-08 减半后口径，政策以最新为准）
    transfer_fee_rate: float = 0.00001  # 过户费 0.001% 双向（沪市，需确认）

    # ETF 豁免项
    etf_exempt_stamp_tax: bool = True  # ETF 免印花税（需确认）
    etf_exempt_transfer_fee: bool = True  # ETF 免过户费（需确认）

    # ------------------------------------------------------------------
    # 买卖价差（半价差模型，按流动性可配）
    # ------------------------------------------------------------------
    spread_rate: float = 0.0002  # 全价差 2bp（买卖各承担半价差 1bp，按流动性可配）

    # ------------------------------------------------------------------
    # 冲击成本（按成交量占比的冲击函数）
    # ------------------------------------------------------------------
    impact_model: str = IMPACT_SQRT  # sqrt | linear
    impact_coef: float = 0.05  # 冲击系数（100% 参与率时 sqrt 冲击 ≈ 5%，需确认）
    participation_cap: float = 1.0  # 参与率上限（防止异常成交量导致冲击爆炸）

    # ------------------------------------------------------------------
    # 篮子成本（ETF 篮子同步执行）
    # ------------------------------------------------------------------
    basket_slippage_bps: float = 1.0  # 篮子整体滑点加成（bp，同步执行冲击，需确认）

    def validate(self) -> None:
        """校验参数合法性，非法直接抛 ValueError。"""
        if self.commission_rate < 0:
            raise ValueError("commission_rate 不能为负")
        if self.min_commission < 0:
            raise ValueError("min_commission 不能为负")
        if self.stamp_tax_rate < 0:
            raise ValueError("stamp_tax_rate 不能为负")
        if self.transfer_fee_rate < 0:
            raise ValueError("transfer_fee_rate 不能为负")
        if self.spread_rate < 0:
            raise ValueError("spread_rate 不能为负")
        if self.impact_model not in IMPACT_MODELS:
            raise ValueError(f"impact_model 必须为 {IMPACT_MODELS} 之一，得到 {self.impact_model}")
        if self.impact_coef < 0:
            raise ValueError("impact_coef 不能为负")
        if not (0.0 < self.participation_cap <= 1.0):
            raise ValueError("participation_cap 必须在 (0, 1] 区间")
        if self.basket_slippage_bps < 0:
            raise ValueError("basket_slippage_bps 不能为负")

    def fee_rates_summary(self) -> dict:
        """返回固定费率汇总（便于对账审计）。"""
        return {
            "佣金(万)": self.commission_rate * 10000,
            "最低佣金(元)": self.min_commission,
            "印花税(卖出)": self.stamp_tax_rate,
            "过户费": self.transfer_fee_rate,
            "价差(bp)": self.spread_rate * 10000,
            "冲击模型": self.impact_model,
            "冲击系数": self.impact_coef,
            "篮子滑点(bp)": self.basket_slippage_bps,
        }
