"""B3 执行配置：时延范围、部分成交概率、申赎参数。

本模块只定义参数，不含计算逻辑（计算在 basket_execution.py）。
申赎为**开关**，默认关闭（enable_creation_redemption=False），关闭时走二级市场
成交路径；开启后对 ETF 腿按 AP 实物/现金申赎模拟（需 PCF 篮子与最小申赎单位）。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ExecutionConfig:
    """篮子同步执行与申赎参数。"""

    # ------------------------------------------------------------------
    # 逐笔时延（分钟）
    # ------------------------------------------------------------------
    delay_minutes_min: float = 1.0   # 每笔最小成交时延（分钟）
    delay_minutes_max: float = 5.0   # 每笔最大成交时延（分钟）

    # ------------------------------------------------------------------
    # 部分成交
    # ------------------------------------------------------------------
    partial_fill_prob: float = 0.2       # 每笔发生部分成交的概率
    partial_fill_ratio_min: float = 0.3  # 部分成交时成交比例下限
    partial_fill_ratio_max: float = 0.9  # 部分成交时成交比例上限

    # ------------------------------------------------------------------
    # 申赎机制（开关，默认关闭）
    # ------------------------------------------------------------------
    enable_creation_redemption: bool = False  # 是否启用 AP 申赎（默认关闭，走二级市场）
    creation_unit: float = 900000.0           # 最小申赎单位（份）
    creation_fee: float = 0.0                 # 申赎固定费（元/次，以基金公司为准需确认）
    cash_substitute_fee_bp: float = 0.0       # 现金替代费（bp，对现金差额计，需确认）
    confirm_day: str = "T"                    # 申赎确认日（T 日确认，T+1 到账需配置）

    # ------------------------------------------------------------------
    # 复现
    # ------------------------------------------------------------------
    seed: int = 42

    def validate(self) -> None:
        """校验参数合法性，非法直接抛 ValueError。"""
        if self.delay_minutes_min < 0 or self.delay_minutes_max < self.delay_minutes_min:
            raise ValueError("时延范围非法：需 0 <= min <= max")
        if not (0.0 <= self.partial_fill_prob <= 1.0):
            raise ValueError("partial_fill_prob 必须在 [0, 1]")
        if not (0.0 < self.partial_fill_ratio_min <= self.partial_fill_ratio_max < 1.0):
            raise ValueError("部分成交比例范围非法：需 0 < min <= max < 1")
        if self.creation_unit <= 0:
            raise ValueError("creation_unit 必须为正")
        if self.creation_fee < 0 or self.cash_substitute_fee_bp < 0:
            raise ValueError("申赎费用不能为负")


__all__ = ["ExecutionConfig"]
