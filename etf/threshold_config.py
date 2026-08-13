"""B2 阈值参数配置：开仓缓冲、平仓规则、止损、收盘强平、网格参数范围。

本模块只定义参数，不含任何计算逻辑（计算在 etf_signal.py / signal_grid.py）。
开仓阈值 = 单位交易成本 + 缓冲，其中单位成本来自 A3（cost_model.py），
保证"折溢价幅度必须覆盖成本才开仓"这一硬约束。

所有默认值均为可调的初始估计，上线前应与实盘成本（券商费率、实际滑点）对账。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

# 平仓原因
EXIT_MEAN_REVERT = "mean_revert"   # 均值回归 / 归零平仓
EXIT_STOP_LOSS = "stop_loss"       # 止损平仓
EXIT_FORCE_CLOSE = "force_close"   # 收盘前强制平仓

# 套利方向
DIR_LONG = "long"     # 折价 → 买 ETF、卖篮子（ETF 腿 = BUY）
DIR_SHORT = "short"   # 溢价 → 卖 ETF、买篮子（ETF 腿 = SELL）


@dataclass
class ThresholdConfig:
    """折溢价信号阈值参数。"""

    # ------------------------------------------------------------------
    # 滚动窗口与开仓判据
    # ------------------------------------------------------------------
    zscore_window: int = 60           # 滚动 z-score / 分位数窗口（bar 数，只用窗口内数据）
    use_quantile: bool = True         # True=分位数判据；False=z-score 判据
    zscore_entry: float = 2.0         # 开仓 z-score 阈值（|z| >= 该值）
    quantile_entry: float = 0.95      # 开仓分位数阈值（上侧 >=；下侧 <= 1 - 该值）
    entry_buffer: float = 0.0005      # 开仓缓冲：溢价幅度需 > 单位成本 + 缓冲

    # ------------------------------------------------------------------
    # 平仓规则
    # ------------------------------------------------------------------
    zscore_exit: float = 0.0          # 均值回归平仓阈值：|z| 回落到该值即平仓（0=归零平仓）
    stop_loss_bp: float = 30.0        # 止损（bp）：溢价朝不利方向扩大该幅度即平仓
    force_close_time: str = "14:45"   # 收盘前强制平仓时点（HH:MM，盘中分钟数据用）

    # ------------------------------------------------------------------
    # 风控
    # ------------------------------------------------------------------
    min_holding_minutes: int = 5      # 最小持仓时长（分钟），防止同 bar 反复开平

    # ------------------------------------------------------------------
    # 网格寻优范围（signal_grid.py 使用，仅限样本内寻优，禁止全样本）
    # ------------------------------------------------------------------
    grid_zscore_entry: Tuple[float, ...] = (1.5, 2.0, 2.5, 3.0)
    grid_entry_buffer: Tuple[float, ...] = (0.0003, 0.0005, 0.0010)
    grid_stop_loss_bp: Tuple[float, ...] = (20.0, 30.0, 40.0)

    def validate(self) -> None:
        """校验参数合法性，非法直接抛 ValueError。"""
        if self.zscore_window < 2:
            raise ValueError("zscore_window 至少为 2")
        if self.zscore_entry < 0:
            raise ValueError("zscore_entry 不能为负（0=不要求 z-score 极端，仅按成本幅度开仓）")
        if not (0.5 <= self.quantile_entry < 1.0):
            raise ValueError("quantile_entry 必须在 [0.5, 1) 区间")
        if self.entry_buffer < 0:
            raise ValueError("entry_buffer 不能为负")
        if self.stop_loss_bp < 0:
            raise ValueError("stop_loss_bp 不能为负")
        if self.min_holding_minutes < 0:
            raise ValueError("min_holding_minutes 不能为负")
        hh, mm = self._parse_force_close()
        if not (0 <= hh < 24 and 0 <= mm < 60):
            raise ValueError(f"force_close_time 非法: {self.force_close_time!r}")

    def _parse_force_close(self) -> Tuple[int, int]:
        parts = str(self.force_close_time).split(":")
        if len(parts) != 2:
            raise ValueError(f"force_close_time 需为 HH:MM，得到 {self.force_close_time!r}")
        return int(parts[0]), int(parts[1])

    def force_close_minute_of_day(self) -> int:
        """强平时点换算为当日分钟数（0~1439）。"""
        hh, mm = self._parse_force_close()
        return hh * 60 + mm


__all__ = [
    "ThresholdConfig",
    "EXIT_MEAN_REVERT",
    "EXIT_STOP_LOSS",
    "EXIT_FORCE_CLOSE",
    "DIR_LONG",
    "DIR_SHORT",
]
