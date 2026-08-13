"""引擎配置：种子、初始资金、撮合模式、日志等。

参数集中配置，保证复现（同种子两次运行结果一致）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

# 撮合模式
FILL_OPEN = "open"              # 当 bar 开盘价撮合（可能前视，需注意）
FILL_NEXT_BAR = "next_bar"      # 下一 bar 开盘价撮合（无前视，默认）
FILL_MODES = (FILL_OPEN, FILL_NEXT_BAR)


@dataclass
class EngineConfig:
    """事件驱动回测引擎配置。"""

    initial_cash: float = 1_000_000.0     # 初始资金
    seed: int = 42                          # 随机种子（复现用）
    fill_mode: str = FILL_NEXT_BAR          # 撮合模式：next_bar | open
    default_order_type: str = "market"      # 信号未指定时的默认单类型 market | limit
    allow_fractional: bool = True           # 是否允许小数仓位（ETF 可拆）；False 则向下取整到整数
    log_level: str = "INFO"                 # 日志级别
    log_file: Optional[str] = None          # 日志文件路径（None 则只输出到 stderr）
    etf_symbols: Optional[Tuple[str, ...]] = None  # 显式指定 ETF 代码（None 则按代码前缀启发式判断）

    def validate(self) -> None:
        if self.initial_cash <= 0:
            raise ValueError("initial_cash 必须为正")
        if self.fill_mode not in FILL_MODES:
            raise ValueError(f"fill_mode 必须为 {FILL_MODES} 之一，得到 {self.fill_mode}")
        if self.default_order_type not in ("market", "limit"):
            raise ValueError("default_order_type 必须为 market 或 limit")
