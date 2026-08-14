"""逐笔核算与权益曲线：记录成交、每 bar 盯市，输出权益曲线 DataFrame。"""

from __future__ import annotations

import pandas as pd

from engine.events import FillEvent
from engine.portfolio import Portfolio


class Accounting:
    """逐笔核算器：累积成交记录与逐 bar 权益记录。"""

    def __init__(self):
        self.fills: list[FillEvent] = []  # 全部成交事件
        self.records: list[dict] = []  # 逐 bar 权益记录

    # ------------------------------------------------------------------
    # 记录
    # ------------------------------------------------------------------
    def record_fill(self, fill: FillEvent) -> None:
        """记录一笔成交。"""
        self.fills.append(fill)

    def mark_to_market(
        self,
        timestamp: pd.Timestamp,
        portfolio: Portfolio,
        prices: dict[str, float],
    ) -> dict:
        """在 bar 期末按收盘价盯市，记录现金/市值/权益快照。"""
        snap = portfolio.snapshot(prices)
        rec = {"timestamp": timestamp}
        rec.update(snap)
        self.records.append(rec)
        return rec

    # ------------------------------------------------------------------
    # 输出
    # ------------------------------------------------------------------
    def equity_curve(self) -> pd.DataFrame:
        """返回权益曲线 DataFrame（index=timestamp）。"""
        if not self.records:
            return pd.DataFrame(
                columns=["cash", "market_value", "equity", "available_cash", "total_fees"]
            )
        df = pd.DataFrame(self.records).set_index("timestamp")
        df["returns"] = df["equity"].pct_change().fillna(0.0)
        return df

    def fills_frame(self) -> pd.DataFrame:
        """返回逐笔成交明细 DataFrame。"""
        if not self.fills:
            return pd.DataFrame(
                columns=[
                    "timestamp",
                    "order_id",
                    "symbol",
                    "side",
                    "quantity",
                    "fill_price",
                    "exec_price",
                    "commission",
                    "stamp_tax",
                    "transfer_fee",
                    "total_fee",
                    "cash_flow",
                ]
            )
        return pd.DataFrame([f.to_dict() for f in self.fills])
