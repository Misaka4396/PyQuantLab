"""Aggregated backtest report builder for UI rendering."""

import json
from typing import Dict

import pandas as pd

from core.types import BacktestResult


class BacktestReport:
    def __init__(self, result: BacktestResult):
        self.result = result

    def summary_text(self) -> str:
        m = self.result.metrics
        lines = [
            f"总收益率: {m.total_return * 100:.2f}%",
            f"年化收益: {m.annualized_return * 100:.2f}%",
            f"夏普比率: {m.sharpe_ratio:.2f}",
            f"最大回撤: {m.max_drawdown * 100:.2f}%",
            f"胜率: {m.win_rate * 100:.1f}%",
            f"交易次数: {m.total_trades}",
        ]
        return "\n".join(lines)

    def metrics_table(self) -> pd.DataFrame:
        m = self.result.metrics
        data = {
            "总收益率": f"{m.total_return * 100:.2f}%",
            "年化收益": f"{m.annualized_return * 100:.2f}%",
            "年化波动率": f"{m.annualized_volatility * 100:.2f}%",
            "夏普比率": f"{m.sharpe_ratio:.2f}",
            "索提诺比率": f"{m.sortino_ratio:.2f}",
            "最大回撤": f"{m.max_drawdown * 100:.2f}%",
            "最大回撤天数": str(m.max_drawdown_duration),
            "卡玛比率": f"{m.calmar_ratio:.2f}",
            "Alpha": f"{m.alpha:.4f}",
            "Beta": f"{m.beta:.4f}",
            "VaR 95%": f"{m.var_95 * 100:.2f}%",
            "CVaR 95%": f"{m.cvar_95 * 100:.2f}%",
            "胜率": f"{m.win_rate * 100:.1f}%",
            "盈亏比": f"{m.profit_factor:.2f}",
            "交易次数": str(m.total_trades),
            "盈利次数": str(m.winning_trades),
            "亏损次数": str(m.losing_trades),
            "平均盈利": f"{m.avg_win * 100:.2f}%",
            "平均亏损": f"{m.avg_loss * 100:.2f}%",
        }
        df = pd.DataFrame(list(data.items()), columns=["指标", "数值"])
        return df

    def to_dict(self) -> dict:
        return {
            "config": {
                "initial_capital": self.result.config.initial_capital,
                "commission": self.result.config.commission,
                "slippage": self.result.config.slippage,
            },
            "metrics": {
                k: v for k, v in self.result.metrics.__dict__.items()
                if not k.startswith("_")
            },
            "total_trades": len(self.result.trades),
        }

    def to_json(self, filepath: str) -> None:
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
