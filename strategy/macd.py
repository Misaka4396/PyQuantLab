"""MACD Signal Line Crossover Strategy."""

import pandas as pd

from strategy.base import BaseStrategy


class MACDStrategy(BaseStrategy):
    @classmethod
    def get_name(cls) -> str:
        return "MACD 信号线交叉"

    @classmethod
    def get_description(cls) -> str:
        return (
            "当 MACD 线上穿信号线时买入（看涨动能），"
            "当 MACD 线下穿信号线时卖出（看跌动能）。"
            "一种趋势跟踪型动量指标。"
        )

    @classmethod
    def get_param_spec(cls) -> dict:
        return {
            "fast": {
                "type": "int", "default": 12, "min": 2, "max": 50, "step": 1,
                "label": "快线周期", "help": "快速 EMA 周期",
            },
            "slow": {
                "type": "int", "default": 26, "min": 5, "max": 100, "step": 1,
                "label": "慢线周期", "help": "慢速 EMA 周期",
            },
            "signal": {
                "type": "int", "default": 9, "min": 2, "max": 30, "step": 1,
                "label": "信号线周期", "help": "信号线 EMA 周期",
            },
        }

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close = self._get_close(data)
        fast = self.params["fast"]
        slow = self.params["slow"]
        signal_p = self.params["signal"]

        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_p, adjust=False).mean()

        signals = pd.Series(0, index=data.index)
        signals[macd_line > signal_line] = 1
        signals[macd_line < signal_line] = -1
        signals = signals.diff().fillna(0)
        signals = signals.clip(-1, 1)
        return signals

    def _get_close(self, data: pd.DataFrame) -> pd.Series:
        if isinstance(data.columns, pd.MultiIndex):
            ticker = data.columns.levels[0][0]
            return data[(ticker, "Close")]
        return data["Close"]
