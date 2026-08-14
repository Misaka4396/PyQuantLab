"""Dual Moving Average Crossover Strategy."""

import pandas as pd

from strategy.base import BaseStrategy


class MACrossoverStrategy(BaseStrategy):
    @classmethod
    def get_name(cls) -> str:
        return "双均线交叉"

    @classmethod
    def get_description(cls) -> str:
        return (
            "当快速均线上穿慢速均线时买入（金叉），"
            "当快速均线下穿慢速均线时卖出（死叉）。"
            "适用于趋势明显的市场行情。"
        )

    @classmethod
    def get_param_spec(cls) -> dict:
        return {
            "fast_period": {
                "type": "int",
                "default": 20,
                "min": 2,
                "max": 200,
                "step": 1,
                "label": "快线周期",
                "help": "短期均线窗口",
            },
            "slow_period": {
                "type": "int",
                "default": 50,
                "min": 5,
                "max": 500,
                "step": 1,
                "label": "慢线周期",
                "help": "长期均线窗口",
            },
            "ma_type": {
                "type": "choice",
                "default": "sma",
                "choices": ["sma", "ema"],
                "label": "均线类型",
                "help": "SMA 简单移动平均 / EMA 指数移动平均",
            },
        }

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close = self._get_close(data)
        fast = self.params["fast_period"]
        slow = self.params["slow_period"]
        ma_type = self.params["ma_type"]

        if ma_type == "ema":
            ma_fast = close.ewm(span=fast, adjust=False).mean()
            ma_slow = close.ewm(span=slow, adjust=False).mean()
        else:
            ma_fast = close.rolling(window=fast).mean()
            ma_slow = close.rolling(window=slow).mean()

        signals = pd.Series(0, index=data.index)
        signals[ma_fast > ma_slow] = 1
        signals[ma_fast < ma_slow] = -1
        signals = signals.diff().fillna(0)
        signals = signals.clip(-1, 1)
        return signals

    def _get_close(self, data: pd.DataFrame) -> pd.Series:
        if isinstance(data.columns, pd.MultiIndex):
            ticker = data.columns.levels[0][0]
            return data[(ticker, "Close")]
        return data["Close"]
