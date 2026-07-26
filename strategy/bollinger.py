"""Bollinger Bands Mean Reversion Strategy."""

import pandas as pd

from strategy.base import BaseStrategy


class BollingerStrategy(BaseStrategy):
    @classmethod
    def get_name(cls) -> str:
        return "布林带均值回归"

    @classmethod
    def get_description(cls) -> str:
        return (
            "当价格跌破布林带下轨时买入（超卖），"
            "当价格突破布林带上轨时卖出（超买）。"
            "一种均值回归策略，假设价格会向移动平均线回归。"
        )

    @classmethod
    def get_param_spec(cls) -> dict:
        return {
            "period": {
                "type": "int", "default": 20, "min": 5, "max": 100, "step": 1,
                "label": "布林带周期", "help": "移动平均窗口",
            },
            "num_std": {
                "type": "int", "default": 2, "min": 1, "max": 4, "step": 1,
                "label": "标准差倍数", "help": "带宽乘数",
            },
        }

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close = self._get_close(data)
        period = self.params["period"]
        num_std = self.params["num_std"]

        middle = close.rolling(window=period).mean()
        std = close.rolling(window=period).std()
        upper = middle + num_std * std
        lower = middle - num_std * std

        signals = pd.Series(0, index=data.index)
        signals[close < lower] = 1
        signals[close > upper] = -1

        prev_buy = signals.shift(1).fillna(0)
        signals = signals.where(
            (signals == 1) & (prev_buy != 1) | (signals == -1) & (prev_buy != -1), 0
        )
        return signals

    def _get_close(self, data: pd.DataFrame) -> pd.Series:
        if isinstance(data.columns, pd.MultiIndex):
            ticker = data.columns.levels[0][0]
            return data[(ticker, "Close")]
        return data["Close"]
