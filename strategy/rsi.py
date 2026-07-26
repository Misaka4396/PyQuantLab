"""RSI Overbought/Oversold Strategy."""

import pandas as pd

from strategy.base import BaseStrategy


class RSIStrategy(BaseStrategy):
    @classmethod
    def get_name(cls) -> str:
        return "RSI 超买超卖"

    @classmethod
    def get_description(cls) -> str:
        return (
            "当 RSI 从超卖区域向上突破时买入（超卖反弹），"
            "当 RSI 从超买区域向下跌破时卖出。"
            "适用于震荡行情。"
        )

    @classmethod
    def get_param_spec(cls) -> dict:
        return {
            "period": {
                "type": "int", "default": 14, "min": 2, "max": 50, "step": 1,
                "label": "RSI 周期", "help": "RSI 计算回看窗口",
            },
            "oversold": {
                "type": "int", "default": 30, "min": 10, "max": 40, "step": 1,
                "label": "超卖阈值", "help": "RSI 低于此值为超卖（买入信号）",
            },
            "overbought": {
                "type": "int", "default": 70, "min": 60, "max": 90, "step": 1,
                "label": "超买阈值", "help": "RSI 高于此值为超买（卖出信号）",
            },
        }

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close = self._get_close(data)
        period = self.params["period"]
        oversold = self.params["oversold"]
        overbought = self.params["overbought"]

        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, float("nan"))
        rsi = 100 - (100 / (1 + rs))

        signals = pd.Series(0, index=data.index)
        below_os = rsi < oversold
        above_ob = rsi > overbought

        cross_above_os = below_os.shift(1).fillna(False) & (~below_os)
        signals[cross_above_os] = 1

        cross_below_ob = above_ob.shift(1).fillna(False) & (~above_ob)
        signals[cross_below_ob] = -1

        return signals

    def _get_close(self, data: pd.DataFrame) -> pd.Series:
        if isinstance(data.columns, pd.MultiIndex):
            ticker = data.columns.levels[0][0]
            return data[(ticker, "Close")]
        return data["Close"]
