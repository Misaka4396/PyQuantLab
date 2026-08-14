"""B1 IOPV 合成估算：IOPV 缺失时用成分股实时行情合成。

IOPV（Indicative Optimized Portfolio Value，盘中参考净值）由交易所约每 15s
发布一次；当交易所未发布（停牌、数据缺失）时，可用 PCF 篮子 + 成分股实时价
合成估算：

    IOPV_est = (Σ 数量_i × 价格_i + 现金差额) / 最小申赎单位

其中"价格_i"优先用实时价，缺失（停牌）时用最近有效价（前收盘价兜底），
这与基金公司在停牌股上采用"最近成交价"的实务口径一致。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from etf.pcf_parser import PCFBasket


@dataclass
class IOPVResult:
    """一次 IOPV 估算结果。"""

    iopv: float
    used_constituents: int  # 实际参与计算的成分股数量
    missing_symbols: list[str] = field(default_factory=list)  # 无价格且无兜底的成分股


class IOPVEstimator:
    """由 PCF 篮子 + 成分股价格估算 IOPV。"""

    def estimate(
        self,
        basket: PCFBasket,
        prices: dict[str, float],
        fallback_prices: dict[str, float] | None = None,
    ) -> IOPVResult:
        """估算单时点 IOPV。

        - prices：实时价 {symbol: price}
        - fallback_prices：缺失时的兜底价（如前收盘价）
        """
        fallback = fallback_prices or {}
        if basket.creation_unit <= 0:
            raise ValueError("最小申赎单位必须为正")
        total_value = basket.cash_component
        used = 0
        missing: list[str] = []
        for c in basket.constituents:
            price = prices.get(c.symbol)
            if price is None or not np.isfinite(price) or price <= 0:
                price = fallback.get(c.symbol)
            if price is None or not np.isfinite(price) or price <= 0:
                missing.append(c.symbol)
                continue
            total_value += c.quantity * float(price)
            used += 1
        return IOPVResult(
            iopv=total_value / basket.creation_unit, used_constituents=used, missing_symbols=missing
        )

    def estimate_series(
        self,
        basket: PCFBasket,
        price_df: pd.DataFrame,
        fallback_prices: dict[str, float] | None = None,
    ) -> pd.Series:
        """估算 IOPV 时间序列。

        price_df：index=时间，列=成分股代码，值为实时价。缺失值前向填充
        （停牌沿用最近价），仍缺失的用 fallback_prices 兜底。
        """
        df = price_df.copy()
        df = df.ffill()
        results = []
        for _ts, row in df.iterrows():
            prices = {str(k): float(v) for k, v in row.items() if pd.notna(v)}
            res = self.estimate(basket, prices, fallback_prices=fallback_prices)
            results.append(res.iopv)
        return pd.Series(results, index=df.index, name="iopv_est")


__all__ = ["IOPVEstimator", "IOPVResult"]
