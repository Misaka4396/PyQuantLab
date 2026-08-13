"""B2 折溢价信号层：滚动 z-score/分位数 + 开平仓阈值 + 强平 + 方向判定。

防前视约束（硬约束）：
- 所有滚动统计（z-score / 分位数）只在窗口 [t-window+1, t] 内计算，绝不使用
  t 之后的数据（``rolling`` 天然因果，无全样本标准化）。
- 开仓阈值 = 单位交易成本 + 缓冲（成本来自 A3 cost_model），逻辑上覆盖成本。

方向约定：
- 溢价（premium > 0）：ETF 二级价高于净值 → 卖 ETF、买篮子（direction=short）。
- 折价（premium < 0）：ETF 二级价低于净值 → 买 ETF、卖篮子（direction=long）。

输出：开/平仓信号事件流 DataFrame（字段含 ts/action/direction/threshold_basis），
并附 ``signal_to_engine_events`` 把一行信号转换为 A2 引擎的 ``SignalEvent`` 列表。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from cost_model import CostModel
from engine.events import BUY, SELL, SignalEvent
from etf.threshold_config import (
    ThresholdConfig,
    DIR_LONG,
    DIR_SHORT,
    EXIT_MEAN_REVERT,
    EXIT_STOP_LOSS,
    EXIT_FORCE_CLOSE,
)

# 输出事件列名
COL_TS = "ts"
COL_ACTION = "action"
COL_DIRECTION = "direction"
COL_ETF_SIDE = "etf_side"
COL_PREMIUM = "premium"
COL_ZSCORE = "zscore"
COL_QUANTILE = "quantile"
COL_ENTRY_THRESHOLD = "entry_threshold"
COL_REASON = "threshold_basis"
COL_STOP_LEVEL = "stop_loss_level"

ACTION_OPEN = "open"
ACTION_CLOSE = "close"

_EVENT_COLUMNS = [
    COL_TS, COL_ACTION, COL_DIRECTION, COL_ETF_SIDE, COL_PREMIUM,
    COL_ZSCORE, COL_QUANTILE, COL_ENTRY_THRESHOLD, COL_REASON, COL_STOP_LEVEL,
]


def rolling_zscore(premium: pd.Series, window: int) -> pd.Series:
    """滚动 z-score：z_t = (x_t - mean(x_{t-w+1..t})) / std(x_{t-w+1..t})。

    只在滚动窗口内计算（无前视）；窗口内标准差为 0（常数序列）时 z 置 NaN。
    """
    x = pd.to_numeric(premium, errors="coerce")
    mu = x.rolling(window).mean()
    sd = x.rolling(window).std()
    return (x - mu) / sd.replace(0, np.nan)


def rolling_quantile(premium: pd.Series, window: int) -> pd.Series:
    """滚动分位数：当前值在窗口 [t-window+1, t] 内的百分位（0~1，无前视）。"""
    x = pd.to_numeric(premium, errors="coerce")

    def _rank(w: np.ndarray) -> float:
        return float((w <= w[-1]).mean())

    return x.rolling(window).apply(_rank, raw=True)


def entry_threshold(config: ThresholdConfig, unit_cost_rate: float) -> float:
    """开仓阈值 = 单位成本 + 缓冲（逻辑上覆盖成本）。"""
    return float(unit_cost_rate) + float(config.entry_buffer)


def round_trip_cost_rate(
    cost_model: CostModel,
    price: float,
    quantity: float,
    is_etf: bool = True,
    volume: float = 0.0,
) -> float:
    """单位（双边往返）成本率 = (一次买 + 一次卖的总成本) / 双边成交额。

    成本 = 固定费用（佣金/印花税/过户费）+ 滑点（价差 + 冲击），与 A3 口径一致。
    返回比例（如 0.0006 = 6bp），用于开仓阈值"覆盖成本"。
    """
    buy = cost_model.compute("BUY", quantity, price, is_etf=is_etf, volume=volume)
    sell = cost_model.compute("SELL", quantity, price, is_etf=is_etf, volume=volume)
    notional = 2.0 * float(price) * float(quantity)
    cost = (buy.total_fee + buy.slippage_cost) + (sell.total_fee + sell.slippage_cost)
    return float(cost / notional) if notional > 0 else 0.0


class ETFSignalGenerator:
    """折溢价信号生成器：滚动判据 + 开/平/强平状态机。"""

    def __init__(
        self,
        config: Optional[ThresholdConfig] = None,
        cost_model: Optional[CostModel] = None,
        unit_cost_rate: Optional[float] = None,
    ):
        self.config = config or ThresholdConfig()
        self.config.validate()
        if unit_cost_rate is None:
            model = cost_model or CostModel()
            unit_cost_rate = round_trip_cost_rate(
                model, price=3.0, quantity=10000, is_etf=True
            )
        self.unit_cost_rate = float(unit_cost_rate)
        self.threshold = entry_threshold(self.config, self.unit_cost_rate)

    # ------------------------------------------------------------------
    # 主入口：生成信号事件流
    # ------------------------------------------------------------------
    def generate(self, premium: pd.Series) -> pd.DataFrame:
        """由折溢价序列生成开/平仓信号事件流。

        返回 DataFrame，仅含 open/close 事件行，字段：
        ts / action / direction / etf_side / premium / zscore / quantile /
        entry_threshold / threshold_basis / stop_loss_level。
        """
        cfg = self.config
        x = pd.to_numeric(premium, errors="coerce")
        z = rolling_zscore(x, cfg.zscore_window)
        q = rolling_quantile(x, cfg.zscore_window)

        events: List[dict] = []
        position: Optional[dict] = None  # {direction, entry_ts, entry_premium}
        force_close_min = cfg.force_close_minute_of_day()
        min_hold = pd.Timedelta(minutes=cfg.min_holding_minutes)

        for ts, prem, zs, qt in zip(x.index, x.values, z.values, q.values):
            if not np.isfinite(prem) or not np.isfinite(zs) or not np.isfinite(qt):
                continue  # 数据不足（warmup / 缺失），不产生信号

            force_close_now = (ts.hour * 60 + ts.minute) >= force_close_min

            if position is None:
                # 收盘强平只作用于持仓；空仓时收盘前不再新开仓（避免隔夜敞口）
                if force_close_now:
                    continue
                direction, basis = self._entry_signal(prem, zs, qt)
                if direction is not None:
                    position = {
                        "direction": direction,
                        "entry_ts": ts,
                        "entry_premium": prem,
                    }
                    events.append(self._make_event(
                        ts, ACTION_OPEN, direction, prem, zs, qt, basis, position=None
                    ))
            else:
                direction = position["direction"]
                stop_level = self._stop_level(position)
                hit_stop = (
                    (direction == DIR_LONG and prem <= stop_level)
                    or (direction == DIR_SHORT and prem >= stop_level)
                )
                reverted = (
                    (direction == DIR_LONG and zs >= cfg.zscore_exit)
                    or (direction == DIR_SHORT and zs <= -cfg.zscore_exit)
                )
                holding_ok = (ts - position["entry_ts"]) >= min_hold

                if holding_ok and force_close_now:
                    reason = f"平仓:收盘强平(>{cfg.force_close_time})"
                elif holding_ok and hit_stop:
                    reason = f"平仓:止损(突破{cfg.stop_loss_bp}bp)"
                elif holding_ok and reverted:
                    reason = "平仓:均值回归/归零"
                else:
                    reason = None

                if reason is not None:
                    events.append(self._make_event(
                        ts, ACTION_CLOSE, direction, prem, zs, qt, reason, position=position
                    ))
                    position = None

        out = pd.DataFrame(events, columns=_EVENT_COLUMNS)
        if out.empty:
            return out
        out[COL_TS] = pd.to_datetime(out[COL_TS])
        return out.sort_values(COL_TS).reset_index(drop=True)

    # ------------------------------------------------------------------
    # 内部：开仓 / 止损 / 事件行
    # ------------------------------------------------------------------
    def _entry_signal(self, prem: float, zs: float, qt: float):
        """开仓判据：|溢价| 覆盖成本 + z-score/分位数极端。返回 (direction, basis)。"""
        cfg = self.config
        if prem >= self.threshold:
            if cfg.use_quantile:
                cond, desc = qt >= cfg.quantile_entry, f"分位数>={cfg.quantile_entry:.2f}"
            else:
                cond, desc = zs >= cfg.zscore_entry, f"z>={cfg.zscore_entry:.2f}"
            if cond:
                basis = (f"开仓:溢价{prem:.5f}>=阈值{self.threshold:.5f}"
                         f"(成本{self.unit_cost_rate:.5f}+缓冲{cfg.entry_buffer:.5f}) 且 {desc}")
                return DIR_SHORT, basis
        elif prem <= -self.threshold:
            if cfg.use_quantile:
                cond, desc = qt <= 1.0 - cfg.quantile_entry, f"分位数<={1.0 - cfg.quantile_entry:.2f}"
            else:
                cond, desc = zs <= -cfg.zscore_entry, f"z<={-cfg.zscore_entry:.2f}"
            if cond:
                basis = (f"开仓:折价{prem:.5f}<=-阈值{-self.threshold:.5f}"
                         f"(成本{self.unit_cost_rate:.5f}+缓冲{cfg.entry_buffer:.5f}) 且 {desc}")
                return DIR_LONG, basis
        return None, None

    def _stop_level(self, position: dict) -> float:
        """持仓止损线（溢价朝不利方向移动 stop_loss_bp 即触发）。"""
        stop_bp = self.config.stop_loss_bp / 1e4
        if position["direction"] == DIR_LONG:
            return position["entry_premium"] - stop_bp
        return position["entry_premium"] + stop_bp

    def _make_event(
        self,
        ts: pd.Timestamp,
        action: str,
        direction: str,
        prem: float,
        zs: float,
        qt: float,
        reason: str,
        position: Optional[dict],
    ) -> dict:
        etf_side = SELL if direction == DIR_SHORT else BUY
        return {
            COL_TS: ts,
            COL_ACTION: action,
            COL_DIRECTION: direction,
            COL_ETF_SIDE: etf_side,
            COL_PREMIUM: float(prem),
            COL_ZSCORE: float(zs),
            COL_QUANTILE: float(qt),
            COL_ENTRY_THRESHOLD: self.threshold,
            COL_REASON: reason,
            COL_STOP_LEVEL: self._stop_level(position) if position is not None else np.nan,
        }


def signal_to_engine_events(
    row,
    etf_symbol: str,
    quantity: float,
    basket: Optional[Dict[str, float]] = None,
) -> List[SignalEvent]:
    """把一行信号事件转换为 A2 引擎的 ``SignalEvent`` 列表（可被 EventEngine 消费）。

    - 开仓 long（折价）：买 ETF（BUY），卖篮子成分（SELL）。
    - 开仓 short（溢价）：卖 ETF（SELL），买篮子成分（BUY）。
    - 平仓：与开仓方向相反。
    - basket：{symbol: quantity} 篮子成分股目标数量；缺省只输出 ETF 腿。
    """
    ts = pd.Timestamp(row[COL_TS])
    action = str(row[COL_ACTION])
    direction = str(row[COL_DIRECTION])
    reason = str(row[COL_REASON])

    if action == ACTION_OPEN:
        etf_side = SELL if direction == DIR_SHORT else BUY
        basket_side = BUY if direction == DIR_SHORT else SELL
    else:  # close：与开仓相反
        etf_side = BUY if direction == DIR_SHORT else SELL
        basket_side = SELL if direction == DIR_SHORT else BUY

    events = [SignalEvent(
        timestamp=ts, symbol=etf_symbol, side=etf_side,
        quantity=float(quantity), order_type="market", reason=reason,
    )]
    if basket:
        for sym, qty in basket.items():
            events.append(SignalEvent(
                timestamp=ts, symbol=str(sym), side=basket_side,
                quantity=float(qty), order_type="market", reason=reason,
            ))
    return events


__all__ = [
    "ETFSignalGenerator",
    "rolling_zscore",
    "rolling_quantile",
    "entry_threshold",
    "round_trip_cost_rate",
    "signal_to_engine_events",
    "ACTION_OPEN",
    "ACTION_CLOSE",
    "DIR_LONG",
    "DIR_SHORT",
    "COL_TS",
    "COL_ACTION",
    "COL_DIRECTION",
    "COL_REASON",
]
