"""B2 阈值网格优化：滚动样本内寻优，输出参数-绩效表，注明过拟合风险。

与 C4 联动：网格寻优**只在样本内（IS）**进行，样本外（OOS）仅对选出的最优参数
评估一次并单独报告，禁止全样本寻优（否则会选中过拟合 IS 的参数，OOS 必然衰减）。

绩效口径（无杠杆，名义=1 的套利收益）：
- long（折价买入）：profit = exit_premium - entry_premium - unit_cost
- short（溢价卖出）：profit = entry_premium - exit_premium - unit_cost
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from etf.etf_signal import (
    ETFSignalGenerator,
    ACTION_OPEN,
    ACTION_CLOSE,
    DIR_LONG,
    DIR_SHORT,
    COL_TS,
    COL_ACTION,
    COL_DIRECTION,
    COL_PREMIUM,
)
from etf.threshold_config import ThresholdConfig


def evaluate_signals(signals: pd.DataFrame, unit_cost_rate: float) -> dict:
    """由开/平仓信号事件流计算绩效指标。

    按时间顺序把 open→close 配对为一笔往返交易，计算净收益（率）。
    返回 {n_trades, n_opens, n_closes, win_rate, total_return, avg_profit, sharpe}。
    """
    empty = {
        "n_trades": 0, "n_opens": 0, "n_closes": 0,
        "win_rate": 0.0, "total_return": 0.0, "avg_profit": 0.0, "sharpe": 0.0,
    }
    if signals is None or len(signals) == 0:
        return empty

    df = signals.sort_values(COL_TS).reset_index(drop=True)
    n_opens = int((df[COL_ACTION] == ACTION_OPEN).sum())
    n_closes = int((df[COL_ACTION] == ACTION_CLOSE).sum())

    trades: list = []
    entry = None
    for _, row in df.iterrows():
        if row[COL_ACTION] == ACTION_OPEN:
            entry = row
        elif row[COL_ACTION] == ACTION_CLOSE and entry is not None:
            if entry[COL_DIRECTION] == DIR_LONG:
                profit = row[COL_PREMIUM] - entry[COL_PREMIUM]
            else:
                profit = entry[COL_PREMIUM] - row[COL_PREMIUM]
            trades.append(float(profit) - float(unit_cost_rate))
            entry = None

    if not trades:
        return {**empty, "n_opens": n_opens, "n_closes": n_closes}

    arr = np.asarray(trades, dtype=float)
    wins = arr[arr > 0]
    return {
        "n_trades": int(len(arr)),
        "n_opens": n_opens,
        "n_closes": n_closes,
        "win_rate": float(len(wins) / len(arr)),
        "total_return": float(arr.sum()),
        "avg_profit": float(arr.mean()),
        "sharpe": float(arr.mean() / arr.std(ddof=1)) if arr.std(ddof=1) > 0 else 0.0,
    }


@dataclass
class GridResult:
    """网格寻优结果。"""

    param_table: pd.DataFrame = field(default_factory=pd.DataFrame)  # 各参数组合 + IS 指标
    best_params: Dict = field(default_factory=dict)                  # 样本内最优参数
    best_is_metrics: Dict = field(default_factory=dict)              # 最优参数 IS 指标
    oos_metrics: Dict = field(default_factory=dict)                  # 最优参数 OOS 指标（只评估一次）
    is_overfit_risk: bool = False                                    # OOS 相对 IS 是否显著衰减
    note: str = ""


def grid_search(
    premium: pd.Series,
    unit_cost_rate: float,
    base_config: Optional[ThresholdConfig] = None,
    split_ratio: float = 0.7,
    grid: Optional[Dict[str, Sequence]] = None,
) -> GridResult:
    """样本内网格寻优 + 样本外一次验证。

    - 仅前 ``split_ratio`` 的样本内（IS）参与寻优；后 (1-split_ratio) 样本外（OOS）
      只对最优参数评估一次，输出 IS/OOS 两组指标。
    - grid 缺省使用 ThresholdConfig 的 grid_* 范围（zscore_entry / entry_buffer /
      stop_loss_bp）；可显式传入更小网格加速。
    """
    base = base_config or ThresholdConfig()
    if grid is None:
        grid = {
            "zscore_entry": base.grid_zscore_entry,
            "entry_buffer": base.grid_entry_buffer,
            "stop_loss_bp": base.grid_stop_loss_bp,
        }

    x = pd.to_numeric(premium, errors="coerce").dropna()
    if len(x) < 20:
        raise ValueError("样本过短，无法做样本内网格寻优")
    split_at = int(len(x) * split_ratio)
    is_x, oos_x = x.iloc[:split_at], x.iloc[split_at:]

    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))

    rows = []
    best_params: Optional[dict] = None
    best_score = -np.inf
    for combo in combos:
        params = dict(zip(keys, combo))
        cfg = ThresholdConfig(**{**base.__dict__, **params})
        try:
            cfg.validate()
        except ValueError:
            continue
        gen = ETFSignalGenerator(cfg, unit_cost_rate=unit_cost_rate)
        sig = gen.generate(is_x)
        m = evaluate_signals(sig, unit_cost_rate)
        rows.append({**params, **m})
        if m["total_return"] > best_score:
            best_score = m["total_return"]
            best_params = params

    table = pd.DataFrame(rows)
    if best_params is None:
        raise ValueError("无合法参数组合（网格全部非法）")
    table = table.sort_values("total_return", ascending=False).reset_index(drop=True)

    best_cfg = ThresholdConfig(**{**base.__dict__, **best_params})
    best_gen = ETFSignalGenerator(best_cfg, unit_cost_rate=unit_cost_rate)
    is_metrics = evaluate_signals(best_gen.generate(is_x), unit_cost_rate)
    oos_metrics = evaluate_signals(best_gen.generate(oos_x), unit_cost_rate)

    is_total = is_metrics["total_return"]
    oos_total = oos_metrics["total_return"]
    is_overfit = (oos_total < 0) or (is_total > 0 and oos_total < 0.5 * is_total)

    note = (
        "网格只在样本内(前 {:.0%})寻优，样本外仅对最优参数评估一次；"
        "若 OOS 显著弱于 IS，说明参数过拟合，应扩大样本、减少网格维度或做 C4 过拟合检测。"
    ).format(split_ratio)

    return GridResult(
        param_table=table,
        best_params=best_params,
        best_is_metrics=is_metrics,
        oos_metrics=oos_metrics,
        is_overfit_risk=is_overfit,
        note=note,
    )


__all__ = ["evaluate_signals", "grid_search", "GridResult"]
