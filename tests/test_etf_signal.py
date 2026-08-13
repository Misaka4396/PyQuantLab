"""B2 折溢价信号层单元测试。

覆盖验收标准：
- 信号无前视（截断不变性 + 滚动 z-score/分位数与手工因果实现一致）
- 开仓阈值逻辑上覆盖成本（开仓阈值 = 单位成本 + 缓冲）
- 输出开平仓信号事件流（字段含 ts/action/direction/threshold_basis）
- 方向判定（溢价→卖 ETF；折价→买 ETF）
- 止损 / 均值回归平仓
- 网格优化只在样本内寻优，输出参数-绩效表并注明过拟合风险
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cost_config import CostConfig
from cost_model import CostModel
from etf.threshold_config import ThresholdConfig
from etf.etf_signal import (
    ETFSignalGenerator,
    rolling_zscore,
    rolling_quantile,
    entry_threshold,
    round_trip_cost_rate,
    signal_to_engine_events,
    ACTION_OPEN,
    ACTION_CLOSE,
    DIR_LONG,
    DIR_SHORT,
)
from etf.signal_grid import grid_search, evaluate_signals


def _minute_index(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-02 09:30", periods=n, freq="1min")


# ---------------------------------------------------------------------------
# 滚动统计：无前视（手工因果实现对照）
# ---------------------------------------------------------------------------
def test_rolling_zscore_matches_manual_causal():
    idx = _minute_index(10)
    premium = pd.Series([1.0, 2.0, 1.5, 2.5, 3.0, 2.8, 3.2, 2.9, 3.1, 3.5], index=idx)
    z = rolling_zscore(premium, window=3)
    assert z.iloc[:2].isna().all()  # warmup
    for i in range(2, 10):
        w = premium.iloc[i - 2:i + 1]
        expected = (w.iloc[-1] - w.mean()) / w.std(ddof=1)
        assert z.iloc[i] == pytest.approx(expected)


def test_rolling_quantile_matches_manual_causal():
    idx = _minute_index(10)
    premium = pd.Series([1.0, 2.0, 1.5, 2.5, 3.0, 2.8, 3.2, 2.9, 3.1, 3.5], index=idx)
    q = rolling_quantile(premium, window=3)
    assert q.iloc[:2].isna().all()
    for i in range(2, 10):
        w = premium.iloc[i - 2:i + 1]
        expected = float((w <= w.iloc[-1]).mean())
        assert q.iloc[i] == pytest.approx(expected)


def test_signal_no_lookahead_truncation_invariance():
    """信号在 t 时刻只用 t 及之前数据：截断未来数据不改变截断点之前的信号。"""
    idx = _minute_index(200)
    premium = pd.Series(np.sin(np.linspace(0, 20, 200)) * 0.005, index=idx)
    cfg = ThresholdConfig(zscore_window=20, use_quantile=False, zscore_entry=0.5, entry_buffer=0.0)
    gen = ETFSignalGenerator(cfg, unit_cost_rate=0.0)

    full = gen.generate(premium)
    trunc = gen.generate(premium.iloc[:150])

    cutoff = idx[140]
    f = full[full["ts"] < cutoff].reset_index(drop=True)
    t = trunc[trunc["ts"] < cutoff].reset_index(drop=True)
    assert not f.empty
    pd.testing.assert_frame_equal(f, t)


# ---------------------------------------------------------------------------
# 开仓阈值覆盖成本
# ---------------------------------------------------------------------------
def test_entry_threshold_covers_cost():
    cfg = ThresholdConfig(entry_buffer=0.0005)
    cost = 0.0006
    assert entry_threshold(cfg, cost) == pytest.approx(cost + cfg.entry_buffer)
    assert entry_threshold(cfg, cost) >= cost  # 至少覆盖成本


def test_round_trip_cost_rate_uses_a3():
    model = CostModel(CostConfig(
        commission_rate=0.001, min_commission=0.0, stamp_tax_rate=0.0,
        transfer_fee_rate=0.0, spread_rate=0.002, impact_coef=0.0,
    ))
    rate = round_trip_cost_rate(model, price=10.0, quantity=1000, is_etf=False, volume=0.0)
    # 买+卖总成本 40 元 / 双边成交额 20000 = 0.002
    assert rate == pytest.approx(0.002)


# ---------------------------------------------------------------------------
# 信号事件流 + 方向
# ---------------------------------------------------------------------------
def test_signal_event_stream_format():
    idx = _minute_index(100)
    premium = pd.Series(np.sin(np.linspace(0, 10, 100)) * 0.004, index=idx)
    cfg = ThresholdConfig(zscore_window=10, use_quantile=False, zscore_entry=0.5, entry_buffer=0.0)
    gen = ETFSignalGenerator(cfg, unit_cost_rate=0.0)
    df = gen.generate(premium)
    for col in ("ts", "action", "direction", "threshold_basis"):
        assert col in df.columns
    assert set(df["action"]).issubset({ACTION_OPEN, ACTION_CLOSE})
    assert set(df["direction"]).issubset({DIR_LONG, DIR_SHORT})
    # 事件按时间升序，开平交替
    assert df["ts"].is_monotonic_increasing


def test_direction_premium_short_discount_long():
    idx = _minute_index(120)
    rng = np.random.default_rng(1)
    noise = rng.normal(0, 1e-5, 120)
    base = np.where(np.arange(120) < 60, 0.010, -0.010)  # 前段溢价、后段折价
    premium = pd.Series(base + noise, index=idx)
    cfg = ThresholdConfig(zscore_window=10, use_quantile=False, zscore_entry=0.0, entry_buffer=0.0)
    gen = ETFSignalGenerator(cfg, unit_cost_rate=0.0)
    df = gen.generate(premium)
    opens = df[df["action"] == ACTION_OPEN]
    assert opens.iloc[0]["direction"] == DIR_SHORT       # 溢价 → 卖 ETF
    assert opens.iloc[0]["etf_side"] == "SELL"
    assert (opens["direction"] == DIR_LONG).any()        # 折价 → 买 ETF


def test_stop_loss_close_event():
    idx = _minute_index(120)
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 1e-5, 120)
    base = np.where(np.arange(120) < 60, -0.010, -0.020)  # 折价加深（对 long 不利）
    premium = pd.Series(base + noise, index=idx)
    cfg = ThresholdConfig(zscore_window=10, use_quantile=False, zscore_entry=0.0,
                          entry_buffer=0.0, stop_loss_bp=50.0, zscore_exit=10.0)
    gen = ETFSignalGenerator(cfg, unit_cost_rate=0.0)
    df = gen.generate(premium)
    closes = df[df["action"] == ACTION_CLOSE]
    assert not closes.empty
    assert any("止损" in str(r) for r in closes["threshold_basis"])


# ---------------------------------------------------------------------------
# A2 引擎兼容：信号 → SignalEvent
# ---------------------------------------------------------------------------
def test_signal_to_engine_events_direction():
    row = {
        "ts": pd.Timestamp("2024-01-02 10:00"), "action": "open", "direction": "short",
        "threshold_basis": "x", "premium": 0.01, "zscore": 2.0, "quantile": 0.99,
    }
    events = signal_to_engine_events(row, etf_symbol="510300", quantity=1000,
                                     basket={"600000": 500})
    assert events[0].symbol == "510300" and events[0].side == "SELL"
    assert events[1].symbol == "600000" and events[1].side == "BUY"

    # 平仓反向
    row_close = {**row, "action": "close"}
    events2 = signal_to_engine_events(row_close, etf_symbol="510300", quantity=1000,
                                      basket={"600000": 500})
    assert events2[0].side == "BUY"
    assert events2[1].side == "SELL"


# ---------------------------------------------------------------------------
# 网格优化
# ---------------------------------------------------------------------------
def test_evaluate_signals_round_trip_pnl():
    idx = _minute_index(4)
    signals = pd.DataFrame({
        "ts": idx,
        "action": [ACTION_OPEN, ACTION_CLOSE, ACTION_OPEN, ACTION_CLOSE],
        "direction": [DIR_LONG, DIR_LONG, DIR_SHORT, DIR_SHORT],
        "premium": [-0.010, -0.001, 0.010, 0.001],
    })
    m = evaluate_signals(signals, unit_cost_rate=0.001)
    # long: -0.001 - (-0.010) - 0.001 = 0.008；short: 0.010 - 0.001 - 0.001 = 0.008
    assert m["n_trades"] == 2
    assert m["total_return"] == pytest.approx(0.016)
    assert m["win_rate"] == pytest.approx(1.0)


def test_grid_search_in_sample_optimization():
    idx = _minute_index(300)
    premium = pd.Series(np.sin(np.linspace(0, 30, 300)) * 0.004, index=idx)
    base = ThresholdConfig(use_quantile=False, zscore_window=20)
    res = grid_search(
        premium, unit_cost_rate=0.0005, base_config=base, split_ratio=0.7,
        grid={"zscore_entry": (0.5, 1.0), "entry_buffer": (0.0, 0.0005),
              "stop_loss_bp": (30.0, 50.0)},
    )
    assert not res.param_table.empty
    for col in ("zscore_entry", "entry_buffer", "stop_loss_bp", "total_return", "n_trades"):
        assert col in res.param_table.columns
    assert res.best_params
    assert "total_return" in res.best_is_metrics
    assert "total_return" in res.oos_metrics
    assert "样本内" in res.note and "样本外" in res.note
