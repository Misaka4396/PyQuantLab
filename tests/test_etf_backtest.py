"""B4 ETF 套利回测集成测试。

覆盖验收标准：
- 理想 vs 真实执行对比清晰（两条权益曲线存在且不同）
- 扣成本收益可复现（固定种子两次运行一致）
- 容量分析输出容量上限估算；压力测试输出最大敞口
- 报告含折溢价机会频率/幅度统计
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from etf.basket_execution import BasketExecutor
from etf.etf_backtest import (
    run_dual_mode,
    capacity_analysis,
    stress_test,
    premium_opportunity_stats,
    generate_report,
)
from etf.execution_config import ExecutionConfig
from etf.threshold_config import ThresholdConfig


# ---------------------------------------------------------------------------
# 合成数据：均值回归折溢价 + 联动 ETF 价（溢价→卖高，回归→买低）
# ---------------------------------------------------------------------------
def _synthetic_data(n_days: int = 4, seed: int = 1, base: float = 3.0):
    days = pd.bdate_range("2024-01-02", periods=n_days)
    idx_parts = []
    for d in days:
        r = pd.date_range(
            d.replace(hour=9, minute=30), d.replace(hour=15, minute=0), freq="1min"
        )
        mask = (r.hour >= 9) & (r.hour <= 15)
        mask &= ~((r.hour == 11) & (r.minute > 30))
        mask &= r.hour != 12
        idx_parts.append(r[mask])
    idx = pd.DatetimeIndex(np.concatenate([list(x) for x in idx_parts]))

    t = np.arange(len(idx))
    premium = pd.Series(0.006 * np.sin(2 * np.pi * t / 90.0), index=idx)
    close = base * (1.0 + premium)
    open_ = close.shift(1).fillna(base)
    high = np.maximum(open_, close) * 1.0001
    low = np.minimum(open_, close) * 0.9999
    volume = np.full(len(idx), 1_000_000.0)
    amount = volume * close
    quotes = pd.DataFrame({
        "open": open_, "high": high, "low": low,
        "close": close, "volume": volume, "amount": amount,
    }, index=idx)
    quotes.index.name = "datetime"
    return quotes, premium


def _cfg() -> ThresholdConfig:
    return ThresholdConfig(
        use_quantile=False, zscore_window=20, zscore_entry=0.5, entry_buffer=0.0,
        stop_loss_bp=30.0, zscore_exit=0.0, min_holding_minutes=5,
        force_close_time="14:45",
    )


def _sync_executor(seed: int = 42) -> BasketExecutor:
    return BasketExecutor(ExecutionConfig(
        partial_fill_prob=0.5,
        partial_fill_ratio_min=0.4,
        partial_fill_ratio_max=0.8,
        seed=seed,
    ))


# ---------------------------------------------------------------------------
# 双模式对比
# ---------------------------------------------------------------------------
def test_dual_mode_two_equity_curves_exist_and_differ():
    quotes, premium = _synthetic_data()
    result = run_dual_mode(
        quotes, premium, "510300", etf_quantity=100_000.0,
        threshold_config=_cfg(), unit_cost_rate=0.0, seed=42,
        executor=_sync_executor(42),
    )

    ideal_eq = result.ideal.equity_curve["equity"]
    sync_eq = result.sync_risk.equity_curve["equity"]
    assert len(ideal_eq) > 0
    assert len(sync_eq) > 0
    # 两条权益曲线必须存在且不同（执行风险导致成交/收益差异）
    assert not np.allclose(ideal_eq.values, sync_eq.values)
    # 有成交（策略确实产生了交易）
    assert len(result.ideal.fills) > 0
    assert len(result.sync_risk.fills) > 0
    # 对比指标存在
    assert "execution_drag_return" in result.comparison


# ---------------------------------------------------------------------------
# 可复现（固定种子）
# ---------------------------------------------------------------------------
def test_net_of_cost_reproducible_same_seed():
    quotes, premium = _synthetic_data()

    def run_once():
        return run_dual_mode(
            quotes, premium, "510300", etf_quantity=100_000.0,
            threshold_config=_cfg(), unit_cost_rate=0.0, seed=42,
            executor=_sync_executor(42),
        )

    r1, r2 = run_once(), run_once()
    pd.testing.assert_frame_equal(r1.sync_risk.equity_curve, r2.sync_risk.equity_curve)
    pd.testing.assert_frame_equal(r1.sync_risk.fills, r2.sync_risk.fills)
    # 扣成本收益（累计收益率）复现
    assert r1.sync_risk_metrics["total_return"] == pytest.approx(
        r2.sync_risk_metrics["total_return"]
    )
    assert r1.ideal_metrics["total_return"] == pytest.approx(
        r2.ideal_metrics["total_return"]
    )


# ---------------------------------------------------------------------------
# 容量 + 压力测试结论
# ---------------------------------------------------------------------------
def test_capacity_analysis_outputs_upper_bound():
    quotes, premium = _synthetic_data()
    result = run_dual_mode(
        quotes, premium, "510300", etf_quantity=100_000.0,
        threshold_config=_cfg(), unit_cost_rate=0.0, seed=42,
        executor=_sync_executor(42),
    )
    cap = result.capacity
    assert "capacity_capital" in cap
    assert "max_participation" in cap
    assert cap["max_participation"] > 0
    # 容量上限为正（可复现）
    assert cap["capacity_capital"] > 0


def test_stress_test_outputs_max_exposure():
    legs = [
        {"symbol": "510300", "side": "SELL", "quantity": 100_000.0},
        {"symbol": "600000", "side": "BUY", "quantity": 200_000.0},
    ]
    prices = {"510300": 3.0, "600000": 7.5}
    stress = stress_test(legs, prices, seed=42)
    assert "max_exposure" in stress
    assert "max_exposure_scenario" in stress
    assert stress["max_exposure"] >= 0
    assert set(stress["scenarios"]) == {"limit_wave", "suspension", "extreme_premium"}
    # 停牌场景敞口比例为 1（全部无法成交）
    assert stress["scenarios"]["suspension"]["exposure_ratio"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 折溢价机会统计 + 报告联动
# ---------------------------------------------------------------------------
def test_premium_opportunity_stats_frequency_amplitude():
    quotes, premium = _synthetic_data()
    stats = premium_opportunity_stats(premium, threshold=0.001)
    for key in ("n_opportunities", "opportunity_frequency",
                "mean_opportunity_amplitude", "max_opportunity_amplitude"):
        assert key in stats
    assert 0.0 <= stats["opportunity_frequency"] <= 1.0
    assert stats["n_opportunities"] > 0
    assert stats["max_opportunity_amplitude"] >= stats["mean_opportunity_amplitude"]


def test_report_contains_premium_stats(tmp_path):
    quotes, premium = _synthetic_data()
    result = run_dual_mode(
        quotes, premium, "510300", etf_quantity=100_000.0,
        threshold_config=_cfg(), unit_cost_rate=0.0, seed=42,
        executor=_sync_executor(42),
    )
    paths = generate_report(result, tmp_path)
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert "premium_stats" in summary
    assert "capacity" in summary
    assert "stress" in summary
    assert summary["premium_stats"]["n_opportunities"] > 0
    # 报告产物落盘
    assert (tmp_path / "ideal.png").exists()
    assert (tmp_path / "sync_risk.png").exists()
    assert (tmp_path / "comparison.png").exists()
