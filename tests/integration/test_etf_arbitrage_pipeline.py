"""B 链集成测试：A1 数据 → B1 折溢价 → B2 信号 → B3 执行 → A2 引擎回测 → A4 评估。

跨模块全链路验证（tests/integration/ 分层，P9 规范）：
- 合成数据 + 固定种子，离线确定性（无网络/时间依赖，防 flaky）
- 折溢价序列注入"机会窗口"（幅度 > 成本+缓冲阈值）驱动信号与交易
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cost_model import CostConfig, CostModel
from etf.basket_execution import BasketExecutor, ExecutionConfig, make_basket_legs
from etf.etf_backtest import run_dual_mode
from etf.etf_data import synthetic_etf_minute
from etf.etf_signal import ETFSignalGenerator
from etf.pcf_parser import parse_pcf

DEMO_PCF = Path(__file__).resolve().parents[2] / "etf" / "demo_data" / "pcf_510300_20240102.txt"
ETF = "510300"
# demo PCF 真实成分股代码
STOCKS = ["600000", "600036", "601318", "600519", "000858"]


def _minute_index(n: int) -> pd.DatetimeIndex:
    """分钟级时间轴（09:30 起，跳过午休以保持连续）。"""
    start = pd.Timestamp("2024-01-05 09:30:00")
    return pd.date_range(start, periods=n, freq="min")


def _quotes(n: int = 240, seed: int = 42) -> pd.DataFrame:
    """合成 ETF 分钟行情（index=分钟）。"""
    idx = _minute_index(n)
    return synthetic_etf_minute(ETF, idx[0], idx[-1], seed=seed)


def _premium_with_opportunities(n: int = 240, amp: float = 0.003) -> pd.Series:
    """人工折溢价序列：两个机会窗口（正/负超阈值）+ 其余回归 0。"""
    idx = _minute_index(n)
    base = np.zeros(n)
    base[100:140] = amp  # 窗口 1：正折溢价（溢价套利机会）
    base[180:220] = -amp  # 窗口 2：负折溢价（折价套利机会）
    # 平滑进出场：加小噪声避免常数窗口
    noise = np.random.default_rng(1).normal(0, 1e-5, n)
    return pd.Series(base + noise, index=idx, name="premium")


# ---------------------------------------------------------------------------
# A1/B1 → B2：折溢价信号流水线
# ---------------------------------------------------------------------------
def test_b1_b2_signal_pipeline_produces_open_close_events():
    """折溢价序列 → 信号事件流：开仓方向正确 + 平仓齐全。"""
    premium = _premium_with_opportunities()
    assert premium.notna().all()

    gen = ETFSignalGenerator()
    events = gen.generate(premium)
    assert not events.empty, "机会窗口应触发信号"
    actions = set(events["action"].unique())
    assert "open" in actions, f"开仓信号缺失: {actions}"
    assert "close" in actions, f"平仓信号缺失: {actions}"
    # 方向列存在且含 long/short（溢价/折价双向机会）
    assert "direction" in events.columns
    assert set(events["direction"].unique()) & {"long", "short"}, "应出现双向套利方向"
    # 阈值依据列存在（B2 验收：开仓阈值逻辑可追溯）
    assert "threshold_basis" in events.columns
    assert events["threshold_basis"].notna().all()


def test_b1_b2_no_lookahead_rolling_window():
    """前视防护：数据不足滚动窗口（默认 60）时不产生信号。"""
    premium = _premium_with_opportunities(n=30)  # 短于窗口
    gen = ETFSignalGenerator()
    assert gen.generate(premium).empty


# ---------------------------------------------------------------------------
# B3：篮子执行成本与 A3 口径一致
# ---------------------------------------------------------------------------
def test_b3_basket_execution_cost_matches_a3():
    """篮子执行逐腿计费（A3 注入），敞口/费用明细可审计。"""
    basket = parse_pcf(DEMO_PCF.read_text(encoding="utf-8"))
    assert set(basket.weights().keys()) >= set(STOCKS[:3]), "PCF 应解析出成分股权重"

    executor = BasketExecutor(
        ExecutionConfig(seed=3, partial_fill_prob=0.2),
        CostModel(CostConfig()),
        seed=3,
    )
    stock_prices = {c: float(10 + i) for i, c in enumerate(STOCKS)}  # 10,11,12...
    legs = make_basket_legs(
        basket,
        etf_quantity=1000,
        etf_price=3.0,
        stock_prices=stock_prices,
        direction="long",
    )
    prices = {ETF: 3.0, **stock_prices}
    result = executor.execute(legs, prices, basket=basket, etf_symbol=ETF)

    s = result.summary()
    assert s["total_fee"] > 0, "A3 计费应产生佣金"
    assert s["total_filled_value"] <= s["total_target_value"] + 1e-6
    assert result.to_dataframe().shape[0] == 1 + len(STOCKS), "ETF 腿 + 全部成分股腿"


# ---------------------------------------------------------------------------
# B4：双模式回测集成（A2 引擎 + A4 评估）
# ---------------------------------------------------------------------------
def test_b4_dual_mode_backtest_full_pipeline():
    """理想 vs 含同步风险：产生交易、双模式指标可对比、容量/压力/机会统计齐全。"""
    quotes = _quotes(n=240, seed=42)
    premium = _premium_with_opportunities(n=240)
    result = run_dual_mode(
        quotes,
        premium,
        ETF,
        etf_quantity=100_000.0,
        cost_model=CostModel(CostConfig()),
        seed=42,
    )
    # 交易确实发生（信号触发 → 引擎成交）
    assert result.signals.shape[0] > 0, "机会窗口应产生信号"
    assert len(result.ideal.fills) > 0, "理想模式应产生成交"
    # 双模式指标可对比
    assert result.ideal_metrics and result.sync_risk_metrics
    assert result.comparison
    # 容量分析有上限结论（成交后参与率>0 → 上限有限）
    assert np.isfinite(result.capacity.get("capacity_capital", np.inf)), "有成交时容量上限应有限"
    # 压力测试场景齐全
    assert set(result.stress["scenarios"]) >= {"limit_wave", "suspension", "extreme_premium"}
    # 折溢价机会统计（机会频率 > 0）
    assert result.premium_stats["n_opportunities"] > 0
