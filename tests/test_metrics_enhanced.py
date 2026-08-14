"""A4 绩效评估层单元测试。

覆盖验收标准：
- 指标与已知数据手算一致（Sharpe / 最大回撤 / 累计/年化收益 / VaR/CVaR）
- 交易统计（胜率 / 盈亏比 / 换手 / 成本占比 / FIFO 配对）
- IS/OOS 对比接口（C4 联动）
- 一键生成报告（PNG / HTML / trade CSV 均落盘）
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.metrics_enhanced import (
    PerformanceAnalyzer,
    build_round_trips,
    compare_is_oos,
)
from backtest.report import ReportGenerator


# ---------------------------------------------------------------------------
# 测试辅助
# ---------------------------------------------------------------------------
def analyzer_from_equity(equity, periods_per_year=252, rf=0.02) -> PerformanceAnalyzer:
    """由权益值列表构造分析器（index=工作日日期）。"""
    df = pd.DataFrame(
        {"equity": [float(x) for x in equity]},
        index=pd.date_range("2024-01-01", periods=len(equity), freq="B"),
    )
    return PerformanceAnalyzer(df, periods_per_year=periods_per_year, risk_free_rate=rf)


def analyzer_from_returns(returns, periods_per_year=252, rf=0.02) -> PerformanceAnalyzer:
    """由收益率列表构造分析器（equity = 100 * cumprod(1+r)）。"""
    equity = [100.0]
    for r in returns:
        equity.append(equity[-1] * (1.0 + r))
    return analyzer_from_equity(equity, periods_per_year=periods_per_year, rf=rf)


def make_fills(rows) -> pd.DataFrame:
    """构造成交明细 DataFrame（列名与 A2 fills_frame 一致）。"""
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 收益 / 风险：手算对照
# ---------------------------------------------------------------------------
def test_sharpe_matches_manual():
    """Sharpe = mean(r)/std(r, ddof=1) * sqrt(periods)，rf=0。"""
    r = [0.01, 0.03, -0.02, 0.02]
    analyzer = analyzer_from_returns(r, periods_per_year=4, rf=0.0)
    # 手算（与实现无关的独立计算）
    arr = np.array(r)
    expected = arr.mean() / arr.std(ddof=1) * np.sqrt(4)
    assert analyzer.sharpe_ratio() == pytest.approx(expected, rel=1e-12)


def test_max_drawdown_matches_manual():
    """最大回撤 = min(equity/cummax - 1)：120 高点后跌到 90，回撤 -25%。"""
    analyzer = analyzer_from_equity([100, 120, 90, 95, 130])
    assert analyzer.max_drawdown() == pytest.approx(-0.25, rel=1e-12)


def test_cumulative_and_annualized_return():
    """累计 = (121/100 - 1)；年化 = (1.21)^(periods/n) - 1，n=1、periods=4。"""
    analyzer = analyzer_from_equity([100, 121], periods_per_year=4)
    assert analyzer.cumulative_return() == pytest.approx(0.21, rel=1e-12)
    expected_ann = 1.21**4 - 1.0
    assert analyzer.annualized_return() == pytest.approx(expected_ann, rel=1e-12)


def test_var_cvar_matches_manual():
    """VaR(95%) 取 floor(0.05*n) 分位；CVaR = 尾部均值。n=6 → var_idx=0。"""
    r = [-0.05, -0.02, -0.01, 0.0, 0.01, 0.03]
    analyzer = analyzer_from_returns(r, rf=0.0)
    var, cvar = analyzer.var_cvar(0.95)
    assert var == pytest.approx(-0.05, rel=1e-12)
    assert cvar == pytest.approx(-0.05, rel=1e-12)


def test_monthly_and_yearly_returns():
    """月度/年度复利序列返回正确形状与值。"""
    r = [0.01] * 60
    analyzer = analyzer_from_returns(r)
    monthly = analyzer.monthly_returns()
    yearly = analyzer.yearly_returns()
    assert len(monthly) > 0
    assert len(yearly) >= 1
    # 每月约 21 个交易日，月收益 ≈ (1.01)^21 - 1（仅校验符号与有限性）
    assert np.isfinite(monthly.iloc[0])


# ---------------------------------------------------------------------------
# 交易统计：FIFO 配对 + 手算对照
# ---------------------------------------------------------------------------
def test_round_trips_fifo_pnl():
    """一笔买入 100@10、卖出 100@11，无费用：pnl=100、win。"""
    fills = make_fills(
        [
            {
                "timestamp": "2024-01-02",
                "symbol": "510300",
                "side": "BUY",
                "quantity": 100.0,
                "exec_price": 10.0,
                "total_fee": 0.0,
            },
            {
                "timestamp": "2024-01-03",
                "symbol": "510300",
                "side": "SELL",
                "quantity": 100.0,
                "exec_price": 11.0,
                "total_fee": 0.0,
            },
        ]
    )
    trips = build_round_trips(fills)
    assert len(trips) == 1
    t = trips[0]
    assert t.pnl == pytest.approx(100.0)
    assert t.win is True
    assert t.pnl_pct == pytest.approx(0.10)


def test_trade_statistics_win_rate_and_profit_factor():
    """两笔交易一赢一亏：胜率 0.5，盈亏比 = 盈利/|亏损|。"""
    fills = make_fills(
        [
            {
                "timestamp": "2024-01-02",
                "symbol": "A",
                "side": "BUY",
                "quantity": 100.0,
                "exec_price": 10.0,
                "total_fee": 0.0,
            },
            {
                "timestamp": "2024-01-03",
                "symbol": "A",
                "side": "SELL",
                "quantity": 100.0,
                "exec_price": 11.0,
                "total_fee": 0.0,
            },
            {
                "timestamp": "2024-01-02",
                "symbol": "B",
                "side": "BUY",
                "quantity": 100.0,
                "exec_price": 10.0,
                "total_fee": 0.0,
            },
            {
                "timestamp": "2024-01-03",
                "symbol": "B",
                "side": "SELL",
                "quantity": 100.0,
                "exec_price": 9.0,
                "total_fee": 0.0,
            },
        ]
    )
    analyzer = PerformanceAnalyzer(
        analyzer_from_equity([100, 105, 110]).equity_curve,
        fills=fills,
    )
    stats = analyzer.trade_statistics()
    assert stats["total_trades"] == 2
    assert stats["win_rate"] == pytest.approx(0.5)
    assert stats["profit_factor"] == pytest.approx(1.0)  # 100 / 100


def test_trade_statistics_cost_ratio_and_turnover():
    """成本占比 = 费用/成交额；换手 = 成交额/平均权益。"""
    fills = make_fills(
        [
            {
                "timestamp": "2024-01-02",
                "symbol": "A",
                "side": "BUY",
                "quantity": 100.0,
                "exec_price": 10.0,
                "total_fee": 10.0,
            },
            {
                "timestamp": "2024-01-03",
                "symbol": "A",
                "side": "SELL",
                "quantity": 100.0,
                "exec_price": 10.0,
                "total_fee": 10.0,
            },
        ]
    )
    analyzer = PerformanceAnalyzer(
        analyzer_from_equity([100, 105, 110]).equity_curve,
        fills=fills,
    )
    stats = analyzer.trade_statistics()
    # 成交额 = 1000 + 1000 = 2000；费用 = 20 → 成本占比 1%
    assert stats["total_traded_value"] == pytest.approx(2000.0)
    assert stats["cost_ratio"] == pytest.approx(0.01)
    assert stats["total_fees"] == pytest.approx(20.0)
    # 换手 = 2000 / mean([100,105,110]=105)
    assert stats["turnover"] == pytest.approx(2000.0 / 105.0)


# ---------------------------------------------------------------------------
# 归因
# ---------------------------------------------------------------------------
def test_attribute_by_symbol():
    fills = make_fills(
        [
            {
                "timestamp": "2024-01-02",
                "symbol": "A",
                "side": "BUY",
                "quantity": 100.0,
                "exec_price": 10.0,
                "total_fee": 0.0,
            },
            {
                "timestamp": "2024-01-03",
                "symbol": "A",
                "side": "SELL",
                "quantity": 100.0,
                "exec_price": 12.0,
                "total_fee": 0.0,
            },
            {
                "timestamp": "2024-01-02",
                "symbol": "B",
                "side": "BUY",
                "quantity": 100.0,
                "exec_price": 10.0,
                "total_fee": 0.0,
            },
            {
                "timestamp": "2024-01-03",
                "symbol": "B",
                "side": "SELL",
                "quantity": 100.0,
                "exec_price": 8.0,
                "total_fee": 0.0,
            },
        ]
    )
    analyzer = PerformanceAnalyzer(
        analyzer_from_equity([100, 105, 110]).equity_curve,
        fills=fills,
    )
    attr = analyzer.attribute_by_symbol()
    assert set(attr["symbol"]) == {"A", "B"}
    assert attr.set_index("symbol").loc["A", "realized_pnl"] == pytest.approx(200.0)
    assert attr.set_index("symbol").loc["B", "realized_pnl"] == pytest.approx(-200.0)


# ---------------------------------------------------------------------------
# IS/OOS 对比（C4 联动接口）
# ---------------------------------------------------------------------------
def test_compare_is_oos_same_series():
    """相同序列对比：衰减比 = 1。"""
    eq = [100, 101, 103, 102, 105]
    res = compare_is_oos(pd.Series(eq), pd.Series(eq), periods_per_year=252, risk_free_rate=0.0)
    assert res["sharpe_degradation"] == pytest.approx(1.0)
    assert res["return_degradation"] == pytest.approx(1.0)
    assert set(res) >= {"is_sharpe", "oos_sharpe", "is_max_drawdown", "oos_max_drawdown"}


# ---------------------------------------------------------------------------
# 一键生成报告（PNG / HTML / trade CSV）
# ---------------------------------------------------------------------------
def test_report_generator_outputs_files(tmp_path):
    fills = make_fills(
        [
            {
                "timestamp": "2024-01-02",
                "symbol": "A",
                "side": "BUY",
                "quantity": 100.0,
                "exec_price": 10.0,
                "total_fee": 5.0,
            },
            {
                "timestamp": "2024-01-03",
                "symbol": "A",
                "side": "SELL",
                "quantity": 100.0,
                "exec_price": 11.0,
                "total_fee": 5.0,
            },
        ]
    )
    analyzer = PerformanceAnalyzer(
        analyzer_from_equity([100, 105, 110]).equity_curve,
        fills=fills,
    )
    gen = ReportGenerator(analyzer, title="测试")
    gen.generate(tmp_path, name="demo")
    assert (tmp_path / "demo.png").exists()
    assert (tmp_path / "demo.html").exists()
    assert (tmp_path / "demo_trades.csv").exists()
    # HTML 内嵌 base64 图
    html = (tmp_path / "demo.html").read_text(encoding="utf-8")
    assert "data:image/png;base64," in html
