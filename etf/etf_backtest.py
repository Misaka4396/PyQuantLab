"""B4 ETF 套利回测集成与评估（B 专项收口）。

把 B1（数据）、B2（信号）、B3（执行）接入 A2 引擎，跑完整回测并评估：

1. 套利策略挂载 A2（``ETFArbitrageStrategy``，消费 B2 信号流，生成 A2 订单）。
2. 双模式对比：**理想执行**（满量成交） vs **含同步风险**（B3 时延 + 部分成交），
   凸显执行风险对收益的影响。
3. 多年分钟级回测，输出扣成本收益（引擎注入 A3 成本模型，固定种子可复现）。
4. 容量分析：按成交额占比估算容量上限（参与率封顶）。
5. 压力测试：涨跌停潮 / 停牌 / 极端折溢价下最大敞口。
6. 与 A4 联动出报告（``backtest.report.ReportGenerator`` + 折溢价机会统计）。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from backtest.metrics_enhanced import PerformanceAnalyzer
from backtest.report import ReportGenerator
from cost_model import CostModel
from engine import EngineConfig, EventEngine
from etf.basket_execution import BasketExecutor
from etf.etf_signal import ETFSignalGenerator, entry_threshold
from etf.etf_strategy import (
    ETFArbitrageStrategy,
    IDEAL,
    SYNC_RISK,
    _leg_notional,
)
from etf.execution_config import ExecutionConfig
from etf.threshold_config import ThresholdConfig


@dataclass
class ETFBacktestResult:
    """B4 完整回测与评估结果。"""

    ideal: EventEngine
    sync_risk: EventEngine
    ideal_metrics: dict
    sync_risk_metrics: dict
    comparison: dict
    premium_stats: dict
    capacity: dict
    stress: dict
    signals: pd.DataFrame = field(default_factory=pd.DataFrame)


# ---------------------------------------------------------------------------
# 主回测
# ---------------------------------------------------------------------------
def run_arbitrage_backtest(
    quotes: pd.DataFrame,
    premium: pd.Series,
    etf_symbol: str,
    etf_quantity: float = 100_000.0,
    execution_mode: str = IDEAL,
    *,
    basket: Optional[Dict[str, float]] = None,
    stock_quotes: Optional[Dict[str, pd.DataFrame]] = None,
    cost_model: Optional[CostModel] = None,
    initial_cash: float = 1_000_000.0,
    threshold_config: Optional[ThresholdConfig] = None,
    unit_cost_rate: Optional[float] = None,
    executor: Optional[BasketExecutor] = None,
    seed: int = 42,
) -> EventEngine:
    """把 ETF 套利策略挂载到 A2 引擎并跑一次回测，返回 EngineResult。

    - quotes：ETF 分钟 OHLCV（index=datetime，含 volume/amount）。
    - premium：折溢价序列（index 与 quotes 对齐；内部会 reindex 到 quotes 时间轴）。
    - basket：{成分股代码: 数量} 静态篮子（缺省只做 ETF 腿）。
    - stock_quotes：篮子成分股行情（引擎核算需要，缺省 None 表示不做成分股腿）。
    - executor：真实执行模式的 B3 执行器（缺省按 seed 构造默认执行器）。
    """
    quotes = quotes.copy()
    premium = pd.to_numeric(premium, errors="coerce").reindex(quotes.index)

    data: Dict[str, pd.DataFrame] = {etf_symbol: quotes}
    if stock_quotes:
        data.update(stock_quotes)

    strategy = ETFArbitrageStrategy(
        premium,
        etf_symbol,
        etf_quantity,
        basket=basket,
        execution_mode=execution_mode,
        executor=executor,
        threshold_config=threshold_config,
        unit_cost_rate=unit_cost_rate,
        seed=seed,
    )

    engine = EventEngine(
        config=EngineConfig(
            initial_cash=initial_cash,
            seed=seed,
            fill_mode="next_bar",
            etf_symbols=(etf_symbol,),
        ),
        data=data,
        cost_model=cost_model or CostModel(),
        strategy=strategy,
    )
    result = engine.run()
    result.strategy = strategy  # 附上策略（执行/敞口记录供容量与压力测试）
    return result


def run_dual_mode(
    quotes: pd.DataFrame,
    premium: pd.Series,
    etf_symbol: str,
    etf_quantity: float = 100_000.0,
    *,
    basket: Optional[Dict[str, float]] = None,
    stock_quotes: Optional[Dict[str, pd.DataFrame]] = None,
    cost_model: Optional[CostModel] = None,
    initial_cash: float = 1_000_000.0,
    threshold_config: Optional[ThresholdConfig] = None,
    unit_cost_rate: Optional[float] = None,
    executor: Optional[BasketExecutor] = None,
    seed: int = 42,
    participation_cap: float = 0.05,
) -> ETFBacktestResult:
    """双模式回测：理想执行 vs 含同步风险，输出评估结果（含容量/压力/折溢价统计）。"""
    # 折溢价机会阈值与信号一致（entry_threshold = 单位成本 + 缓冲）
    gen = ETFSignalGenerator(config=threshold_config, unit_cost_rate=unit_cost_rate)
    threshold = gen.threshold

    ideal = run_arbitrage_backtest(
        quotes, premium, etf_symbol, etf_quantity, IDEAL,
        basket=basket, stock_quotes=stock_quotes, cost_model=cost_model,
        initial_cash=initial_cash, threshold_config=threshold_config,
        unit_cost_rate=unit_cost_rate, seed=seed,
    )
    sync = run_arbitrage_backtest(
        quotes, premium, etf_symbol, etf_quantity, SYNC_RISK,
        basket=basket, stock_quotes=stock_quotes, cost_model=cost_model,
        initial_cash=initial_cash, threshold_config=threshold_config,
        unit_cost_rate=unit_cost_rate, executor=executor, seed=seed,
    )

    ideal_metrics = PerformanceAnalyzer(ideal.equity_curve, ideal.fills).summary()
    sync_metrics = PerformanceAnalyzer(sync.equity_curve, sync.fills).summary()

    exec_frame = sync.strategy.exposure_frame()
    avg_exposure = float(exec_frame["exposure_ratio"].mean()) if len(exec_frame) else 0.0
    avg_tracking = float(exec_frame["tracking_error"].mean()) if len(exec_frame) else 0.0

    comparison = {
        "ideal_total_return": float(ideal_metrics["total_return"]),
        "sync_risk_total_return": float(sync_metrics["total_return"]),
        "ideal_sharpe": float(ideal_metrics["sharpe_ratio"]),
        "sync_risk_sharpe": float(sync_metrics["sharpe_ratio"]),
        "execution_drag_return": float(ideal_metrics["total_return"] - sync_metrics["total_return"]),
        "ideal_total_fees": float(ideal_metrics["total_fees"]),
        "sync_risk_total_fees": float(sync_metrics["total_fees"]),
        "sync_risk_avg_exposure_ratio": avg_exposure,
        "sync_risk_avg_tracking_error": avg_tracking,
    }

    premium_stats = premium_opportunity_stats(premium, threshold)
    capacity = capacity_analysis(
        sync.fills, quotes, etf_symbol, participation_cap, initial_cash
    )
    stress = _stress_from_result(sync, etf_symbol, etf_quantity, basket, seed)

    return ETFBacktestResult(
        ideal=ideal,
        sync_risk=sync,
        ideal_metrics=ideal_metrics,
        sync_risk_metrics=sync_metrics,
        comparison=comparison,
        premium_stats=premium_stats,
        capacity=capacity,
        stress=stress,
        signals=sync.strategy.signals,
    )


# ---------------------------------------------------------------------------
# 容量分析（按成交额占比）
# ---------------------------------------------------------------------------
def capacity_analysis(
    fills: pd.DataFrame,
    quotes: pd.DataFrame,
    etf_symbol: str,
    participation_cap: float = 0.05,
    initial_cash: float = 1_000_000.0,
) -> dict:
    """按成交额占比估算容量上限。

    参与率 = 单笔成交名义 / 该 bar 市场成交额（amount）。容量上限 = 当前资金 ×
    participation_cap / 最大参与率（即把资金放大到参与率触及上限为止）。
    """
    if fills is None or len(fills) == 0:
        return {
            "n_fills": 0,
            "max_participation": 0.0,
            "capacity_capital": float("inf"),
            "avg_daily_participation": 0.0,
            "participation_cap": participation_cap,
        }

    q = quotes.copy()
    if "amount" not in q.columns:
        q["amount"] = pd.to_numeric(q.get("volume", 0.0), errors="coerce") * pd.to_numeric(q["close"], errors="coerce")
    amount_map = pd.to_numeric(q["amount"], errors="coerce").to_dict()

    f = fills.copy()
    f["timestamp"] = pd.to_datetime(f["timestamp"])
    etf_fills = f[f["symbol"].astype(str) == str(etf_symbol)]
    if len(etf_fills) == 0:
        return {
            "n_fills": 0,
            "max_participation": 0.0,
            "capacity_capital": float("inf"),
            "avg_daily_participation": 0.0,
            "participation_cap": participation_cap,
        }

    parts = []
    daily: Dict[pd.Timestamp, List[tuple]] = {}
    for _, row in etf_fills.iterrows():
        ts = pd.Timestamp(row["timestamp"])
        notional = float(row["exec_price"]) * float(row["quantity"])
        amount = amount_map.get(ts, np.nan)
        if np.isfinite(amount) and amount > 0:
            p = notional / float(amount)
            parts.append(p)
            day = ts.normalize()
            daily.setdefault(day, []).append((notional, float(amount)))

    max_participation = float(max(parts)) if parts else 0.0
    capacity_capital = (
        float(initial_cash) * float(participation_cap) / max_participation
        if max_participation > 0 else float("inf")
    )

    daily_parts = [
        sum(n for n, _ in v) / sum(a for _, a in v)
        for v in daily.values() if sum(a for _, a in v) > 0
    ]
    avg_daily = float(np.mean(daily_parts)) if daily_parts else 0.0

    return {
        "n_fills": int(len(etf_fills)),
        "max_participation": max_participation,
        "capacity_capital": capacity_capital,
        "avg_daily_participation": avg_daily,
        "participation_cap": participation_cap,
    }


# ---------------------------------------------------------------------------
# 压力测试（涨跌停潮 / 停牌 / 极端折溢价）
# ---------------------------------------------------------------------------
def _scenario(note: str, result) -> dict:
    return {
        "note": note,
        "exposure_ratio": float(result.exposure_ratio),
        "unfilled_value": float(result.total_unfilled_value),
        "target_value": float(result.total_target_value),
    }


def stress_test(
    legs: List[dict],
    prices: Dict[str, float],
    volumes: Optional[Dict[str, float]] = None,
    seed: int = 42,
) -> dict:
    """三场景压力测试，输出各场景敞口与最大敞口。

    - 涨跌停潮：约 60% 腿无法成交（部分成交比例固定 0.4）。
    - 停牌：ETF/成分股停牌，全部无法成交，敞口 = 目标名义。
    - 极端折溢价：目标名义放大 5 倍（极端溢价下的仓位规模），流动性骤降（0.7 成交）。
    """
    volumes = volumes or {}
    target = _leg_notional(legs, prices)
    scenarios: Dict[str, dict] = {}

    cfg_limit = ExecutionConfig(
        partial_fill_prob=1.0,
        partial_fill_ratio_min=0.4,
        partial_fill_ratio_max=0.4,
        seed=seed,
    )
    r_limit = BasketExecutor(cfg_limit).execute(legs, prices, volumes)
    scenarios["limit_wave"] = _scenario("涨跌停潮：60% 腿无法成交", r_limit)

    scenarios["suspension"] = {
        "note": "停牌：ETF/成分股停牌无法成交",
        "exposure_ratio": 1.0,
        "unfilled_value": target,
        "target_value": target,
    }

    legs_extreme = [
        {"symbol": l["symbol"], "side": l["side"], "quantity": float(l["quantity"]) * 5.0}
        for l in legs
    ]
    cfg_extreme = ExecutionConfig(
        partial_fill_prob=1.0,
        partial_fill_ratio_min=0.7,
        partial_fill_ratio_max=0.7,
        seed=seed,
    )
    r_extreme = BasketExecutor(cfg_extreme).execute(legs_extreme, prices, volumes)
    scenarios["extreme_premium"] = _scenario("极端折溢价：目标名义放大 5 倍", r_extreme)

    max_scenario = max(scenarios, key=lambda k: scenarios[k]["unfilled_value"])
    return {
        "scenarios": scenarios,
        "max_exposure": scenarios[max_scenario]["unfilled_value"],
        "max_exposure_ratio": scenarios[max_scenario]["exposure_ratio"],
        "max_exposure_scenario": max_scenario,
    }


def _stress_from_result(
    result: EventEngine,
    etf_symbol: str,
    etf_quantity: float,
    basket: Optional[Dict[str, float]],
    seed: int,
) -> dict:
    """由回测结果抽取代表性腿与价格，构造压力测试。"""
    strat = result.strategy
    prices = dict(strat._last_prices) if strat._last_prices else {etf_symbol: 1.0}
    volumes = dict(strat._last_volumes) if strat._last_volumes else {}

    legs = [{"symbol": etf_symbol, "side": "SELL", "quantity": float(etf_quantity)}]
    if basket:
        for sym, qty in basket.items():
            legs.append({"symbol": str(sym), "side": "BUY", "quantity": float(qty)})
    return stress_test(legs, prices, volumes, seed=seed)


# ---------------------------------------------------------------------------
# 折溢价机会统计（频率/幅度）
# ---------------------------------------------------------------------------
def premium_opportunity_stats(premium: pd.Series, threshold: float) -> dict:
    """统计折溢价套利机会的频率与幅度（|premium| >= threshold 视为机会）。"""
    x = pd.to_numeric(premium, errors="coerce").dropna()
    if len(x) == 0:
        return {
            "n_obs": 0, "n_opportunities": 0, "opportunity_frequency": 0.0,
            "mean_premium": 0.0, "max_premium": 0.0, "min_premium": 0.0,
            "mean_abs_premium": 0.0, "mean_opportunity_amplitude": 0.0,
            "max_opportunity_amplitude": 0.0, "threshold": threshold,
        }
    opps = x[abs(x) >= threshold]
    return {
        "n_obs": int(len(x)),
        "n_opportunities": int(len(opps)),
        "opportunity_frequency": float(len(opps) / len(x)),
        "mean_premium": float(x.mean()),
        "max_premium": float(x.max()),
        "min_premium": float(x.min()),
        "mean_abs_premium": float(x.abs().mean()),
        "mean_opportunity_amplitude": float(opps.abs().mean()) if len(opps) else 0.0,
        "max_opportunity_amplitude": float(opps.abs().max()) if len(opps) else 0.0,
        "threshold": float(threshold),
    }


# ---------------------------------------------------------------------------
# A4 报告联动
# ---------------------------------------------------------------------------
def generate_report(
    result: ETFBacktestResult,
    output_dir: Union[str, Path],
    title: str = "ETF 套利回测报告",
) -> Dict[str, str]:
    """与 A4 联动：双模式绩效报告 + 对比图 + 汇总 JSON（含折溢价机会/容量/压力）。"""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 双模式 A4 报告（权益/回撤/月度热力图/成本占比）
    ideal_rep = ReportGenerator(
        PerformanceAnalyzer(result.ideal.equity_curve, result.ideal.fills),
        title=f"{title}（理想执行）",
    ).generate(out, name="ideal")
    sync_rep = ReportGenerator(
        PerformanceAnalyzer(result.sync_risk.equity_curve, result.sync_risk.fills),
        title=f"{title}（含同步风险）",
    ).generate(out, name="sync_risk")

    # 对比图（两条权益曲线）
    compare_png = out / "comparison.png"
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(result.ideal.equity_curve["equity"].values, label="理想执行", lw=1.2)
    ax.plot(result.sync_risk.equity_curve["equity"].values, label="含同步风险", lw=1.2)
    ax.set_title("理想 vs 含同步风险 权益曲线对比")
    ax.set_xlabel("bar")
    ax.set_ylabel("权益")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(compare_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 汇总 JSON（含折溢价机会频率/幅度、容量、压力、双模式对比）
    summary = {
        "comparison": result.comparison,
        "premium_stats": result.premium_stats,
        "capacity": result.capacity,
        "stress": result.stress,
        "n_signals": int(len(result.signals)),
    }
    summary_path = out / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    return {
        **ideal_rep,
        **{f"sync_{k}": v for k, v in sync_rep.items()},
        "comparison_png": str(compare_png),
        "summary_json": str(summary_path),
    }


# ---------------------------------------------------------------------------
# 演示入口
# ---------------------------------------------------------------------------
def main() -> None:
    """合成数据跑一遍完整 B4 流程（演示/自检）。"""
    from etf.etf_data import synthetic_etf_minute

    quotes = synthetic_etf_minute(
        "510300", "2024-01-02", "2024-02-28", seed=7, freq="1min"
    )
    # 构造均值回归的折溢价（溢价→回归），与 ETF 价联动
    close = quotes["close"]
    t = np.arange(len(close))
    premium = pd.Series(
        0.004 * np.sin(2 * np.pi * t / 120.0) + np.random.default_rng(7).normal(0, 0.0002, len(t)),
        index=close.index,
    )
    result = run_dual_mode(quotes, premium, "510300", etf_quantity=100_000.0, seed=42)

    print("=== 双模式对比 ===")
    for k, v in result.comparison.items():
        print(f"  {k}: {v}")
    print("=== 折溢价机会 ===")
    for k, v in result.premium_stats.items():
        print(f"  {k}: {v}")
    print("=== 容量 ===")
    for k, v in result.capacity.items():
        print(f"  {k}: {v}")
    print("=== 压力测试最大敞口 ===")
    print(f"  最大敞口: {result.stress['max_exposure']:.2f} 元 "
          f"({result.stress['max_exposure_scenario']})")
    paths = generate_report(result, "./data_cache/etf_report")
    print("报告已生成：", list(paths.values()))


__all__ = [
    "ETFBacktestResult",
    "run_arbitrage_backtest",
    "run_dual_mode",
    "capacity_analysis",
    "stress_test",
    "premium_opportunity_stats",
    "generate_report",
    "entry_threshold",
]
