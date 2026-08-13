"""A4 绩效评估层（增强版指标，供 B4/C4 联动）。

与旧 ``backtest/metrics.py``（向量化引擎用）解耦，本模块直接消费 A2 事件驱动
引擎的输出：``engine/accounting.py`` 的权益曲线 DataFrame 与 ``fills_frame()``
成交明细 DataFrame。

指标公式（全部在此注明定义，方便与手算对照 / 审计）：

收益
- 累计收益率 cumulative_return = equity[-1] / equity[0] - 1
- 年化收益率 annualized_return = (1 + total_return) ** (periods_per_year / n_periods) - 1
  其中 n_periods = 收益率观测数（len(returns)，首根 bar 无收益已丢弃）
- 月度收益 monthly_returns：把日收益率按自然月复利 (1+r).prod()-1
- 年度收益 yearly_returns：把日收益率按自然年复利

风险
- 年化波动率 annualized_volatility = std(returns, ddof=1) * sqrt(periods_per_year)
- 最大回撤 max_drawdown = min(equity / equity.cummax() - 1)
- Sharpe = mean(excess) / std(excess, ddof=1) * sqrt(periods_per_year)，
  excess = returns - rf / periods_per_year（rf 为年化无风险利率）
- Sortino = mean(excess) / std(excess[excess < 0], ddof=1) * sqrt(periods_per_year)
- Calmar = annualized_return / |max_drawdown|
- VaR(α) = 收益升序排列后的第 floor((1-α)*n) 个（经验分位数，取更保守一侧）
- CVaR(α) = 收益 <= VaR 的尾部收益的均值

交易（由成交明细 FIFO 配对得到逐笔平仓交易）
- 胜率 win_rate = 盈利笔数 / 平仓总笔数
- 盈亏比 profit_factor = 盈利总额 / |亏损总额|
- 平均盈亏比 avg_win_loss_ratio = 平均盈利 / |平均亏损|
- 换手率 turnover = 期间总成交额 / 期间平均权益（双边口径）
- 平均持仓周期 avg_holding_bars = 平仓交易的平均持仓 bar 数
- 成本占比 cost_ratio = 累计费用 / 累计成交额

与 C4 联动：``compare_is_oos`` 接收 IS/OOS 两组权益序列做对比（过拟合检测的输入）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

BUY = "BUY"
SELL = "SELL"

DEFAULT_PERIODS_PER_YEAR = 252
DEFAULT_RISK_FREE_RATE = 0.02


# ---------------------------------------------------------------------------
# 工具：从 A2 权益曲线 / 成交明细抽取标准输入
# ---------------------------------------------------------------------------
def _equity_series(equity_curve: pd.DataFrame) -> pd.Series:
    """从权益曲线 DataFrame 抽取 equity 列（缺失则报错）。"""
    if "equity" not in equity_curve.columns:
        raise ValueError("equity_curve 必须包含 equity 列")
    s = pd.to_numeric(equity_curve["equity"], errors="coerce")
    return s


def _returns_from_equity(equity: pd.Series) -> pd.Series:
    """由权益序列计算收益率：r_t = equity_t / equity_{t-1} - 1，首元素 NaN 丢弃。"""
    r = equity.pct_change(fill_method=None)
    return r.dropna()


# ---------------------------------------------------------------------------
# FIFO 平仓配对：把 A2 成交明细还原为逐笔交易（已实现盈亏）
# ---------------------------------------------------------------------------
@dataclass
class RoundTrip:
    """一笔平仓交易（FIFO 配对后）。"""

    symbol: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    quantity: float
    entry_price: float
    exit_price: float
    pnl: float          # 已扣双边费用的净盈亏
    pnl_pct: float      # pnl / 建仓成本
    win: bool
    holding_bars: int


def build_round_trips(
    fills: pd.DataFrame,
    equity_index: Optional[pd.DatetimeIndex] = None,
) -> List[RoundTrip]:
    """按 symbol 分组、FIFO 配对平仓，返回逐笔交易列表。

    费用分摊：买入单位成本 = exec_price + 买入费/买入量；卖出单位收入 = exec_price - 卖出费/卖出量。
    未平仓（只有买入没有卖出）的部分不计入已实现交易。
    """
    if fills is None or len(fills) == 0:
        return []
    df = fills.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    # 缺列兜底，兼容只给必要列的最简输入
    for c in ("side", "symbol", "quantity", "exec_price", "total_fee"):
        if c not in df.columns:
            raise ValueError(f"成交明细缺少列: {c}")

    pos_map: Dict[pd.Timestamp, int] = {}
    if equity_index is not None:
        pos_map = {ts: i for i, ts in enumerate(pd.DatetimeIndex(equity_index))}

    trips: List[RoundTrip] = []
    for symbol, grp in df.groupby("symbol"):
        grp = grp.sort_values("timestamp")
        open_lots: List[dict] = []  # FIFO 队列：{qty, cost_per_share, entry_time}
        for _, row in grp.iterrows():
            side = str(row["side"]).upper()
            qty = float(row["quantity"])
            exec_price = float(row["exec_price"])
            fee = float(row["total_fee"])
            ts = pd.Timestamp(row["timestamp"])
            if qty <= 0:
                continue
            if side == BUY:
                cost_per_share = exec_price + fee / qty
                open_lots.append({"qty": qty, "cost": cost_per_share, "entry": ts, "entry_price": exec_price})
            elif side == SELL:
                proceeds_per_share = exec_price - fee / qty
                remaining = qty
                while remaining > 1e-12 and open_lots:
                    lot = open_lots[0]
                    matched = min(lot["qty"], remaining)
                    pnl = (proceeds_per_share - lot["cost"]) * matched
                    entry_cost = lot["entry_price"] * matched
                    pnl_pct = pnl / entry_cost if entry_cost > 0 else 0.0
                    hb = 1
                    if pos_map:
                        ei = pos_map.get(lot["entry"])
                        xi = pos_map.get(ts)
                        if ei is not None and xi is not None:
                            hb = max(1, xi - ei)
                    trips.append(RoundTrip(
                        symbol=symbol,
                        entry_time=lot["entry"],
                        exit_time=ts,
                        quantity=matched,
                        entry_price=lot["entry_price"],
                        exit_price=exec_price,
                        pnl=float(pnl),
                        pnl_pct=float(pnl_pct),
                        win=pnl > 0,
                        holding_bars=hb,
                    ))
                    lot["qty"] -= matched
                    remaining -= matched
                    if lot["qty"] <= 1e-12:
                        open_lots.pop(0)
    return trips


# ---------------------------------------------------------------------------
# 绩效分析器
# ---------------------------------------------------------------------------
class PerformanceAnalyzer:
    """绩效与风险指标计算器（消费 A2 引擎输出的权益曲线 + 成交明细）。"""

    def __init__(
        self,
        equity_curve: pd.DataFrame,
        fills: Optional[pd.DataFrame] = None,
        periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
        risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    ):
        self.equity_curve = equity_curve
        self.equity = _equity_series(equity_curve)
        self.returns = _returns_from_equity(self.equity)
        self.fills = fills if fills is not None else pd.DataFrame()
        self.periods_per_year = int(periods_per_year)
        self.risk_free_rate = float(risk_free_rate)
        self._trips: Optional[List[RoundTrip]] = None

    # ------------------------------ 收益 ------------------------------
    def cumulative_return(self) -> float:
        """累计收益率 = equity[-1]/equity[0] - 1。"""
        if len(self.equity) < 2:
            return 0.0
        first = float(self.equity.iloc[0])
        if first <= 0:
            return 0.0
        return float(self.equity.iloc[-1] / first - 1.0)

    def annualized_return(self) -> float:
        """年化收益率 = (1 + 累计收益) ** (periods_per_year / n_periods) - 1。"""
        total = self.cumulative_return()
        n = len(self.returns)
        if n <= 0 or total <= -1.0:
            return 0.0 if n <= 0 else -1.0
        return float((1.0 + total) ** (self.periods_per_year / n) - 1.0)

    def monthly_returns(self) -> pd.Series:
        """月度收益：日收益率按自然月复利。"""
        if self.returns.empty:
            return pd.Series(dtype=float)
        return self.returns.resample("ME").apply(lambda x: (1.0 + x).prod() - 1.0)

    def yearly_returns(self) -> pd.Series:
        """年度收益：日收益率按自然年复利。"""
        if self.returns.empty:
            return pd.Series(dtype=float)
        return self.returns.resample("YE").apply(lambda x: (1.0 + x).prod() - 1.0)

    # ------------------------------ 风险 ------------------------------
    def annualized_volatility(self) -> float:
        """年化波动率 = std(returns, ddof=1) * sqrt(periods_per_year)。"""
        if len(self.returns) < 2:
            return 0.0
        return float(self.returns.std(ddof=1) * np.sqrt(self.periods_per_year))

    def max_drawdown(self) -> float:
        """最大回撤 = min(equity / cummax(equity) - 1)。"""
        if len(self.equity) < 2:
            return 0.0
        dd = self.equity / self.equity.cummax() - 1.0
        return float(dd.min())

    def drawdown_series(self) -> pd.Series:
        """回撤序列（每个时点相对历史最高权益的回撤）。"""
        if len(self.equity) < 2:
            return pd.Series(dtype=float)
        return self.equity / self.equity.cummax() - 1.0

    def max_drawdown_duration(self) -> int:
        """最大回撤持续时间（bar 数，最长一段处于回撤状态的连续长度）。"""
        dd = self.drawdown_series()
        if dd.empty:
            return 0
        in_dd = dd < 0
        if not in_dd.any():
            return 0
        groups = (in_dd != in_dd.shift()).cumsum()
        durations = groups[in_dd].value_counts()
        return int(durations.max()) if not durations.empty else 0

    def sharpe_ratio(self) -> float:
        """Sharpe = mean(excess)/std(excess, ddof=1) * sqrt(periods)。"""
        excess = self.returns - self.risk_free_rate / self.periods_per_year
        std = excess.std(ddof=1)
        if len(excess) < 2 or std == 0 or np.isnan(std):
            return 0.0
        return float(excess.mean() / std * np.sqrt(self.periods_per_year))

    def sortino_ratio(self) -> float:
        """Sortino = mean(excess)/std(excess[excess<0], ddof=1) * sqrt(periods)。"""
        excess = self.returns - self.risk_free_rate / self.periods_per_year
        downside = excess[excess < 0]
        if len(excess) < 2 or len(downside) < 1:
            return 0.0
        dstd = downside.std(ddof=1)
        if dstd == 0 or np.isnan(dstd):
            return 0.0
        return float(excess.mean() / dstd * np.sqrt(self.periods_per_year))

    def calmar_ratio(self) -> float:
        """Calmar = 年化收益率 / |最大回撤|。"""
        mdd = self.max_drawdown()
        if mdd == 0:
            return 0.0
        return self.annualized_return() / abs(mdd)

    def var_cvar(self, confidence: float = 0.95) -> tuple:
        """VaR/CVaR：经验分位数口径。

        var_idx = floor((1-confidence)*n)，clamp 到 [0, n-1]；CVaR = 收益 <= VaR 的均值。
        """
        r = self.returns.sort_values()
        n = len(r)
        if n == 0:
            return 0.0, 0.0
        var_idx = int((1.0 - confidence) * n)
        var_idx = max(0, min(var_idx, n - 1))
        var = float(r.iloc[var_idx])
        tail = r.iloc[: var_idx + 1]
        cvar = float(tail.mean()) if len(tail) > 0 else var
        return var, cvar

    # ------------------------------ 交易 ------------------------------
    def round_trips(self) -> List[RoundTrip]:
        """返回 FIFO 配对后的平仓交易列表（懒加载缓存）。"""
        if self._trips is None:
            self._trips = build_round_trips(self.fills, self.equity.index)
        return self._trips

    def trade_statistics(self) -> dict:
        """交易统计：胜率、盈亏比、平均盈亏比、换手、持仓周期、成本占比。"""
        trips = self.round_trips()
        total = len(trips)
        wins = [t for t in trips if t.win]
        losses = [t for t in trips if not t.win]
        n_wins, n_losses = len(wins), len(losses)
        gross_profit = sum(t.pnl for t in wins)
        gross_loss = abs(sum(t.pnl for t in losses))

        win_rate = n_wins / total if total else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)
        avg_win = gross_profit / n_wins if n_wins else 0.0
        avg_loss = -gross_loss / n_losses if n_losses else 0.0
        avg_win_loss_ratio = avg_win / abs(avg_loss) if avg_loss != 0 else (float("inf") if avg_win > 0 else 0.0)
        avg_holding = float(np.mean([t.holding_bars for t in trips])) if total else 0.0

        # 换手率 / 成本占比（基于成交额口径）
        buy_value = sell_value = 0.0
        total_fees = 0.0
        total_traded_value = 0.0
        if len(self.fills):
            f = self.fills.copy()
            f["exec_price"] = pd.to_numeric(f["exec_price"], errors="coerce")
            f["quantity"] = pd.to_numeric(f["quantity"], errors="coerce")
            f["total_fee"] = pd.to_numeric(f["total_fee"], errors="coerce")
            value = f["exec_price"] * f["quantity"]
            side = f["side"].astype(str).str.upper()
            buy_value = float(value[side == BUY].sum())
            sell_value = float(value[side == SELL].sum())
            total_traded_value = buy_value + sell_value
            total_fees = float(f["total_fee"].sum())
        avg_equity = float(self.equity.mean()) if len(self.equity) else 0.0
        turnover = total_traded_value / avg_equity if avg_equity > 0 else 0.0
        cost_ratio = total_fees / total_traded_value if total_traded_value > 0 else 0.0

        return {
            "total_trades": total,
            "winning_trades": n_wins,
            "losing_trades": n_losses,
            "win_rate": win_rate,
            "profit_factor": _finite(profit_factor, 999.0),
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "avg_win_loss_ratio": _finite(avg_win_loss_ratio, 999.0),
            "turnover": turnover,
            "avg_holding_bars": avg_holding,
            "cost_ratio": cost_ratio,
            "total_fees": total_fees,
            "total_traded_value": total_traded_value,
        }

    # ------------------------------ 归因 ------------------------------
    def attribute_by_symbol(self) -> pd.DataFrame:
        """按标的归因：各 symbol 的平仓笔数、已实现盈亏、费用贡献。"""
        trips = self.round_trips()
        rows = []
        for symbol in sorted({t.symbol for t in trips}):
            st = [t for t in trips if t.symbol == symbol]
            rows.append({
                "symbol": symbol,
                "trades": len(st),
                "realized_pnl": float(sum(t.pnl for t in st)),
                "win_rate": sum(1 for t in st if t.win) / len(st),
            })
        return pd.DataFrame(rows, columns=["symbol", "trades", "realized_pnl", "win_rate"])

    # ------------------------------ 汇总 ------------------------------
    def summary(self) -> dict:
        """一次性输出全部指标（供报告与 C4 联动）。"""
        var, cvar = self.var_cvar()
        trade = self.trade_statistics()
        start = self.equity.index.min() if len(self.equity) else pd.NaT
        end = self.equity.index.max() if len(self.equity) else pd.NaT
        return {
            "start": start,
            "end": end,
            "n_periods": int(len(self.returns)),
            "total_return": self.cumulative_return(),
            "annualized_return": self.annualized_return(),
            "annualized_volatility": self.annualized_volatility(),
            "sharpe_ratio": self.sharpe_ratio(),
            "sortino_ratio": self.sortino_ratio(),
            "calmar_ratio": self.calmar_ratio(),
            "max_drawdown": self.max_drawdown(),
            "max_drawdown_duration": self.max_drawdown_duration(),
            "var_95": var,
            "cvar_95": cvar,
            **trade,
        }

    def to_dataframe(self) -> pd.DataFrame:
        """把汇总指标转成两列 DataFrame（指标名/数值），供报告渲染。"""
        return pd.DataFrame(list(self.summary().items()), columns=["指标", "数值"])


def _finite(x: float, cap: float) -> float:
    """把 inf/nan 归一为有限值（供盈亏比等无亏损时的兜底）。"""
    if x is None or np.isnan(x):
        return 0.0
    if np.isinf(x):
        return cap
    return float(x)


# ---------------------------------------------------------------------------
# 与 C4 联动：IS/OOS 两组权益序列对比（过拟合检测输入）
# ---------------------------------------------------------------------------
def compare_is_oos(
    is_equity: Union[pd.Series, pd.DataFrame],
    oos_equity: Union[pd.Series, pd.DataFrame],
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
) -> dict:
    """对比样本内（IS）与样本外（OOS）两组权益序列。

    接受 Series（权益值）或含 equity 列的 DataFrame。返回 IS/OOS 各自的
    年化收益、Sharpe、最大回撤，以及 OOS 相对 IS 的衰减比（供 C4 判断过拟合）：
    - sharpe_degradation = OOS Sharpe / IS Sharpe（<1 说明样本外衰减）
    - return_degradation = OOS 年化收益 / IS 年化收益
    """
    def _curve(x) -> pd.DataFrame:
        if isinstance(x, pd.DataFrame):
            return x
        return pd.DataFrame({"equity": pd.Series(x)})

    is_a = PerformanceAnalyzer(_curve(is_equity), periods_per_year=periods_per_year, risk_free_rate=risk_free_rate)
    oos_a = PerformanceAnalyzer(_curve(oos_equity), periods_per_year=periods_per_year, risk_free_rate=risk_free_rate)

    is_sharpe = is_a.sharpe_ratio()
    oos_sharpe = oos_a.sharpe_ratio()
    is_ret = is_a.annualized_return()
    oos_ret = oos_a.annualized_return()
    return {
        "is_annualized_return": is_ret,
        "oos_annualized_return": oos_ret,
        "is_sharpe": is_sharpe,
        "oos_sharpe": oos_sharpe,
        "is_max_drawdown": is_a.max_drawdown(),
        "oos_max_drawdown": oos_a.max_drawdown(),
        "sharpe_degradation": oos_sharpe / is_sharpe if is_sharpe != 0 else 0.0,
        "return_degradation": oos_ret / is_ret if is_ret != 0 else 0.0,
    }


__all__ = [
    "PerformanceAnalyzer",
    "RoundTrip",
    "build_round_trips",
    "compare_is_oos",
    "DEFAULT_PERIODS_PER_YEAR",
    "DEFAULT_RISK_FREE_RATE",
]
