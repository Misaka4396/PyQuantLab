"""Performance metrics calculator — all static methods, independently testable."""

import numpy as np
import pandas as pd

from config import RISK_FREE_RATE
from core.types import PerformanceMetrics


class MetricsCalculator:
    TRADING_DAYS = 252

    @staticmethod
    def compute_all(
        equity_curve: pd.DataFrame,
        trades: pd.DataFrame,
        benchmark_returns: pd.Series | None = None,
        strategy_returns: pd.Series | None = None,
    ) -> PerformanceMetrics:
        if strategy_returns is None:
            strategy_returns = equity_curve["returns"]
        equity = equity_curve["equity"]
        rf = RISK_FREE_RATE

        total_ret = MetricsCalculator.total_return(equity)
        ann_ret = MetricsCalculator.annualized_return(equity)
        ann_vol = MetricsCalculator.annualized_volatility(strategy_returns)
        sharpe = MetricsCalculator.sharpe_ratio(strategy_returns, rf)
        sortino = MetricsCalculator.sortino_ratio(strategy_returns, rf)
        max_dd = MetricsCalculator.max_drawdown(equity)
        max_dd_dur = MetricsCalculator.max_drawdown_duration(equity)
        calmar = MetricsCalculator.calmar_ratio(ann_ret, max_dd)

        alpha, beta = 0.0, 0.0
        if benchmark_returns is not None and len(benchmark_returns) == len(strategy_returns):
            alpha, beta = MetricsCalculator.alpha_beta(strategy_returns, benchmark_returns, rf)

        var_95, cvar_95 = MetricsCalculator.var_cvar(strategy_returns)

        trade_stats = MetricsCalculator.trade_statistics(trades)

        return PerformanceMetrics(
            total_return=round(total_ret, 4),
            annualized_return=round(ann_ret, 4),
            annualized_volatility=round(ann_vol, 4),
            sharpe_ratio=round(sharpe, 4),
            sortino_ratio=round(sortino, 4),
            max_drawdown=round(max_dd, 4),
            max_drawdown_duration=int(max_dd_dur),
            calmar_ratio=round(calmar, 4),
            alpha=round(alpha, 4),
            beta=round(beta, 4),
            var_95=round(var_95, 4),
            cvar_95=round(cvar_95, 4),
            total_trades=trade_stats["total_trades"],
            winning_trades=trade_stats["winning_trades"],
            losing_trades=trade_stats["losing_trades"],
            win_rate=round(trade_stats["win_rate"], 4),
            profit_factor=round(trade_stats["profit_factor"], 4),
            avg_win=round(trade_stats["avg_win"], 4),
            avg_loss=round(trade_stats["avg_loss"], 4),
        )

    @staticmethod
    def total_return(equity: pd.Series) -> float:
        return float((equity.iloc[-1] / equity.iloc[0]) - 1)

    @staticmethod
    def annualized_return(equity: pd.Series) -> float:
        total = MetricsCalculator.total_return(equity)
        years = len(equity) / MetricsCalculator.TRADING_DAYS
        if years <= 0:
            return 0.0
        return float((1 + total) ** (1 / years) - 1)

    @staticmethod
    def annualized_volatility(returns: pd.Series) -> float:
        return float(returns.std() * np.sqrt(MetricsCalculator.TRADING_DAYS))

    @staticmethod
    def sharpe_ratio(returns: pd.Series, rf: float = RISK_FREE_RATE) -> float:
        excess = returns - rf / MetricsCalculator.TRADING_DAYS
        std = excess.std()
        if std == 0:
            return 0.0
        return float(excess.mean() / std * np.sqrt(MetricsCalculator.TRADING_DAYS))

    @staticmethod
    def sortino_ratio(returns: pd.Series, rf: float = RISK_FREE_RATE) -> float:
        excess = returns - rf / MetricsCalculator.TRADING_DAYS
        downside = excess[excess < 0].std()
        if downside == 0 or np.isnan(downside):
            return 0.0
        return float(excess.mean() / downside * np.sqrt(MetricsCalculator.TRADING_DAYS))

    @staticmethod
    def max_drawdown(equity: pd.Series) -> float:
        peak = equity.expanding().max()
        dd = equity / peak - 1
        return float(dd.min())

    @staticmethod
    def max_drawdown_duration(equity: pd.Series) -> int:
        peak = equity.expanding().max()
        dd = equity / peak - 1
        in_dd = dd < 0
        if not in_dd.any():
            return 0
        groups = (in_dd != in_dd.shift()).cumsum()
        durations = groups[in_dd].value_counts()
        return int(durations.max()) if not durations.empty else 0

    @staticmethod
    def calmar_ratio(ann_return: float, max_dd: float) -> float:
        if max_dd == 0:
            return 0.0
        return ann_return / abs(max_dd)

    @staticmethod
    def alpha_beta(returns: pd.Series, benchmark: pd.Series, rf: float = RISK_FREE_RATE) -> tuple:
        aligned = pd.concat([returns, benchmark], axis=1).dropna()
        if len(aligned) < 2:
            return 0.0, 0.0
        r = aligned.iloc[:, 0]
        b = aligned.iloc[:, 1]
        cov = np.cov(r, b)
        if cov[1, 1] == 0:
            return 0.0, 0.0
        beta = cov[0, 1] / cov[1, 1]
        alpha = (
            r.mean()
            - rf / MetricsCalculator.TRADING_DAYS
            - beta * (b.mean() - rf / MetricsCalculator.TRADING_DAYS)
        )
        return float(alpha * MetricsCalculator.TRADING_DAYS), float(beta)

    @staticmethod
    def var_cvar(returns: pd.Series, confidence: float = 0.95) -> tuple:
        sorted_ret = returns.dropna().sort_values()
        if len(sorted_ret) == 0:
            return 0.0, 0.0
        var_idx = int((1 - confidence) * len(sorted_ret))
        var_idx = max(0, var_idx)
        var = float(sorted_ret.iloc[var_idx])
        tail = sorted_ret.iloc[: var_idx + 1]
        cvar = float(tail.mean()) if len(tail) > 0 else var
        return var, cvar

    @staticmethod
    def trade_statistics(trades: pd.DataFrame) -> dict:
        total = len(trades)
        if total == 0:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
            }

        wins = trades[trades["win"]]
        losses = trades[~trades["win"]]
        total_wins = len(wins)
        total_losses = len(losses)
        avg_win = wins["pnl_pct"].mean() if total_wins > 0 else 0.0
        avg_loss = losses["pnl_pct"].mean() if total_losses > 0 else 0.0

        gross_profit = wins["pnl"].sum() if total_wins > 0 else 0.0
        gross_loss = abs(losses["pnl"].sum()) if total_losses > 0 else 0.0
        pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        return {
            "total_trades": total,
            "winning_trades": total_wins,
            "losing_trades": total_losses,
            "win_rate": total_wins / total if total > 0 else 0.0,
            "profit_factor": pf if pf != float("inf") else 999.0,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
        }

    @staticmethod
    def win_rate(trades: pd.DataFrame) -> float:
        return MetricsCalculator.trade_statistics(trades)["win_rate"]

    @staticmethod
    def profit_factor(trades: pd.DataFrame) -> float:
        return MetricsCalculator.trade_statistics(trades)["profit_factor"]
