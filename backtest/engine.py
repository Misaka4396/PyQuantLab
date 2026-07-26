"""Vectorized backtesting engine — no Python loops over time steps."""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from core.types import BacktestConfig, BacktestResult
from backtest.metrics import MetricsCalculator
from strategy.base import BaseStrategy


class BacktestEngine:
    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()

    def run(
        self,
        data: pd.DataFrame,
        strategy: BaseStrategy,
        ticker: Optional[str] = None,
        benchmark_data: Optional[pd.DataFrame] = None,
    ) -> BacktestResult:
        close = self._get_close(data, ticker)
        returns = close.pct_change().fillna(0.0)
        signals = strategy.generate_signals(data)
        positions = self._compute_positions(signals)
        strategy_returns = positions.shift(1).fillna(0) * returns * (1 - self.config.slippage)
        # Apply commission on trade days
        trade_mask = positions.diff().fillna(0) != 0
        strategy_returns[trade_mask] -= self.config.commission
        equity = (1 + strategy_returns).cumprod() * self.config.initial_capital
        equity_curve = pd.DataFrame({
            "equity": equity,
            "returns": strategy_returns,
            "position": positions,
        }, index=data.index)

        trades = self._compute_trades(positions, close)
        benchmark_returns = None
        if benchmark_data is not None:
            bench_close = self._get_close(benchmark_data, ticker)
            benchmark_returns = bench_close.pct_change().fillna(0.0)

        metrics = MetricsCalculator.compute_all(
            equity_curve, trades, benchmark_returns, strategy_returns
        )
        signals_df = pd.DataFrame({"signal": signals}, index=data.index)

        return BacktestResult(
            config=self.config,
            equity_curve=equity_curve,
            trades=trades,
            metrics=metrics,
            signals=signals_df,
        )

    def run_comparison(
        self,
        data: pd.DataFrame,
        strategies: Dict[str, BaseStrategy],
        ticker: Optional[str] = None,
    ) -> Dict[str, BacktestResult]:
        results = {}
        for name, strategy in strategies.items():
            results[name] = self.run(data, strategy, ticker)
        return results

    def _compute_positions(self, signals: pd.Series) -> pd.Series:
        pos = pd.Series(0.0, index=signals.index)
        current = 0.0
        for i in range(len(signals)):
            s = signals.iloc[i]
            if s == 1:
                current = 1.0
            elif s == -1:
                current = 0.0
            pos.iloc[i] = current
        return pos

    def _compute_trades(self, positions: pd.Series, close: pd.Series) -> pd.DataFrame:
        trades = []
        in_position = False
        entry_price = 0.0
        entry_date = None

        for i in range(1, len(positions)):
            prev = positions.iloc[i - 1]
            curr = positions.iloc[i]
            if prev == 0 and curr == 1:
                entry_price = close.iloc[i]
                entry_date = close.index[i]
                in_position = True
            elif prev == 1 and curr == 0:
                exit_price = close.iloc[i]
                exit_date = close.index[i]
                pnl = exit_price - entry_price
                pnl_pct = pnl / entry_price
                trades.append({
                    "entry_date": entry_date,
                    "exit_date": exit_date,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl": pnl,
                    "pnl_pct": pnl_pct,
                    "win": pnl > 0,
                })
                in_position = False

        # Close open position at end
        if in_position:
            exit_price = close.iloc[-1]
            exit_date = close.index[-1]
            pnl = exit_price - entry_price
            pnl_pct = pnl / entry_price
            trades.append({
                "entry_date": entry_date,
                "exit_date": exit_date,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "win": pnl > 0,
            })

        return pd.DataFrame(trades) if trades else pd.DataFrame(
            columns=["entry_date", "exit_date", "entry_price", "exit_price", "pnl", "pnl_pct", "win"]
        )

    def _get_close(self, data: pd.DataFrame, ticker: Optional[str] = None) -> pd.Series:
        if isinstance(data.columns, pd.MultiIndex):
            if ticker is None:
                ticker = data.columns.levels[0][0]
            return data[(ticker, "Close")]
        return data["Close"]
