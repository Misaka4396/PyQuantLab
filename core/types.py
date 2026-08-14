"""Shared dataclasses for inter-module communication."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Signal:
    date: datetime
    ticker: str
    action: str  # BUY | SELL | HOLD
    price: float
    quantity: int
    reason: str = ""


@dataclass
class BacktestConfig:
    initial_capital: float = 100_000.0
    commission: float = 0.001
    slippage: float = 0.0005
    start_date: str | None = None
    end_date: str | None = None


@dataclass
class PerformanceMetrics:
    total_return: float = 0.0
    annualized_return: float = 0.0
    annualized_volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    calmar_ratio: float = 0.0
    alpha: float = 0.0
    beta: float = 0.0
    var_95: float = 0.0
    cvar_95: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    avg_win: float = 0.0
    avg_loss: float = 0.0


@dataclass
class BacktestResult:
    config: BacktestConfig = field(default_factory=BacktestConfig)
    equity_curve: pd.DataFrame = field(default_factory=pd.DataFrame)
    trades: pd.DataFrame = field(default_factory=pd.DataFrame)
    metrics: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    signals: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass
class PortfolioWeights:
    assets: list = field(default_factory=list)
    weights: Optional["np.ndarray"] = None
    expected_return: float = 0.0
    expected_volatility: float = 0.0
    sharpe_ratio: float = 0.0
