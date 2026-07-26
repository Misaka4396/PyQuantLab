# PyQuantLab

A Python quantitative trading platform with a native desktop GUI, covering the full quant workflow: live market data, strategy development, backtesting, risk analysis, and portfolio optimization.

## Features

- **Live Market Data** — Real-time quotes and intraday charts with auto-refresh
- **Native Desktop GUI** — PyQt5-based application, no browser required
- **Data Management** — Download historical stock data via yfinance, with local Parquet caching
- **Strategy Engine** — 4 built-in strategies: MA Crossover, RSI, MACD, Bollinger Bands
- **Plugin Architecture** — Add new strategies by implementing a base class and registering it
- **Vectorized Backtesting** — Fast backtesting engine with commission and slippage modeling
- **Performance Metrics** — Sharpe, Sortino, Calmar ratios, max drawdown, VaR/CVaR, alpha/beta
- **Risk Analysis** — Drawdown analysis, rolling metrics, stress testing, tail risk
- **Portfolio Optimization** — Mean-variance (max Sharpe, min variance), risk parity, equal weight
- **Monte Carlo Simulation** — Forward portfolio simulation with Cholesky decomposition

## Installation

### Option 1: Run from Source

```bash
git clone https://github.com/yourusername/PyQuantLab.git
cd PyQuantLab
pip install -r requirements.txt
python launcher_qt.py
```

### Option 2: Standalone EXE (no Python required)

```bash
pip install pyinstaller
pyinstaller PyQuantLab_Qt.spec --noconfirm
# EXE located at: dist/PyQuantLab/PyQuantLab.exe
```

Double-click `dist/PyQuantLab/PyQuantLab.exe` to launch.

### Option 3: Web UI (Streamlit)

```bash
streamlit run app.py
# Open http://localhost:8501 in browser
```

## Requirements

```
PyQt5>=5.15
matplotlib>=3.5
pandas>=1.3
numpy>=1.21
yfinance>=0.1.70
scipy>=1.7
```

## Usage Guide

### 1. Live Market
Enter ticker symbols to monitor real-time prices. Toggle auto-refresh to poll every 60 seconds. View intraday charts with configurable intervals.

### 2. Data Management
Enter ticker symbols and date range, then click "Load Data" to download historical data. Data is cached locally as Parquet files.

### 3. Strategy Config
Select a strategy and adjust parameters. Buy/sell signals are previewed on the price chart in real-time.

### 4. Run Backtest
Set capital, commission, and slippage. Run a single strategy or compare all strategies side-by-side.

### 5. Results & Risk
Six analysis tabs: Overview, Equity Curve, Drawdown, Trade Analysis, Risk Metrics with stress testing.

### 6. Portfolio Simulator
Optimize portfolio weights (max Sharpe / min variance / risk parity) and run Monte Carlo simulations with interactive charts.

## Adding a New Strategy

1. Create `strategy/my_strategy.py` implementing `BaseStrategy`:

```python
from strategy.base import BaseStrategy
import pandas as pd

class MyStrategy(BaseStrategy):
    @classmethod
    def get_name(cls) -> str:
        return "My Strategy"

    @classmethod
    def get_description(cls) -> str:
        return "Strategy description."

    @classmethod
    def get_param_spec(cls) -> dict:
        return {
            "param1": {
                "type": "int", "default": 10, "min": 1, "max": 100, "step": 1,
                "label": "Parameter 1", "help": "Description",
            },
        }

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        # Return: 1 (BUY), -1 (SELL), 0 (HOLD) per time step
        pass
```

2. Register in `strategy/registry.py`:
```python
from strategy.my_strategy import MyStrategy
registry.register(MyStrategy)
```

The strategy automatically appears in the UI dropdown with auto-generated parameter controls.

## Project Structure

```
PyQuantLab/
├── launcher_qt.py       # Desktop app entry point
├── launcher.py          # Streamlit web entry point
├── app.py               # Streamlit app
├── config.py            # Global configuration
├── data/
│   ├── live_data.py     # Real-time data fetcher
│   ├── downloader.py    # yfinance wrapper
│   ├── cache.py         # Parquet cache
│   ├── manager.py       # Data orchestration
│   └── transform.py     # Technical indicators
├── strategy/            # Strategy base + 4 built-in strategies
├── backtest/            # Vectorized engine + metrics + risk
├── portfolio/           # Portfolio optimization + Monte Carlo
└── ui/
    ├── main_window.py   # PyQt5 main window
    ├── charts_qt.py     # Matplotlib chart factories
    ├── pages_qt/        # 6 PyQt5 page widgets
    ├── pages/           # 5 Streamlit pages (web UI)
    └── components/      # Streamlit chart components
```

## License

MIT License — see [LICENSE](LICENSE) for details.
