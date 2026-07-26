"""Global configuration constants for PyQuantLab."""

DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
CACHE_DIR = "./data_cache"
CACHE_TTL_DAYS = 1
DEFAULT_INITIAL_CAPITAL = 100_000.0
DEFAULT_COMMISSION = 0.001
DEFAULT_SLIPPAGE = 0.0005
RISK_FREE_RATE = 0.02
MC_SIMULATIONS_DEFAULT = 1000
MAX_RETRIES = 3
RETRY_DELAY = 2.0
SESSION_KEYS = {
    "data_df": "data_df",
    "strategy_instance": "strategy_instance",
    "backtest_result": "backtest_result",
    "backtest_results_dict": "backtest_results_dict",
    "portfolio_weights": "portfolio_weights",
}
