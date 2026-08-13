"""Custom exception hierarchy for PyQuantLab."""


class PyQuantLabError(Exception):
    """Base exception for all PyQuantLab errors."""


class DataError(PyQuantLabError):
    """Errors related to data fetching or caching."""


class DataDownloadError(DataError):
    """yfinance API failures, network errors."""


class DataNotFoundError(DataError):
    """Requested ticker or data not available."""


class CacheError(DataError):
    """Corrupted cache or IO errors."""


class StrategyError(PyQuantLabError):
    """Invalid strategy parameters or signal computation failure."""


class BacktestError(PyQuantLabError):
    """Backtest execution failures."""


class EngineError(PyQuantLabError):
    """Event-driven backtest engine failures."""


class PortfolioError(PyQuantLabError):
    """Optimization convergence failures."""
