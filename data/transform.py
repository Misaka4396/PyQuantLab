"""Resampling, normalization, and technical indicator computations."""

import numpy as np
import pandas as pd


def compute_returns(prices: pd.Series, method: str = "simple") -> pd.Series:
    if method == "log":
        return np.log(prices / prices.shift(1))
    return prices.pct_change()


def normalize_prices(prices: pd.Series, base: float = 100.0) -> pd.Series:
    return prices / prices.iloc[0] * base


def compute_drawdown(prices: pd.Series) -> pd.DataFrame:
    peak = prices.expanding().max()
    drawdown = prices / peak - 1
    return pd.DataFrame({"price": prices, "peak": peak, "drawdown": drawdown})


def add_sma(df: pd.DataFrame, column: str, period: int) -> pd.Series:
    return df[column].rolling(window=period).mean()


def add_ema(df: pd.DataFrame, column: str, period: int) -> pd.Series:
    return df[column].ewm(span=period, adjust=False).mean()


def add_rsi(df: pd.DataFrame, column: str, period: int = 14) -> pd.Series:
    delta = df[column].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def add_macd(
    df: pd.DataFrame, column: str, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    ema_fast = df[column].ewm(span=fast, adjust=False).mean()
    ema_slow = df[column].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return pd.DataFrame(
        {"macd": macd_line, "signal": signal_line, "histogram": histogram},
        index=df.index,
    )


def add_bollinger_bands(
    df: pd.DataFrame, column: str, period: int = 20, num_std: float = 2.0
) -> pd.DataFrame:
    sma = df[column].rolling(window=period).mean()
    std = df[column].rolling(window=period).std()
    return pd.DataFrame(
        {"middle": sma, "upper": sma + num_std * std, "lower": sma - num_std * std},
        index=df.index,
    )
