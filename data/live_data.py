"""Real-time / live data fetching via yfinance with polling support."""

import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf


@dataclass
class Quote:
    symbol: str
    price: float
    change: float
    change_pct: float
    volume: int
    timestamp: datetime


class LiveDataFetcher:
    def __init__(self):
        self._tickers_cache: Dict[str, yf.Ticker] = {}

    def get_current_quotes(self, symbols: List[str]) -> Dict[str, Quote]:
        results = {}
        for sym in symbols:
            try:
                t = self._get_ticker(sym)
                info = t.info or {}

                price = info.get('currentPrice') or info.get('regularMarketPrice') or info.get('previousClose') or 0.0
                prev_close = info.get('previousClose') or info.get('regularMarketPreviousClose') or price or 1.0
                change = price - prev_close if price and prev_close else 0.0
                change_pct = change / prev_close if prev_close else 0.0
                volume = info.get('volume') or info.get('regularMarketVolume') or 0

                results[sym] = Quote(
                    symbol=sym, price=price, change=change,
                    change_pct=change_pct, volume=int(volume),
                    timestamp=datetime.now(),
                )
            except Exception as e:
                results[sym] = Quote(
                    symbol=sym, price=0.0, change=0.0,
                    change_pct=0.0, volume=0, timestamp=datetime.now(),
                )
        return results

    def get_intraday(self, symbols: List[str], period: str = "1d", interval: str = "5m") -> pd.DataFrame:
        frames = {}
        for sym in symbols:
            try:
                t = self._get_ticker(sym)
                df = t.history(period=period, interval=interval)
                if not df.empty:
                    frames[sym] = df["Close"]
            except Exception:
                pass
        if not frames:
            return pd.DataFrame()
        return pd.DataFrame(frames)

    def get_quote_table(self, symbols: List[str]) -> pd.DataFrame:
        quotes = self.get_current_quotes(symbols)
        rows = []
        for q in quotes.values():
            rows.append({
                "Symbol": q.symbol,
                "Price": f"${q.price:.2f}",
                "Change": f"${q.change:+.2f}",
                "Change %": f"{q.change_pct:+.2%}",
                "Volume": f"{q.volume:,}",
                "Time": q.timestamp.strftime("%H:%M:%S"),
            })
        return pd.DataFrame(rows)

    def _get_ticker(self, symbol: str) -> yf.Ticker:
        if symbol not in self._tickers_cache:
            self._tickers_cache[symbol] = yf.Ticker(symbol)
        return self._tickers_cache[symbol]
