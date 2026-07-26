"""Local Parquet-based cache manager for OHLCV data."""

import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from config import CACHE_DIR, CACHE_TTL_DAYS
from core.exceptions import CacheError


class DataCache:
    def __init__(self, cache_dir: str = CACHE_DIR, ttl_days: int = CACHE_TTL_DAYS):
        self.cache_dir = Path(cache_dir)
        self.ttl_days = ttl_days
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _make_path(self, ticker: str, start: str, end: str, interval: str) -> Path:
        filename = f"{ticker}_{start}_{end}_{interval}.parquet"
        return self.cache_dir / ticker / filename

    def get(
        self, ticker: str, start: str, end: str, interval: str
    ) -> Optional[pd.DataFrame]:
        path = self._make_path(ticker, start, end, interval)
        if not path.exists():
            return None
        if self._is_stale(path):
            path.unlink()
            return None
        try:
            return pd.read_parquet(path)
        except Exception as e:
            path.unlink(missing_ok=True)
            raise CacheError(f"Corrupted cache for {ticker}: {e}") from e

    def put(
        self, ticker: str, start: str, end: str, interval: str, data: pd.DataFrame
    ) -> None:
        path = self._make_path(ticker, start, end, interval)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            data.to_parquet(path, index=True)
        except Exception as e:
            raise CacheError(f"Failed to write cache for {ticker}: {e}") from e

    def exists(self, ticker: str, start: str, end: str, interval: str) -> bool:
        path = self._make_path(ticker, start, end, interval)
        return path.exists()

    def invalidate(self, ticker: Optional[str] = None) -> int:
        count = 0
        if ticker:
            ticker_dir = self.cache_dir / ticker
            if ticker_dir.exists():
                for f in ticker_dir.iterdir():
                    f.unlink()
                    count += 1
                ticker_dir.rmdir()
        else:
            for f in self.cache_dir.rglob("*.parquet"):
                f.unlink()
                count += 1
        return count

    def get_stats(self) -> dict:
        total_size = 0
        tickers = set()
        count = 0
        for f in self.cache_dir.rglob("*.parquet"):
            total_size += f.stat().st_size
            tickers.add(f.parent.name)
            count += 1
        return {
            "total_files": count,
            "total_tickers": len(tickers),
            "tickers": sorted(tickers),
            "total_size_mb": round(total_size / (1024 * 1024), 2),
        }

    def _is_stale(self, path: Path) -> bool:
        mtime = datetime.fromtimestamp(path.stat().st_mtime)
        return datetime.now() - mtime > timedelta(days=self.ttl_days)
