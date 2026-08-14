"""Orchestrator: coordinates download + cache for multi-ticker data requests."""

from collections.abc import Callable

import numpy as np
import pandas as pd

from core.exceptions import DataDownloadError, DataNotFoundError
from data.cache import DataCache
from data.downloader import DataDownloader


class DataManager:
    def __init__(
        self,
        downloader: DataDownloader | None = None,
        cache: DataCache | None = None,
    ):
        self.downloader = downloader or DataDownloader()
        self.cache = cache or DataCache()

    def get_data(
        self,
        tickers: list[str],
        start: str,
        end: str,
        interval: str = "1d",
        force_download: bool = False,
        progress_callback: Callable[[str, str], None] | None = None,
    ) -> pd.DataFrame:
        frames = {}
        errors = []

        for ticker in tickers:
            if progress_callback:
                progress_callback(ticker, "checking cache")

            if not force_download:
                cached = self.cache.get(ticker, start, end, interval)
                if cached is not None:
                    frames[ticker] = cached
                    continue

            if progress_callback:
                progress_callback(ticker, "downloading")

            try:
                df = self.downloader.download_single(ticker, start, end, interval)
                if df.empty:
                    errors.append((ticker, "No data returned"))
                    continue
                self.cache.put(ticker, start, end, interval, df)
                frames[ticker] = df
            except (DataDownloadError, DataNotFoundError) as e:
                errors.append((ticker, str(e)))

        if not frames:
            raise DataNotFoundError(f"No data available for any ticker. Errors: {errors}")

        combined = pd.concat(frames, axis=1)
        combined.columns = pd.MultiIndex.from_tuples(combined.columns)
        return combined

    def get_single(self, ticker: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame:
        return self.get_data([ticker], start, end, interval)

    def list_cached_tickers(self) -> list[str]:
        return self.cache.get_stats()["tickers"]

    def get_data_overview(self, data: pd.DataFrame) -> dict:
        return {
            "shape": data.shape,
            "start_date": str(data.index.min().date()),
            "end_date": str(data.index.max().date()),
            "trading_days": len(data),
            "missing_values": int(data.isna().sum().sum()),
        }

    def get_returns(self, data: pd.DataFrame, ticker: str, method: str = "log") -> pd.Series:
        close = self.get_price_column(data, ticker, "Close")
        if method == "log":
            return np.log(close / close.shift(1)).dropna()
        return close.pct_change().dropna()

    def get_price_column(self, data: pd.DataFrame, ticker: str, column: str = "Close") -> pd.Series:
        try:
            return data[(ticker, column)]
        except KeyError:
            return data[column]
