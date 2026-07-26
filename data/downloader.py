"""yfinance wrapper with retry logic and exponential backoff."""

import time
from typing import List, Optional

import pandas as pd
import yfinance as yf

from config import MAX_RETRIES, RETRY_DELAY
from core.exceptions import DataDownloadError


class DataDownloader:
    def __init__(self, max_retries: int = MAX_RETRIES, retry_delay: float = RETRY_DELAY):
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def download(
        self,
        tickers: List[str],
        start: str,
        end: str,
        interval: str = "1d",
        auto_adjust: bool = True,
    ) -> pd.DataFrame:
        for attempt in range(self.max_retries):
            try:
                data = yf.download(
                    tickers=tickers,
                    start=start,
                    end=end,
                    interval=interval,
                    auto_adjust=auto_adjust,
                    progress=False,
                    group_by="ticker",
                )
                if data.empty:
                    raise DataDownloadError(f"No data returned for {tickers}")
                if len(tickers) == 1:
                    data = data.copy()
                    data.columns = pd.MultiIndex.from_product(
                        [tickers, data.columns]
                    )
                return data
            except DataDownloadError:
                raise
            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait = self.retry_delay * (2 ** attempt)
                    time.sleep(wait)
                else:
                    raise DataDownloadError(
                        f"Failed after {self.max_retries} attempts: {e}"
                    ) from e

    def download_single(
        self, ticker: str, start: str, end: str, interval: str = "1d"
    ) -> pd.DataFrame:
        return self.download([ticker], start, end, interval)

    def get_info(self, ticker: str) -> dict:
        try:
            t = yf.Ticker(ticker)
            return t.info or {}
        except Exception:
            return {}

    def validate_ticker(self, ticker: str) -> bool:
        try:
            info = self.get_info(ticker)
            return bool(info.get("symbol"))
        except Exception:
            return False
