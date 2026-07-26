"""Multi-asset portfolio management and position tracking."""

from typing import List, Optional

import numpy as np
import pandas as pd


class PortfolioManager:
    def __init__(self, assets: List[str], weights: Optional[np.ndarray] = None):
        self.assets = assets
        if weights is None:
            weights = np.ones(len(assets)) / len(assets)
        self.weights = np.array(weights, dtype=float)

    def compute_portfolio_returns(self, returns_df: pd.DataFrame) -> pd.Series:
        common = returns_df.columns.intersection(self.assets)
        w = self.weights[:len(common)]
        w = w / w.sum()
        return returns_df[common].dot(w)

    def compute_portfolio_value(
        self, prices_df: pd.DataFrame, initial: float = 100_000.0
    ) -> pd.Series:
        common_assets = [a for a in self.assets if a in prices_df.columns]
        if not common_assets:
            return pd.Series(initial, index=prices_df.index)
        norm = prices_df[common_assets] / prices_df[common_assets].iloc[0]
        w = self.weights[:len(common_assets)] / self.weights[:len(common_assets)].sum()
        port_norm = norm.dot(w)
        return port_norm * initial

    def rebalance(self, target_weights: np.ndarray) -> pd.Series:
        turnover = np.abs(target_weights - self.weights).sum() / 2
        self.weights = target_weights
        return turnover

    def get_current_weights(self) -> np.ndarray:
        return self.weights.copy()

    def turnover(self) -> float:
        return 0.0
