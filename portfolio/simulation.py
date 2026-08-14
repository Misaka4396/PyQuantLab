"""Monte Carlo portfolio simulation using Cholesky decomposition."""

import numpy as np
import pandas as pd

from config import MC_SIMULATIONS_DEFAULT
from portfolio.optimizer import TRADING_DAYS


class PortfolioSimulation:
    def __init__(
        self,
        returns: pd.DataFrame,
        num_simulations: int = MC_SIMULATIONS_DEFAULT,
    ):
        self.returns = returns.dropna()
        self.num_simulations = num_simulations
        self.mean = self.returns.mean().values
        self.cov = self.returns.cov().values
        self.n_assets = len(self.returns.columns)

    def run(
        self,
        weights: np.ndarray,
        initial_value: float = 100_000.0,
        days: int = TRADING_DAYS,
        seed: int | None = None,
    ) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed)
        L = np.linalg.cholesky(self.cov)
        dt = 1.0
        paths = np.zeros((self.num_simulations, days + 1))
        paths[:, 0] = initial_value
        for t in range(1, days + 1):
            z = np.random.randn(self.num_simulations, self.n_assets)
            r = self.mean * dt + (z @ L.T) * np.sqrt(dt)
            port_r = r @ weights
            paths[:, t] = paths[:, t - 1] * (1 + port_r)
        return paths

    def compute_terminal_stats(self, simulation: np.ndarray) -> dict:
        terminal = simulation[:, -1]
        return {
            "mean": float(np.mean(terminal)),
            "median": float(np.median(terminal)),
            "std": float(np.std(terminal)),
            "p5": float(np.percentile(terminal, 5)),
            "p25": float(np.percentile(terminal, 25)),
            "p75": float(np.percentile(terminal, 75)),
            "p95": float(np.percentile(terminal, 95)),
            "min": float(np.min(terminal)),
            "max": float(np.max(terminal)),
        }

    def compute_path_percentiles(self, simulation: np.ndarray) -> pd.DataFrame:
        p5 = np.percentile(simulation, 5, axis=0)
        p25 = np.percentile(simulation, 25, axis=0)
        p50 = np.percentile(simulation, 50, axis=0)
        p75 = np.percentile(simulation, 75, axis=0)
        p95 = np.percentile(simulation, 95, axis=0)
        return pd.DataFrame(
            {
                "p5": p5,
                "p25": p25,
                "p50": p50,
                "p75": p75,
                "p95": p95,
            }
        )
