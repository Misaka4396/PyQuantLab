"""Mean-variance and risk-parity portfolio optimization using scipy."""

from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from config import RISK_FREE_RATE
from core.types import PortfolioWeights

TRADING_DAYS = 252


class PortfolioOptimizer:
    def __init__(self, returns: pd.DataFrame):
        self.returns = returns.dropna()
        self.mean_returns = self.returns.mean() * TRADING_DAYS
        self.cov_matrix = self.returns.cov() * TRADING_DAYS
        self.assets = list(self.returns.columns)
        self.n = len(self.assets)

    def efficient_frontier(self, num_points: int = 50) -> pd.DataFrame:
        points = []
        target_returns = np.linspace(
            self.mean_returns.min() * 0.5,
            self.mean_returns.max() * 1.5,
            num_points,
        )
        for tr in target_returns:
            try:
                w = self._min_vol_for_return(tr)
                r, v, s = self.annualize_metrics(w)
                points.append({"return": r, "volatility": v, "sharpe": s, **dict(zip(self.assets, w))})
            except Exception:
                continue
        return pd.DataFrame(points)

    def max_sharpe(self) -> PortfolioWeights:
        constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
        bounds = tuple((0, 1) for _ in range(self.n))
        x0 = np.ones(self.n) / self.n

        def neg_sharpe(w):
            r, v = self._portfolio_metrics(w)
            return -(r - RISK_FREE_RATE) / v if v > 0 else 1e6

        result = minimize(neg_sharpe, x0, bounds=bounds, constraints=constraints, method="SLSQP")
        if not result.success:
            return self.equal_weight()
        r, v, s = self.annualize_metrics(result.x)
        return PortfolioWeights(assets=self.assets, weights=result.x, expected_return=r,
                                expected_volatility=v, sharpe_ratio=s)

    def min_variance(self) -> PortfolioWeights:
        constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
        bounds = tuple((0, 1) for _ in range(self.n))
        x0 = np.ones(self.n) / self.n

        def port_vol(w):
            return np.sqrt(w.T @ self.cov_matrix.values @ w)

        result = minimize(port_vol, x0, bounds=bounds, constraints=constraints, method="SLSQP")
        if not result.success:
            return self.equal_weight()
        r, v, s = self.annualize_metrics(result.x)
        return PortfolioWeights(assets=self.assets, weights=result.x, expected_return=r,
                                expected_volatility=v, sharpe_ratio=s)

    def risk_parity(self) -> PortfolioWeights:
        def risk_budget_error(w):
            w = np.abs(w)
            port_vol = np.sqrt(w.T @ self.cov_matrix.values @ w)
            marginal_contrib = self.cov_matrix.values @ w
            risk_contrib = w * marginal_contrib / port_vol
            target = port_vol / self.n
            return np.sum((risk_contrib - target) ** 2)

        constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
        bounds = tuple((1e-6, 1) for _ in range(self.n))
        x0 = np.ones(self.n) / self.n

        result = minimize(risk_budget_error, x0, bounds=bounds, constraints=constraints, method="SLSQP")
        if not result.success:
            return self.equal_weight()
        w = np.abs(result.x) / np.sum(np.abs(result.x))
        r, v, s = self.annualize_metrics(w)
        return PortfolioWeights(assets=self.assets, weights=w, expected_return=r,
                                expected_volatility=v, sharpe_ratio=s)

    def equal_weight(self) -> PortfolioWeights:
        w = np.ones(self.n) / self.n
        r, v, s = self.annualize_metrics(w)
        return PortfolioWeights(assets=self.assets, weights=w, expected_return=r,
                                expected_volatility=v, sharpe_ratio=s)

    def _min_vol_for_return(self, target_return: float) -> np.ndarray:
        constraints = [
            {"type": "eq", "fun": lambda x: np.sum(x) - 1},
            {"type": "eq", "fun": lambda x: x @ self.mean_returns.values - target_return},
        ]
        bounds = tuple((0, 1) for _ in range(self.n))
        x0 = np.ones(self.n) / self.n

        def port_vol(w):
            return np.sqrt(w.T @ self.cov_matrix.values @ w)

        result = minimize(port_vol, x0, bounds=bounds, constraints=constraints, method="SLSQP")
        return result.x if result.success else x0

    def _portfolio_metrics(self, weights: np.ndarray) -> Tuple[float, float]:
        r = float(weights @ self.mean_returns.values)
        v = float(np.sqrt(weights.T @ self.cov_matrix.values @ weights))
        return r, v

    def annualize_metrics(self, weights: np.ndarray) -> Tuple[float, float, float]:
        r, v = self._portfolio_metrics(weights)
        s = (r - RISK_FREE_RATE) / v if v > 0 else 0.0
        return r, v, s
