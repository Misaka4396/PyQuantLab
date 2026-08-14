"""Risk analysis utilities — rolling metrics, drawdown analysis, stress testing."""

import numpy as np
import pandas as pd

from config import RISK_FREE_RATE


class RiskAnalyzer:
    TRADING_DAYS = 252

    @staticmethod
    def rolling_sharpe(returns: pd.Series, window: int = 252) -> pd.Series:
        rf_daily = RISK_FREE_RATE / RiskAnalyzer.TRADING_DAYS
        excess = returns - rf_daily
        roll_mean = excess.rolling(window=window).mean()
        roll_std = excess.rolling(window=window).std()
        sharpe = (roll_mean / roll_std.replace(0, np.nan)) * np.sqrt(RiskAnalyzer.TRADING_DAYS)
        return sharpe

    @staticmethod
    def rolling_volatility(returns: pd.Series, window: int = 21) -> pd.Series:
        return returns.rolling(window=window).std() * np.sqrt(RiskAnalyzer.TRADING_DAYS)

    @staticmethod
    def rolling_beta(returns: pd.Series, benchmark: pd.Series, window: int = 252) -> pd.Series:
        aligned = pd.concat([returns, benchmark], axis=1).dropna()
        r = aligned.iloc[:, 0]
        b = aligned.iloc[:, 1]
        roll_cov = r.rolling(window=window).cov(b)
        roll_var = b.rolling(window=window).var()
        return roll_cov / roll_var.replace(0, np.nan)

    @staticmethod
    def drawdown_analysis(equity: pd.Series) -> pd.DataFrame:
        peak = equity.expanding().max()
        dd = equity / peak - 1
        is_dd = dd < 0
        if not is_dd.any():
            return pd.DataFrame()

        group = (is_dd != is_dd.shift()).cumsum()
        dd_periods = []
        for g in group[is_dd].unique():
            mask = group == g
            if not mask.any():
                continue
            start = mask.idxmax()
            end = mask[::-1].idxmax()
            dd_periods.append(
                {
                    "start": start,
                    "end": end,
                    "duration": (end - start).days
                    if hasattr(end - start, "days")
                    else len(equity.loc[start:end]),
                    "max_drawdown": float(dd.loc[start:end].min()),
                    "recovery": float((equity.loc[end] / peak.loc[start]) - 1) if end else 0.0,
                }
            )
        return pd.DataFrame(dd_periods).sort_values("max_drawdown")

    @staticmethod
    def tail_risk(returns: pd.Series) -> dict:
        clean = returns.dropna()
        if len(clean) < 2:
            return {"skewness": 0.0, "kurtosis": 0.0, "var_99": 0.0, "max_daily_loss": 0.0}
        return {
            "skewness": float(clean.skew()),
            "kurtosis": float(clean.kurtosis()),
            "var_99": float(clean.quantile(0.01)),
            "max_daily_loss": float(clean.min()),
        }

    @staticmethod
    def correlation_heatmap(prices_df: pd.DataFrame) -> pd.DataFrame:
        returns = prices_df.pct_change().dropna()
        return returns.corr()

    @staticmethod
    def stress_test(returns: pd.Series, scenarios: dict[str, float] | None = None) -> pd.DataFrame:
        if scenarios is None:
            scenarios = {"-5% 下跌": -0.05, "-10% 下跌": -0.10, "-20% 暴跌": -0.20}
        results = []
        current_value = 100_000.0
        for name, shock in scenarios.items():
            shocked_ret = returns + shock
            final = current_value * (1 + shocked_ret).prod()
            results.append(
                {
                    "scenario": name,
                    "shock": shock,
                    "final_value": round(final, 2),
                    "total_return": round(float((1 + shocked_ret).prod() - 1), 4),
                }
            )
        return pd.DataFrame(results)
