"""Abstract base class for all trading strategies."""

from abc import ABC, abstractmethod
from typing import Dict

import pandas as pd

from core.exceptions import StrategyError


class BaseStrategy(ABC):
    def __init__(self, params: dict):
        self.params = params
        self.validate_parameters()

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Return a Series of -1 (SELL), 0 (HOLD), +1 (BUY) indexed by date."""

    @classmethod
    @abstractmethod
    def get_name(cls) -> str:
        """Human-readable strategy name."""

    @classmethod
    @abstractmethod
    def get_description(cls) -> str:
        """One-paragraph explanation of the strategy logic."""

    @classmethod
    @abstractmethod
    def get_param_spec(cls) -> Dict[str, dict]:
        """Parameter schema for auto-generating UI controls.
        Format: {param_name: {type, default, min, max, step, label, help, choices}}
        """

    def validate_parameters(self) -> None:
        spec = self.get_param_spec()
        for key, info in spec.items():
            value = self.params.get(key, info["default"])
            if info["type"] == "int":
                if not isinstance(value, (int, float)):
                    raise StrategyError(f"{key} must be a number, got {type(value).__name__}")
                if "min" in info and value < info["min"]:
                    raise StrategyError(f"{key} must be >= {info['min']}, got {value}")
                if "max" in info and value > info["max"]:
                    raise StrategyError(f"{key} must be <= {info['max']}, got {value}")
            elif info["type"] == "choice":
                if value not in info["choices"]:
                    raise StrategyError(f"{key} must be one of {info['choices']}, got {value}")

    def to_dict(self) -> dict:
        return {"name": self.get_name(), "params": self.params}
