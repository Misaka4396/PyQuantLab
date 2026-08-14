"""Strategy plugin registry for dynamic strategy discovery."""

from __future__ import annotations

from strategy.base import BaseStrategy


class StrategyRegistry:
    def __init__(self):
        self._strategies: dict[str, type[BaseStrategy]] = {}

    def register(self, strategy_cls: type[BaseStrategy]) -> None:
        name = strategy_cls.get_name()
        self._strategies[name] = strategy_cls

    def register_defaults(self) -> None:
        from strategy.bollinger import BollingerStrategy
        from strategy.ma_crossover import MACrossoverStrategy
        from strategy.macd import MACDStrategy
        from strategy.rsi import RSIStrategy

        self.register(MACrossoverStrategy)
        self.register(RSIStrategy)
        self.register(MACDStrategy)
        self.register(BollingerStrategy)

    def get(self, name: str) -> type[BaseStrategy]:
        if name not in self._strategies:
            raise KeyError(f"Strategy '{name}' not found. Available: {self.list_names()}")
        return self._strategies[name]

    def create(self, name: str, **params) -> BaseStrategy:
        cls = self.get(name)
        spec = cls.get_param_spec()
        merged = {}
        for key, info in spec.items():
            merged[key] = params.get(key, info["default"])
        return cls(merged)

    def list_all(self) -> list[type[BaseStrategy]]:
        return list(self._strategies.values())

    def list_names(self) -> list[str]:
        return list(self._strategies.keys())

    def get_param_spec(self, name: str) -> dict:
        return self.get(name).get_param_spec()


registry = StrategyRegistry()
registry.register_defaults()
