"""事件驱动回测引擎（A2 核心）。

事件循环：market data → signal → order → fill → portfolio → accounting。

设计原则：
- 引擎不做策略决策：策略通过 ``Strategy.on_bar`` 回调挂载。
- 撮合与成本模型解耦：成本模型通过 ``MatchingEngine`` 的接口注入。
- 多标的支持：master 时间轴为各标的 time index 的并集，按标的各自的
  时间轴做 next-bar 撮合调度（支持日历不对齐）。
- 回调/插件机制：``register_callback`` 按事件类型注册回调（策略、成本模型、
  未来 ML 重训调度均可挂载）。
- 复现：种子固定 + 确定性撮合顺序 + 可配置日志。

为 ETF 篮子下单（B）与 ML 滚动重训（C）预留的扩展点：
- 篮子下单：策略在单 bar 内可一次性发出多标的 SignalEvent，引擎逐单撮合，
  成本模型对篮子整体加滑点（见 cost_model.compute_basket）。
- 重训调度：通过 register_callback 挂载 MarketDataEvent / PortfolioEvent 回调，
  在指定时点触发重训，不改动引擎循环。
"""

from __future__ import annotations

import itertools
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from core.exceptions import EngineError
from data import schemas as sc
from engine.accounting import Accounting
from engine.config import FILL_OPEN, EngineConfig
from engine.events import (
    Event,
    MarketDataEvent,
    OrderEvent,
    PortfolioEvent,
    SignalEvent,
)
from engine.matching import CostModelLike, MatchingEngine
from engine.order import Order, OrderSide, OrderStatus, OrderType, coerce_side
from engine.portfolio import Portfolio

# A 股 ETF 代码前缀启发式：沪市 51/56/58，深市 15/16（159xxx 归入 15）
_ETF_PREFIXES = ("51", "56", "58", "15", "16")


def is_etf_symbol(symbol: str) -> bool:
    """按代码前缀启发式判断是否 ETF（供成本模型豁免印花税/过户费）。"""
    s = str(symbol).zfill(6)
    return s.startswith(_ETF_PREFIXES)


class Strategy(ABC):
    """策略插件接口：引擎在每个 bar 调用一次，返回信号列表。"""

    @abstractmethod
    def on_bar(
        self,
        timestamp: pd.Timestamp,
        bars: dict[str, MarketDataEvent],
        portfolio: Portfolio,
    ) -> list[SignalEvent]:
        """基于当前 bar 数据与组合状态生成信号（不直接下单，只返回意图）。"""


@dataclass
class EngineResult:
    """引擎运行结果。"""

    equity_curve: pd.DataFrame = field(default_factory=pd.DataFrame)
    fills: pd.DataFrame = field(default_factory=pd.DataFrame)
    orders: pd.DataFrame = field(default_factory=pd.DataFrame)
    portfolio: Portfolio | None = None
    config: EngineConfig | None = None


class EventEngine:
    """事件驱动回测主引擎。"""

    def __init__(
        self,
        config: EngineConfig | None = None,
        data: dict[str, pd.DataFrame] | pd.DataFrame | None = None,
        cost_model: CostModelLike | None = None,
        strategy: Strategy | None = None,
    ):
        self.config = config or EngineConfig()
        self.config.validate()

        self.data = self._normalize_data({} if data is None else data)
        self.cost_model = cost_model
        self.matching = MatchingEngine(cost_model) if cost_model is not None else None
        self.strategy = strategy

        self.rng = np.random.default_rng(self.config.seed)
        self._callbacks: dict[str, list[Callable]] = defaultdict(list)
        self._order_seq = 0
        self.logger = self._setup_logger()

        # 预计算 master 时间轴与各标的 next-bar 映射
        self._master_index: pd.DatetimeIndex = self._build_master_index()
        self._tsets: dict[str, set] = {sym: set(df.index) for sym, df in self.data.items()}
        self._next_map: dict[str, dict[pd.Timestamp, pd.Timestamp]] = {}
        for sym, df in self.data.items():
            idx = df.index
            self._next_map[sym] = dict(itertools.pairwise(idx))

    # ------------------------------------------------------------------
    # 插件/回调注册
    # ------------------------------------------------------------------
    def register_strategy(self, strategy: Strategy) -> None:
        """挂载策略。"""
        self.strategy = strategy

    def register_callback(self, event_cls: type, fn: Callable) -> None:
        """按事件类型注册回调（如 FillEvent、PortfolioEvent、MarketDataEvent）。"""
        self._callbacks[event_cls.__name__].append(fn)

    def _emit(self, event: Event) -> None:
        """把事件分发给对应类型的回调。"""
        for fn in self._callbacks.get(type(event).__name__, []):
            fn(event, self)

    # ------------------------------------------------------------------
    # 运行
    # ------------------------------------------------------------------
    def run(self) -> EngineResult:
        """执行事件循环，返回 EngineResult。"""
        if self.matching is None:
            raise EngineError("引擎未注入成本模型，无法撮合")

        portfolio = Portfolio(self.config.initial_cash)
        accounting = Accounting()
        pending: list[Order] = []
        orders_log: list[Order] = []
        self._order_seq = 0
        last_prices: dict[str, float] = {}

        for t in self._master_index:
            # 1) 构造当前 bar 的行情事件
            bars_t: dict[str, MarketDataEvent] = {}
            for sym, df in self.data.items():
                if t in self._tsets[sym]:
                    row = df.loc[t]
                    bars_t[sym] = self._make_market_event(t, sym, row)
                    last_prices[sym] = bars_t[sym].close
            if not bars_t:
                continue
            self._emit_market_events(bars_t)

            # 2) 撮合到期订单（next_bar 单在下一 bar 开盘成交）
            remaining: list[Order] = []
            for order in pending:
                if order.fill_timestamp == t:
                    self._execute(order, bars_t, portfolio, accounting)
                else:
                    remaining.append(order)
            pending = remaining

            # 3) 策略生成信号 → 订单
            if self.strategy is not None:
                signals = self.strategy.on_bar(t, bars_t, portfolio)
                for sig in signals:
                    order = self._create_order(sig, t)
                    orders_log.append(order)
                    self._emit(
                        OrderEvent(
                            timestamp=order.timestamp,
                            order_id=order.order_id,
                            symbol=order.symbol,
                            side=order.side.value,
                            quantity=order.quantity,
                            order_type=order.order_type.value,
                            limit_price=order.limit_price,
                            status=order.status.value,
                            fill_timestamp=order.fill_timestamp,
                        )
                    )
                    if self.config.fill_mode == FILL_OPEN:
                        # 当 bar 开盘价撮合（注意：可能与信号使用当 bar 收盘价产生前视）
                        self._execute(order, bars_t, portfolio, accounting)
                    else:  # next_bar
                        nt = self._next_map.get(order.symbol, {}).get(t)
                        if nt is None:
                            order.status = OrderStatus.REJECTED
                            self.logger.debug("订单 %s 无下一 bar，拒绝", order.order_id)
                        else:
                            order.status = OrderStatus.PENDING
                            order.fill_timestamp = nt
                            pending.append(order)

            # 4) 期末盯市（用各标的最近已知收盘价）
            prices = dict(last_prices)
            rec = accounting.mark_to_market(t, portfolio, prices)
            self._emit(
                PortfolioEvent(
                    timestamp=t,
                    cash=rec["cash"],
                    positions=rec["positions"],
                    market_value=rec["market_value"],
                    equity=rec["equity"],
                    available_cash=rec["available_cash"],
                    total_fees=rec["total_fees"],
                )
            )

        return EngineResult(
            equity_curve=accounting.equity_curve(),
            fills=accounting.fills_frame(),
            orders=pd.DataFrame([o.to_dict() for o in orders_log]),
            portfolio=portfolio,
            config=self.config,
        )

    # ------------------------------------------------------------------
    # 内部：订单执行
    # ------------------------------------------------------------------
    def _execute(
        self,
        order: Order,
        bars_t: dict[str, MarketDataEvent],
        portfolio: Portfolio,
        accounting: Accounting,
    ) -> None:
        """撮合单笔订单并应用成交（含资金校验）。"""
        bar = bars_t.get(order.symbol)
        if bar is None:
            order.status = OrderStatus.CANCELLED
            self.logger.debug("订单 %s 在计划时点无行情，取消", order.order_id)
            return

        fill = self.matching.match(order, bar, is_etf=self._is_etf(order.symbol))
        if fill is None:
            order.status = OrderStatus.REJECTED
            self.logger.debug("订单 %s 限价未成交，拒绝", order.order_id)
            return

        if order.side == OrderSide.BUY:
            cash_out = fill.exec_price * fill.quantity + fill.total_fee
            if not portfolio.can_afford(cash_out):
                order.status = OrderStatus.REJECTED
                self.logger.debug("订单 %s 资金不足（需 %.2f），拒绝", order.order_id, cash_out)
                return

        order.status = OrderStatus.FILLED
        portfolio.apply_fill(fill)
        accounting.record_fill(fill)
        self._emit(fill)

    def _create_order(self, sig: SignalEvent, t: pd.Timestamp) -> Order:
        """由信号构造订单对象。"""
        self._order_seq += 1
        order_id = f"ORD-{self._order_seq:06d}"
        side = coerce_side(sig.side)
        quantity = float(sig.quantity)
        if not self.config.allow_fractional:
            quantity = float(int(quantity))
        is_limit = (sig.order_type == "limit") or (sig.limit_price is not None)
        order_type = OrderType.LIMIT if is_limit else OrderType.MARKET
        return Order(
            order_id=order_id,
            symbol=sig.symbol,
            side=side,
            quantity=quantity,
            order_type=order_type,
            limit_price=sig.limit_price,
            timestamp=t,
            status=OrderStatus.CREATED,
        )

    def _is_etf(self, symbol: str) -> bool:
        if self.config.etf_symbols is not None:
            return symbol in set(self.config.etf_symbols)
        return is_etf_symbol(symbol)

    # ------------------------------------------------------------------
    # 内部：数据与时间轴
    # ------------------------------------------------------------------
    def _normalize_data(self, data) -> dict[str, pd.DataFrame]:
        """把输入数据归一化为 {symbol: DataFrame}（宽表 dict 或含 symbol 列的长表）。"""
        if isinstance(data, dict):
            return {sym: self._prepare_symbol_df(sym, df) for sym, df in data.items()}
        if isinstance(data, pd.DataFrame) and sc.COL_SYMBOL in data.columns:
            out = {}
            for sym, sub in data.groupby(sc.COL_SYMBOL):
                out[sym] = self._prepare_symbol_df(sym, sub)
            return out
        raise EngineError("data 需为 dict[symbol, DataFrame] 或含 symbol 列的长表 DataFrame")

    def _prepare_symbol_df(self, symbol: str, df: pd.DataFrame) -> pd.DataFrame:
        """归一化单标的行情表：时间索引、必填列、去重排序、缺失兜底。"""
        df = df.copy()
        if sc.COL_DATETIME in df.columns:
            df = df.set_index(sc.COL_DATETIME)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df = df[~df.index.duplicated(keep="last")].sort_index()

        for c in (sc.COL_OPEN, sc.COL_HIGH, sc.COL_LOW, sc.COL_CLOSE):
            if c not in df.columns:
                df[c] = np.nan
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df[sc.COL_CLOSE] = df[sc.COL_CLOSE].ffill()
        for c in (sc.COL_OPEN, sc.COL_HIGH, sc.COL_LOW):
            df[c] = df[c].fillna(df[sc.COL_CLOSE])
        for c in (sc.COL_VOLUME, sc.COL_AMOUNT):
            if c not in df.columns:
                df[c] = 0.0
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        return df

    def _build_master_index(self) -> pd.DatetimeIndex:
        """多标的时间轴并集（排序、去重）。"""
        if not self.data:
            return pd.DatetimeIndex([])
        return pd.DatetimeIndex(sorted(set().union(*[set(df.index) for df in self.data.values()])))

    def _make_market_event(self, t: pd.Timestamp, symbol: str, row: pd.Series) -> MarketDataEvent:
        return MarketDataEvent(
            timestamp=t,
            symbol=symbol,
            open=float(row[sc.COL_OPEN]),
            high=float(row[sc.COL_HIGH]),
            low=float(row[sc.COL_LOW]),
            close=float(row[sc.COL_CLOSE]),
            volume=float(row[sc.COL_VOLUME]),
            amount=float(row.get(sc.COL_AMOUNT, 0.0)),
        )

    def _emit_market_events(self, bars_t: dict[str, MarketDataEvent]) -> None:
        for bar in bars_t.values():
            self._emit(bar)

    # ------------------------------------------------------------------
    # 日志
    # ------------------------------------------------------------------
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(f"PyQuantLab.engine.{id(self)}")
        logger.setLevel(getattr(logging, self.config.log_level.upper(), logging.INFO))
        logger.propagate = False
        if not logger.handlers:
            handler: logging.Handler
            if self.config.log_file:
                handler = logging.FileHandler(self.config.log_file, encoding="utf-8")
            else:
                handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            logger.addHandler(handler)
        return logger
