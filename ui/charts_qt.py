"""Matplotlib chart factories for PyQt5 embedding."""

from typing import Optional

import matplotlib
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

matplotlib.use("Qt5Agg")

# Microsoft YaHei font setup for Chinese rendering
FONT_SIZE = 12
_font_configured = False
for f in fm.findSystemFonts():
    if 'msyh' in f.lower() or 'yahei' in f.lower():
        try:
            fm.fontManager.addfont(f)
            matplotlib.rcParams['font.family'] = 'Microsoft YaHei'
            _font_configured = True
        except Exception:
            pass
        break
matplotlib.rcParams['font.size'] = FONT_SIZE
matplotlib.rcParams['axes.unicode_minus'] = False


def _make_fig_ax(rows=1, cols=1, figsize=(8, 5)):
    fig = Figure(figsize=figsize, dpi=100)
    if rows == 1 and cols == 1:
        ax = fig.add_subplot(111)
        return fig, ax
    axes = fig.subplots(rows, cols)
    return fig, axes


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, fig=None, figsize=(8, 5)):
        self.fig = fig or Figure(figsize=figsize, dpi=100)
        super().__init__(self.fig)
        self.setMinimumHeight(300)


def equity_curve_chart(equity_curve: pd.DataFrame) -> MplCanvas:
    fig, (ax1, ax2) = _make_fig_ax(rows=2, cols=1, figsize=(10, 5.5))
    fig.subplots_adjust(hspace=0.3)

    equity = equity_curve["equity"]
    ax1.plot(equity.index, equity.values, color="#26a69a", linewidth=1.5)
    ax1.set_ylabel("权益")
    ax1.set_title("权益曲线")
    ax1.grid(True, alpha=0.3)

    peak = equity.expanding().max()
    dd = (equity / peak - 1) * 100
    ax2.fill_between(range(len(dd)), dd.values, 0, color="#ef5350", alpha=0.3)
    ax2.plot(range(len(dd)), dd.values, color="#ef5350", linewidth=0.8)
    ax2.set_ylabel("回撤 %")
    ax2.grid(True, alpha=0.3)
    return MplCanvas(fig)


def price_chart(data: pd.DataFrame, signals: Optional[pd.DataFrame] = None) -> MplCanvas:
    fig, ax = _make_fig_ax(figsize=(10, 4.5))

    if isinstance(data.columns, pd.MultiIndex):
        ticker = data.columns.levels[0][0]
        close = data[(ticker, "Close")]
    elif "Close" in data.columns:
        close = data["Close"]
    else:
        close = data.iloc[:, 0]

    ax.plot(close.index, close.values, color="#42a5f5", linewidth=1.2, label="收盘价")
    ax.set_ylabel("价格")
    ax.set_title("价格走势图")
    ax.grid(True, alpha=0.3)

    if signals is not None and len(signals) > 0:
        buy_idx = signals[signals["signal"] == 1].index
        sell_idx = signals[signals["signal"] == -1].index
        close_aligned = close.reindex(signals.index)
        buy_aligned = buy_idx.intersection(close_aligned.dropna().index)
        sell_aligned = sell_idx.intersection(close_aligned.dropna().index)
        if len(buy_aligned) > 0:
            ax.scatter(buy_aligned, close_aligned.loc[buy_aligned], marker="^",
                       color="green", s=80, zorder=5, label="买入")
        if len(sell_aligned) > 0:
            ax.scatter(sell_aligned, close_aligned.loc[sell_aligned], marker="v",
                       color="red", s=80, zorder=5, label="卖出")

    ax.legend(loc="upper left")
    ax.tick_params(axis="x", rotation=30)
    return MplCanvas(fig)


def returns_histogram(returns: pd.Series) -> MplCanvas:
    fig, ax = _make_fig_ax(figsize=(8, 4))
    clean = returns.dropna()
    ax.hist(clean, bins=60, color="#42a5f5", alpha=0.7, density=True)
    x = np.linspace(clean.min(), clean.max(), 200)
    y = (1 / (clean.std() * np.sqrt(2 * np.pi))) * np.exp(
        -(x - clean.mean()) ** 2 / (2 * clean.std() ** 2)
    )
    ax.plot(x, y, color="orange", linewidth=2, label="正态拟合")
    ax.set_title("收益率分布")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return MplCanvas(fig)


def efficient_frontier_chart(frontier: pd.DataFrame, optimal=None) -> MplCanvas:
    fig, ax = _make_fig_ax(figsize=(8, 5))
    ax.plot(frontier["volatility"], frontier["return"], "o-",
            color="steelblue", markersize=4, alpha=0.7, label="有效前沿")
    if optimal is not None:
        ax.scatter([optimal.expected_volatility], [optimal.expected_return],
                   marker="*", s=300, color="gold", edgecolors="darkgoldenrod",
                   linewidths=1.5, zorder=5, label="最大夏普比率")
    ax.set_xlabel("年化波动率")
    ax.set_ylabel("年化收益率")
    ax.set_title("有效前沿")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return MplCanvas(fig)


def monte_carlo_chart(percentiles: pd.DataFrame) -> MplCanvas:
    fig, ax = _make_fig_ax(figsize=(8, 4.5))
    x = range(len(percentiles))
    ax.fill_between(x, percentiles["p5"], percentiles["p95"],
                    color="#42a5f5", alpha=0.1, label="90% 置信区间")
    ax.fill_between(x, percentiles["p25"], percentiles["p75"],
                    color="#42a5f5", alpha=0.2, label="50% 置信区间")
    ax.plot(x, percentiles["p50"], color="#26a69a", linewidth=2, label="中位数")
    ax.set_ylabel("组合价值")
    ax.set_title("蒙特卡洛模拟")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    return MplCanvas(fig)


def rolling_chart(series: pd.Series, title: str = "", color: str = "#42a5f5") -> MplCanvas:
    fig, ax = _make_fig_ax(figsize=(8, 3.5))
    s = series.dropna()
    ax.plot(s.index, s.values, color=color, linewidth=1.2)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="x", rotation=30)
    return MplCanvas(fig)


def live_price_chart(prices: pd.DataFrame) -> MplCanvas:
    fig, ax = _make_fig_ax(figsize=(10, 4))
    colors = ["#42a5f5", "#ef5350", "#26a69a", "#ffa726", "#ab47bc"]
    for i, col in enumerate(prices.columns):
        s = prices[col].dropna()
        if len(s) > 0:
            ax.plot(s.index, s.values, color=colors[i % len(colors)],
                    linewidth=1.5, label=col)
    ax.set_title("日内价格走势")
    ax.set_ylabel("价格 ($)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="x", rotation=30)
    return MplCanvas(fig)
