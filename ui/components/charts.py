"""Reusable Plotly chart factories for the UI."""

from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def price_chart(
    data: pd.DataFrame,
    ticker: str,
    signals: Optional[pd.DataFrame] = None,
) -> go.Figure:
    """OHLC candlestick chart with optional buy/sell markers."""
    if isinstance(data.columns, pd.MultiIndex):
        close = data[(ticker, "Close")]
        try:
            o = data[(ticker, "Open")]
            h = data[(ticker, "High")]
            l = data[(ticker, "Low")]
            c = data[(ticker, "Close")]
        except KeyError:
            o = h = l = c = close
    else:
        try:
            o, h, l, c = data["Open"], data["High"], data["Low"], data["Close"]
        except KeyError:
            o = h = l = c = data["Close"]

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=data.index, open=o, high=h, low=l, close=c,
        name=ticker, increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
    ))

    if signals is not None and len(signals) > 0:
        buy_dates = signals[signals["signal"] == 1].index
        sell_dates = signals[signals["signal"] == -1].index
        if len(buy_dates) > 0:
            fig.add_trace(go.Scatter(
                x=buy_dates, y=c.loc[buy_dates] * 0.98,
                mode="markers", marker=dict(symbol="triangle-up", size=12, color="green"),
                name="Buy Signal",
            ))
        if len(sell_dates) > 0:
            fig.add_trace(go.Scatter(
                x=sell_dates, y=c.loc[sell_dates] * 1.02,
                mode="markers", marker=dict(symbol="triangle-down", size=12, color="red"),
                name="Sell Signal",
            ))

    fig.update_layout(height=500, xaxis_rangeslider_visible=False, template="plotly_dark",
                      margin=dict(l=0, r=0, t=30, b=0))
    return fig


def equity_curve_chart(equity_curve: pd.DataFrame, benchmark: Optional[pd.Series] = None) -> go.Figure:
    """Interactive equity curve with drawdown subplot."""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
    )

    equity = equity_curve["equity"]
    fig.add_trace(go.Scatter(
        x=equity.index, y=equity.values,
        mode="lines", name="Equity Curve", line=dict(color="#26a69a", width=2),
    ), row=1, col=1)

    if benchmark is not None:
        bench_norm = benchmark / benchmark.iloc[0] * equity.iloc[0]
        fig.add_trace(go.Scatter(
            x=bench_norm.index, y=bench_norm.values,
            mode="lines", name="Benchmark", line=dict(color="gray", width=1, dash="dash"),
        ), row=1, col=1)

    # Drawdown
    peak = equity.expanding().max()
    dd = (equity / peak - 1) * 100
    fig.add_trace(go.Scatter(
        x=dd.index, y=dd.values,
        mode="lines", name="Drawdown %", fill="tozeroy",
        line=dict(color="#ef5350", width=1), fillcolor="rgba(239,83,80,0.2)",
    ), row=2, col=1)

    fig.update_layout(height=600, template="plotly_dark", margin=dict(l=0, r=0, t=30, b=0),
                      showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02))
    fig.update_yaxes(title_text="Equity", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown %", row=2, col=1)
    return fig


def returns_distribution(returns: pd.Series) -> go.Figure:
    """Histogram of returns with normal overlay."""
    clean = returns.dropna()
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=clean, nbinsx=50, name="Returns",
                               marker_color="#42a5f5", opacity=0.7, histnorm="probability density"))
    x = np.linspace(clean.min(), clean.max(), 200)
    y = (1 / (clean.std() * np.sqrt(2 * np.pi))) * np.exp(
        -(x - clean.mean()) ** 2 / (2 * clean.std() ** 2)
    )
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="Normal Fit",
                             line=dict(color="orange", width=2)))
    fig.update_layout(height=350, template="plotly_dark", margin=dict(l=0, r=0, t=30, b=0),
                      bargap=0.05)
    return fig


def efficient_frontier_chart(frontier: pd.DataFrame, optimal=None) -> go.Figure:
    """Efficient frontier scatter with optimal portfolio marker."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=frontier["volatility"], y=frontier["return"],
        mode="markers+lines", name="Efficient Frontier",
        marker=dict(size=6, color="lightblue"), line=dict(color="steelblue", width=1),
    ))
    if optimal is not None:
        fig.add_trace(go.Scatter(
            x=[optimal.expected_volatility], y=[optimal.expected_return],
            mode="markers", name="Max Sharpe",
            marker=dict(symbol="star", size=20, color="gold", line=dict(width=2, color="darkgoldenrod")),
        ))
    fig.update_layout(
        height=450, template="plotly_dark", margin=dict(l=0, r=0, t=30, b=0),
        xaxis_title="Annualized Volatility", yaxis_title="Annualized Return",
    )
    return fig


def monte_carlo_chart(percentiles: pd.DataFrame) -> go.Figure:
    """Monte Carlo fan chart with percentile bands."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=percentiles.index, y=percentiles["p95"],
        mode="lines", name="P95", line=dict(width=0), showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=percentiles.index, y=percentiles["p5"],
        mode="lines", name="P5", fill="tonexty",
        fillcolor="rgba(66,165,245,0.15)", line=dict(width=0), showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=percentiles.index, y=percentiles["p75"],
        mode="lines", name="P75", line=dict(width=0), showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=percentiles.index, y=percentiles["p25"],
        mode="lines", name="P25", fill="tonexty",
        fillcolor="rgba(66,165,245,0.25)", line=dict(width=0), showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=percentiles.index, y=percentiles["p50"],
        mode="lines", name="Median", line=dict(color="#26a69a", width=2),
    ))
    fig.update_layout(height=450, template="plotly_dark", margin=dict(l=0, r=0, t=30, b=0),
                      yaxis_title="Portfolio Value")
    return fig


def correlation_heatmap_chart(corr: pd.DataFrame) -> go.Figure:
    """Correlation matrix heatmap."""
    fig = go.Figure(data=go.Heatmap(
        z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
        zmin=-1, zmax=1, colorscale="RdBu_r", text=np.round(corr.values, 2),
        texttemplate="%{text}", textfont=dict(size=12),
    ))
    fig.update_layout(height=400, template="plotly_dark", margin=dict(l=0, r=0, t=30, b=0))
    return fig


def monthly_returns_heatmap(equity_curve: pd.DataFrame) -> Optional[go.Figure]:
    """Monthly returns heatmap."""
    if "returns" not in equity_curve.columns:
        return None
    returns = equity_curve["returns"]
    if len(returns) < 21:
        return None
    monthly = returns.groupby([returns.index.year, returns.index.month]).apply(
        lambda x: (1 + x).prod() - 1
    ).unstack()
    fig = go.Figure(data=go.Heatmap(
        z=monthly.values * 100, x=[f"M{m}" for m in monthly.columns],
        y=monthly.index, zmin=-15, zmax=15, colorscale="RdYlGn",
        text=[[f"{v:.1f}%" if not np.isnan(v) else "" for v in row] for row in monthly.values],
        texttemplate="%{text}", textfont=dict(size=10),
    ))
    fig.update_layout(height=400, template="plotly_dark", margin=dict(l=0, r=0, t=30, b=0),
                      xaxis_title="Month", yaxis_title="Year")
    return fig
