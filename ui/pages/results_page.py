"""Page 4: Results & Risk Analysis — detailed performance breakdown."""

import streamlit as st

from backtest.report import BacktestReport
from backtest.risk import RiskAnalyzer
from ui.components.charts import (
    equity_curve_chart,
    monthly_returns_heatmap,
    returns_distribution,
)
from ui.session_state import KEYS


def render() -> None:
    st.title("Results & Risk Analysis")
    st.markdown("Detailed performance analysis, risk metrics, and trade breakdown.")

    result = st.session_state.get(KEYS["backtest_result"])
    if result is None:
        st.info("Run a backtest first to see results here.")
        return

    m = result.metrics
    report = BacktestReport(result)
    returns = result.equity_curve["returns"]

    tabs = st.tabs(
        [
            "Overview",
            "Equity Curve",
            "Drawdown Analysis",
            "Trade Analysis",
            "Risk Metrics",
            "Monthly Returns",
        ]
    )

    with tabs[0]:
        st.subheader("Performance Summary")
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(report.metrics_table(), use_container_width=True, hide_index=True)
        with col2:
            st.markdown("**Equity Curve**")
            fig = equity_curve_chart(result.equity_curve)
            st.plotly_chart(fig, use_container_width=True)

    with tabs[1]:
        st.subheader("Equity Curve & Returns")
        fig = equity_curve_chart(result.equity_curve)
        st.plotly_chart(fig, use_container_width=True)
        st.subheader("Returns Distribution")
        fig2 = returns_distribution(returns)
        st.plotly_chart(fig2, use_container_width=True)

    with tabs[2]:
        st.subheader("Drawdown Analysis")
        dd_df = RiskAnalyzer.drawdown_analysis(result.equity_curve["equity"])
        if not dd_df.empty:
            st.dataframe(
                dd_df.style.format(
                    {
                        "max_drawdown": "{:.2%}",
                        "recovery": "{:.2%}",
                    }
                ),
                use_container_width=True,
            )
            st.caption(
                f"**Maximum Drawdown:** {m.max_drawdown * 100:.2f}% | "
                f"**Longest DD Period:** {m.max_drawdown_duration} days"
            )
        else:
            st.info("No drawdown periods detected.")

        # Rolling metrics
        st.subheader("Rolling Sharpe Ratio (252-day)")
        roll_sharpe = RiskAnalyzer.rolling_sharpe(returns)
        st.line_chart(roll_sharpe.dropna(), use_container_width=True)

    with tabs[3]:
        st.subheader("Trade Analysis")
        trades = result.trades
        if not trades.empty:
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Total Trades", m.total_trades)
            col_b.metric("Winning Trades", m.winning_trades)
            col_c.metric("Losing Trades", m.losing_trades)

            col_d, col_e, col_f = st.columns(3)
            col_d.metric("Win Rate", f"{m.win_rate * 100:.1f}%")
            col_e.metric("Avg Win", f"{m.avg_win * 100:.2f}%")
            col_f.metric("Avg Loss", f"{m.avg_loss * 100:.2f}%")

            st.metric("Profit Factor", f"{m.profit_factor:.2f}")

            st.subheader("Trade List")
            display_trades = trades.copy()
            for col in ["pnl_pct"]:
                if col in display_trades.columns:
                    display_trades[col] = display_trades[col].apply(lambda x: f"{x * 100:.2f}%")
            st.dataframe(display_trades, use_container_width=True)
        else:
            st.info("No trades were executed during this backtest.")

    with tabs[4]:
        st.subheader("Risk Metrics")
        tail = RiskAnalyzer.tail_risk(returns)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Skewness", f"{tail['skewness']:.4f}")
        col2.metric("Kurtosis", f"{tail['kurtosis']:.4f}")
        col3.metric("VaR 99%", f"{tail['var_99'] * 100:.2f}%")
        col4.metric("Max Daily Loss", f"{tail['max_daily_loss'] * 100:.2f}%")

        st.divider()
        st.subheader("Stress Test")
        stress_df = RiskAnalyzer.stress_test(returns)
        st.dataframe(
            stress_df.style.format({"shock": "{:.0%}", "total_return": "{:.2%}"}),
            use_container_width=True,
            hide_index=True,
        )

        st.divider()
        st.subheader("Rolling Volatility (21-day)")
        roll_vol = RiskAnalyzer.rolling_volatility(returns)
        st.line_chart(roll_vol.dropna(), use_container_width=True)

    with tabs[5]:
        st.subheader("Monthly Returns Heatmap")
        fig = monthly_returns_heatmap(result.equity_curve)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough data for monthly returns heatmap.")
