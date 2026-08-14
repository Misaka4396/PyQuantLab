"""Page 5: Portfolio Simulation — optimization and Monte Carlo simulation."""

import pandas as pd
import streamlit as st

from config import MC_SIMULATIONS_DEFAULT
from portfolio.optimizer import PortfolioOptimizer
from portfolio.simulation import PortfolioSimulation
from ui.components.charts import (
    correlation_heatmap_chart,
    efficient_frontier_chart,
    monte_carlo_chart,
)
from ui.session_state import KEYS


def render() -> None:
    st.title("Portfolio Simulator")
    st.markdown("Portfolio optimization (mean-variance / risk parity) and Monte Carlo simulation.")

    data = st.session_state.get(KEYS["data_df"])
    if data is None:
        st.warning("Please load data first in the Data Management page.")
        return

    # Extract close prices for all tickers
    if isinstance(data.columns, pd.MultiIndex):
        tickers = list(data.columns.levels[0])
        closes = pd.DataFrame({t: data[(t, "Close")] for t in tickers})
    else:
        tickers = ["Stock"]
        closes = pd.DataFrame({"Stock": data["Close"]})

    if len(tickers) < 2:
        st.warning("Please load at least 2 tickers for portfolio analysis.")
        return

    returns = closes.pct_change().dropna()

    # Optimization
    st.subheader("Portfolio Optimization")
    opt_method = st.selectbox(
        "Optimization Method",
        ["Max Sharpe Ratio", "Minimum Variance", "Risk Parity", "Equal Weight"],
    )

    optimizer = PortfolioOptimizer(returns)

    col_opt, col_chart = st.columns([1, 2])

    with col_opt:
        if st.button("Run Optimization", type="primary", use_container_width=True):
            with st.spinner("Optimizing..."):
                if opt_method == "Max Sharpe Ratio":
                    pw = optimizer.max_sharpe()
                elif opt_method == "Minimum Variance":
                    pw = optimizer.min_variance()
                elif opt_method == "Risk Parity":
                    pw = optimizer.risk_parity()
                else:
                    pw = optimizer.equal_weight()

                st.session_state[KEYS["portfolio_weights"]] = pw
                st.success("Optimization complete!")

        pw = st.session_state.get(KEYS["portfolio_weights"])
        if pw is not None:
            st.markdown("### Optimal Weights")
            w_df = pd.DataFrame(
                {
                    "Asset": pw.assets,
                    "Weight": pw.weights,
                }
            )
            w_df["Weight %"] = w_df["Weight"].apply(lambda x: f"{x * 100:.1f}%")
            st.dataframe(w_df.set_index("Asset"), use_container_width=True)

            st.metric("Expected Return", f"{pw.expected_return * 100:.2f}%")
            st.metric("Expected Volatility", f"{pw.expected_volatility * 100:.2f}%")
            st.metric("Sharpe Ratio", f"{pw.sharpe_ratio:.2f}")

    with col_chart:
        # Efficient frontier
        st.markdown("### Efficient Frontier")
        with st.spinner("Computing efficient frontier..."):
            frontier = optimizer.efficient_frontier(30)
            if not frontier.empty:
                optimal = st.session_state.get(KEYS["portfolio_weights"])
                fig = efficient_frontier_chart(frontier, optimal)
                st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Monte Carlo
    st.subheader("Monte Carlo Simulation")
    pw_mc = st.session_state.get(KEYS["portfolio_weights"])
    if pw_mc is None:
        st.info("Run optimization first to enable Monte Carlo simulation.")
        return

    col_mc1, col_mc2, col_mc3 = st.columns(3)
    with col_mc1:
        sims = st.number_input("Simulations", 100, 10_000, MC_SIMULATIONS_DEFAULT, 100)
    with col_mc2:
        days = st.number_input("Horizon (days)", 21, 1260, 252, 21)
    with col_mc3:
        initial = st.number_input("Initial Value ($)", 1000, 10_000_000, 100_000, 10000)

    if st.button("Run Monte Carlo", type="primary", use_container_width=True):
        with st.spinner(f"Running {sims} simulations..."):
            ps = PortfolioSimulation(returns, num_simulations=sims)
            simulation = ps.run(pw_mc.weights, initial_value=initial, days=days)
            percentiles = ps.compute_path_percentiles(simulation)
            stats = ps.compute_terminal_stats(simulation)

            st.session_state["mc_percentiles"] = percentiles
            st.session_state["mc_stats"] = stats
            st.session_state["mc_simulation"] = simulation

    mc_percentiles = st.session_state.get("mc_percentiles")
    if mc_percentiles is not None:
        st.markdown("### Simulation Results")
        fig = monte_carlo_chart(mc_percentiles)
        st.plotly_chart(fig, use_container_width=True)

        stats = st.session_state.get("mc_stats", {})
        cols = st.columns(6)
        cols[0].metric("Mean", f"${stats.get('mean', 0):,.0f}")
        cols[1].metric("Median", f"${stats.get('median', 0):,.0f}")
        cols[2].metric("Std", f"${stats.get('std', 0):,.0f}")
        cols[3].metric("P5", f"${stats.get('p5', 0):,.0f}")
        cols[4].metric("P95", f"${stats.get('p95', 0):,.0f}")
        cols[5].metric("Min/Max", f"${stats.get('min', 0):,.0f} / ${stats.get('max', 0):,.0f}")

    # Correlation
    st.divider()
    st.subheader("Asset Correlations")
    if len(tickers) >= 2:
        corr = returns.corr()
        fig_corr = correlation_heatmap_chart(corr)
        st.plotly_chart(fig_corr, use_container_width=True)
