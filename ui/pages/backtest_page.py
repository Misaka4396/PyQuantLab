"""Page 3: Backtest Runner — configure and execute backtests."""

import streamlit as st

from backtest.engine import BacktestEngine
from backtest.report import BacktestReport
from config import DEFAULT_COMMISSION, DEFAULT_INITIAL_CAPITAL, DEFAULT_SLIPPAGE
from core.types import BacktestConfig
from strategy.registry import registry
from ui.components.charts import equity_curve_chart
from ui.session_state import KEYS


def render() -> None:
    st.title("Run Backtest")
    st.markdown("Configure backtest parameters and evaluate your strategy.")

    data = st.session_state.get(KEYS["data_df"])
    if data is None:
        st.warning("Please load data first in the Data Management page.")
        return

    strategy_name = st.session_state.get(KEYS["strategy_name"])
    strategy_params = st.session_state.get(KEYS["strategy_params"], {})
    if not strategy_name:
        st.warning("Please select a strategy first in the Strategy Config page.")
        return

    # Config
    col1, col2, col3 = st.columns(3)
    with col1:
        capital = st.number_input(
            "Initial Capital ($)",
            min_value=1000,
            max_value=10_000_000,
            value=DEFAULT_INITIAL_CAPITAL,
            step=10000,
            format="%d",
        )
    with col2:
        commission = (
            st.number_input(
                "Commission (%)",
                min_value=0.0,
                max_value=1.0,
                value=DEFAULT_COMMISSION * 100,
                step=0.01,
                format="%.3f",
            )
            / 100
        )
    with col3:
        slippage = (
            st.number_input(
                "Slippage (%)",
                min_value=0.0,
                max_value=1.0,
                value=DEFAULT_SLIPPAGE * 100,
                step=0.01,
                format="%.3f",
            )
            / 100
        )

    # Strategy info
    st.caption(f"**Strategy:** {strategy_name} | **Params:** {strategy_params}")

    col_run, col_compare = st.columns([2, 1])
    with col_run:
        run_clicked = st.button("Run Backtest", type="primary", use_container_width=True)
    with col_compare:
        compare_clicked = st.button("Compare All Strategies", use_container_width=True)

    if run_clicked or compare_clicked:
        config = BacktestConfig(initial_capital=capital, commission=commission, slippage=slippage)
        engine = BacktestEngine(config)

        if compare_clicked:
            all_results = {}
            progress = st.progress(0)
            names = registry.list_names()
            for i, name in enumerate(names):
                spec = registry.get_param_spec(name)
                default_params = {k: v["default"] for k, v in spec.items()}
                strategy = registry.create(name, **default_params)
                result = engine.run(data, strategy)
                all_results[name] = result
                progress.progress((i + 1) / len(names))
            st.session_state[KEYS["backtest_results_dict"]] = all_results
            st.success(f"Compared {len(all_results)} strategies.")
        else:
            strategy = registry.create(strategy_name, **strategy_params)
            with st.spinner("Running backtest..."):
                result = engine.run(data, strategy)
                st.session_state[KEYS["backtest_result"]] = result
                st.session_state[KEYS["backtest_results_dict"]] = {strategy_name: result}
            st.success("Backtest complete!")

    # Display results
    result = st.session_state.get(KEYS["backtest_result"])
    results_dict = st.session_state.get(KEYS["backtest_results_dict"], {})

    if result is not None:
        st.divider()
        BacktestReport(result)

        # KPI cards
        m = result.metrics
        cols_kpi = st.columns(6)
        kpis = [
            ("Total Return", f"{m.total_return * 100:.2f}%"),
            ("Ann. Return", f"{m.annualized_return * 100:.2f}%"),
            ("Sharpe", f"{m.sharpe_ratio:.2f}"),
            ("Max DD", f"{m.max_drawdown * 100:.2f}%"),
            ("Win Rate", f"{m.win_rate * 100:.1f}%"),
            ("Trades", str(m.total_trades)),
        ]
        for col, (label, value) in zip(cols_kpi, kpis, strict=False):
            col.metric(label, value)

        # Equity curve
        fig = equity_curve_chart(result.equity_curve)
        st.plotly_chart(fig, use_container_width=True)

        # Comparison table
        if len(results_dict) > 1:
            st.subheader("Strategy Comparison")
            comp_rows = []
            for name, r in results_dict.items():
                rm = r.metrics
                comp_rows.append(
                    {
                        "Strategy": name,
                        "Total Return": f"{rm.total_return * 100:.2f}%",
                        "Sharpe": f"{rm.sharpe_ratio:.2f}",
                        "Max DD": f"{rm.max_drawdown * 100:.2f}%",
                        "Win Rate": f"{rm.win_rate * 100:.1f}%",
                        "Trades": rm.total_trades,
                    }
                )
            import pandas as pd

            st.dataframe(
                pd.DataFrame(comp_rows).set_index("Strategy"),
                use_container_width=True,
            )
