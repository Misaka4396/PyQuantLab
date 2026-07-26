"""Page 2: Strategy Configuration — select strategy, tune parameters, preview signals."""

import pandas as pd
import streamlit as st

from strategy.registry import registry
from ui.components.charts import price_chart
from ui.session_state import KEYS


def render() -> None:
    st.title("Strategy Configuration")
    st.markdown("Select a trading strategy, adjust its parameters, and preview the signals.")

    data = st.session_state.get(KEYS["data_df"])
    if data is None:
        st.warning("Please load data first in the Data Management page.")
        return

    strategy_names = registry.list_names()
    selected_name = st.selectbox(
        "Strategy", strategy_names,
        index=strategy_names.index(st.session_state.get(KEYS["strategy_name"], strategy_names[0])),
    )
    strategy_cls = registry.get(selected_name)

    # Show description
    with st.expander("Strategy Description", expanded=False):
        st.markdown(strategy_cls.get_description())

    # Auto-generate parameter controls from param_spec
    spec = strategy_cls.get_param_spec()
    params = {}
    cols = st.columns(min(len(spec), 3))
    for i, (key, info) in enumerate(spec.items()):
        col_idx = i % len(cols)
        with cols[col_idx]:
            if info["type"] == "int":
                params[key] = st.slider(
                    info["label"], info["min"], info["max"],
                    value=info["default"], step=info["step"], help=info["help"],
                    key=f"param_{key}",
                )
            elif info["type"] == "choice":
                params[key] = st.selectbox(
                    info["label"], info["choices"],
                    index=info["choices"].index(info["default"]),
                    help=info["help"], key=f"param_{key}",
                )

    # Create strategy instance and store
    strategy = strategy_cls(params)
    st.session_state[KEYS["strategy_name"]] = selected_name
    st.session_state[KEYS["strategy_params"]] = params

    st.divider()

    # Signal preview
    st.subheader("Signal Preview")
    signals = strategy.generate_signals(data)

    if isinstance(data.columns, pd.MultiIndex):
        ticker = data.columns.levels[0][0]
    else:
        ticker = "Stock"

    signals_df = pd.DataFrame({"signal": signals}, index=data.index)

    # Show last N days
    preview_days = st.slider("Preview days", 30, min(500, len(data)), 200, 30)
    preview = signals_df.tail(preview_days)

    buy_count = int((preview["signal"] == 1).sum())
    sell_count = int((preview["signal"] == -1).sum())
    col_b, col_s, col_t = st.columns(3)
    col_b.metric("Buy Signals", buy_count)
    col_s.metric("Sell Signals", sell_count)
    col_t.metric("Total Trading Days", len(preview))

    fig = price_chart(data.tail(preview_days), ticker, signals_df.tail(preview_days))
    st.plotly_chart(fig, use_container_width=True)
