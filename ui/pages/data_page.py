"""Page 1: Data Management — download, view, and manage stock data."""

from datetime import datetime, timedelta

import pandas as pd
import streamlit as st

from config import DEFAULT_TICKERS
from core.exceptions import DataNotFoundError
from data.manager import DataManager
from ui.components.charts import price_chart, returns_distribution
from ui.session_state import KEYS


def render() -> None:
    st.title("Data Management")
    st.markdown("Download historical stock data, manage cache, and inspect data quality.")

    dm = DataManager()

    col1, col2, col3 = st.columns([3, 2, 1])
    with col1:
        tickers_str = st.text_input(
            "Stock Tickers (comma-separated)",
            value=",".join(DEFAULT_TICKERS),
            help="e.g. AAPL, MSFT, GOOGL",
        )
    with col2:
        end_date = st.date_input("End Date", value=datetime.now().date())
        start_date = st.date_input(
            "Start Date", value=datetime.now().date() - timedelta(days=365 * 3)
        )
    with col3:
        interval = st.selectbox("Interval", ["1d", "1wk", "1mo"], index=0)
        force = st.checkbox("Force Download", value=False)

    if st.button("Load Data", type="primary", use_container_width=True):
        tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]
        if not tickers:
            st.warning("Please enter at least one ticker.")
            return
        with st.spinner(f"Loading data for {', '.join(tickers)}..."):
            try:
                data = dm.get_data(
                    tickers,
                    start=str(start_date),
                    end=str(end_date),
                    interval=interval,
                    force_download=force,
                )
                st.session_state[KEYS["data_df"]] = data
                st.success(f"Loaded {data.shape[0]} rows x {data.shape[1]} columns")
            except DataNotFoundError as e:
                st.error(str(e))

    data = st.session_state.get(KEYS["data_df"])
    if data is None:
        st.info("No data loaded. Enter tickers and click 'Load Data' to begin.")
        return

    st.divider()

    # Data overview and cache info
    col_a, col_b = st.columns([3, 1])
    with col_a:
        overview = dm.get_data_overview(data)
        st.caption(
            f"**{overview['shape'][0]}** rows | "
            f"**{overview['start_date']}** to **{overview['end_date']}** | "
            f"**{overview.get('missing_values', 0)}** missing values"
        )
    with col_b:
        stats = dm.cache.get_stats()
        st.caption(f"Cache: **{stats['total_files']}** files, **{stats['total_size_mb']}** MB")

    tab1, tab2, tab3 = st.tabs(["Data Preview", "Price Chart", "Returns Distribution"])

    with tab1:
        if isinstance(data.columns, pd.MultiIndex):
            ticker = data.columns.levels[0][0]
            display_data = data[ticker]
        else:
            display_data = data
        st.dataframe(display_data.tail(500), use_container_width=True, height=400)

    with tab2:
        if isinstance(data.columns, pd.MultiIndex):
            ticker = data.columns.levels[0][0]
        else:
            ticker = "Stock"
        fig = price_chart(data, ticker)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        if isinstance(data.columns, pd.MultiIndex):
            ticker = data.columns.levels[0][0]
            close = data[(ticker, "Close")]
        else:
            close = data["Close"]
        ret = close.pct_change().dropna()
        fig = returns_distribution(ret)
        st.plotly_chart(fig, use_container_width=True)
