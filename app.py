"""PyQuantLab — Quantitative Trading Platform with Streamlit UI."""

import streamlit as st

from ui.pages import (
    backtest_page,
    data_page,
    portfolio_page,
    results_page,
    strategy_page,
)

st.set_page_config(
    page_title="PyQuantLab",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAGE_OPTIONS = {
    "📥  Data Management": data_page.render,
    "📊  Strategy Config": strategy_page.render,
    "▶️   Run Backtest": backtest_page.render,
    "📈  Results & Risk": results_page.render,
    "🏦  Portfolio Simulator": portfolio_page.render,
}

with st.sidebar:
    st.title("PyQuantLab")
    st.markdown("Quantitative Trading Platform")
    st.markdown("---")
    page = st.radio(
        "Navigate to",
        list(PAGE_OPTIONS.keys()),
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption(
        "Workflow: Load Data → Configure Strategy → "
        "Run Backtest → Analyze Results"
    )

PAGE_OPTIONS[page]()
