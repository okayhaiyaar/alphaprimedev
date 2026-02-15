"""
============================================================
ALPHA-PRIME v2.0 - War Room Dashboard
============================================================
Module 4: Interactive Streamlit command center.

Features:
- Live signal feed with confidence scores
- Interactive candlestick charts with indicators
- Portfolio management and trade execution
- Performance analytics and equity curves
- Intelligence viewer (SEC, news, sentiment)
- Manual override controls

Usage:
    streamlit run app.py

    # Custom port
    streamlit run app.py --server.port 8502

    # Headless mode (production)
    streamlit run app.py --server.headless true

Design Philosophy:
- Dark Bloomberg Terminal aesthetic
- Real-time updates (auto-refresh)
- Intuitive navigation
- Mobile-responsive (where possible)

Navigation:
1. 🏠 Home: Live signals + main chart
2. 💼 Portfolio: Positions, P&L, equity curve
3. 📊 Performance: Metrics, trade history
4. 🔬 Research: Deep intelligence viewer
5. ⚙️ Execute: Manual trade interface
6. 🛠️ Settings: Configuration

============================================================
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import platform
from typing import Dict, List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from alerts import send_discord_alert
from brain import consult_oracle
from alphaprime import __version__
from alphaprime.cli import collect_doctor_info
from config import get_logger, get_run_id, get_settings
from data_engine import calculate_hard_technicals, get_market_data
from portfolio import PaperTrader
from research_engine import get_god_tier_intel
from scheduler import scan_market_movers

logger = get_logger(__name__)
settings = get_settings()

st.set_page_config(
    page_title="ALPHA-PRIME War Room",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ──────────────────────────────────────────────────────────
# CUSTOM STYLING (Bloomberg Terminal Theme)
# ──────────────────────────────────────────────────────────


def inject_custom_css() -> None:
    """Inject custom CSS for dark Bloomberg-style aesthetic."""
    st.markdown(
        """
    <style>
    :root {
        --bg-primary: #0e1117;
        --bg-secondary: #1a1d24;
        --text-primary: #e6e6e6;
        --text-secondary: #a8a8a8;
        --accent-green: #00ff00;
        --accent-red: #ff0000;
        --accent-blue: #00aaff;
    }

    .main {
        background-color: var(--bg-primary);
        color: var(--text-primary);
    }

    section[data-testid="stSidebar"] {
        background-color: var(--bg-secondary);
        border-right: 1px solid #333;
    }

    div[data-testid="stMetricValue"] {
        font-size: 1.6rem;
        font-weight: 700;
        font-family: "Courier New", monospace;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background-color: var(--bg-secondary);
    }

    .stTabs [data-baseweb="tab"] {
        color: var(--text-secondary);
        font-weight: 600;
    }

    .stTabs [aria-selected="true"] {
        color: var(--accent-blue);
        border-bottom-color: var(--accent-blue);
    }

    .stButton > button {
        background-color: var(--bg-secondary);
        color: var(--text-primary);
        border: 1px solid #444;
        font-weight: 600;
    }

    .stButton > button:hover {
        border-color: var(--accent-blue);
        color: var(--accent-blue);
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """,
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────
# UTILITY FUNCTIONS
# ──────────────────────────────────────────────────────────


@st.cache_data(ttl=300)
def get_cached_intel(ticker: str) -> Dict[str, object]:
    """Fetch and cache intelligence for a ticker."""
    return get_god_tier_intel(ticker.upper())


@st.cache_data(ttl=300)
def get_cached_market_data(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """Fetch and cache market data."""
    try:
        return get_market_data(ticker.upper(), period, interval)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Market data fetch failed for %s: %s", ticker, exc)
        return pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Volume"])


@st.cache_data(ttl=600)
def get_cached_movers(limit: int) -> List[str]:
    """Fetch and cache list of market movers."""
    return scan_market_movers(limit=limit)


def format_currency(value: float) -> str:
    """Return Markdown-formatted currency with color."""
    color = "green" if value >= 0 else "red"
    sign = "+" if value > 0 else ""
    return f":{color}[{sign}${value:,.2f}]"


def format_percentage(value: float) -> str:
    """Return Markdown-formatted percentage with color."""
    color = "green" if value >= 0 else "red"
    sign = "+" if value > 0 else ""
    return f":{color}[{sign}{value:.2f}%]"


def get_action_emoji(action: str) -> str:
    """Map action string to emoji."""
    return {
        "BUY": "🟢",
        "SELL": "🔴",
        "WAIT": "⚪",
        "SKIP": "⏸️",
        "ERROR": "❌",
    }.get(action.upper(), "❓")


def render_diagnostics_panel() -> None:
    """Render lightweight diagnostics summary for common setup issues."""
    with st.expander("🩺 Diagnostics", expanded=False):
        try:
            info = collect_doctor_info()
            st.write(
                {
                    "version": __version__,
                    "python": info.get("python"),
                    "platform": platform.platform(),
                    "run_id": info.get("run_id"),
                    "mock_mode": info.get("mock_mode"),
                    "api_deps_installed": info.get("api_deps_installed"),
                    "wheelhouse_present": info.get("wheelhouse_present"),
                }
            )
            if not info.get("api_deps_installed", False):
                st.info("API dependencies are optional for UI. Install requirements-api.txt to run API.")
            st.caption("Need help? Run `alphaprime doctor` in PowerShell for full diagnostics.")
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Diagnostics unavailable: {exc}. Try `alphaprime doctor`.")


# ──────────────────────────────────────────────────────────
# CHART BUILDERS
# ──────────────────────────────────────────────────────────


def build_candlestick_chart(
    df: pd.DataFrame,
    ticker: str,
    technicals: Optional[Dict[str, object]] = None,
) -> go.Figure:
    """
    Build an interactive candlestick chart with volume and optional indicators.

    Args:
        df: OHLCV DataFrame with 'Date', 'Open', 'High', 'Low', 'Close', 'Volume'.
        ticker: Symbol for the title.
        technicals: Optional hard technicals dict (unused for now but available).

    Returns:
        Plotly Figure.
    """
    df_plot = df.copy()
    if "Date" in df_plot.columns:
        df_plot["Date"] = pd.to_datetime(df_plot["Date"])

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
        subplot_titles=(f"{ticker} Price Action", "Volume"),
    )

    fig.add_trace(
        go.Candlestick(
            x=df_plot["Date"],
            open=df_plot["Open"],
            high=df_plot["High"],
            low=df_plot["Low"],
            close=df_plot["Close"],
            name="OHLC",
            increasing_line_color="#00ff00",
            decreasing_line_color="#ff0000",
        ),
        row=1,
        col=1,
    )

    colors = [
        "#ff0000" if c < o else "#00ff00"
        for c, o in zip(df_plot["Close"], df_plot["Open"])
    ]
    fig.add_trace(
        go.Bar(
            x=df_plot["Date"],
            y=df_plot["Volume"],
            name="Volume",
            marker_color=colors,
            opacity=0.5,
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        template="plotly_dark",
        height=600,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="#1a1d24")

    return fig


def build_equity_curve(trader: PaperTrader) -> Optional[go.Figure]:
    """
    Build equity curve chart from portfolio history.

    Args:
        trader: PaperTrader instance.

    Returns:
        Plotly Figure or None if no history.
    """
    equity_points = trader.get_equity_curve()
    if not equity_points:
        return None

    df = pd.DataFrame(
        {
            "Date": [p.timestamp_utc for p in equity_points],
            "Value": [p.portfolio_value for p in equity_points],
            "CumulativePnL": [p.cumulative_pnl for p in equity_points],
        }
    )

    df["Date"] = pd.to_datetime(df["Date"])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["Value"],
            mode="lines",
            name="Portfolio Value",
            line=dict(color="#00aaff", width=2),
            fill="tozeroy",
            fillcolor="rgba(0,170,255,0.1)",
        )
    )

    start_cash = float(getattr(settings, "starting_cash", df["Value"].iloc[0]))
    fig.add_hline(
        y=start_cash,
        line_dash="dash",
        line_color="gray",
        annotation_text="Starting Capital",
    )

    fig.update_layout(
        template="plotly_dark",
        height=400,
        title="Equity Curve",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        hovermode="x unified",
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
    )
    return fig


# ──────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────


def render_sidebar(trader: PaperTrader) -> None:
    """Render sidebar showing portfolio summary and quick actions."""
    with st.sidebar:
        st.markdown(
            "<h3 style='color:#00ff00;'>ALPHA-PRIME v2.0</h3>",
            unsafe_allow_html=True,
        )
        st.markdown("#### 📈 War Room Dashboard")
        st.markdown("---")

        portfolio = trader.get_portfolio_state()

        st.markdown("### 💼 Portfolio")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Cash", f"${portfolio.cash:,.0f}")
        with c2:
            st.metric("Positions", portfolio.position_count)
        st.metric(
            "Total Value",
            f"${portfolio.total_value:,.2f}",
            delta=f"{portfolio.total_pnl:+,.2f} ({portfolio.total_pnl_pct:+.2f}%)",
        )

        metrics = trader.calculate_metrics()
        st.markdown("---")
        st.markdown("### 📊 Quick Stats")
        st.metric("Win Rate", f"{metrics.win_rate:.1f}%")
        st.metric("Total Trades", metrics.total_trades)

        st.markdown("---")
        st.markdown("### ⚡ Quick Actions")
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        if st.button("📤 Backup Portfolio", use_container_width=True):
            if trader.backup_portfolio():
                st.success("Backup created.")
            else:
                st.error("Backup failed.")

        st.markdown("---")
        st.markdown("### 🟢 System Status")
        st.caption(f"Oracle model: {settings.openai_model}")
        mode_label = "Paper Trading" if getattr(settings, "paper_trading_only", True) else "LIVE"
        st.caption(f"Mode: {mode_label}")
        st.caption(f"Version: {__version__}")
        st.caption(f"RUN_ID: {get_run_id()}")


# ──────────────────────────────────────────────────────────
# TAB: HOME (LIVE SIGNALS)
# ──────────────────────────────────────────────────────────


def render_home_tab(trader: PaperTrader) -> None:
    """Render home tab: live signal feed, chart, quick execute."""
    st.header("🏠 Live Signal Feed")

    col1, col2 = st.columns([2, 1])
    with col1:
        ticker = st.text_input("Ticker", value="AAPL", max_chars=10, key="home_ticker").upper().strip()
    with col2:
        mover_limit = st.selectbox("Scan Movers (N)", [5, 10, 15, 20], index=1)

    st.caption("Tip: Use scanner below to pick active tickers.")

    with st.expander("📡 Market Movers", expanded=False):
        try:
            movers = get_cached_movers(mover_limit)
            st.write(", ".join(movers) or "No movers found.")
            chosen = st.selectbox("Select from movers", [""] + movers)
            if chosen:
                ticker = chosen
        except Exception as exc:  # noqa: BLE001
            st.error("Scanner temporarily unavailable. Try again or run `alphaprime doctor` to check network/proxy settings.")
            logger.error("Scanner error in dashboard: %s", exc, exc_info=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        period = st.selectbox("Period", ["1d", "5d", "1mo", "3mo", "6mo", "1y"], index=3)
    with c2:
        interval = st.selectbox("Interval", ["1m", "5m", "15m", "1h", "1d"], index=4)
    with c3:
        analyze = st.button("🔍 Analyze", use_container_width=True)

    if not ticker:
        st.info("Enter a ticker or select from movers to analyze.")
        return

    if analyze:
        with st.spinner(f"Analyzing {ticker}..."):
            try:
                intel = get_cached_intel(ticker)
                df = get_cached_market_data(ticker, period, interval)
                if df.empty:
                    st.warning("No market data returned. Check network/proxy and try again. For offline setup help, run `alphaprime doctor`.")
                    return
                technicals = calculate_hard_technicals(df, ticker=ticker, timeframe=interval)
                decision = consult_oracle(
                    ticker=ticker,
                    intel=intel,
                    technicals=technicals,
                    regime="UNKNOWN",
                    events=[],
                )

                st.markdown("---")
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    emoji = get_action_emoji(decision.action)
                    st.markdown(f"### {emoji} **{decision.action}**")
                    st.caption("Oracle verdict")
                with c2:
                    st.metric("Confidence", f"{decision.confidence}%")
                with c3:
                    st.metric("Horizon", decision.time_horizon)
                with c4:
                    st.metric("Stop Loss", f"${decision.stop_loss:.2f}")

                if decision.action != "WAIT":
                    st.markdown("#### 💰 Risk Parameters")
                    e1, e2, e3, e4 = st.columns(4)
                    with e1:
                        st.metric(
                            "Entry Min",
                            f"${decision.entry_zone[0]:.2f}",
                        )
                    with e2:
                        st.metric(
                            "Entry Max",
                            f"${decision.entry_zone[1]:.2f}",
                        )
                    with e3:
                        st.metric("TP1", f"${decision.take_profit[0]:.2f}")
                    with e4:
                        st.metric("TP2/TP3", f"${decision.take_profit[1]:.2f}/{decision.take_profit[2]:.2f}")

                st.markdown("#### 📋 Rationale")
                for idx, point in enumerate(decision.rationale, start=1):
                    st.markdown(f"{idx}. {point}")

                if decision.risk_notes:
                    with st.expander("⚠️ Risk Warnings"):
                        for note in decision.risk_notes:
                            st.warning(note)

                st.markdown("#### 📈 Chart")
                fig = build_candlestick_chart(df, ticker, technicals)
                st.plotly_chart(fig, use_container_width=True)

                if decision.action in ("BUY", "SELL") and decision.confidence >= 70:
                    st.markdown("---")
                    st.subheader("⚡ Quick Execute (Paper)")
                    pf = trader.get_portfolio_state()
                    last_price = technicals["price_action"]["last_price"]
                    default_risk_pct = float(getattr(settings, "max_risk_per_trade_pct", 1.0))

                    risk_pct = st.slider(
                        "Risk per trade (%)",
                        min_value=0.5,
                        max_value=5.0,
                        value=default_risk_pct,
                        step=0.5,
                    )

                    risk_amount = pf.total_value * (risk_pct / 100.0)
                    price_risk = abs(last_price - decision.stop_loss)
                    qty_preview = int(risk_amount / price_risk) if price_risk > 0 else 0
                    st.caption(
                        f"Preview: risk=${risk_amount:,.2f}, "
                        f"price risk=${price_risk:.2f}, qty≈{qty_preview}"
                    )

                    if st.button(
                        f"Execute {decision.action} {qty_preview} {ticker}",
                        type="primary",
                    ):
                        if qty_preview <= 0:
                            st.error("Quantity is zero; adjust risk or stop loss.")
                        else:
                            trade_result = trader.execute_trade(
                                action=decision.action,
                                ticker=ticker,
                                price=last_price,
                                quantity=qty_preview,
                                notes=f"Dashboard execution (conf={decision.confidence}%)",
                            )
                            if trade_result.success:
                                st.success(trade_result.message)
                                pf = trader.get_portfolio_state()
                                send_discord_alert(
                                    decision=decision.to_dict(),
                                    portfolio_summary={
                                        "cash": pf.cash,
                                        "positions": pf.position_count,
                                        "total_value": pf.total_value,
                                    },
                                )
                            else:
                                st.error(trade_result.message)

            except Exception as exc:  # noqa: BLE001
                st.error(f"Error during analysis: {exc}")
                logger.error("Dashboard error: %s", exc, exc_info=True)


# ──────────────────────────────────────────────────────────
# TAB: PORTFOLIO
# ──────────────────────────────────────────────────────────


def render_portfolio_tab(trader: PaperTrader) -> None:
    """Render portfolio tab: summary, equity curve, open positions."""
    st.header("💼 Portfolio")

    portfolio = trader.get_portfolio_state()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Value", f"${portfolio.total_value:,.2f}")
    with c2:
        st.metric("Cash", f"${portfolio.cash:,.2f}")
    with c3:
        st.metric(
            "Total P&L",
            f"${portfolio.total_pnl:+,.2f}",
            delta=f"{portfolio.total_pnl_pct:+.2f}%",
        )
    with c4:
        st.metric("Positions", portfolio.position_count)

    st.markdown("---")
    st.subheader("📈 Equity Curve")
    fig = build_equity_curve(trader)
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No trade history yet.")

    st.markdown("---")
    st.subheader("📊 Open Positions")

    if portfolio.positions:
        rows: List[Dict[str, object]] = []
        for ticker, pos in portfolio.positions.items():
            ticker_u = ticker.upper()
            try:
                df = get_cached_market_data(ticker_u, "1d", "1d")
                current_price = float(df.iloc[-1]["Close"])
            except Exception:  # noqa: BLE001
                current_price = pos.avg_entry_price

            unreal_pnl = pos.unrealized_pnl(current_price)
            unreal_pct = pos.unrealized_pnl_pct(current_price)

            rows.append(
                {
                    "Ticker": ticker_u,
                    "Quantity": pos.quantity,
                    "Entry Price": f"${pos.avg_entry_price:.2f}",
                    "Current Price": f"${current_price:.2f}",
                    "Cost Basis": f"${pos.cost_basis:.2f}",
                    "Unrealized P&L": f"${unreal_pnl:+,.2f}",
                    "P&L %": f"{unreal_pct:+.2f}%",
                }
            )

        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("No open positions.")


# ──────────────────────────────────────────────────────────
# TAB: PERFORMANCE
# ──────────────────────────────────────────────────────────


def render_performance_tab(trader: PaperTrader) -> None:
    """Render performance metrics and trade history."""
    st.header("📊 Performance Analytics")

    metrics = trader.calculate_metrics()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Trades", metrics.total_trades)
    with c2:
        st.metric("Win Rate", f"{metrics.win_rate:.1f}%")
    with c3:
        st.metric("Profit Factor", f"{metrics.profit_factor:.2f}")
    with c4:
        st.metric("Sharpe Ratio", f"{metrics.sharpe_ratio:.2f}")

    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Winning Trades", metrics.winning_trades)
    with c2:
        st.metric("Losing Trades", metrics.losing_trades)
    with c3:
        st.metric("Avg Win", f"${metrics.average_win:+,.2f}")
    with c4:
        st.metric("Avg Loss", f"${metrics.average_loss:+,.2f}")

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Max Drawdown", f"{metrics.max_drawdown_pct:.2f}%")
    with c2:
        st.metric("Largest Win", f"${metrics.largest_win:+,.2f}")
    with c3:
        st.metric("Largest Loss", f"${metrics.largest_loss:+,.2f}")

    st.markdown("---")
    st.subheader("📜 Trade History")
    try:
        if trader.trade_history_path.exists():
            df = pd.read_csv(trader.trade_history_path)
            df = df.sort_values("timestamp_utc", ascending=False)
            st.dataframe(df.head(100), use_container_width=True, hide_index=True)
        else:
            st.info("No trades recorded yet.")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Unable to load trade history: {exc}")
        logger.error("Trade history load error: %s", exc, exc_info=True)


# ──────────────────────────────────────────────────────────
# TAB: RESEARCH (INTELLIGENCE VIEWER)
# ──────────────────────────────────────────────────────────


def render_research_tab() -> None:
    """Render deep intelligence viewer for a ticker."""
    st.header("🔬 Research & Intelligence")

    c1, c2 = st.columns([2, 1])
    with c1:
        ticker = st.text_input("Ticker", value="AAPL", max_chars=10, key="research_ticker").upper().strip()
    with c2:
        analyze = st.button("Fetch Intelligence")

    if not ticker:
        st.info("Enter a ticker to view research.")
        return

    if analyze:
        with st.spinner(f"Fetching intelligence for {ticker}..."):
            try:
                intel = get_cached_intel(ticker)

                st.subheader("Fundamentals")
                fundamentals = intel.get("fundamentals", {})
                st.json(
                    {
                        "going_concern_flags": fundamentals.get("going_concern_flags", []),
                        "debt_maturity_flags": fundamentals.get("debt_maturity_flags", []),
                        "insider_selling_summary": fundamentals.get(
                            "insider_selling_summary", ""
                        ),
                    }
                )

                st.subheader("News & Catalysts")
                news = intel.get("news_catalysts", {})
                headlines = news.get("headlines", [])[:15]
                for h in headlines:
                    st.markdown(
                        f"- **{h.get('title','')}**  \n"
                        f"  {h.get('url','')}  \n"
                        f"  _{h.get('published_at_utc','')}_"
                    )

                st.subheader("Sentiment Snapshot")
                sentiment = intel.get("sentiment_score", {})
                s1, s2, s3, s4 = st.columns(4)
                with s1:
                    st.metric("Hype Score", sentiment.get("hype_score", 0))
                with s2:
                    st.metric("Polarity", f"{sentiment.get('polarity', 0.0):.2f}")
                with s3:
                    st.metric("Volume", sentiment.get("volume", 0))
                with s4:
                    st.metric(
                        "Availability",
                        str(sentiment.get("availability", {})),
                    )

            except Exception as exc:  # noqa: BLE001
                st.error("Intelligence feed unavailable. The UI can continue without it.")
                st.caption("Try `alphaprime doctor` and check proxy settings if you are on a restricted network.")
                logger.error("Research tab error: %s", exc, exc_info=True)


# ──────────────────────────────────────────────────────────
# TAB: EXECUTE (MANUAL TRADES)
# ──────────────────────────────────────────────────────────


def render_execute_tab(trader: PaperTrader) -> None:
    """Render manual trade execution interface with risk preview."""
    st.header("⚙️ Manual Trade Execution (Paper)")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        action = st.selectbox("Action", ["BUY", "SELL"])
    with c2:
        ticker = st.text_input("Ticker", value="AAPL", max_chars=10, key="execute_ticker").upper().strip()
    with c3:
        price = st.number_input("Price", min_value=0.0, value=150.0, step=0.01)
    with c4:
        quantity = st.number_input("Quantity", min_value=0, value=10, step=1)

    st.markdown("#### 🧮 Risk Preview")
    pf = trader.get_portfolio_state()
    risk_pct = st.slider(
        "Risk per trade (%)",
        min_value=0.5,
        max_value=5.0,
        value=float(getattr(settings, "max_risk_per_trade_pct", 1.0)),
        step=0.5,
    )
    stop_loss_price = st.number_input(
        "Stop Loss Price",
        min_value=0.0,
        value=max(price * 0.95, 0.01),
        step=0.01,
    )

    risk_amount = pf.total_value * (risk_pct / 100.0)
    price_risk = abs(price - stop_loss_price)
    qty_by_risk = int(risk_amount / price_risk) if price_risk > 0 else 0

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Risk Amount", f"${risk_amount:,.2f}")
    with c2:
        st.metric("Price Risk", f"${price_risk:.2f}")
    with c3:
        st.metric("Size by Risk", f"{qty_by_risk} shares")

    st.markdown("---")
    note = st.text_input("Notes", value="Manual trade via dashboard", key="execute_notes")

    if st.button("Submit Trade", type="primary"):
        if quantity <= 0:
            st.error("Quantity must be positive.")
        elif price <= 0:
            st.error("Price must be positive.")
        else:
            try:
                result = trader.execute_trade(
                    action=action,
                    ticker=ticker,
                    price=price,
                    quantity=quantity,
                    notes=note,
                )
                if result.success:
                    st.success(result.message)
                else:
                    st.error(result.message)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Execution error: {exc}")
                logger.error("Manual execution error: %s", exc, exc_info=True)


# ──────────────────────────────────────────────────────────
# TAB: SETTINGS
# ──────────────────────────────────────────────────────────


def render_settings_tab() -> None:
    """Render settings and configuration panel (read-only knobs)."""
    st.header("🛠️ Settings & Configuration")

    st.subheader("Trading Parameters")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(
            "Risk per Trade (%)",
            f"{getattr(settings, 'max_risk_per_trade_pct', 1.0):.1f}%",
        )
    with c2:
        st.metric(
            "Alert Min Confidence",
            f"{getattr(settings, 'alert_min_confidence', 70)}%",
        )
    with c3:
        st.metric(
            "Starting Cash",
            f"${getattr(settings, 'starting_cash', 10000):,.2f}",
        )

    st.subheader("System Flags")
    flags = {
        "Paper Trading Only": getattr(settings, "paper_trading_only", True),
        "Circuit Breaker Enabled": getattr(settings, "circuit_breaker_enabled", False),
        "Regime Filter": getattr(settings, "enable_regime_filter", False),
        "Earnings Filter": getattr(settings, "enable_earnings_filter", False),
        "Drift Monitoring": getattr(settings, "enable_drift_monitoring", False),
    }
    for name, value in flags.items():
        st.checkbox(name, value=value, disabled=True)

    st.subheader("Environment")
    st.json(
        {
            "OpenAI Model": getattr(settings, "openai_model", ""),
            "Timezone": getattr(settings, "timezone", "IST"),
            "Market Open": getattr(settings, "market_open_time", "09:30"),
        }
    )


# ──────────────────────────────────────────────────────────
# MAIN APP
# ──────────────────────────────────────────────────────────


def main() -> None:
    """Main Streamlit entry point."""
    inject_custom_css()

    if "trader" not in st.session_state:
        st.session_state["trader"] = PaperTrader()
    trader: PaperTrader = st.session_state["trader"]

    render_sidebar(trader)
    render_diagnostics_panel()

    tabs = st.tabs(
        [
            "🏠 Home",
            "💼 Portfolio",
            "📊 Performance",
            "🔬 Research",
            "⚙️ Execute",
            "🛠️ Settings",
        ]
    )

    with tabs[0]:
        render_home_tab(trader)
    with tabs[1]:
        render_portfolio_tab(trader)
    with tabs[2]:
        render_performance_tab(trader)
    with tabs[3]:
        render_research_tab()
    with tabs[4]:
        render_execute_tab(trader)
    with tabs[5]:
        render_settings_tab()

    st.markdown("---")
    st.caption("ALPHA-PRIME v2.0 | AI-Powered Trading System | Paper Trading Mode")


if __name__ == "__main__":
    main()
