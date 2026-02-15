"""
============================================================
ALPHA-PRIME v2.0 - Correlation Monitor (Risk Management)
============================================================

Prevents over-concentration risk by monitoring correlations:

1. Position Correlation Check:
   - Calculate correlation between new ticker and existing positions
   - Block trades if correlation > threshold (default: 0.7)
   - Prevents portfolio from becoming too concentrated

2. Sector Exposure Analysis:
   - Group positions by sector
   - Warn if too many positions in single sector
   - Promotes diversification across industries

3. Diversification Metrics:
   - Calculate portfolio-wide correlation
   - Herfindahl-Hirschman Index (HHI)
   - Effective number of positions

4. Correlation Matrix:
   - Visual heatmap of all position correlations
   - Identify clusters of correlated assets
   - Historical rolling correlations (via lookback period)

Why It Matters:
- Correlated positions = concentrated risk.
- If one large tech name falls 10%, highly correlated names likely drop too.
- Diversification reduces portfolio volatility and tail risk.
- Protects against sector-specific or factor-specific shocks.

Usage:
    from risk.correlation_monitor import check_correlation_risk
    from portfolio import PaperTrader

    trader = PaperTrader()
    allowed, reason = check_correlation_risk(trader, "MSFT")

    if not allowed:
        logger.warning(f"High correlation: {reason}")

Integration:
- Called by scheduler.py before new positions
- Generates heatmaps for dashboard
- Alerts on over-concentration
============================================================
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf

from config import get_logger, get_settings

logger = get_logger(__name__)
settings = get_settings()


# ──────────────────────────────────────────────────────────
# DATA SCHEMAS
# ──────────────────────────────────────────────────────────


@dataclass
class DiversificationMetrics:
    """
    Portfolio diversification metrics.

    Attributes:
        num_positions: Number of open positions.
        effective_positions: Effective number of positions (1 / HHI).
        avg_correlation: Portfolio-wide average pairwise correlation.
        max_correlation: Highest pairwise correlation observed.
        hhi: Herfindahl-Hirschman Index based on dollar weights (0–1).
        diversification_score: 0–100 score, higher = more diversified.
        sector_concentration: Mapping of sector → position count.
        most_concentrated_sector: Name of most concentrated sector.
    """

    num_positions: int
    effective_positions: float
    avg_correlation: float
    max_correlation: float
    hhi: float
    diversification_score: float
    sector_concentration: Dict[str, int]
    most_concentrated_sector: str


@dataclass
class CorrelationCheckResult:
    """
    Result of correlation check for a prospective new position.

    Attributes:
        allowed: True if correlation constraints permit the new trade.
        ticker: New ticker being evaluated.
        max_correlation: Maximum observed correlation vs portfolio tickers.
        correlated_with: List of tickers above correlation threshold.
        message: Human-readable summary of the decision.
    """

    allowed: bool
    ticker: str
    max_correlation: float
    correlated_with: List[str]
    message: str


# ──────────────────────────────────────────────────────────
# CORRELATION MATRIX CALCULATION
# ──────────────────────────────────────────────────────────


def _download_price_history(
    tickers: List[str],
    period: str = "3mo",
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Internal helper to download OHLCV data for multiple tickers.

    Args:
        tickers: List of tickers to download.
        period: History period string understood by yfinance (e.g. "3mo").
        interval: Bar interval ("1d", "1h", etc.).

    Returns:
        DataFrame with multi-index columns (ticker, field) from yfinance.
    """
    if not tickers:
        return pd.DataFrame()

    try:
        data = yf.download(
            tickers=tickers,
            period=period,
            interval=interval,
            progress=False,
            group_by="ticker",
            auto_adjust=False,
        )
        return data
    except Exception as exc:  # noqa: BLE001
        logger.error("Error downloading price history for %s: %s", tickers, exc)
        return pd.DataFrame()


def calculate_correlation_matrix(
    tickers: List[str],
    period: str = "3mo",
    min_periods: int = 20,
) -> pd.DataFrame:
    """
    Calculate correlation matrix for a list of tickers.

    Uses daily close prices over the specified period and computes the
    Pearson correlation coefficient for each pair.

    Args:
        tickers: List of stock symbols.
        period: Historical period (e.g., "1mo", "3mo", "6mo", "1y").
        min_periods: Minimum number of non-NaN daily closes required.

    Returns:
        DataFrame representing correlation matrix (tickers x tickers).
        Empty DataFrame if insufficient data.
    """
    if not tickers or len(tickers) < 2:
        logger.warning("Correlation matrix requires at least 2 tickers.")
        return pd.DataFrame()

    logger.info("Calculating correlation matrix for %d tickers.", len(tickers))

    data = _download_price_history(tickers, period=period, interval="1d")
    if data.empty:
        logger.error("No historical data returned for tickers: %s", tickers)
        return pd.DataFrame()

    try:
        # yfinance returns multi-index columns when multiple tickers are requested.
        if isinstance(data.columns, pd.MultiIndex):
            close_prices = {}
            for ticker in tickers:
                if ticker in data.columns.get_level_values(0):
                    close_prices[ticker] = data[ticker]["Close"]
            close_df = pd.DataFrame(close_prices)
        else:
            # Single ticker case
            close_df = data["Close"].to_frame(name=tickers[0])

        close_df = close_df.dropna()
        if len(close_df) < min_periods:
            logger.warning(
                "Insufficient price history for correlation (%d < %d).",
                len(close_df),
                min_periods,
            )
            return pd.DataFrame()

        corr_matrix = close_df.corr()
        logger.info("Correlation matrix computed over %d trading days.", len(close_df))
        return corr_matrix
    except Exception as exc:  # noqa: BLE001
        logger.error("Error constructing correlation matrix: %s", exc, exc_info=True)
        return pd.DataFrame()


# ──────────────────────────────────────────────────────────
# NEW POSITION CORRELATION CHECK
# ──────────────────────────────────────────────────────────


def check_position_correlation(
    ticker: str,
    portfolio_tickers: List[str],
    threshold: Optional[float] = None,
    period: str = "3mo",
) -> CorrelationCheckResult:
    """
    Check if a prospective new ticker is too correlated with the portfolio.

    Args:
        ticker: New ticker being considered for a position.
        portfolio_tickers: List of existing portfolio tickers.
        threshold: Maximum allowed correlation. Defaults to
                   settings.correlation_limit (fallback 0.7).
        period: Lookback period for correlation (e.g., "3mo").

    Returns:
        CorrelationCheckResult summarizing the outcome.
    """
    if threshold is None:
        threshold = float(getattr(settings, "correlation_limit", 0.7))

    ticker = ticker.upper()
    portfolio_tickers = [t.upper() for t in portfolio_tickers if t]

    if not portfolio_tickers:
        return CorrelationCheckResult(
            allowed=True,
            ticker=ticker,
            max_correlation=0.0,
            correlated_with=[],
            message="First position, correlation check not applicable.",
        )

    try:
        all_tickers = [ticker] + portfolio_tickers
        corr_matrix = calculate_correlation_matrix(all_tickers, period=period)

        if corr_matrix.empty or ticker not in corr_matrix.columns:
            logger.warning(
                "Correlation matrix empty or missing ticker %s; allowing trade.", ticker
            )
            return CorrelationCheckResult(
                allowed=True,
                ticker=ticker,
                max_correlation=0.0,
                correlated_with=[],
                message="Correlation check skipped (data unavailable).",
            )

        corr_series = corr_matrix[ticker].drop(index=ticker, errors="ignore")
        if corr_series.empty:
            return CorrelationCheckResult(
                allowed=True,
                ticker=ticker,
                max_correlation=0.0,
                correlated_with=[],
                message="Correlation check: no overlapping history.",
            )

        max_corr = float(corr_series.max())
        high_corr = corr_series[corr_series > threshold].index.tolist()

        if high_corr:
            msg = (
                f"❌ High correlation: {ticker} vs "
                f"{', '.join(high_corr)} (max={max_corr:.2f} > {threshold:.2f}). "
                "Adding this would increase concentration risk."
            )
            logger.warning(msg)
            return CorrelationCheckResult(
                allowed=False,
                ticker=ticker,
                max_correlation=max_corr,
                correlated_with=high_corr,
                message=msg,
            )

        msg = f"Correlation OK for {ticker} (max={max_corr:.2f} <= {threshold:.2f})."
        logger.info(msg)
        return CorrelationCheckResult(
            allowed=True,
            ticker=ticker,
            max_correlation=max_corr,
            correlated_with=[],
            message=msg,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Error during position correlation check: %s", exc, exc_info=True)
        return CorrelationCheckResult(
            allowed=True,
            ticker=ticker,
            max_correlation=0.0,
            correlated_with=[],
            message=f"Correlation check failed (allowing trade): {exc}",
        )


# ──────────────────────────────────────────────────────────
# SECTOR EXPOSURE ANALYSIS
# ──────────────────────────────────────────────────────────


def get_ticker_sector(ticker: str) -> str:
    """
    Retrieve GICS sector for a ticker via yfinance.

    Args:
        ticker: Stock symbol.

    Returns:
        Sector name string, or "Unknown" on failure.
    """
    try:
        yf_ticker = yf.Ticker(ticker)
        info = getattr(yf_ticker, "info", {}) or {}
        sector = info.get("sector", "Unknown")
        return sector or "Unknown"
    except Exception as exc:  # noqa: BLE001
        logger.debug("Failed to fetch sector for %s: %s", ticker, exc)
        return "Unknown"


def get_sector_exposure(trader) -> Dict[str, float]:
    """
    Compute sector exposure as percentage of portfolio value.

    Args:
        trader: PaperTrader instance.

    Returns:
        Dictionary mapping sector name to portfolio percentage.
    """
    portfolio = trader.get_portfolio_state()
    if not portfolio.positions:
        return {}

    sector_values: Dict[str, float] = {}
    for ticker, position in portfolio.positions.items():
        sector = get_ticker_sector(ticker.upper())
        sector_values[sector] = sector_values.get(sector, 0.0) + float(
            position.cost_basis
        )

    total_value = sum(sector_values.values())
    if total_value <= 0:
        return {}

    sector_pct = {sec: val / total_value * 100.0 for sec, val in sector_values.items()}
    sector_pct = dict(sorted(sector_pct.items(), key=lambda kv: kv[1], reverse=True))

    logger.info("Sector exposure: %s", sector_pct)
    return sector_pct


def check_sector_concentration(
    trader,
    new_ticker: str,
    max_positions_per_sector: int = 3,
) -> Tuple[bool, str]:
    """
    Check whether adding a new position over-concentrates a sector.

    Args:
        trader: PaperTrader instance.
        new_ticker: Proposed new ticker.
        max_positions_per_sector: Maximum allowed positions in a single sector.

    Returns:
        (allowed, message) pair.
    """
    portfolio = trader.get_portfolio_state()
    if not portfolio.positions:
        return True, "First position; sector concentration not applicable."

    sector_counts: Dict[str, int] = {}
    for existing_ticker in portfolio.positions.keys():
        sec = get_ticker_sector(existing_ticker.upper())
        sector_counts[sec] = sector_counts.get(sec, 0) + 1

    new_sector = get_ticker_sector(new_ticker.upper())
    current_count = sector_counts.get(new_sector, 0)

    if current_count >= max_positions_per_sector:
        msg = (
            f"⚠️ Sector concentration: {current_count} positions in {new_sector} "
            f"(limit={max_positions_per_sector})."
        )
        logger.warning(msg)
        return False, msg

    return True, (
        f"Sector OK for {new_ticker} ({new_sector}: "
        f"{current_count}/{max_positions_per_sector} positions)."
    )


# ──────────────────────────────────────────────────────────
# PORTFOLIO DIVERSIFICATION METRICS
# ──────────────────────────────────────────────────────────


def calculate_herfindahl_index(position_values: List[float]) -> float:
    """
    Compute Herfindahl-Hirschman Index (HHI) from position values.

    HHI = sum(w_i^2) where w_i are portfolio weights (0–1).

    Args:
        position_values: List of position market values.

    Returns:
        HHI in [0, 1]. 0 = perfectly diversified, 1 = single position.
    """
    if not position_values:
        return 1.0

    total = float(sum(position_values))
    if total <= 0:
        return 1.0

    weights = [v / total for v in position_values]
    hhi = float(sum(w * w for w in weights))
    return hhi


def calculate_portfolio_diversification(trader) -> DiversificationMetrics:
    """
    Calculate diversification metrics for the current portfolio.

    Metrics:
        - num_positions
        - effective_positions (1 / HHI)
        - avg_correlation (pairwise)
        - max_correlation
        - hhi
        - diversification_score (0–100)
        - sector_concentration and most_concentrated_sector

    Args:
        trader: PaperTrader instance.

    Returns:
        DiversificationMetrics instance.
    """
    portfolio = trader.get_portfolio_state()
    if not portfolio.positions:
        return DiversificationMetrics(
            num_positions=0,
            effective_positions=0.0,
            avg_correlation=0.0,
            max_correlation=0.0,
            hhi=1.0,
            diversification_score=0.0,
            sector_concentration={},
            most_concentrated_sector="None",
        )

    tickers = list(portfolio.positions.keys())
    values = [float(pos.cost_basis) for pos in portfolio.positions.values()]
    num_positions = len(values)

    hhi = calculate_herfindahl_index(values)
    effective_positions = 1.0 / hhi if hhi > 0 else float(num_positions)

    avg_correlation = 0.0
    max_correlation = 0.0
    if len(tickers) >= 2:
        corr_matrix = calculate_correlation_matrix(tickers)
        if not corr_matrix.empty:
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
            upper = corr_matrix.where(mask)
            correlations = upper.stack().values
            if correlations.size > 0:
                avg_correlation = float(np.mean(correlations))
                max_correlation = float(np.max(correlations))

    sector_counts: Dict[str, int] = {}
    for t in tickers:
        sec = get_ticker_sector(t.upper())
        sector_counts[sec] = sector_counts.get(sec, 0) + 1
    if sector_counts:
        most_concentrated_sector = max(sector_counts.items(), key=lambda kv: kv[1])[0]
    else:
        most_concentrated_sector = "None"

    # Diversification score heuristic: combine HHI, avg correlation, and num positions.
    hhi_score = max(0.0, (1.0 - hhi)) * 100.0  # lower HHI → higher score
    corr_score = max(0.0, (1.0 - max(avg_correlation, 0.0))) * 100.0
    position_score = min(num_positions / 5.0 * 100.0, 100.0)  # saturates at 5+

    diversification_score = hhi_score * 0.4 + corr_score * 0.4 + position_score * 0.2

    metrics = DiversificationMetrics(
        num_positions=num_positions,
        effective_positions=effective_positions,
        avg_correlation=avg_correlation,
        max_correlation=max_correlation,
        hhi=hhi,
        diversification_score=diversification_score,
        sector_concentration=sector_counts,
        most_concentrated_sector=most_concentrated_sector,
    )

    logger.info(
        "Diversification metrics: positions=%d, eff=%.2f, hhi=%.3f, "
        "avg_corr=%.2f, max_corr=%.2f, score=%.1f.",
        metrics.num_positions,
        metrics.effective_positions,
        metrics.hhi,
        metrics.avg_correlation,
        metrics.max_correlation,
        metrics.diversification_score,
    )
    return metrics


# ──────────────────────────────────────────────────────────
# MASTER CORRELATION RISK CHECK
# ──────────────────────────────────────────────────────────


def check_correlation_risk(
    trader,
    new_ticker: str,
    correlation_threshold: Optional[float] = None,
    max_positions_per_sector: int = 3,
) -> Tuple[bool, str]:
    """
    Master function to check correlation and diversification risks.

    Checks:
        1. Position correlation vs existing positions.
        2. Sector concentration for the new ticker.
        3. Overall diversification (informational logging).

    Args:
        trader: PaperTrader instance.
        new_ticker: Ticker for proposed new position.
        correlation_threshold: Max allowed correlation (default from settings).
        max_positions_per_sector: Max positions allowed in one sector.

    Returns:
        (allowed, message) pair summarizing decision.
    """
    new_ticker = new_ticker.upper()
    logger.info("Checking correlation risk for proposed ticker: %s.", new_ticker)

    portfolio = trader.get_portfolio_state()
    portfolio_tickers = list(portfolio.positions.keys())

    corr_result = check_position_correlation(
        ticker=new_ticker,
        portfolio_tickers=portfolio_tickers,
        threshold=correlation_threshold,
        period="3mo",
    )
    if not corr_result.allowed:
        return False, corr_result.message

    sector_allowed, sector_msg = check_sector_concentration(
        trader=trader,
        new_ticker=new_ticker,
        max_positions_per_sector=max_positions_per_sector,
    )
    if not sector_allowed:
        return False, sector_msg

    metrics = calculate_portfolio_diversification(trader)
    logger.info(
        "Post-check diversification (pre-trade): score=%.1f, avg_corr=%.2f, HHI=%.3f.",
        metrics.diversification_score,
        metrics.avg_correlation,
        metrics.hhi,
    )

    return True, (
        f"Correlation risk OK for {new_ticker} "
        f"(max_corr={corr_result.max_correlation:.2f})."
    )


# ──────────────────────────────────────────────────────────
# CORRELATION HEATMAP VISUALIZATION
# ──────────────────────────────────────────────────────────


def visualize_correlation_heatmap(
    correlation_matrix: pd.DataFrame,
) -> Optional[go.Figure]:
    """
    Generate a Plotly correlation heatmap for dashboard visualization.

    Args:
        correlation_matrix: DataFrame of correlations (tickers x tickers).

    Returns:
        Plotly Figure instance, or None if matrix is empty.
    """
    if correlation_matrix.empty:
        logger.warning("Empty correlation matrix – cannot build heatmap.")
        return None

    fig = go.Figure(
        data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            colorscale=[
                [0.0, "#00ff00"],  # green (low/negative)
                [0.5, "#ffff00"],  # yellow (medium)
                [1.0, "#ff0000"],  # red (high)
            ],
            zmin=-1,
            zmax=1,
            zmid=0,
            text=correlation_matrix.values,
            texttemplate="%{text:.2f}",
            textfont={"size": 10},
            colorbar=dict(title="Correlation"),
        )
    )

    fig.update_layout(
        title="Portfolio Correlation Matrix",
        template="plotly_dark",
        height=500,
        xaxis={"side": "bottom"},
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
    )

    return fig


# ──────────────────────────────────────────────────────────
# CLI TOOL & TESTING
# ──────────────────────────────────────────────────────────


if __name__ == "__main__":
    from portfolio import PaperTrader

    print("\n" + "=" * 70)
    print("ALPHA-PRIME Correlation Monitor - Test Tool")
    print("=" * 70 + "\n")

    trader = PaperTrader()
    portfolio_state = trader.get_portfolio_state()

    if not portfolio_state.positions:
        print("❌ No positions in portfolio. Execute some trades first.\n")
    else:
        tickers = list(portfolio_state.positions.keys())
        print(f"Current Positions: {', '.join(tickers)}\n")

        print("Calculating correlation matrix...")
        corr = calculate_correlation_matrix(tickers)
        if not corr.empty:
            print("\nCorrelation Matrix:\n")
            print(corr.to_string())
            print()

        print("-" * 70)
        print("Diversification Metrics:\n")
        metrics = calculate_portfolio_diversification(trader)
        print(f"Number of Positions   : {metrics.num_positions}")
        print(f"Effective Positions   : {metrics.effective_positions:.2f}")
        print(f"HHI                   : {metrics.hhi:.3f}")
        print(f"Avg Correlation       : {metrics.avg_correlation:.2f}")
        print(f"Max Correlation       : {metrics.max_correlation:.2f}")
        print(f"Diversification Score : {metrics.diversification_score:.1f} / 100")
        print(f"\nSector Concentration  : {metrics.sector_concentration}")
        print(f"Most Concentrated     : {metrics.most_concentrated_sector}")

        print("\n" + "-" * 70)
        test_ticker = input(
            "\nEnter ticker to test correlation risk (e.g., MSFT): "
        ).strip().upper()

        if test_ticker:
            print(f"\nChecking correlation risk for {test_ticker}...\n")
            allowed, reason = check_correlation_risk(trader, test_ticker)
            if allowed:
                print(f"✅ {reason}")
            else:
                print(f"❌ {reason}")

    print("\n" + "=" * 70 + "\n")
