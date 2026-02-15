"""
============================================================
ALPHA-PRIME v2.0 - Risk Metrics Calculator
============================================================

Comprehensive risk-adjusted performance metrics:

Core Metrics:
1. Sharpe Ratio: (Return - Risk_Free) / Volatility
   - Industry-standard risk-adjusted return  [web:141][web:192][web:193]
   - > 1.0 = good, > 2.0 = excellent

2. Sortino Ratio: (Return - Risk_Free) / Downside_Deviation
   - Only penalizes downside volatility
   - Better for asymmetric return distributions  [web:141][web:192][web:193]

3. Max Drawdown (MDD): Largest peak-to-trough decline
   - Shows worst-case loss from peak
   - Duration = time to recover  [web:141][web:199]

4. Value at Risk (VaR): Maximum expected loss at X% confidence
   - VaR(95%) = worst loss in 95% of scenarios
   - Example: VaR(95%) = -2.5% means 5% chance of losing > 2.5%  [web:197][web:200][web:203]

5. Conditional VaR (CVaR): Average loss beyond VaR
   - Also called Expected Shortfall (ES)
   - More informative than VaR for tail risk  [web:197][web:200][web:203]

6. Beta: Sensitivity to market movements
   - Beta = 1.0 → moves with market
   - Beta > 1.0 → amplifies market moves
   - Beta < 1.0 → dampens market moves

7. Alpha: Excess return vs benchmark
   - Alpha > 0 → outperforming market
   - Measures skill vs luck

8. Calmar Ratio: Annual Return / Max Drawdown
   - Higher = better return per unit of drawdown risk
   - > 3.0 is excellent  [web:141][web:192]

9. Ulcer Index: Depth and duration of drawdowns
   - Square-root of the mean squared drawdowns  [web:196][web:199][web:202]

10. Omega Ratio: Probability-weighted gains vs losses
    - Ratio of gains above threshold to losses below threshold

Usage:
    from risk.risk_metrics import get_comprehensive_metrics
    from portfolio import PaperTrader

    trader = PaperTrader()
    metrics = get_comprehensive_metrics(trader)

    print(f"Sharpe: {metrics.sharpe_ratio:.2f}")
    print(f"Max DD: {metrics.max_drawdown_pct:.2f}%")
    print(f"VaR(95%): {metrics.var_95:.2f}%")

Integration:
- Dashboard performance tab
- Automated performance reports
- Position-level risk analysis
- Drift monitoring validation
============================================================
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

from config import get_logger, get_settings

logger = get_logger(__name__)
settings = get_settings()


# ──────────────────────────────────────────────────────────
# DATA SCHEMAS
# ──────────────────────────────────────────────────────────


@dataclass
class RiskMetrics:
    """
    Comprehensive risk metrics for portfolio-level performance.

    All rates and percentages are expressed as percentages where noted (e.g. 10.0 = 10%).
    """

    # Return metrics
    total_return: float
    annualized_return: float

    # Risk-adjusted returns
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    omega_ratio: float

    # Drawdown metrics
    max_drawdown_pct: float
    max_drawdown_duration_days: int
    current_drawdown_pct: float
    avg_drawdown_pct: float
    ulcer_index: float

    # Volatility metrics
    volatility_annual: float
    downside_deviation: float

    # Risk metrics
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float

    # Market comparison
    beta: Optional[float]
    alpha: Optional[float]
    correlation_to_spy: Optional[float]

    # Distribution metrics
    skewness: float
    kurtosis: float
    winning_months_pct: float

    # Metadata
    days_traded: int
    total_trades: int
    calculated_at_utc: str


@dataclass
class DrawdownPeriod:
    """
    Single drawdown episode captured from an equity curve.

    Attributes:
        start_date: Index label or pseudo-date where drawdown starts.
        end_date: Index label or pseudo-date where drawdown reaches bottom.
        recovery_date: Index label or pseudo-date where equity recovers prior high.
        drawdown_pct: Magnitude of maximum drawdown during this episode (positive %).
        duration_days: Number of periods from start to recovery (or to current).
        recovered: True if drawdown has fully recovered at end of sample.
    """

    start_date: str
    end_date: str
    recovery_date: Optional[str]
    drawdown_pct: float
    duration_days: int
    recovered: bool


# ──────────────────────────────────────────────────────────
# RISK-ADJUSTED RETURN METRICS
# ──────────────────────────────────────────────────────────


def calculate_sharpe_ratio(
    returns: List[float],
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252,
) -> float:
    """
    Calculate annualized Sharpe Ratio from periodic returns.  [web:141][web:192]

    Formula (for periodic returns r_t):
        Sharpe = (E[r_t - rf_t]) / std(r_t - rf_t),
    annualized using periods_per_year.

    Args:
        returns: Sequence of periodic returns (e.g. daily decimal returns).
        risk_free_rate: Annual risk-free rate (e.g. 0.02 = 2%).
        periods_per_year: Number of periods per year (252 for daily).

    Returns:
        Annualized Sharpe ratio. Returns 0.0 if insufficient data.
    """
    if not returns or len(returns) < 2:
        return 0.0

    r = np.array(returns, dtype=float)
    rf_periodic = risk_free_rate / periods_per_year
    excess = r - rf_periodic

    std_excess = np.std(excess, ddof=1)
    if std_excess == 0:
        return 0.0

    sharpe = np.mean(excess) / std_excess * np.sqrt(periods_per_year)
    return float(sharpe)


def calculate_sortino_ratio(
    returns: List[float],
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252,
    target_return: float = 0.0,
) -> float:
    """
    Calculate annualized Sortino Ratio from periodic returns.  [web:141][web:192]

    Sortino focuses on downside risk:
        Sortino = (E[r_t - rf_t] - target) / downside_deviation,
    where downside deviation is computed from returns below target_return.

    Args:
        returns: Sequence of periodic returns.
        risk_free_rate: Annual risk-free rate.
        periods_per_year: Number of periods per year.
        target_return: Minimum acceptable return per period (decimal).

    Returns:
        Annualized Sortino ratio. Returns inf if no downside volatility.
    """
    if not returns or len(returns) < 2:
        return 0.0

    r = np.array(returns, dtype=float)
    rf_periodic = risk_free_rate / periods_per_year
    excess = r - rf_periodic

    downside = excess[excess < target_return]
    if downside.size == 0:
        return float("inf")

    downside_dev = np.sqrt(np.mean((downside - target_return) ** 2)) * np.sqrt(
        periods_per_year
    )
    if downside_dev == 0:
        return 0.0

    mean_excess = np.mean(excess) * np.sqrt(periods_per_year)
    sortino = mean_excess / downside_dev
    return float(sortino)


# ──────────────────────────────────────────────────────────
# DRAWDOWN ANALYSIS
# ──────────────────────────────────────────────────────────


def calculate_max_drawdown(
    equity_curve: List[float],
) -> Tuple[float, int, int]:
    """
    Calculate maximum drawdown (MDD) and its start/end indices.  [web:141][web:199]

    Drawdown at time t:
        DD_t = (Equity_t - Peak_t) / Peak_t

    Args:
        equity_curve: Sequence of portfolio values over time.

    Returns:
        (max_dd_pct, start_idx, end_idx) where:
            max_dd_pct is positive (% drawdown),
            start_idx is index of peak before MDD,
            end_idx is index where MDD occurs.
    """
    if not equity_curve or len(equity_curve) < 2:
        return 0.0, 0, 0

    equity = np.array(equity_curve, dtype=float)
    running_max = np.maximum.accumulate(equity)
    drawdowns = (equity - running_max) / running_max

    end_idx = int(np.argmin(drawdowns))
    max_dd = float(drawdowns[end_idx])

    # find peak index before max DD
    peak_value = running_max[end_idx]
    start_candidates = np.where(running_max[: end_idx + 1] == peak_value)[0]
    start_idx = int(start_candidates[0]) if start_candidates.size > 0 else 0

    return abs(max_dd * 100.0), start_idx, end_idx


def calculate_drawdown_duration(
    equity_curve: List[float],
    start_idx: int,
    end_idx: int,
) -> int:
    """
    Calculate drawdown duration in periods from start to recovery.

    Args:
        equity_curve: Portfolio values.
        start_idx: Index where drawdown starts (peak).
        end_idx: Index of drawdown trough.

    Returns:
        Duration (in periods) from start_idx to recovery (inclusive).
        If not recovered, returns length from start_idx to last point.
    """
    if not equity_curve or start_idx >= len(equity_curve):
        return 0

    equity = np.array(equity_curve, dtype=float)
    peak_value = equity[start_idx]

    for i in range(end_idx, len(equity)):
        if equity[i] >= peak_value:
            return i - start_idx

    return len(equity) - start_idx


def calculate_ulcer_index(equity_curve: List[float]) -> float:
    """
    Calculate Ulcer Index for an equity curve.  [web:196][web:199][web:202]

    Ulcer Index measures the depth and duration of drawdowns:
        UI = sqrt(average of squared percentage drawdowns).

    Args:
        equity_curve: Portfolio values.

    Returns:
        Ulcer Index as positive percentage value.
    """
    if not equity_curve or len(equity_curve) < 2:
        return 0.0

    equity = np.array(equity_curve, dtype=float)
    running_max = np.maximum.accumulate(equity)
    dd_pct = (equity - running_max) / running_max * 100.0
    squared = dd_pct**2
    ui = float(np.sqrt(np.mean(squared)))
    return ui


def get_all_drawdown_periods(equity_curve: List[float]) -> List[DrawdownPeriod]:
    """
    Identify all drawdown periods from an equity curve.

    A new drawdown starts when equity falls below prior running max
    and ends when equity recovers that high.

    Args:
        equity_curve: Portfolio values.

    Returns:
        List of DrawdownPeriod objects describing each episode.
    """
    if not equity_curve or len(equity_curve) < 2:
        return []

    equity = np.array(equity_curve, dtype=float)
    running_max = np.maximum.accumulate(equity)
    dd = (equity - running_max) / running_max * 100.0

    periods: List[DrawdownPeriod] = []
    in_dd = False
    start_idx = 0

    for i, dd_val in enumerate(dd):
        if dd_val < -0.01 and not in_dd:
            in_dd = True
            start_idx = i
        elif dd_val >= 0 and in_dd:
            end_idx = i - 1
            segment = dd[start_idx:i]
            max_dd = abs(float(np.min(segment)))
            duration = i - start_idx
            periods.append(
                DrawdownPeriod(
                    start_date=str(start_idx),
                    end_date=str(end_idx),
                    recovery_date=str(i),
                    drawdown_pct=max_dd,
                    duration_days=duration,
                    recovered=True,
                )
            )
            in_dd = False

    if in_dd:
        segment = dd[start_idx:]
        max_dd = abs(float(np.min(segment)))
        duration = len(equity) - start_idx
        periods.append(
            DrawdownPeriod(
                start_date=str(start_idx),
                end_date=str(len(equity) - 1),
                recovery_date=None,
                drawdown_pct=max_dd,
                duration_days=duration,
                recovered=False,
            )
        )

    return periods


# ──────────────────────────────────────────────────────────
# VALUE AT RISK (VaR) & CONDITIONAL VaR
# ──────────────────────────────────────────────────────────


def calculate_var(
    returns: List[float],
    confidence: float = 0.95,
    method: str = "historical",
) -> float:
    """
    Calculate Value at Risk (VaR) using historical or parametric methods.  [web:197][web:200]

    VaR at confidence level c (e.g. 0.95) is the loss threshold
    such that losses worse than VaR occur with probability (1 - c).

    Args:
        returns: List of periodic returns (decimal).
        confidence: Confidence level in (0, 1).
        method: "historical" (empirical quantile) or "parametric" (normal assumption).

    Returns:
        VaR as a positive percentage (e.g. 2.5 for 2.5%).
    """
    if not returns or len(returns) < 10:
        return 0.0

    r = np.array(returns, dtype=float)

    if method == "historical":
        q = np.percentile(r, (1.0 - confidence) * 100.0)
        var_value = abs(float(q))
    elif method == "parametric":
        mean = float(np.mean(r))
        std = float(np.std(r, ddof=1))
        if std == 0:
            return 0.0
        z = float(stats.norm.ppf(1.0 - confidence))
        var_value = abs(mean + z * std)
    else:
        raise ValueError(f"Unknown VaR method: {method}")

    return var_value * 100.0


def calculate_cvar(
    returns: List[float],
    confidence: float = 0.95,
) -> float:
    """
    Calculate Conditional VaR (CVaR, Expected Shortfall).  [web:197][web:200][web:203]

    CVaR is the mean loss conditional on being worse than VaR
    at the same confidence level.

    Args:
        returns: List of periodic returns (decimal).
        confidence: Confidence level.

    Returns:
        CVaR as positive percentage (e.g. 3.1 for 3.1%).
    """
    if not returns or len(returns) < 10:
        return 0.0

    r = np.array(returns, dtype=float)
    threshold = np.percentile(r, (1.0 - confidence) * 100.0)
    tail = r[r <= threshold]

    if tail.size == 0:
        return calculate_var(returns, confidence=confidence, method="historical")

    cvar_value = abs(float(np.mean(tail)))
    return cvar_value * 100.0


# ──────────────────────────────────────────────────────────
# BETA & ALPHA (Market Comparison)
# ──────────────────────────────────────────────────────────


def calculate_beta(
    asset_returns: List[float],
    market_returns: List[float],
) -> float:
    """
    Calculate portfolio Beta relative to market returns.

    Beta = Cov(asset, market) / Var(market)

    Args:
        asset_returns: Portfolio periodic returns.
        market_returns: Benchmark periodic returns.

    Returns:
        Beta coefficient. Returns 1.0 on failure/degenerate inputs.
    """
    if not asset_returns or not market_returns:
        return 1.0

    n = min(len(asset_returns), len(market_returns))
    if n < 2:
        return 1.0

    a = np.array(asset_returns[:n], dtype=float)
    m = np.array(market_returns[:n], dtype=float)

    cov = np.cov(a, m)[0, 1]
    var_m = np.var(m, ddof=1)

    if var_m == 0:
        return 1.0

    beta = cov / var_m
    return float(beta)


def calculate_alpha(
    portfolio_return_annual: float,
    market_return_annual: float,
    beta: float,
    risk_free_rate: float = 0.02,
) -> float:
    """
    Calculate annualized Alpha vs a benchmark.

    Alpha = Portfolio_Return - [rf + Beta * (Market_Return - rf)]

    Args:
        portfolio_return_annual: Portfolio annualized return (decimal).
        market_return_annual: Benchmark annualized return (decimal).
        beta: Portfolio beta.
        risk_free_rate: Annual risk-free rate (decimal).

    Returns:
        Alpha as annualized percentage (e.g. 2.3 for 2.3%).
    """
    expected = risk_free_rate + beta * (market_return_annual - risk_free_rate)
    alpha_dec = portfolio_return_annual - expected
    return alpha_dec * 100.0


def get_market_returns(
    period: str = "1y",
    benchmark: str = "SPY",
) -> List[float]:
    """
    Fetch daily benchmark returns via yfinance for comparison.

    Args:
        period: Lookback period (e.g. "1y").
        benchmark: Benchmark ticker (default: "SPY").

    Returns:
        List of daily decimal returns. Empty list on failure.
    """
    try:
        data = yf.download(
            benchmark,
            period=period,
            interval="1d",
            progress=False,
        )
        if data.empty or "Close" not in data.columns:
            return []
        ret = data["Close"].pct_change().dropna()
        return ret.tolist()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not fetch %s returns: %s", benchmark, exc)
        return []


# ──────────────────────────────────────────────────────────
# ADDITIONAL RISK METRICS
# ──────────────────────────────────────────────────────────


def calculate_calmar_ratio(
    annual_return: float,
    max_drawdown_pct: float,
) -> float:
    """
    Calculate Calmar Ratio.  [web:141][web:192][web:193]

    Calmar = Annual_Return / Max_Drawdown

    Args:
        annual_return: Annualized return as decimal (e.g. 0.15 for 15%).
        max_drawdown_pct: Maximum drawdown as positive percent.

    Returns:
        Calmar ratio. Returns 0.0 if max_drawdown_pct is zero.
    """
    if max_drawdown_pct <= 0:
        return 0.0
    return float(annual_return * 100.0 / max_drawdown_pct)


def calculate_omega_ratio(
    returns: List[float],
    threshold: float = 0.0,
) -> float:
    """
    Calculate Omega Ratio.

    Omega = sum(max(r - threshold, 0)) / sum(max(threshold - r, 0))

    Args:
        returns: List of periodic returns (decimal).
        threshold: Minimum acceptable return per period.

    Returns:
        Omega ratio. Returns inf if no losses below threshold.
    """
    if not returns:
        return 1.0

    r = np.array(returns, dtype=float)
    gains = np.maximum(r - threshold, 0.0)
    losses = np.maximum(threshold - r, 0.0)

    total_losses = float(np.sum(losses))
    if total_losses == 0:
        return float("inf")

    omega = float(np.sum(gains) / total_losses)
    return omega


def calculate_skewness_kurtosis(
    returns: List[float],
) -> Tuple[float, float]:
    """
    Calculate skewness and kurtosis of return distribution.

    Args:
        returns: List of periodic returns.

    Returns:
        (skewness, kurtosis) using Pearson kurtosis (3 = normal).
    """
    if not returns or len(returns) < 4:
        return 0.0, 0.0

    r = np.array(returns, dtype=float)
    skew = float(stats.skew(r))
    kurt = float(stats.kurtosis(r, fisher=False))
    return skew, kurt


# ──────────────────────────────────────────────────────────
# COMPREHENSIVE METRICS CALCULATION
# ──────────────────────────────────────────────────────────


def get_comprehensive_metrics(trader) -> RiskMetrics:
    """
    Compute the full set of risk and performance metrics for a portfolio.

    This function queries the PaperTrader for its equity curve and trade stats,
    then derives:
        - total & annualized returns,
        - Sharpe, Sortino, Calmar, Omega ratios,
        - max drawdown & duration, ulcer index,
        - VaR & CVaR (95% and 99%),
        - beta, alpha, correlation vs SPY,
        - skewness, kurtosis, winning-period percentage.

    Args:
        trader: PaperTrader instance with:
                - get_equity_curve()
                - get_portfolio_state()

    Returns:
        RiskMetrics dataclass populated with calculated metrics.
    """
    logger.info("Computing comprehensive risk metrics for portfolio...")

    equity_points = trader.get_equity_curve()
    if not equity_points or len(equity_points) < 2:
        logger.warning("Insufficient equity history for risk metrics.")
        return RiskMetrics(
            total_return=0.0,
            annualized_return=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            omega_ratio=0.0,
            max_drawdown_pct=0.0,
            max_drawdown_duration_days=0,
            current_drawdown_pct=0.0,
            avg_drawdown_pct=0.0,
            ulcer_index=0.0,
            volatility_annual=0.0,
            downside_deviation=0.0,
            var_95=0.0,
            var_99=0.0,
            cvar_95=0.0,
            cvar_99=0.0,
            beta=None,
            alpha=None,
            correlation_to_spy=None,
            skewness=0.0,
            kurtosis=0.0,
            winning_months_pct=0.0,
            days_traded=0,
            total_trades=0,
            calculated_at_utc=datetime.now(timezone.utc).isoformat(),
        )

    equity_curve = [pt.portfolio_value for pt in equity_points]
    equity = np.array(equity_curve, dtype=float)

    if hasattr(settings, "starting_cash"):
        starting_value = float(getattr(settings, "starting_cash"))
    else:
        starting_value = float(equity[0])

    returns = np.diff(equity) / equity[:-1]
    returns_list = returns.tolist()

    total_return = (equity[-1] / starting_value - 1.0) * 100.0 if starting_value > 0 else 0.0
    days_traded = len(equity)
    years_traded = days_traded / 252.0 if days_traded > 0 else 0.0
    if years_traded > 0 and starting_value > 0:
        annualized_return_dec = (equity[-1] / starting_value) ** (1.0 / years_traded) - 1.0
        annualized_return = annualized_return_dec * 100.0
    else:
        annualized_return_dec = 0.0
        annualized_return = 0.0

    sharpe = calculate_sharpe_ratio(returns_list)
    sortino = calculate_sortino_ratio(returns_list)

    max_dd_pct, dd_start, dd_end = calculate_max_drawdown(equity_curve)
    dd_duration = calculate_drawdown_duration(equity_curve, dd_start, dd_end)

    dd_periods = get_all_drawdown_periods(equity_curve)
    avg_dd = float(
        np.mean([p.drawdown_pct for p in dd_periods])
    ) if dd_periods else 0.0

    run_max = np.maximum.accumulate(equity)
    current_dd_pct = (
        (equity[-1] / run_max[-1] - 1.0) * 100.0 if run_max[-1] > 0 else 0.0
    )
    ulcer = calculate_ulcer_index(equity_curve)

    calmar = calculate_calmar_ratio(annualized_return_dec, max_dd_pct)
    omega = calculate_omega_ratio(returns_list)

    if returns.size > 1:
        volatility_annual = float(np.std(returns, ddof=1) * np.sqrt(252.0) * 100.0)
    else:
        volatility_annual = 0.0

    downside = returns[returns < 0]
    if downside.size > 0:
        downside_deviation = float(
            np.std(downside, ddof=1) * np.sqrt(252.0) * 100.0
        )
    else:
        downside_deviation = 0.0

    var_95 = calculate_var(returns_list, confidence=0.95, method="historical")
    var_99 = calculate_var(returns_list, confidence=0.99, method="historical")
    cvar_95 = calculate_cvar(returns_list, confidence=0.95)
    cvar_99 = calculate_cvar(returns_list, confidence=0.99)

    beta_val: Optional[float] = None
    alpha_val: Optional[float] = None
    correlation: Optional[float] = None
    try:
        market_returns = get_market_returns(period="1y", benchmark="SPY")
        if market_returns and len(market_returns) >= len(returns_list) and len(returns_list) > 10:
            n = len(returns_list)
            m = np.array(market_returns[-n:], dtype=float)
            r = np.array(returns_list, dtype=float)
            beta_est = calculate_beta(r.tolist(), m.tolist())
            beta_val = float(beta_est)

            market_annual_ret_dec = float(np.mean(m) * 252.0)
            alpha_val = calculate_alpha(
                annualized_return_dec,
                market_annual_ret_dec,
                beta_val,
                risk_free_rate=float(getattr(settings, "risk_free_rate", 0.02)),
            )
            if np.std(m) > 0 and np.std(r) > 0:
                correlation = float(np.corrcoef(r, m)[0, 1])
    except Exception as exc:  # noqa: BLE001
        logger.debug("Failed to compute beta/alpha/correlation: %s", exc)

    skew, kurt = calculate_skewness_kurtosis(returns_list)

    pos_periods = int(np.sum(returns > 0))
    winning_pct = float(pos_periods / len(returns) * 100.0) if len(returns) > 0 else 0.0

    portfolio_state = trader.get_portfolio_state()
    total_trades = int(getattr(portfolio_state, "total_trades", 0))

    metrics = RiskMetrics(
        total_return=float(total_return),
        annualized_return=float(annualized_return),
        sharpe_ratio=float(sharpe),
        sortino_ratio=float(sortino),
        calmar_ratio=float(calmar),
        omega_ratio=float(omega),
        max_drawdown_pct=float(max_dd_pct),
        max_drawdown_duration_days=int(dd_duration),
        current_drawdown_pct=abs(float(current_dd_pct)),
        avg_drawdown_pct=float(avg_dd),
        ulcer_index=float(ulcer),
        volatility_annual=float(volatility_annual),
        downside_deviation=float(downside_deviation),
        var_95=float(var_95),
        var_99=float(var_99),
        cvar_95=float(cvar_95),
        cvar_99=float(cvar_99),
        beta=beta_val,
        alpha=alpha_val,
        correlation_to_spy=correlation,
        skewness=float(skew),
        kurtosis=float(kurt),
        winning_months_pct=float(winning_pct),
        days_traded=int(days_traded),
        total_trades=total_trades,
        calculated_at_utc=datetime.now(timezone.utc).isoformat(),
    )

    logger.info(
        "Risk metrics: Sharpe=%.2f, Sortino=%.2f, MaxDD=%.2f%%, "
        "AnnRet=%.2f%%, Var95=%.2f%%.",
        metrics.sharpe_ratio,
        metrics.sortino_ratio,
        metrics.max_drawdown_pct,
        metrics.annualized_return,
        metrics.var_95,
    )
    return metrics


# ──────────────────────────────────────────────────────────
# CLI TOOL & TESTING
# ──────────────────────────────────────────────────────────


if __name__ == "__main__":
    from portfolio import PaperTrader

    print("\n" + "=" * 70)
    print("ALPHA-PRIME Risk Metrics Calculator - Test Tool")
    print("=" * 70 + "\n")

    trader = PaperTrader()
    metrics = get_comprehensive_metrics(trader)

    print("RETURN METRICS")
    print("-" * 70)
    print(f"Total Return         : {metrics.total_return:+.2f}%")
    print(f"Annualized Return    : {metrics.annualized_return:+.2f}%")
    print(f"Days Traded          : {metrics.days_traded}")
    print(f"Total Trades         : {metrics.total_trades}\n")

    print("RISK-ADJUSTED RETURNS")
    print("-" * 70)
    print(f"Sharpe Ratio         : {metrics.sharpe_ratio:.2f}")
    print(f"Sortino Ratio        : {metrics.sortino_ratio:.2f}")
    print(f"Calmar Ratio         : {metrics.calmar_ratio:.2f}")
    print(f"Omega Ratio          : {metrics.omega_ratio:.2f}\n")

    print("DRAWDOWN ANALYSIS")
    print("-" * 70)
    print(f"Max Drawdown         : {metrics.max_drawdown_pct:.2f}%")
    print(f"Max DD Duration      : {metrics.max_drawdown_duration_days} days")
    print(f"Current Drawdown     : {metrics.current_drawdown_pct:.2f}%")
    print(f"Average Drawdown     : {metrics.avg_drawdown_pct:.2f}%")
    print(f"Ulcer Index          : {metrics.ulcer_index:.2f}\n")

    print("VOLATILITY")
    print("-" * 70)
    print(f"Annual Volatility    : {metrics.volatility_annual:.2f}%")
    print(f"Downside Deviation   : {metrics.downside_deviation:.2f}%\n")

    print("RISK METRICS")
    print("-" * 70)
    print(f"VaR (95%%)           : {metrics.var_95:.2f}%")
    print(f"VaR (99%%)           : {metrics.var_99:.2f}%")
    print(f"CVaR (95%%)          : {metrics.cvar_95:.2f}%")
    print(f"CVaR (99%%)          : {metrics.cvar_99:.2f}%\n")

    if metrics.beta is not None:
        print("MARKET COMPARISON (vs SPY)")
        print("-" * 70)
        print(f"Beta                 : {metrics.beta:.2f}")
        if metrics.alpha is not None:
            print(f"Alpha                : {metrics.alpha:+.2f}%")
        if metrics.correlation_to_spy is not None:
            print(f"Correlation to SPY   : {metrics.correlation_to_spy:.2f}\n")

    print("DISTRIBUTION")
    print("-" * 70)
    print(f"Skewness             : {metrics.skewness:.2f}")
    print(f"Kurtosis             : {metrics.kurtosis:.2f}")
    print(f"Winning Periods      : {metrics.winning_months_pct:.1f}%\n")

    print("=" * 70 + "\n")
