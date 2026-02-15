"""
============================================================
ALPHA-PRIME v2.0 - Walk-Forward Analysis
============================================================

Rigorous backtesting methodology for strategy validation:

What is Walk-Forward Analysis?
Walk-forward analysis is a backtesting technique that:
1. Divides historical data into training/testing windows.
2. Optimizes strategy parameters on training data (in-sample).
3. Validates performance on testing data (out-of-sample).
4. Rolls the window forward and repeats.
5. Aggregates out-of-sample results for a realistic performance estimate. [web:339][web:341][web:343][web:344]

Why Walk-Forward?
- Traditional single-sample backtests are prone to curve-fitting.
- Optimizing on the full dataset risks exploiting noise, not signal. [web:336][web:340][web:347]
- Walk-forward enforces a strict time split between optimisation and evaluation.
- It approximates a live workflow: optimize today, trade tomorrow.
- It reveals whether a strategy degrades across regimes and recent periods.

Example:
Historical Data: Jan 2023 – Dec 2025 (~3 years)

Window 1: Train (Jan–Jun 2023) → Test (Jul–Aug 2023)
Window 2: Train (Apr–Sep 2023) → Test (Oct–Nov 2023)
Window 3: Train (Jul–Dec 2023) → Test (Jan–Feb 2024)
...continue rolling forward.

Aggregate all test segments to approximate live performance.

Performance Degradation Test:
- In-Sample (IS): Performance on optimisation data.
- Out-of-Sample (OOS): Performance on unseen validation data.
- Degradation Ratio: OOS / IS statistic (e.g. Sharpe, return). [web:338][web:344][web:346]

Interpreting Degradation Ratio:
- >1.0: Strategy appears to improve OOS (rare; check for instability).
- 0.7–1.0: Healthy (expected degradation).
- 0.5–0.7: Moderate overfitting risk.
- <0.5: Severe overfitting (reject or rework strategy).

Key Metrics (OOS-focused):
1. Sharpe ratio (risk-adjusted return).
2. Win rate and profit factor.
3. Maximum drawdown and volatility.
4. Stability across windows.
5. Recent-window behaviour (no sharp deterioration). [web:336][web:339][web:340][web:344]

Usage:
    from walk_forward import run_walk_forward_analysis, WalkForwardConfig

    config = WalkForwardConfig(
        in_sample_days=180,
        out_sample_days=60,
        step_size_days=30,
        min_trades=10,
    )

    result = run_walk_forward_analysis(
        ticker="AAPL",
        strategy="momentum",
        config=config,
    )

    print(f"OOS Sharpe: {result.aggregate_oos.sharpe_ratio:.2f}")
    print(f"Degradation: {result.degradation_ratio:.2%}")
    if result.overfitting_detected:
        print("⚠️ Strategy appears overfit to in-sample data.")

Integration:
- Standalone validation tool for ALPHA-PRIME strategies.
- Run prior to promotion to live or paper trading.
- Re-run periodically to monitor strategy drift and degradation.
============================================================
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from itertools import product
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf  # optional fallback, primary data via data_engine

from config import get_logger, get_settings
from data_engine import get_market_data

logger = get_logger(__name__)
settings = get_settings()


# ──────────────────────────────────────────────────────────
# DATA SCHEMAS
# ──────────────────────────────────────────────────────────


@dataclass
class WalkForwardConfig:
    """
    Configuration for walk-forward optimisation and validation.

    Attributes:
        in_sample_days: Training window length in trading days.
        out_sample_days: Testing window length in trading days.
        step_size_days: Forward step between consecutive windows.
        min_data_points: Minimum total bars required to run WFA.
        min_trades: Minimum trades per window to consider metrics meaningful.
        initial_capital: Starting capital for each window backtest.
        commission: Commission rate per side (fraction of traded notional).
        slippage: Slippage fraction applied to entry and exit prices.
        param_grid: Optional hyperparameter grid for optimisation.
    """

    in_sample_days: int = 180
    out_sample_days: int = 60
    step_size_days: int = 30
    min_data_points: int = 200
    min_trades: int = 5
    initial_capital: float = 10_000.0
    commission: float = 0.001
    slippage: float = 0.001
    param_grid: Optional[Dict[str, List[Any]]] = None


@dataclass
class Trade:
    """Individual trade record used for performance statistics."""

    entry_date: datetime
    exit_date: datetime
    entry_price: float
    exit_price: float
    shares: int
    pnl: float
    pnl_pct: float
    holding_days: int
    signal: str  # e.g. "LONG", "SHORT"


@dataclass
class PerformanceMetrics:
    """
    Comprehensive performance metrics for a backtest segment.

    All returns and ratios are expressed as percentages where applicable.
    """

    total_return: float
    cagr: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    volatility: float
    win_rate: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    avg_trade: float
    expectancy: float
    equity_curve: List[float]


@dataclass
class DataWindow:
    """Represents a single in-sample/out-of-sample window."""

    window_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_data: pd.DataFrame
    test_data: pd.DataFrame


@dataclass
class WindowResult:
    """Results for a single walk-forward window."""

    window_id: int
    train_period: Tuple[datetime, datetime]
    test_period: Tuple[datetime, datetime]
    optimized_params: Dict[str, Any]
    in_sample_metrics: PerformanceMetrics
    out_sample_metrics: PerformanceMetrics
    degradation_ratio: float
    trades: List[Trade]


@dataclass
class OverfitReport:
    """
    Overfitting analysis across all windows.

    Attributes:
        overfitting_detected: Flag indicating likely overfit.
        degradation_ratio: Average OOS/IS Sharpe ratio.
        in_sample_sharpe: Average in-sample Sharpe.
        out_sample_sharpe: Average out-of-sample Sharpe.
        in_sample_win_rate: Average in-sample win rate.
        out_sample_win_rate: Average out-of-sample win rate.
        consistency_score: 0–100, higher is more consistent across windows.
        warnings: Descriptive warnings and notes.
    """

    overfitting_detected: bool
    degradation_ratio: float
    in_sample_sharpe: float
    out_sample_sharpe: float
    in_sample_win_rate: float
    out_sample_win_rate: float
    consistency_score: float
    warnings: List[str]


@dataclass
class AggregateResults:
    """
    Aggregate metrics across windows.

    Attributes:
        total_windows: Number of valid windows.
        aggregate_metrics: PerformanceMetrics based on concatenated OOS equity.
        avg_degradation_ratio: Mean window-level degradation.
        consistency_score: Stability of OOS Sharpe across windows.
        best_window: Window ID with highest OOS Sharpe.
        worst_window: Window ID with lowest OOS Sharpe.
        recent_performance_trend: IMPROVING | STABLE | DECLINING.
    """

    total_windows: int
    aggregate_metrics: PerformanceMetrics
    avg_degradation_ratio: float
    consistency_score: float
    best_window: int
    worst_window: int
    recent_performance_trend: str


@dataclass
class WalkForwardResult:
    """
    Top-level walk-forward analysis result.

    Attributes:
        ticker: Symbol tested.
        strategy: Strategy identifier.
        config: WalkForwardConfig used.
        window_results: List of per-window results.
        aggregate_is: Aggregate in-sample metrics (concatenated).
        aggregate_oos: Aggregate out-of-sample metrics (concatenated).
        degradation_ratio: Global OOS/IS Sharpe ratio.
        overfitting_detected: From OverfitReport.
        overfit_report: Detailed overfitting diagnostics.
        aggregate_results: AggregateResults helper.
        analysis_timestamp: ISO timestamp when analysis completed.
    """

    ticker: str
    strategy: str
    config: WalkForwardConfig
    window_results: List[WindowResult]
    aggregate_is: PerformanceMetrics
    aggregate_oos: PerformanceMetrics
    degradation_ratio: float
    overfitting_detected: bool
    overfit_report: OverfitReport
    aggregate_results: AggregateResults
    analysis_timestamp: str


# ──────────────────────────────────────────────────────────
# DATA WINDOWING
# ──────────────────────────────────────────────────────────


def split_data_windows(
    data: pd.DataFrame,
    in_sample_days: int,
    out_sample_days: int,
    step_size_days: int,
    min_data_points: int,
) -> List[DataWindow]:
    """
    Split historical data into rolling train/test windows. [web:341][web:343]

    Windows are defined in index order (assumed chronological).
    Training and test windows do not overlap within a window, but
    successive windows can overlap depending on step_size_days.

    Args:
        data: Price/feature DataFrame indexed by datetime.
        in_sample_days: Length of training window in rows.
        out_sample_days: Length of test window in rows.
        step_size_days: Forward step between window starts in rows.
        min_data_points: Minimum bars required to attempt WFA.

    Returns:
        List of DataWindow objects.

    Raises:
        ValueError: If data length is insufficient.
    """
    if len(data) < min_data_points:
        raise ValueError(f"Insufficient data: {len(data)} < {min_data_points}")

    data = data.sort_index()
    windows: List[DataWindow] = []
    total_window = in_sample_days + out_sample_days
    start_idx = 0
    window_id = 1

    while start_idx + total_window <= len(data):
        train_end_idx = start_idx + in_sample_days
        test_end_idx = train_end_idx + out_sample_days

        train_data = data.iloc[start_idx:train_end_idx].copy()
        test_data = data.iloc[train_end_idx:test_end_idx].copy()

        if train_data.empty or test_data.empty:
            break

        window = DataWindow(
            window_id=window_id,
            train_start=train_data.index[0],
            train_end=train_data.index[-1],
            test_start=test_data.index[0],
            test_end=test_data.index[-1],
            train_data=train_data,
            test_data=test_data,
        )
        windows.append(window)

        logger.info(
            "Window %d: Train [%s → %s], Test [%s → %s].",
            window_id,
            window.train_start.date(),
            window.train_end.date(),
            window.test_start.date(),
            window.test_end.date(),
        )

        window_id += 1
        start_idx += step_size_days

    logger.info("Created %d walk-forward windows.", len(windows))
    return windows


# ──────────────────────────────────────────────────────────
# SIMPLE STRATEGY IMPLEMENTATION (EXAMPLE)
# ──────────────────────────────────────────────────────────


def simple_momentum_strategy(
    data: pd.DataFrame,
    ema_fast: int = 9,
    ema_slow: int = 21,
    rsi_period: int = 14,
    rsi_overbought: int = 70,
    rsi_oversold: int = 30,
) -> pd.DataFrame:
    """
    Example momentum strategy used for demonstration.

    Rules (long-only):
        - Buy when fast EMA crosses above slow EMA and RSI < rsi_overbought.
        - Exit when fast EMA crosses below slow EMA or RSI > rsi_overbought.

    Args:
        data: OHLCV DataFrame with 'Close' and 'Volume'.
        ema_fast: Fast EMA lookback.
        ema_slow: Slow EMA lookback.
        rsi_period: RSI period.
        rsi_overbought: Upper RSI threshold.
        rsi_oversold: Lower RSI threshold (not used in exit rule here).

    Returns:
        DataFrame with additional columns and a 'Signal' column
        (1 for long entry, -1 for exit, 0 otherwise).
    """
    df = data.copy()
    close = df["Close"]

    df["EMA_Fast"] = close.ewm(span=ema_fast, adjust=False).mean()
    df["EMA_Slow"] = close.ewm(span=ema_slow, adjust=False).mean()

    delta = close.diff()
    gain = delta.clip(lower=0.0).rolling(window=rsi_period).mean()
    loss = (-delta.clip(upper=0.0)).rolling(window=rsi_period).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI"] = 100.0 - (100.0 / (1.0 + rs))

    df["Signal"] = 0

    buy_mask = (
        (df["EMA_Fast"] > df["EMA_Slow"])
        & (df["EMA_Fast"].shift(1) <= df["EMA_Slow"].shift(1))
        & (df["RSI"] < rsi_overbought)
    )
    df.loc[buy_mask, "Signal"] = 1

    sell_mask = (
        (df["EMA_Fast"] < df["EMA_Slow"])
        & (df["EMA_Fast"].shift(1) >= df["EMA_Slow"].shift(1))
    ) | (df["RSI"] > rsi_overbought)
    df.loc[sell_mask, "Signal"] = -1

    return df


# ──────────────────────────────────────────────────────────
# BACKTESTING ENGINE
# ──────────────────────────────────────────────────────────


def backtest_window(
    data: pd.DataFrame,
    strategy: str,
    params: Optional[Dict[str, Any]],
    initial_capital: float,
    commission: float,
    slippage: float,
) -> Tuple[List[Trade], List[float]]:
    """
    Backtest a single window for a given strategy and parameter set.

    Current implementation:
        - Long-only, single-position strategy.
        - All-in on entry, flat on exit.

    Args:
        data: Price/feature DataFrame.
        strategy: Strategy identifier ("momentum" supported).
        params: Strategy parameters.
        initial_capital: Starting capital for this window.
        commission: Commission fraction per trade.
        slippage: Slippage fraction applied to fills.

    Returns:
        (trades, equity_curve) tuple.
    """
    params = params or {}

    if strategy == "momentum":
        df = simple_momentum_strategy(
            data,
            ema_fast=int(params.get("ema_fast", 9)),
            ema_slow=int(params.get("ema_slow", 21)),
            rsi_period=int(params.get("rsi_period", 14)),
            rsi_overbought=int(params.get("rsi_overbought", 70)),
            rsi_oversold=int(params.get("rsi_oversold", 30)),
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    trades: List[Trade] = []
    equity_curve: List[float] = []

    cash = float(initial_capital)
    position = 0
    entry_price = 0.0
    entry_date: Optional[datetime] = None

    for i in range(1, len(df)):
        price = float(df["Close"].iloc[i])
        signal = int(df["Signal"].iloc[i])
        date = df.index[i].to_pydatetime() if hasattr(df.index[i], "to_pydatetime") else df.index[i]

        if signal == 1 and position == 0:
            fill_price = price * (1.0 + slippage)
            shares = int(cash / (fill_price * (1.0 + commission)))
            if shares > 0:
                cost = shares * fill_price * (1.0 + commission)
                cash -= cost
                position = shares
                entry_price = fill_price
                entry_date = date

        elif signal == -1 and position > 0:
            fill_price = price * (1.0 - slippage)
            proceeds = position * fill_price * (1.0 - commission)
            cash += proceeds

            if entry_date is None:
                entry_date = date

            pnl = proceeds - (position * entry_price * (1.0 + commission))
            pnl_pct = (fill_price / entry_price - 1.0) * 100.0
            holding_days = (date - entry_date).days

            trade = Trade(
                entry_date=entry_date,
                exit_date=date,
                entry_price=entry_price,
                exit_price=fill_price,
                shares=position,
                pnl=float(pnl),
                pnl_pct=float(pnl_pct),
                holding_days=int(holding_days),
                signal="LONG",
            )
            trades.append(trade)
            position = 0
            entry_price = 0.0
            entry_date = None

        portfolio_value = cash + (position * price if position > 0 else 0.0)
        equity_curve.append(float(portfolio_value))

    if position > 0 and entry_date is not None:
        price = float(df["Close"].iloc[-1])
        fill_price = price * (1.0 - slippage)
        proceeds = position * fill_price * (1.0 - commission)
        cash += proceeds
        pnl = proceeds - (position * entry_price * (1.0 + commission))
        pnl_pct = (fill_price / entry_price - 1.0) * 100.0
        holding_days = (df.index[-1] - pd.Timestamp(entry_date)).days
        trades.append(
            Trade(
                entry_date=entry_date,
                exit_date=df.index[-1].to_pydatetime(),
                entry_price=entry_price,
                exit_price=fill_price,
                shares=position,
                pnl=float(pnl),
                pnl_pct=float(pnl_pct),
                holding_days=int(holding_days),
                signal="LONG",
            )
        )
        position = 0
        equity_curve.append(float(cash))

    if not equity_curve:
        equity_curve = [initial_capital]

    return trades, equity_curve


# ──────────────────────────────────────────────────────────
# PERFORMANCE METRICS
# ──────────────────────────────────────────────────────────


def calculate_performance_metrics(
    trades: List[Trade],
    equity_curve: List[float],
    initial_capital: float,
) -> PerformanceMetrics:
    """
    Compute performance metrics for a set of trades and equity curve. [web:336][web:339]

    Args:
        trades: Executed trades.
        equity_curve: Sequence of portfolio values over time.
        initial_capital: Starting capital.

    Returns:
        PerformanceMetrics object.
    """
    if not equity_curve:
        equity_curve = [initial_capital]

    if not trades:
        return PerformanceMetrics(
            total_return=0.0,
            cagr=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            max_drawdown=0.0,
            volatility=0.0,
            win_rate=0.0,
            profit_factor=0.0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            avg_win=0.0,
            avg_loss=0.0,
            avg_trade=0.0,
            expectancy=0.0,
            equity_curve=list(equity_curve),
        )

    total_trades = len(trades)
    winning_trades = sum(1 for t in trades if t.pnl > 0)
    losing_trades = sum(1 for t in trades if t.pnl < 0)

    final_capital = float(equity_curve[-1])
    total_return = (final_capital / float(initial_capital) - 1.0) * 100.0

    wins = [t.pnl for t in trades if t.pnl > 0]
    losses = [abs(t.pnl) for t in trades if t.pnl < 0]
    avg_win = float(np.mean(wins)) if wins else 0.0
    avg_loss = float(np.mean(losses)) if losses else 0.0
    avg_trade = float(np.mean([t.pnl for t in trades])) if trades else 0.0

    win_rate = (winning_trades / total_trades * 100.0) if total_trades > 0 else 0.0
    gross_profit = float(sum(wins)) if wins else 0.0
    gross_loss = float(sum(losses)) if losses else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
    expectancy = (win_rate / 100.0 * avg_win) - ((100.0 - win_rate) / 100.0 * avg_loss)

    equity_series = pd.Series(equity_curve)
    returns = equity_series.pct_change().dropna()

    if len(returns) > 1:
        volatility = float(returns.std() * np.sqrt(252) * 100.0)
        if volatility > 0:
            sharpe_ratio = float((returns.mean() * 252) / returns.std())
        else:
            sharpe_ratio = 0.0

        downside = returns[returns < 0]
        downside_std = float(downside.std() * np.sqrt(252)) if len(downside) > 0 else 0.0
        sortino_ratio = float((returns.mean() * 252) / downside_std) if downside_std > 0 else 0.0

        cumulative_max = equity_series.cummax()
        drawdown = (equity_series - cumulative_max) / cumulative_max * 100.0
        max_drawdown = float(abs(drawdown.min()))
    else:
        volatility = 0.0
        sharpe_ratio = 0.0
        sortino_ratio = 0.0
        max_drawdown = 0.0

    if len(equity_curve) > 1:
        days = len(equity_curve)
        years = days / 252.0
        cagr = ((final_capital / float(initial_capital)) ** (1.0 / years) - 1.0) * 100.0 if years > 0 else 0.0
    else:
        cagr = 0.0

    return PerformanceMetrics(
        total_return=float(total_return),
        cagr=float(cagr),
        sharpe_ratio=float(sharpe_ratio),
        sortino_ratio=float(sortino_ratio),
        max_drawdown=float(max_drawdown),
        volatility=float(volatility),
        win_rate=float(win_rate),
        profit_factor=float(profit_factor),
        total_trades=int(total_trades),
        winning_trades=int(winning_trades),
        losing_trades=int(losing_trades),
        avg_win=float(avg_win),
        avg_loss=float(avg_loss),
        avg_trade=float(avg_trade),
        expectancy=float(expectancy),
        equity_curve=[float(v) for v in equity_curve],
    )


# ──────────────────────────────────────────────────────────
# PARAMETER OPTIMISATION
# ──────────────────────────────────────────────────────────


def optimize_parameters(
    train_data: pd.DataFrame,
    strategy: str,
    param_grid: Optional[Dict[str, List[Any]]],
    config: WalkForwardConfig,
) -> Dict[str, Any]:
    """
    Optimise strategy parameters using in-sample data.

    Currently performs a brute-force grid search on the parameter grid,
    using in-sample Sharpe ratio as the objective. [web:336][web:340][web:342]

    Args:
        train_data: In-sample DataFrame.
        strategy: Strategy identifier.
        param_grid: Parameter grid; falls back to a default for "momentum".
        config: WalkForwardConfig (for capital, costs, min_trades).

    Returns:
        Dict of best parameter values.
    """
    if param_grid is None:
        if strategy == "momentum":
            param_grid = {
                "ema_fast": [8, 10, 12],
                "ema_slow": [18, 21, 26],
                "rsi_period": [10, 14],
                "rsi_overbought": [65, 70],
                "rsi_oversold": [30],
            }
        else:
            raise ValueError(f"No default param_grid defined for strategy '{strategy}'.")

    logger.info("Optimising parameters for strategy '%s' on in-sample data.", strategy)

    best_sharpe = -np.inf
    best_params: Dict[str, Any] = {}

    keys = list(param_grid.keys())
    values_list = list(param_grid.values())
    for combo in product(*values_list):
        params = dict(zip(keys, combo))
        try:
            trades, equity = backtest_window(
                train_data,
                strategy=strategy,
                params=params,
                initial_capital=config.initial_capital,
                commission=config.commission,
                slippage=config.slippage,
            )
            if len(trades) < config.min_trades:
                continue

            metrics = calculate_performance_metrics(trades, equity, config.initial_capital)
            if metrics.sharpe_ratio > best_sharpe:
                best_sharpe = metrics.sharpe_ratio
                best_params = params
        except Exception as exc:  # noqa: BLE001
            logger.debug("Error evaluating params %s: %s", params, exc, exc_info=False)
            continue

    logger.info("Best parameters: %s (IS Sharpe=%.2f).", best_params, best_sharpe)
    return best_params


# ──────────────────────────────────────────────────────────
# OVERFITTING DETECTION
# ──────────────────────────────────────────────────────────


def detect_overfitting(window_results: List[WindowResult]) -> OverfitReport:
    """
    Detect overfitting by comparing IS and OOS metrics across windows. [web:338][web:344][web:346][web:348]

    Criteria:
        - Average degradation ratio (OOS Sharpe / IS Sharpe).
        - Average OOS Sharpe and win rate.
        - Consistency via standard deviation of OOS Sharpe.

    Args:
        window_results: List of per-window results.

    Returns:
        OverfitReport summarising robustness.
    """
    if not window_results:
        return OverfitReport(
            overfitting_detected=True,
            degradation_ratio=0.0,
            in_sample_sharpe=0.0,
            out_sample_sharpe=0.0,
            in_sample_win_rate=0.0,
            out_sample_win_rate=0.0,
            consistency_score=0.0,
            warnings=["No walk-forward windows to analyse."],
        )

    degrs = [w.degradation_ratio for w in window_results]
    avg_degr = float(np.mean(degrs))

    is_sharpes = [w.in_sample_metrics.sharpe_ratio for w in window_results]
    oos_sharpes = [w.out_sample_metrics.sharpe_ratio for w in window_results]
    is_wr = [w.in_sample_metrics.win_rate for w in window_results]
    oos_wr = [w.out_sample_metrics.win_rate for w in window_results]

    avg_is_sharpe = float(np.mean(is_sharpes))
    avg_oos_sharpe = float(np.mean(oos_sharpes))
    avg_is_wr = float(np.mean(is_wr))
    avg_oos_wr = float(np.mean(oos_wr))

    oos_sharpe_std = float(np.std(oos_sharpes))
    consistency_score = max(0.0, 100.0 - oos_sharpe_std * 50.0)

    warnings: List[str] = []
    overfit = False

    if avg_degr < 0.5:
        overfit = True
        warnings.append(f"Severe degradation: average OOS/IS Sharpe = {avg_degr:.2f} (< 0.50).")
    elif avg_degr < 0.7:
        warnings.append(f"Moderate degradation: average OOS/IS Sharpe = {avg_degr:.2f} (0.50–0.70).")

    if avg_oos_sharpe < 0.0:
        overfit = True
        warnings.append(f"Negative average OOS Sharpe ({avg_oos_sharpe:.2f}).")

    if consistency_score < 50.0:
        warnings.append(f"Low OOS consistency: {consistency_score:.0f}/100 (high Sharpe variability).")

    if not warnings:
        warnings.append("No obvious signs of overfitting detected.")

    return OverfitReport(
        overfitting_detected=overfit,
        degradation_ratio=avg_degr,
        in_sample_sharpe=avg_is_sharpe,
        out_sample_sharpe=avg_oos_sharpe,
        in_sample_win_rate=avg_is_wr,
        out_sample_win_rate=avg_oos_wr,
        consistency_score=consistency_score,
        warnings=warnings,
    )


# ──────────────────────────────────────────────────────────
# AGGREGATION HELPERS
# ──────────────────────────────────────────────────────────


def aggregate_results(window_results: List[WindowResult], initial_capital: float) -> AggregateResults:
    """
    Aggregate window results into global metrics.

    Aggregation is performed on out-of-sample equity curves, with
    degradation-based and consistency scores summarised.

    Args:
        window_results: List of WindowResult.
        initial_capital: Starting capital per window.

    Returns:
        AggregateResults instance.
    """
    if not window_results:
        empty_metrics = calculate_performance_metrics([], [initial_capital], initial_capital)
        return AggregateResults(
            total_windows=0,
            aggregate_metrics=empty_metrics,
            avg_degradation_ratio=0.0,
            consistency_score=0.0,
            best_window=0,
            worst_window=0,
            recent_performance_trend="STABLE",
        )

    concat_equity: List[float] = []
    for w in window_results:
        concat_equity.extend(w.out_sample_metrics.equity_curve or [initial_capital])

    aggregate_metrics = calculate_performance_metrics([], concat_equity, initial_capital)

    degrs = [w.degradation_ratio for w in window_results]
    avg_degr = float(np.mean(degrs))

    oos_sharpes = [w.out_sample_metrics.sharpe_ratio for w in window_results]
    oos_sharpe_std = float(np.std(oos_sharpes))
    consistency_score = max(0.0, 100.0 - oos_sharpe_std * 50.0)

    best_idx = int(np.argmax(oos_sharpes))
    worst_idx = int(np.argmin(oos_sharpes))
    best_window = window_results[best_idx].window_id
    worst_window = window_results[worst_idx].window_id

    half = max(1, len(window_results) // 2)
    first_half_sharpe = float(np.mean(oos_sharpes[:half]))
    second_half_sharpe = float(np.mean(oos_sharpes[half:]))

    if second_half_sharpe > first_half_sharpe + 0.2:
        trend = "IMPROVING"
    elif second_half_sharpe < first_half_sharpe - 0.2:
        trend = "DECLINING"
    else:
        trend = "STABLE"

    return AggregateResults(
        total_windows=len(window_results),
        aggregate_metrics=aggregate_metrics,
        avg_degradation_ratio=avg_degr,
        consistency_score=consistency_score,
        best_window=best_window,
        worst_window=worst_window,
        recent_performance_trend=trend,
    )


# ──────────────────────────────────────────────────────────
# MAIN WALK-FORWARD ANALYSIS
# ──────────────────────────────────────────────────────────


def run_walk_forward_analysis(
    ticker: str,
    strategy: str = "momentum",
    config: Optional[WalkForwardConfig] = None,
) -> WalkForwardResult:
    """
    Run a complete walk-forward analysis for a single ticker and strategy. [web:339][web:344][web:345][web:347]

    Steps:
        1. Download market data.
        2. Split into rolling train/test windows.
        3. For each window:
           - Optimise parameters on training segment.
           - Backtest in-sample with optimal parameters.
           - Backtest out-of-sample with same parameters.
           - Compute degradation ratios and store results.
        4. Aggregate per-window results into global IS/OOS statistics.
        5. Detect overfitting and compute degradation analysis.

    Args:
        ticker: Symbol to analyse.
        strategy: Strategy identifier (currently "momentum").
        config: WalkForwardConfig; defaults provided if None.

    Returns:
        WalkForwardResult object.
    """
    cfg = config or WalkForwardConfig()
    ticker_u = ticker.upper()
    logger.info(
        "Starting walk-forward analysis for %s (strategy=%s, IS=%d, OOS=%d, step=%d).",
        ticker_u,
        strategy,
        cfg.in_sample_days,
        cfg.out_sample_days,
        cfg.step_size_days,
    )

    total_req = cfg.in_sample_days + cfg.out_sample_days + cfg.step_size_days * 10
    period_str = f"{max(total_req, cfg.min_data_points)}d"

    data = get_market_data(ticker_u, period=period_str, interval="1d")
    if data is None or data.empty:
        raise ValueError(f"No historical data available for {ticker_u}.")

    windows = split_data_windows(
        data=data,
        in_sample_days=cfg.in_sample_days,
        out_sample_days=cfg.out_sample_days,
        step_size_days=cfg.step_size_days,
        min_data_points=cfg.min_data_points,
    )
    if not windows:
        raise ValueError("Unable to create any walk-forward windows with given configuration.")

    window_results: List[WindowResult] = []

    for window in windows:
        logger.info("Processing window %d.", window.window_id)

        params = optimize_parameters(
            train_data=window.train_data,
            strategy=strategy,
            param_grid=cfg.param_grid,
            config=cfg,
        )

        is_trades, is_equity = backtest_window(
            data=window.train_data,
            strategy=strategy,
            params=params,
            initial_capital=cfg.initial_capital,
            commission=cfg.commission,
            slippage=cfg.slippage,
        )
        is_metrics = calculate_performance_metrics(is_trades, is_equity, cfg.initial_capital)

        oos_trades, oos_equity = backtest_window(
            data=window.test_data,
            strategy=strategy,
            params=params,
            initial_capital=cfg.initial_capital,
            commission=cfg.commission,
            slippage=cfg.slippage,
        )
        oos_metrics = calculate_performance_metrics(oos_trades, oos_equity, cfg.initial_capital)

        if is_metrics.sharpe_ratio > 0:
            degradation = oos_metrics.sharpe_ratio / is_metrics.sharpe_ratio
        else:
            degradation = 0.0

        window_result = WindowResult(
            window_id=window.window_id,
            train_period=(window.train_start, window.train_end),
            test_period=(window.test_start, window.test_end),
            optimized_params=params,
            in_sample_metrics=is_metrics,
            out_sample_metrics=oos_metrics,
            degradation_ratio=float(degradation),
            trades=oos_trades,
        )
        window_results.append(window_result)

        logger.info(
            "Window %d: IS Sharpe=%.2f, OOS Sharpe=%.2f, degradation=%.2f.",
            window.window_id,
            is_metrics.sharpe_ratio,
            oos_metrics.sharpe_ratio,
            degradation,
        )

    concat_is_equity: List[float] = []
    concat_oos_equity: List[float] = []
    for w in window_results:
        concat_is_equity.extend(w.in_sample_metrics.equity_curve or [cfg.initial_capital])
        concat_oos_equity.extend(w.out_sample_metrics.equity_curve or [cfg.initial_capital])

    aggregate_is = calculate_performance_metrics([], concat_is_equity, cfg.initial_capital)
    aggregate_oos = calculate_performance_metrics([], concat_oos_equity, cfg.initial_capital)

    if aggregate_is.sharpe_ratio > 0:
        global_degr = aggregate_oos.sharpe_ratio / aggregate_is.sharpe_ratio
    else:
        global_degr = 0.0

    overfit_report = detect_overfitting(window_results)
    agg_results = aggregate_results(window_results, cfg.initial_capital)

    result = WalkForwardResult(
        ticker=ticker_u,
        strategy=strategy,
        config=cfg,
        window_results=window_results,
        aggregate_is=aggregate_is,
        aggregate_oos=aggregate_oos,
        degradation_ratio=float(global_degr),
        overfitting_detected=overfit_report.overfitting_detected,
        overfit_report=overfit_report,
        aggregate_results=agg_results,
        analysis_timestamp=datetime.now(timezone.utc).isoformat(),
    )

    logger.info(
        "Walk-forward complete for %s: OOS Sharpe=%.2f, global degradation=%.2f.",
        ticker_u,
        aggregate_oos.sharpe_ratio,
        global_degr,
    )
    return result


# ──────────────────────────────────────────────────────────
# CLI TOOL
# ──────────────────────────────────────────────────────────


if __name__ == "__main__":
    import sys
    import traceback

    print("\n" + "=" * 70)
    print("ALPHA-PRIME Walk-Forward Analysis")
    print("=" * 70 + "\n")

    if len(sys.argv) < 2:
        print("Usage: python walk_forward.py TICKER [STRATEGY]")
        print("Example: python walk_forward.py AAPL momentum\n")
        sys.exit(1)

    ticker_cli = sys.argv[1].upper()
    strategy_cli = sys.argv[2] if len(sys.argv) > 2 else "momentum"

    print(f"Running walk-forward analysis: {ticker_cli}, strategy={strategy_cli}\n")

    cfg_cli = WalkForwardConfig(
        in_sample_days=180,
        out_sample_days=60,
        step_size_days=30,
        min_data_points=200,
        min_trades=5,
    )

    try:
        wf_result = run_walk_forward_analysis(ticker_cli, strategy_cli, cfg_cli)

        print("RESULTS")
        print("=" * 70)
        print(f"Total Windows        : {wf_result.aggregate_results.total_windows}")
        print("\nAGGREGATE OUT-OF-SAMPLE PERFORMANCE")
        print("-" * 70)
        oos = wf_result.aggregate_oos
        print(f"Sharpe Ratio         : {oos.sharpe_ratio:.2f}")
        print(f"CAGR                 : {oos.cagr:.2f}%")
        print(f"Total Return         : {oos.total_return:.2f}%")
        print(f"Max Drawdown         : {oos.max_drawdown:.2f}%")
        print(f"Volatility           : {oos.volatility:.2f}%")
        print(f"Win Rate             : {oos.win_rate:.1f}%")
        print(f"Profit Factor        : {oos.profit_factor:.2f}")
        print(f"Total Trades         : {oos.total_trades}")

        print("\nDEGRADATION & OVERFITTING")
        print("-" * 70)
        print(f"Global Degradation   : {wf_result.degradation_ratio:.2f}")
        print(f"Avg Window Degradation: {wf_result.aggregate_results.avg_degradation_ratio:.2f}")
        print(f"OOS Consistency      : {wf_result.aggregate_results.consistency_score:.0f}/100")
        print(f"Overfitting Detected : {'⚠️ YES' if wf_result.overfitting_detected else '✅ NO'}")

        print("\nOverfitting Warnings:")
        for w in wf_result.overfit_report.warnings:
            print(f"  - {w}")

        print("\nRecent Performance Trend:", wf_result.aggregate_results.recent_performance_trend)

        print("\n" + "=" * 70 + "\n")
    except Exception as exc:  # noqa: BLE001
        print(f"❌ Error running walk-forward analysis: {exc}")
        traceback.print_exc()
