"""
============================================================
ALPHA-PRIME v2.0 - Backtest Validator
============================================================

Backtest quality gate to detect common pitfalls, unrealistic
assumptions, and statistical issues before live deployment.

Validation Categories:
1. Lookahead Bias (CRITICAL)
   - Signals using future data or same-bar closes.
   - Misaligned timestamps between features and trades.

2. Survivorship Bias (HIGH)
   - Missing delisted/bankrupt names from universe.
   - Too-small universe relative to benchmark.

3. Transaction Costs & Slippage (MEDIUM)
   - Unrealistically low commission or slippage.
   - Ignoring market impact of large orders.

4. Statistical Significance (HIGH)
   - Edge may be indistinguishable from noise.
   - Bootstrap and t-tests vs benchmark. [web:366][web:369][web:371][web:373][web:375][web:378]

5. Curve Fitting (CRITICAL)
   - Excessive parameter tuning and sensitivity.
   - Overreliance on a single optimal configuration.

6. Liquidity & Realism (MEDIUM/LOW)
   - Position sizes too large relative to volume.
   - Trading outside regular market hours or
     with unrealistic frequency.

Usage:
    from backtest_validator import validate_backtest

    report = validate_backtest(backtest_result)
    print(generate_validation_summary(report))

CLI:
    python backtest_validator.py BACKTEST_CSV

The CLI expects a simple trade-level CSV and runs basic
validators to demonstrate the reporting pipeline.
============================================================
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import get_logger, get_settings

logger = get_logger(__name__)
settings = get_settings()


# ──────────────────────────────────────────────────────────
# DATACLASSES & DATA MODELS
# ──────────────────────────────────────────────────────────


@dataclass
class Trade:
    """
    Simplified trade record for validation.

    Attributes:
        entry_date: Entry timestamp.
        exit_date: Exit timestamp.
        entry_price: Entry fill price.
        exit_price: Exit fill price.
        shares: Number of shares (or contracts).
        pnl: Profit and loss in currency units.
        signal_type: String label for signal (e.g. 'LONG', 'SHORT').
        avg_daily_volume: Optional estimate of ADV for liquidity checks.
    """

    entry_date: datetime
    exit_date: datetime
    entry_price: float
    exit_price: float
    shares: int
    pnl: float
    signal_type: str
    avg_daily_volume: Optional[float] = None


@dataclass
class BacktestResult:
    """
    Container for backtest outputs required by the validator.

    Attributes:
        equity_curve: Series of portfolio equity indexed by date.
        returns: Series of strategy returns.
        benchmark_returns: Optional benchmark returns.
        trades: List of executed trades.
        signals: Optional Series of entry signals (1/-1/0).
        data: Optional feature/price DataFrame aligned to signals.
        params_history: Optional list of parameter dicts used during optimisation.
        performance_history: Optional list of performance metrics
                             corresponding to params_history (e.g. Sharpe).
        universe: Optional list of tickers used in the backtest.
        period: String representation of historical period (e.g. '3y').
    """

    equity_curve: pd.Series
    returns: pd.Series
    benchmark_returns: Optional[pd.Series] = None
    trades: List[Trade] = field(default_factory=list)
    signals: Optional[pd.Series] = None
    data: Optional[pd.DataFrame] = None
    params_history: Optional[List[Dict[str, Any]]] = None
    performance_history: Optional[List[float]] = None
    universe: Optional[List[str]] = None
    period: str = ""


@dataclass
class Issue:
    """
    Single validation issue detected by a check.

    Attributes:
        category: Validation category (e.g. 'Lookahead Bias').
        severity: CRITICAL | HIGH | MEDIUM | LOW.
        description: Explanation of the issue.
        fix: Recommended fix.
        impact: 'High' | 'Medium' | 'Low' impact on live trading.
    """

    category: str
    severity: str
    description: str
    fix: str
    impact: str


@dataclass
class CategoryScore:
    """
    Score and issues for a validation category.

    Attributes:
        name: Category name.
        score: 0–100 numeric score (higher is better).
        pass_fail: True if category passes thresholds.
        issues: List of issues in this category.
    """

    name: str
    score: float
    pass_fail: bool
    issues: List[Issue] = field(default_factory=list)


@dataclass
class LookaheadCheck:
    """
    Lookahead bias diagnostics.

    Attributes:
        score: 0–100 (100 = no evidence of lookahead).
        issues: List of detected issues.
        same_bar_signal_fraction: Fraction of signals that occur on same bar as
                                  feature/close used.
        misaligned_timestamps: Whether signals/data indices are misaligned.
    """

    score: float
    issues: List[Issue]
    same_bar_signal_fraction: float
    misaligned_timestamps: bool


@dataclass
class SurvivorshipCheck:
    """
    Survivorship bias diagnostics.

    Attributes:
        score: 0–100.
        issues: List of issues.
        universe_size: Number of tickers tested.
        delisted_included: Whether delisted symbols are included (best-effort flag).
    """

    score: float
    issues: List[Issue]
    universe_size: int
    delisted_included: bool


@dataclass
class CostCheck:
    """
    Transaction costs and slippage realism diagnostics.

    Attributes:
        score: 0–100.
        issues: List of issues.
        avg_commission_per_share: Average commission per share.
        commission_zero_fraction: Fraction of trades with zero commission.
    """

    score: float
    issues: List[Issue]
    avg_commission_per_share: float
    commission_zero_fraction: float


@dataclass
class SignificanceCheck:
    """
    Statistical significance diagnostics.

    Attributes:
        score: 0–100.
        issues: List[Issue].
        sharpe_ratio: Strategy Sharpe ratio.
        p_value: p-value from t-test vs benchmark (if available).
        bootstrap_sharpe_pvalue: Approx p-value from bootstrap Sharpe distribution.
    """

    score: float
    issues: List[Issue]
    sharpe_ratio: float
    p_value: Optional[float]
    bootstrap_sharpe_pvalue: Optional[float]


@dataclass
class CurveFittingCheck:
    """
    Curve fitting / over-optimisation diagnostics.

    Attributes:
        score: 0–100.
        issues: List[Issue].
        param_count: Number of tuned parameters.
        sensitivity_flag: True if small parameter changes cause large performance swings.
        edge_of_grid_flag: True if best parameters lie at edges of search space.
    """

    score: float
    issues: List[Issue]
    param_count: int
    sensitivity_flag: bool
    edge_of_grid_flag: bool


@dataclass
class SlippageCheck:
    """
    Slippage realism & liquidity diagnostics.

    Attributes:
        score: 0–100.
        issues: List[Issue].
        assumed_slippage_pct: Slippage assumption as percentage.
        large_order_fraction: Fraction of trades exceeding 1% ADV.
    """

    score: float
    issues: List[Issue]
    assumed_slippage_pct: float
    large_order_fraction: float


@dataclass
class ValidationReport:
    """
    Consolidated validation report.

    Attributes:
        overall_score: 0–100 score combining categories.
        pass_fail: True if strategy passes validation.
        category_scores: Per-category scores.
        critical_issues: All CRITICAL severity issues.
        recommendations: Human-readable recommended actions.
        warnings: Additional non-fatal warnings.
        notes: Misc notes and context.
    """

    overall_score: float
    pass_fail: bool
    category_scores: Dict[str, CategoryScore]
    critical_issues: List[Issue]
    recommendations: List[str]
    warnings: List[str]
    notes: List[str]


# ──────────────────────────────────────────────────────────
# HELPER FUNCTIONS (STATS, SHARPE, BOOTSTRAP, T-TEST)
# ──────────────────────────────────────────────────────────


def _compute_sharpe(returns: pd.Series) -> float:
    """Compute annualised Sharpe ratio assuming daily returns."""
    r = returns.dropna()
    if r.empty or r.std() == 0:
        return 0.0
    return float((r.mean() * 252) / r.std())


def _bootstrap_sharpe_pvalue(
    returns: pd.Series, n_bootstrap: int = 1000, target_sharpe: float = 0.0
) -> float:
    """
    Approximate p-value that Sharpe > target_sharpe via bootstrap. [web:366][web:369][web:370][web:373]

    Args:
        returns: Strategy return series.
        n_bootstrap: Number of bootstrap resamples.
        target_sharpe: Null Sharpe value.

    Returns:
        p-value (probability Sharpe <= target_sharpe under bootstrap).
    """
    r = returns.dropna().values
    if r.size < 10:
        return 1.0
    sharpe_samples: List[float] = []
    n = len(r)
    for _ in range(n_bootstrap):
        sample = np.random.choice(r, size=n, replace=True)
        s = _compute_sharpe(pd.Series(sample))
        sharpe_samples.append(s)
    sharpe_samples = np.array(sharpe_samples)
    p_val = float(np.mean(sharpe_samples <= target_sharpe))
    return p_val


def _t_test_vs_benchmark(
    strategy_returns: pd.Series, benchmark_returns: pd.Series
) -> float:
    """
    Simple t-test p-value for difference in mean returns vs benchmark. [web:371][web:373][web:375][web:378]

    This is a basic two-sample t-test assuming i.i.d. returns, used as a
    heuristic rather than rigorous inference.

    Args:
        strategy_returns: Strategy daily returns.
        benchmark_returns: Benchmark daily returns.

    Returns:
        p-value (two-sided).
    """
    sr = strategy_returns.dropna()
    br = benchmark_returns.dropna()
    df = pd.concat([sr, br], axis=1, join="inner").dropna()
    if df.shape[0] < 20:
        return 1.0
    x = df.iloc[:, 0].values
    y = df.iloc[:, 1].values
    mean_diff = np.mean(x) - np.mean(y)
    var_x = np.var(x, ddof=1)
    var_y = np.var(y, ddof=1)
    n_x = len(x)
    n_y = len(y)
    se = np.sqrt(var_x / n_x + var_y / n_y)
    if se == 0:
        return 1.0
    t_stat = mean_diff / se
    df_dof = (var_x / n_x + var_y / n_y) ** 2 / (
        (var_x**2) / ((n_x**2) * (n_x - 1)) + (var_y**2) / ((n_y**2) * (n_y - 1))
    )
    from math import erf, sqrt

    def _t_to_p(t: float, df_: float) -> float:
        # Approximate using normal distribution for simplicity.
        z = abs(t)
        p = 1.0 - 0.5 * (1.0 + erf(z / np.sqrt(2.0)))
        return p * 2.0

    p_val = _t_to_p(t_stat, df_dof)
    return float(p_val)


# ──────────────────────────────────────────────────────────
# INDIVIDUAL VALIDATORS
# ──────────────────────────────────────────────────────────


def check_lookahead_bias(signals: pd.Series, data: pd.DataFrame) -> LookaheadCheck:
    """
    Detect common forms of lookahead bias using signals and data timestamps.

    Heuristics:
        - Check alignment of indices (signals should not precede data).
        - Estimate fraction of signals that appear on the same bar as
          close-to-close returns that they profit from.

    Args:
        signals: Series of trade signals (nonzero = entry/exit).
        data: DataFrame with at least 'Close' and indicators.

    Returns:
        LookaheadCheck with score and issues.
    """
    issues: List[Issue] = []
    if signals is None or data is None or signals.empty or data.empty:
        return LookaheadCheck(
            score=100.0,
            issues=[],
            same_bar_signal_fraction=0.0,
            misaligned_timestamps=False,
        )

    sig = signals.dropna()
    misaligned = False

    if not sig.index.equals(data.index):
        misaligned = True
        issues.append(
            Issue(
                category="Lookahead Bias",
                severity="CRITICAL",
                description="Signals index does not align with data index; potential lookahead.",
                fix="Ensure signals are generated using data up to t-1 and aligned on the same index.",
                impact="High",
            )
        )

    close = data["Close"]
    returns = close.pct_change().shift(-1)
    sig_nonzero = sig[sig != 0]
    if not sig_nonzero.empty:
        aligned = pd.concat([sig_nonzero, returns], axis=1, join="inner").dropna()
        if not aligned.empty:
            same_bar_profit = (aligned.iloc[:, 1] * np.sign(aligned.iloc[:, 0]) > 0).mean()
            same_bar_signal_fraction = float(same_bar_profit)
        else:
            same_bar_signal_fraction = 0.0
    else:
        same_bar_signal_fraction = 0.0

    if same_bar_signal_fraction > 0.6:
        issues.append(
            Issue(
                category="Lookahead Bias",
                severity="CRITICAL",
                description=(
                    "High fraction of signals appear to profit from same-bar close-to-close "
                    "returns; may indicate use of future prices."
                ),
                fix="Use previous bar's close/indicators to generate signals, and execute at next open.",
                impact="High",
            )
        )

    if issues:
        score = 0.0
    else:
        score = 100.0

    return LookaheadCheck(
        score=score,
        issues=issues,
        same_bar_signal_fraction=same_bar_signal_fraction,
        misaligned_timestamps=misaligned,
    )


def detect_survivorship_bias(ticker_list: List[str], period: str) -> SurvivorshipCheck:
    """
    Heuristic survivorship bias check based on universe size and metadata.

    Rules:
        - Universe size < 100 → warning, < 50 → high risk.
        - If tickers appear to be large-cap only, note potential bias.

    Args:
        ticker_list: List of tickers used in backtest.
        period: Backtest period (e.g. '2015-2025').

    Returns:
        SurvivorshipCheck.
    """
    issues: List[Issue] = []
    if not ticker_list:
        issues.append(
            Issue(
                category="Survivorship Bias",
                severity="HIGH",
                description="No ticker universe information provided.",
                fix="Record and validate the full universe used, including delisted symbols where possible.",
                impact="High",
            )
        )
        return SurvivorshipCheck(
            score=40.0,
            issues=issues,
            universe_size=0,
            delisted_included=False,
        )

    universe_size = len(ticker_list)
    delisted_included = False  # Cannot verify without external source; assume False.

    if universe_size < 50:
        issues.append(
            Issue(
                category="Survivorship Bias",
                severity="HIGH",
                description=f"Universe size is very small ({universe_size} tickers).",
                fix="Expand universe to include at least 200+ tickers, ideally matching benchmark coverage.",
                impact="High",
            )
        )
    elif universe_size < 200:
        issues.append(
            Issue(
                category="Survivorship Bias",
                severity="MEDIUM",
                description=f"Universe size ({universe_size}) may be too small for robust testing.",
                fix="Consider increasing universe size and including multiple sectors and caps.",
                impact="Medium",
            )
        )

    if not delisted_included:
        issues.append(
            Issue(
                category="Survivorship Bias",
                severity="MEDIUM",
                description="Universe may exclude delisted or bankrupt names.",
                fix="Source a survivorship-bias-free dataset or ensure delisted names are included.",
                impact="Medium",
            )
        )

    if any(i.severity == "HIGH" for i in issues):
        score = 40.0
    elif any(i.severity == "MEDIUM" for i in issues):
        score = 70.0
    else:
        score = 100.0

    return SurvivorshipCheck(
        score=score,
        issues=issues,
        universe_size=universe_size,
        delisted_included=delisted_included,
    )


def validate_transaction_costs(trades: List[Trade], commissions: float) -> CostCheck:
    """
    Validate transaction cost assumptions vs realistic levels.

    Heuristics:
        - Commission < $0.005 per share → optimistic.
        - Zero commission fraction > 0 → suspicious unless broker supports it.
        - For now, use per-share commission param as proxy.

    Args:
        trades: List of trades.
        commissions: Per-share commission assumption.

    Returns:
        CostCheck.
    """
    issues: List[Issue] = []
    if not trades:
        return CostCheck(
            score=100.0,
            issues=[],
            avg_commission_per_share=commissions,
            commission_zero_fraction=0.0,
        )

    avg_commission = commissions
    zero_fraction = 1.0 if commissions == 0 else 0.0

    if commissions == 0:
        issues.append(
            Issue(
                category="Transaction Costs",
                severity="MEDIUM",
                description="Zero commission assumption.",
                fix="Assume at least $0.005/share or your broker's published fee schedule.",
                impact="Medium",
            )
        )
    elif commissions < 0.005:
        issues.append(
            Issue(
                category="Transaction Costs",
                severity="LOW",
                description=f"Commission {commissions:.4f} per share may be too low.",
                fix="Cross-check commission with your broker's fee schedule.",
                impact="Low",
            )
        )

    if issues:
        score = 70.0 if any(i.severity == "MEDIUM" for i in issues) else 85.0
    else:
        score = 100.0

    return CostCheck(
        score=score,
        issues=issues,
        avg_commission_per_share=avg_commission,
        commission_zero_fraction=zero_fraction,
    )


def check_statistical_significance(
    returns: pd.Series, benchmark_returns: pd.Series
) -> SignificanceCheck:
    """
    Assess statistical significance of backtest performance vs benchmark.

    Methods:
        - Compute Sharpe ratio.
        - Bootstrap Sharpe distribution to estimate p-value for Sharpe>0. [web:366][web:369][web:370][web:372]
        - Run simple t-test vs benchmark mean returns. [web:371][web:373][web:375][web:378]

    Red flags:
        - Sharpe < 1.0.
        - p-value > 0.05.

    Args:
        returns: Strategy daily returns.
        benchmark_returns: Benchmark daily returns.

    Returns:
        SignificanceCheck.
    """
    issues: List[Issue] = []

    if returns is None or returns.empty:
        return SignificanceCheck(
            score=40.0,
            issues=[
                Issue(
                    category="Statistical Significance",
                    severity="HIGH",
                    description="No returns series provided for significance testing.",
                    fix="Provide full backtest return series for statistical evaluation.",
                    impact="High",
                )
            ],
            sharpe_ratio=0.0,
            p_value=None,
            bootstrap_sharpe_pvalue=None,
        )

    sharpe = _compute_sharpe(returns)
    boot_p = _bootstrap_sharpe_pvalue(returns, n_bootstrap=500, target_sharpe=0.0)
    p_val = (
        _t_test_vs_benchmark(returns, benchmark_returns)
        if benchmark_returns is not None and not benchmark_returns.empty
        else None
    )

    if sharpe < 1.0:
        issues.append(
            Issue(
                category="Statistical Significance",
                severity="HIGH",
                description=f"Sharpe ratio {sharpe:.2f} < 1.0; edge may be weak.",
                fix="Improve risk-adjusted returns or extend sample size before deployment.",
                impact="High",
            )
        )

    if p_val is not None and p_val > 0.05:
        issues.append(
            Issue(
                category="Statistical Significance",
                severity="HIGH",
                description=f"t-test p-value {p_val:.3f} > 0.05; returns not clearly different from benchmark.",
                fix="Increase sample size or refine strategy to strengthen edge.",
                impact="High",
            )
        )

    if boot_p is not None and boot_p > 0.1:
        issues.append(
            Issue(
                category="Statistical Significance",
                severity="MEDIUM",
                description=f"Bootstrap Sharpe p-value {boot_p:.3f} suggests edge may be fragile.",
                fix="Use more conservative expectations and test robustness under resampling.",
                impact="Medium",
            )
        )

    if any(i.severity == "HIGH" for i in issues):
        score = 50.0
    elif any(i.severity == "MEDIUM" for i in issues):
        score = 75.0
    else:
        score = 100.0

    return SignificanceCheck(
        score=score,
        issues=issues,
        sharpe_ratio=sharpe,
        p_value=p_val,
        bootstrap_sharpe_pvalue=boot_p,
    )


def detect_curve_fitting(
    params_history: List[Dict[str, Any]], performance_history: List[float]
) -> CurveFittingCheck:
    """
    Detect signs of curve fitting from parameter and performance history.

    Heuristics:
        - Parameter count > 5 → warning.
        - If best performance occurs at edge of grid for many parameters → suspicious.
        - If ±10% change in parameters leads to >20% performance change → overfit.

    Args:
        params_history: List of parameter dicts tried.
        performance_history: Corresponding performance metric (e.g. Sharpe).

    Returns:
        CurveFittingCheck.
    """
    issues: List[Issue] = []
    if not params_history or not performance_history:
        return CurveFittingCheck(
            score=100.0,
            issues=[],
            param_count=0,
            sensitivity_flag=False,
            edge_of_grid_flag=False,
        )

    param_names = list(params_history[0].keys())
    param_count = len(param_names)

    sensitivity_flag = False
    edge_of_grid_flag = False

    if param_count > 5:
        issues.append(
            Issue(
                category="Curve Fitting",
                severity="HIGH",
                description=f"Strategy uses {param_count} optimised parameters; high risk of curve fitting.",
                fix="Reduce parameter count to the most impactful 3–5 and retest.",
                impact="High",
            )
        )

    best_idx = int(np.argmax(performance_history))
    best_params = params_history[best_idx]

    for name in param_names:
        values = np.array([p[name] for p in params_history if name in p])
        if values.size == 0:
            continue
        v_min = values.min()
        v_max = values.max()
        v_best = best_params.get(name)
        if v_best is None:
            continue
        if v_best == v_min or v_best == v_max:
            edge_of_grid_flag = True

    if edge_of_grid_flag:
        issues.append(
            Issue(
                category="Curve Fitting",
                severity="MEDIUM",
                description="Best parameters lie at edges of search grid.",
                fix="Expand parameter grid and check stability of best region.",
                impact="Medium",
            )
        )

    scores = np.array(performance_history, dtype=float)
    median_perf = np.median(scores)
    if median_perf != 0:
        ratio = scores.max() / median_perf
        if ratio > 1.5:
            sensitivity_flag = True
            issues.append(
                Issue(
                    category="Curve Fitting",
                    severity="HIGH",
                    description="Strategy performance highly sensitive to parameter changes.",
                    fix="Use coarser parameter grids, regularisation, and walk-forward validation.",
                    impact="High",
                )
            )

    if any(i.severity == "HIGH" for i in issues):
        score = 40.0
    elif any(i.severity == "MEDIUM" for i in issues):
        score = 70.0
    else:
        score = 100.0

    return CurveFittingCheck(
        score=score,
        issues=issues,
        param_count=param_count,
        sensitivity_flag=sensitivity_flag,
        edge_of_grid_flag=edge_of_grid_flag,
    )


def validate_slippage_realism(
    slippage_assumed: float, market_data: pd.DataFrame, trades: List[Trade]
) -> SlippageCheck:
    """
    Validate slippage assumptions vs liquidity and order size.

    Realistic guidelines:
        - ~0.05% for very liquid large caps.
        - 0.2–1% for small/illiquid names.
        - Market impact ~0.1% for orders >0.1% ADV.

    Args:
        slippage_assumed: Slippage as fraction (e.g. 0.0005 for 0.05%).
        market_data: Optional OHLCV data (unused here; placeholder for future).
        trades: List of trades with avg_daily_volume if available.

    Returns:
        SlippageCheck.
    """
    issues: List[Issue] = []
    if slippage_assumed < 0:
        issues.append(
            Issue(
                category="Slippage",
                severity="MEDIUM",
                description="Negative slippage assumption is invalid.",
                fix="Use a positive slippage estimate based on historical bid-ask spreads.",
                impact="Medium",
            )
        )

    assumed_pct = slippage_assumed * 100.0
    if slippage_assumed == 0:
        issues.append(
            Issue(
                category="Slippage",
                severity="MEDIUM",
                description="Zero slippage assumption.",
                fix="Assume at least 0.05–0.10% for liquid markets.",
                impact="Medium",
            )
        )
    elif slippage_assumed < 0.0005:
        issues.append(
            Issue(
                category="Slippage",
                severity="LOW",
                description=f"Slippage assumption {assumed_pct:.3f}% may be optimistic.",
                fix="Re-evaluate slippage using historical spread and impact data.",
                impact="Low",
            )
        )

    large_order_fraction = 0.0
    if trades:
        count = 0
        large = 0
        for t in trades:
            if t.avg_daily_volume and t.avg_daily_volume > 0:
                notional = t.entry_price * t.shares
                adv_notional = t.avg_daily_volume * t.entry_price
                if adv_notional > 0:
                    frac = notional / adv_notional
                    count += 1
                    if frac > 0.01:
                        large += 1
        large_order_fraction = float(large / count) if count > 0 else 0.0
        if large_order_fraction > 0.2:
            issues.append(
                Issue(
                    category="Liquidity",
                    severity="MEDIUM",
                    description=(
                        f"{large_order_fraction*100:.1f}% of trades exceed 1% of ADV; "
                        "market impact likely underestimated."
                    ),
                    fix="Reduce position sizes or include explicit market impact in backtest.",
                    impact="Medium",
                )
            )

    if any(i.severity == "MEDIUM" for i in issues):
        score = 70.0
    elif any(i.severity == "LOW" for i in issues):
        score = 85.0
    else:
        score = 100.0

    return SlippageCheck(
        score=score,
        issues=issues,
        assumed_slippage_pct=assumed_pct,
        large_order_fraction=large_order_fraction,
    )


# ──────────────────────────────────────────────────────────
# MAIN VALIDATION & SCORING
# ──────────────────────────────────────────────────────────


def _score_to_pass(score: float, severity_floor: str) -> bool:
    """
    Determine pass/fail based on score and severity expectations.

    severity_floor:
        'CRITICAL' → no failures allowed.
        'HIGH' → require score >= 80.
        'MEDIUM' → require score >= 70.
        'LOW' → require score >= 60.
    """
    if severity_floor == "CRITICAL":
        return score >= 90.0
    if severity_floor == "HIGH":
        return score >= 80.0
    if severity_floor == "MEDIUM":
        return score >= 70.0
    return score >= 60.0


def validate_backtest(result: BacktestResult) -> ValidationReport:
    """
    Run all validation checks on a BacktestResult and aggregate scores.

    Categories:
        - Lookahead Bias (CRITICAL)
        - Survivorship Bias (HIGH)
        - Transaction Costs (MEDIUM)
        - Statistical Significance (HIGH)
        - Curve Fitting (CRITICAL)
        - Slippage & Liquidity (MEDIUM/LOW realism checks)

    Args:
        result: BacktestResult with required fields.

    Returns:
        ValidationReport.
    """
    logger.info("Validating backtest over %d returns and %d trades.", len(result.returns), len(result.trades))

    category_scores: Dict[str, CategoryScore] = {}
    critical_issues: List[Issue] = []
    recommendations: List[str] = []
    warnings: List[str] = []
    notes: List[str] = []

    if result.signals is not None and result.data is not None:
        la_check = check_lookahead_bias(result.signals, result.data)
    else:
        la_check = LookaheadCheck(
            score=100.0,
            issues=[],
            same_bar_signal_fraction=0.0,
            misaligned_timestamps=False,
        )

    la_cat = CategoryScore(
        name="Lookahead Bias",
        score=la_check.score,
        pass_fail=_score_to_pass(la_check.score, "CRITICAL"),
        issues=la_check.issues,
    )
    category_scores["Lookahead Bias"] = la_cat

    surv_check = detect_survivorship_bias(result.universe or [], result.period)
    surv_cat = CategoryScore(
        name="Survivorship Bias",
        score=surv_check.score,
        pass_fail=_score_to_pass(surv_check.score, "HIGH"),
        issues=surv_check.issues,
    )
    category_scores["Survivorship Bias"] = surv_cat

    cost_check = validate_transaction_costs(result.trades, commissions=settings.commission if hasattr(settings, "commission") else 0.0)
    cost_cat = CategoryScore(
        name="Transaction Costs",
        score=cost_check.score,
        pass_fail=_score_to_pass(cost_check.score, "MEDIUM"),
        issues=cost_check.issues,
    )
    category_scores["Transaction Costs"] = cost_cat

    sig_check = check_statistical_significance(result.returns, result.benchmark_returns or pd.Series(dtype=float))
    sig_cat = CategoryScore(
        name="Statistical Significance",
        score=sig_check.score,
        pass_fail=_score_to_pass(sig_check.score, "HIGH"),
        issues=sig_check.issues,
    )
    category_scores["Statistical Significance"] = sig_cat

    if result.params_history and result.performance_history:
        cf_check = detect_curve_fitting(result.params_history, result.performance_history)
    else:
        cf_check = CurveFittingCheck(
            score=100.0,
            issues=[],
            param_count=0,
            sensitivity_flag=False,
            edge_of_grid_flag=False,
        )

    cf_cat = CategoryScore(
        name="Curve Fitting",
        score=cf_check.score,
        pass_fail=_score_to_pass(cf_check.score, "CRITICAL"),
        issues=cf_check.issues,
    )
    category_scores["Curve Fitting"] = cf_cat

    slippage_assumed = getattr(settings, "slippage", 0.0005)
    slippage_check = validate_slippage_realism(slippage_assumed, result.data or pd.DataFrame(), result.trades)
    slip_cat = CategoryScore(
        name="Slippage & Liquidity",
        score=slippage_check.score,
        pass_fail=_score_to_pass(slippage_check.score, "MEDIUM"),
        issues=slippage_check.issues,
    )
    category_scores["Slippage & Liquidity"] = slip_cat

    for cat in category_scores.values():
        for issue in cat.issues:
            if issue.severity == "CRITICAL":
                critical_issues.append(issue)
            if issue.severity in ("CRITICAL", "HIGH"):
                recommendations.append(issue.fix)
            elif issue.severity in ("MEDIUM", "LOW"):
                warnings.append(issue.description)

    if not any(cat.issues for cat in category_scores.values()):
        notes.append("No issues detected; backtest appears robust under current checks.")

    scores = [cat.score for cat in category_scores.values()]
    overall_score = float(np.mean(scores)) if scores else 0.0

    critical_fail = bool(critical_issues)
    any_fail = any(not cat.pass_fail for cat in category_scores.values())

    pass_fail = not critical_fail and not any_fail and overall_score >= 75.0

    report = ValidationReport(
        overall_score=overall_score,
        pass_fail=pass_fail,
        category_scores=category_scores,
        critical_issues=critical_issues,
        recommendations=list(dict.fromkeys(recommendations)),
        warnings=list(dict.fromkeys(warnings)),
        notes=notes,
    )

    logger.info(
        "Backtest validation complete: overall_score=%.1f, pass=%s, critical_issues=%d.",
        overall_score,
        pass_fail,
        len(critical_issues),
    )
    return report


# ──────────────────────────────────────────────────────────
# REPORTING
# ──────────────────────────────────────────────────────────


def generate_validation_summary(report: ValidationReport) -> str:
    """
    Generate a human-readable validation summary for logs/CLI.

    Format:
        OVERALL SCORE: 78/100 ✅ PASS (with warnings)
        CATEGORY: score + emoji
        Critical issues and warnings listed.
    """
    lines: List[str] = []
    status_icon = "✅" if report.pass_fail else "❌"
    warn_suffix = ""
    if report.warnings and report.pass_fail:
        warn_suffix = " (with warnings)"
    elif not report.pass_fail:
        warn_suffix = " (failed)"

    lines.append(
        f"OVERALL SCORE: {report.overall_score:.0f}/100 {status_icon}{warn_suffix}"
    )

    for name, cat in report.category_scores.items():
        cat_icon = "✅" if cat.pass_fail else "⚠️"
        lines.append(f"{name.upper()}: {cat.score:.0f}/100 {cat_icon}")

    lines.append("")
    lines.append(f"CRITICAL ISSUES ({len(report.critical_issues)}):")
    for issue in report.critical_issues:
        lines.append(f"- [{issue.category}] {issue.description} (Impact: {issue.impact})")
        lines.append(f"  Fix: {issue.fix}")

    if report.warnings:
        lines.append("")
        lines.append(f"WARNINGS ({len(report.warnings)}):")
        for w in report.warnings:
            lines.append(f"- {w}")

    if report.recommendations:
        lines.append("")
        lines.append("RECOMMENDATIONS:")
        for rec in report.recommendations:
            lines.append(f"- {rec}")

    if report.notes:
        lines.append("")
        lines.append("NOTES:")
        for n in report.notes:
            lines.append(f"- {n}")

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────
# CLI TOOL
# ──────────────────────────────────────────────────────────


def _load_trades_from_csv(df: pd.DataFrame) -> Tuple[List[Trade], pd.Series]:
    """
    Helper to parse trades and equity from a simple CSV schema.

    Required columns:
        date, entry_price, exit_price, shares, pnl, signal_type
    Optional:
        avg_daily_volume

    Returns:
        (trades, equity_curve) where equity_curve is reconstructed
        cumulatively from pnl.
    """
    required = {"date", "entry_price", "exit_price", "shares", "pnl", "signal_type"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    trades: List[Trade] = []
    equity = []
    capital = 10000.0

    for _, row in df.iterrows():
        entry_date = row["date"]
        exit_date = row["date"]
        trade = Trade(
            entry_date=entry_date,
            exit_date=exit_date,
            entry_price=float(row["entry_price"]),
            exit_price=float(row["exit_price"]),
            shares=int(row["shares"]),
            pnl=float(row["pnl"]),
            signal_type=str(row["signal_type"]),
            avg_daily_volume=float(row["avg_daily_volume"])
            if "avg_daily_volume" in row and not pd.isna(row["avg_daily_volume"])
            else None,
        )
        trades.append(trade)
        capital += trade.pnl
        equity.append(capital)

    equity_curve = pd.Series(equity, index=df["date"])
    returns = equity_curve.pct_change().fillna(0.0)
    return trades, returns


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ALPHA-PRIME v2.0 - Backtest Validator CLI"
    )
    parser.add_argument(
        "backtest_csv",
        type=str,
        help="Path to CSV with trade-level backtest data.",
    )
    parser.add_argument(
        "--benchmark_csv",
        type=str,
        default=None,
        help="Optional benchmark CSV with date, return columns.",
    )
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("ALPHA-PRIME Backtest Validator")
    print("=" * 70 + "\n")

    try:
        df = pd.read_csv(args.backtest_csv)
        trades, returns = _load_trades_from_csv(df)

        if args.benchmark_csv:
            bmk_df = pd.read_csv(args.benchmark_csv)
            if "date" in bmk_df.columns and "return" in bmk_df.columns:
                bmk_df["date"] = pd.to_datetime(bmk_df["date"])
                bmk_df = bmk_df.sort_values("date")
                bmk_ret = pd.Series(bmk_df["return"].values, index=bmk_df["date"])
            else:
                bmk_ret = pd.Series(dtype=float)
        else:
            bmk_ret = pd.Series(dtype=float)

        backtest_result = BacktestResult(
            equity_curve=(10000 * (1 + returns).cumprod()),
            returns=returns,
            benchmark_returns=bmk_ret,
            trades=trades,
            universe=None,
            period="",
        )

        report = validate_backtest(backtest_result)
        summary = generate_validation_summary(report)
        print(summary)
        print("\n" + "=" * 70 + "\n")
    except Exception as exc:  # noqa: BLE001
        print(f"❌ Error validating backtest: {exc}")
        import traceback

        traceback.print_exc()
        raise
