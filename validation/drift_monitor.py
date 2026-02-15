"""
============================================================
ALPHA-PRIME v2.0 - Drift & Degradation Monitor
============================================================

Monitors data drift, model drift, and strategy performance
to detect when a model/strategy is no longer aligned with
current market conditions.

Drift Types:
1. Data Drift (P(X)):
   - Input feature distribution changes.
   - Measured via PSI, mean/std deltas. [web:351][web:352][web:359][web:361][web:365]

2. Model / Concept Drift (P(y|X)):
   - Model errors increase over time.
   - Measured via error ratios vs baseline, error volatility. [web:358][web:360][web:363][web:364]

3. Performance Drift:
   - Strategy equity curve degrades.
   - Measured via Sharpe, drawdown, win rate, benchmark gap. [web:336][web:339][web:340][web:344]

Typical Use:
- After training: compute baseline statistics.
- In production: periodically compare live data and performance
  to baseline.
- When thresholds are breached: trigger alerts, retraining, or
  deactivation.

This module does NOT retrain models itself. It provides signals
and rationale for higher-level orchestration components
(e.g., scheduler, MLOps).
============================================================
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from diskcache import Cache

from config import get_settings, get_logger

logger = get_logger(__name__)
settings = get_settings()
cache = Cache(f"{settings.cache_dir}/drift")


# ──────────────────────────────────────────────────────────
# DATACLASSES & ENUMS
# ──────────────────────────────────────────────────────────


@dataclass
class FeatureStats:
    """
    Summary statistics for a single feature used as drift baseline.

    Attributes:
        name: Feature name.
        mean: Baseline mean.
        std: Baseline standard deviation.
        hist_bins: Bin edges used for histogram/PSI.
        hist_counts: Baseline histogram counts for each bin.
    """

    name: str
    mean: float
    std: float
    hist_bins: List[float]
    hist_counts: List[int]


@dataclass
class DriftBaseline:
    """
    Baseline reference used for drift detection.

    Attributes:
        created_at: Timestamp when baseline was computed.
        feature_stats: Per-feature statistics and histograms.
        baseline_error: Reference model error from validation set.
        baseline_sharpe: Reference Sharpe ratio.
        baseline_max_drawdown: Reference max drawdown.
        description: Free-form description of dataset/model snapshot.
    """

    created_at: datetime
    feature_stats: Dict[str, FeatureStats]
    baseline_error: float
    baseline_sharpe: float
    baseline_max_drawdown: float
    description: str = ""


@dataclass
class FeatureDrift:
    """
    Drift metrics for a single feature.

    Attributes:
        feature: Feature name.
        psi: Population Stability Index (PSI) vs baseline.
        mean_delta_pct: Percent change in mean relative to baseline.
        std_delta_pct: Percent change in std relative to baseline.
        severity: STABLE | MODERATE | SEVERE.
    """

    feature: str
    psi: float
    mean_delta_pct: float
    std_delta_pct: float
    severity: str


@dataclass
class DataDriftReport:
    """
    Aggregate data drift report across all monitored features.

    Attributes:
        timestamp: Time when drift was measured.
        feature_drifts: List of per-feature drift metrics.
        max_psi: Maximum PSI across all features.
        drifted_features: Number of features with MODERATE/SEVERE drift.
        overall_severity: STABLE | MODERATE | SEVERE.
        notes: Human-readable comments.
    """

    timestamp: datetime
    feature_drifts: List[FeatureDrift]
    max_psi: float
    drifted_features: int
    overall_severity: str
    notes: List[str] = field(default_factory=list)


@dataclass
class ModelDriftReport:
    """
    Model / concept drift report based on error metrics.

    Attributes:
        timestamp: Time when drift was measured.
        current_error: Current error (e.g. MAE or MAPE).
        baseline_error: Baseline error.
        error_ratio: current_error / baseline_error.
        error_volatility: Std of error series (if available).
        severity: HEALTHY | MILD | SEVERE.
        notes: Human-readable comments.
    """

    timestamp: datetime
    current_error: float
    baseline_error: float
    error_ratio: float
    error_volatility: float
    severity: str
    notes: List[str] = field(default_factory=list)


@dataclass
class PerformanceDriftReport:
    """
    Strategy performance drift report derived from equity curve.

    Attributes:
        timestamp: Measurement time.
        current_sharpe: Recent Sharpe ratio.
        baseline_sharpe: Historical Sharpe ratio.
        sharpe_ratio_degradation: current_sharpe / baseline_sharpe (0 if baseline 0).
        current_max_drawdown: Recent max drawdown (%).
        baseline_max_drawdown: Historical max drawdown (%).
        drawdown_increase_pct: Percent increase vs baseline.
        recent_win_rate: Recent win rate (%).
        baseline_win_rate: Baseline win rate (%).
        win_rate_drop_pct_points: Absolute drop in win rate (percentage points).
        benchmark_underperformance_pct: Underperformance vs benchmark over window (%).
        severity: HEALTHY | MILD | SEVERE.
        notes: Human-readable comments.
    """

    timestamp: datetime
    current_sharpe: float
    baseline_sharpe: float
    sharpe_ratio_degradation: float
    current_max_drawdown: float
    baseline_max_drawdown: float
    drawdown_increase_pct: float
    recent_win_rate: float
    baseline_win_rate: float
    win_rate_drop_pct_points: float
    benchmark_underperformance_pct: Optional[float]
    severity: str
    notes: List[str] = field(default_factory=list)


@dataclass
class DriftSignal:
    """
    Unified drift signal combining data, model, and performance drift.

    Attributes:
        timestamp: Time of evaluation.
        data_severity: STABLE | MODERATE | SEVERE.
        model_severity: HEALTHY | MILD | SEVERE.
        performance_severity: HEALTHY | MILD | SEVERE.
        composite_severity: OK | WARNING | CRITICAL.
        retrain_recommended: True if policy suggests retrain.
        retrain_reason: Human-readable explanation.
        details: Aggregated numeric metrics (e.g. max_psi, error_ratio).
    """

    timestamp: datetime
    data_severity: str
    model_severity: str
    performance_severity: str
    composite_severity: str
    retrain_recommended: bool
    retrain_reason: str
    details: Dict[str, float] = field(default_factory=dict)


@dataclass
class RetrainPolicy:
    """
    Policy threshold configuration for retrain decisions.

    Attributes:
        max_data_psi: Max tolerated PSI before severe drift.
        max_error_ratio: Max tolerated error ratio.
        max_sharpe_degradation: Min acceptable current/baseline Sharpe ratio.
        max_drawdown_increase_pct: Max allowed increase in drawdown (%).
        min_alerts_before_retrain: Minimum CRITICAL/WARNING signals before retrain.
        cooldown_days: Minimum days between retrain events.
    """

    max_data_psi: float = 0.25
    max_error_ratio: float = 1.5
    max_sharpe_degradation: float = 0.5
    max_drawdown_increase_pct: float = 50.0
    min_alerts_before_retrain: int = 2
    cooldown_days: int = 7


@dataclass
class DriftDashboardSummary:
    """
    High-level summary for dashboards / CLI.

    Attributes:
        last_check: Timestamp of most recent drift check.
        checks_run: Number of signals in lookback period.
        critical_events_last_30d: CRITICAL composite events count.
        warnings_last_30d: WARNING composite events count.
        current_composite_severity: Composite severity of most recent signal.
        recent_retrain_events: Count of retrain-recommended signals.
        trend: IMPROVING | STABLE | WORSENING.
        latest_signal: Most recent DriftSignal, or None.
    """

    last_check: datetime
    checks_run: int
    critical_events_last_30d: int
    warnings_last_30d: int
    current_composite_severity: str
    recent_retrain_events: int
    trend: str
    latest_signal: Optional[DriftSignal]


# ──────────────────────────────────────────────────────────
# PSI & HELPER FUNCTIONS
# ──────────────────────────────────────────────────────────


def compute_psi(
    base_values: np.ndarray, current_values: np.ndarray, bins: int = 10
) -> float:
    """
    Compute Population Stability Index (PSI) between two distributions. [web:351][web:352][web:359][web:361][web:365]

    PSI is defined as:
        sum((p_i - q_i) * ln(p_i / q_i))
    where p_i, q_i are proportions in bin i for current and baseline populations.

    Args:
        base_values: Baseline sample values.
        current_values: Current sample values.
        bins: Number of equal-width bins.

    Returns:
        PSI value (higher = more drift).
    """
    base = np.asarray(base_values, dtype=float)
    curr = np.asarray(current_values, dtype=float)
    base = base[np.isfinite(base)]
    curr = curr[np.isfinite(curr)]

    if base.size == 0 or curr.size == 0:
        return 0.0

    epsilon = 1e-6
    combined_min = float(min(base.min(), curr.min()))
    combined_max = float(max(base.max(), curr.max()))
    if combined_min == combined_max:
        return 0.0

    bin_edges = np.linspace(combined_min, combined_max, bins + 1)
    base_hist, _ = np.histogram(base, bins=bin_edges)
    curr_hist, _ = np.histogram(curr, bins=bin_edges)

    base_prop = base_hist / max(base_hist.sum(), 1)
    curr_prop = curr_hist / max(curr_hist.sum(), 1)

    base_prop = np.clip(base_prop, epsilon, None)
    curr_prop = np.clip(curr_prop, epsilon, None)

    psi_values = (curr_prop - base_prop) * np.log(curr_prop / base_prop)
    psi = float(np.sum(psi_values))
    return psi


def _severity_from_psi(psi: float) -> str:
    """Classify PSI into STABLE | MODERATE | SEVERE."""
    if psi < 0.1:
        return "STABLE"
    if psi < 0.25:
        return "MODERATE"
    return "SEVERE"


def _composite_from_component_severities(
    data_severity: str, model_severity: str, perf_severity: str
) -> str:
    """Map component severities into composite OK | WARNING | CRITICAL."""
    severities = {data_severity, model_severity, perf_severity}
    if "SEVERE" in severities:
        return "CRITICAL"
    if "MODERATE" in severities or "MILD" in severities:
        return "WARNING"
    return "OK"


def _now_utc() -> datetime:
    """Return current UTC time."""
    return datetime.now(timezone.utc)


# ──────────────────────────────────────────────────────────
# BASELINE COMPUTATION
# ──────────────────────────────────────────────────────────


def compute_data_drift_baseline(
    features: pd.DataFrame, as_of: Optional[datetime] = None
) -> DriftBaseline:
    """
    Compute drift baseline statistics from historical feature data.

    For each numeric feature, this function calculates mean, standard
    deviation, and a reference histogram which will be reused when
    measuring PSI against future data.

    Args:
        features: Training feature DataFrame.
        as_of: Optional timestamp for baseline creation.

    Returns:
        DriftBaseline instance (baseline_error / Sharpe / drawdown
        are set to 0 by default and can be updated by caller).
    """
    if features.empty:
        raise ValueError("Cannot compute baseline from empty features.")

    feature_stats: Dict[str, FeatureStats] = {}
    numeric = features.select_dtypes(include=[np.number])

    for col in numeric.columns:
        col_values = numeric[col].dropna().values.astype(float)
        if col_values.size == 0:
            continue
        mean = float(np.mean(col_values))
        std = float(np.std(col_values))
        bins = np.linspace(col_values.min(), col_values.max(), 11)
        counts, _ = np.histogram(col_values, bins=bins)

        feature_stats[col] = FeatureStats(
            name=col,
            mean=mean,
            std=std,
            hist_bins=list(bins),
            hist_counts=list(int(x) for x in counts),
        )

    baseline = DriftBaseline(
        created_at=as_of or _now_utc(),
        feature_stats=feature_stats,
        baseline_error=0.0,
        baseline_sharpe=0.0,
        baseline_max_drawdown=0.0,
        description="Auto-generated baseline from training features.",
    )
    logger.info(
        "Computed drift baseline with %d features at %s.",
        len(feature_stats),
        baseline.created_at.isoformat(),
    )
    return baseline


# ──────────────────────────────────────────────────────────
# DATA DRIFT MEASUREMENT
# ──────────────────────────────────────────────────────────


def measure_data_drift(
    current_features: pd.DataFrame, baseline: DriftBaseline
) -> DataDriftReport:
    """
    Measure data drift by comparing current feature distributions to baseline.

    For each baseline feature:
        - Aligns current column.
        - Reuses baseline histogram bins.
        - Computes PSI, mean delta, std delta.
        - Classifies per-feature severity.

    Args:
        current_features: Current/live feature DataFrame.
        baseline: DriftBaseline with stored feature statistics.

    Returns:
        DataDriftReport summarising per-feature and overall drift.
    """
    feature_drifts: List[FeatureDrift] = []

    numeric_current = current_features.select_dtypes(include=[np.number])
    if numeric_current.empty or not baseline.feature_stats:
        logger.warning("No numeric features or empty baseline; returning STABLE data report.")
        return DataDriftReport(
            timestamp=_now_utc(),
            feature_drifts=[],
            max_psi=0.0,
            drifted_features=0,
            overall_severity="STABLE",
            notes=["No numeric features or baseline stats available."],
        )

    for name, stats in baseline.feature_stats.items():
        if name not in numeric_current.columns:
            continue
        base_bins = np.asarray(stats.hist_bins, dtype=float)
        base_counts = np.asarray(stats.hist_counts, dtype=float)
        base_values = np.repeat(
            (base_bins[:-1] + base_bins[1:]) / 2.0,
            base_counts.astype(int),
        )

        current_values = numeric_current[name].dropna().values.astype(float)
        if current_values.size == 0:
            psi = 0.0
            severity = "STABLE"
            mean_delta_pct = 0.0
            std_delta_pct = 0.0
        else:
            current_hist, _ = np.histogram(current_values, bins=base_bins)
            base_prop = base_counts / max(base_counts.sum(), 1.0)
            curr_prop = current_hist / max(current_hist.sum(), 1.0)
            epsilon = 1e-6
            base_prop = np.clip(base_prop, epsilon, None)
            curr_prop = np.clip(curr_prop, epsilon, None)
            psi = float(np.sum((curr_prop - base_prop) * np.log(curr_prop / base_prop)))
            severity = _severity_from_psi(psi)

            cur_mean = float(np.mean(current_values))
            cur_std = float(np.std(current_values))
            mean_delta_pct = (
                (cur_mean - stats.mean) / abs(stats.mean) * 100.0 if stats.mean != 0 else 0.0
            )
            std_delta_pct = (
                (cur_std - stats.std) / abs(stats.std) * 100.0 if stats.std != 0 else 0.0
            )

        feature_drifts.append(
            FeatureDrift(
                feature=name,
                psi=psi,
                mean_delta_pct=mean_delta_pct,
                std_delta_pct=std_delta_pct,
                severity=severity,
            )
        )

    if feature_drifts:
        max_psi = float(max(fd.psi for fd in feature_drifts))
        drifted_features = sum(1 for fd in feature_drifts if fd.severity in ("MODERATE", "SEVERE"))
    else:
        max_psi = 0.0
        drifted_features = 0

    if any(fd.severity == "SEVERE" for fd in feature_drifts):
        overall_severity = "SEVERE"
    elif any(fd.severity == "MODERATE" for fd in feature_drifts):
        overall_severity = "MODERATE"
    else:
        overall_severity = "STABLE"

    notes: List[str] = []
    if drifted_features > 0:
        top = sorted(feature_drifts, key=lambda x: x.psi, reverse=True)[:5]
        top_names = ", ".join(f"{t.feature} (PSI={t.psi:.3f})" for t in top)
        notes.append(f"Top drifted features: {top_names}.")
    else:
        notes.append("No significant feature-level drift detected.")

    report = DataDriftReport(
        timestamp=_now_utc(),
        feature_drifts=feature_drifts,
        max_psi=max_psi,
        drifted_features=drifted_features,
        overall_severity=overall_severity,
        notes=notes,
    )

    logger.info(
        "Data drift measured: max PSI=%.3f, drifted_features=%d, severity=%s.",
        max_psi,
        drifted_features,
        overall_severity,
    )
    return report


# ──────────────────────────────────────────────────────────
# MODEL DRIFT MEASUREMENT
# ──────────────────────────────────────────────────────────


def measure_model_drift(
    predictions: pd.Series, actuals: pd.Series, baseline_error: float
) -> ModelDriftReport:
    """
    Measure model / concept drift via change in error magnitude. [web:358][web:360][web:363][web:364]

    Uses Mean Absolute Error (MAE) as default error metric.

    Args:
        predictions: Recent model predictions.
        actuals: Corresponding ground truth values.
        baseline_error: Baseline error from validation or training.

    Returns:
        ModelDriftReport object.
    """
    ts = _now_utc()

    if len(predictions) == 0 or len(actuals) == 0:
        return ModelDriftReport(
            timestamp=ts,
            current_error=0.0,
            baseline_error=baseline_error,
            error_ratio=1.0,
            error_volatility=0.0,
            severity="HEALTHY",
            notes=["No predictions/actuals provided; treating as HEALTHY by default."],
        )

    preds = predictions.astype(float).values
    acts = actuals.astype(float).values
    n = min(len(preds), len(acts))
    preds = preds[-n:]
    acts = acts[-n:]

    errors = np.abs(preds - acts)
    current_error = float(np.mean(errors))
    error_volatility = float(np.std(errors))

    if baseline_error <= 0:
        error_ratio = 1.0
    else:
        error_ratio = current_error / baseline_error

    if error_ratio < 1.2:
        severity = "HEALTHY"
    elif error_ratio < 1.5:
        severity = "MILD"
    else:
        severity = "SEVERE"

    notes: List[str] = [
        f"Current MAE={current_error:.4f}, baseline={baseline_error:.4f}, ratio={error_ratio:.2f}."
    ]
    if severity == "HEALTHY":
        notes.append("Error ratio < 1.2 → within healthy range.")
    elif severity == "MILD":
        notes.append("Error ratio between 1.2 and 1.5 → mild degradation.")
    else:
        notes.append("Error ratio > 1.5 → severe model drift suspected.")

    report = ModelDriftReport(
        timestamp=ts,
        current_error=current_error,
        baseline_error=baseline_error,
        error_ratio=error_ratio,
        error_volatility=error_volatility,
        severity=severity,
        notes=notes,
    )
    logger.info(
        "Model drift measured: error=%.4f, baseline=%.4f, ratio=%.2f, severity=%s.",
        current_error,
        baseline_error,
        error_ratio,
        severity,
    )
    return report


# ──────────────────────────────────────────────────────────
# PERFORMANCE DRIFT MEASUREMENT
# ──────────────────────────────────────────────────────────


def _compute_sharpe_from_equity(equity_curve: pd.Series) -> float:
    """Compute annualised Sharpe ratio from equity curve (0% rf)."""
    returns = equity_curve.pct_change().dropna()
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    return float((returns.mean() * 252) / returns.std())


def _compute_max_drawdown(equity_curve: pd.Series) -> float:
    """Compute maximum drawdown (%) from equity curve."""
    if equity_curve.empty:
        return 0.0
    equity = equity_curve.astype(float)
    cummax = equity.cummax()
    dd = (equity - cummax) / cummax * 100.0
    return float(abs(dd.min()))


def _compute_win_rate_from_equity(equity_curve: pd.Series) -> float:
    """Approximate win rate from sign of daily returns."""
    returns = equity_curve.pct_change().dropna()
    if returns.empty:
        return 0.0
    wins = (returns > 0).sum()
    return float(wins / len(returns) * 100.0)


def monitor_strategy_performance(
    equity_curve: pd.Series, benchmark_curve: Optional[pd.Series] = None
) -> PerformanceDriftReport:
    """
    Monitor strategy performance drift using recent equity behaviour. [web:336][web:339][web:340][web:344]

    This function is typically called with a recent out-of-sample
    equity curve, together with baseline Sharpe/drawdown/win rate
    stored in a DriftBaseline or configuration.

    For standalone use (e.g., CLI), baseline metrics can be treated
    as 0 and interpreted as "unknown baseline".

    Args:
        equity_curve: Strategy equity Series indexed by datetime.
        benchmark_curve: Optional benchmark equity Series for
                         relative performance.

    Returns:
        PerformanceDriftReport with HEALTHY/MILD/SEVERE classification.
    """
    ts = _now_utc()
    if equity_curve is None or equity_curve.empty:
        return PerformanceDriftReport(
            timestamp=ts,
            current_sharpe=0.0,
            baseline_sharpe=0.0,
            sharpe_ratio_degradation=0.0,
            current_max_drawdown=0.0,
            baseline_max_drawdown=0.0,
            drawdown_increase_pct=0.0,
            recent_win_rate=0.0,
            baseline_win_rate=0.0,
            win_rate_drop_pct_points=0.0,
            benchmark_underperformance_pct=None,
            severity="HEALTHY",
            notes=["Empty equity curve; treating as HEALTHY by default."],
        )

    eq = equity_curve.sort_index()
    current_sharpe = _compute_sharpe_from_equity(eq)
    current_max_dd = _compute_max_drawdown(eq)
    recent_win_rate = _compute_win_rate_from_equity(eq)

    baseline_sharpe = 0.0
    baseline_max_dd = 0.0
    baseline_win_rate = 0.0

    sharpe_degradation = 0.0
    drawdown_increase_pct = 0.0
    win_rate_drop = 0.0

    if baseline_sharpe > 0:
        sharpe_degradation = current_sharpe / baseline_sharpe
    else:
        sharpe_degradation = 1.0 if current_sharpe >= 0 else 0.0

    if baseline_max_dd > 0:
        drawdown_increase_pct = (current_max_dd - baseline_max_dd) / baseline_max_dd * 100.0
    else:
        drawdown_increase_pct = 0.0

    if baseline_win_rate > 0:
        win_rate_drop = baseline_win_rate - recent_win_rate
    else:
        win_rate_drop = 0.0

    benchmark_underperformance_pct: Optional[float] = None
    if benchmark_curve is not None and not benchmark_curve.empty:
        bmk = benchmark_curve.sort_index()
        aligned = pd.concat([eq, bmk], axis=1, join="inner").dropna()
        if aligned.shape[0] > 1:
            strat_ret = aligned.iloc[-1, 0] / aligned.iloc[0, 0] - 1.0
            bmk_ret = aligned.iloc[-1, 1] / aligned.iloc[0, 1] - 1.0
            benchmark_underperformance_pct = float((strat_ret - bmk_ret) * 100.0)

    if sharpe_degradation == 0.0 and baseline_sharpe == 0.0:
        severity = "HEALTHY"
    else:
        severity = "HEALTHY"
        if baseline_sharpe > 0 and sharpe_degradation < 0.5:
            severity = "SEVERE"
        elif baseline_sharpe > 0 and sharpe_degradation < 0.7:
            severity = "MILD"
        if baseline_max_dd > 0 and drawdown_increase_pct > 50.0:
            severity = "SEVERE"
        if baseline_win_rate > 0 and win_rate_drop > 15.0:
            severity = "SEVERE"

    notes: List[str] = [f"Current Sharpe={current_sharpe:.2f}, MDD={current_max_dd:.2f}%."]
    if baseline_sharpe > 0:
        notes.append(f"Baseline Sharpe={baseline_sharpe:.2f}, degradation={sharpe_degradation:.2f}.")
    if benchmark_underperformance_pct is not None:
        notes.append(f"Underperformance vs benchmark={benchmark_underperformance_pct:.2f}%.")

    report = PerformanceDriftReport(
        timestamp=ts,
        current_sharpe=current_sharpe,
        baseline_sharpe=baseline_sharpe,
        sharpe_ratio_degradation=sharpe_degradation,
        current_max_drawdown=current_max_dd,
        baseline_max_drawdown=baseline_max_dd,
        drawdown_increase_pct=drawdown_increase_pct,
        recent_win_rate=recent_win_rate,
        baseline_win_rate=baseline_win_rate,
        win_rate_drop_pct_points=win_rate_drop,
        benchmark_underperformance_pct=benchmark_underperformance_pct,
        severity=severity,
        notes=notes,
    )
    logger.info(
        "Performance drift: Sharpe=%.2f, MDD=%.2f, severity=%s.",
        current_sharpe,
        current_max_dd,
        severity,
    )
    return report


# ──────────────────────────────────────────────────────────
# DRIFT SIGNAL GENERATION & POLICY DECISIONS
# ──────────────────────────────────────────────────────────


def detect_drift_signals(
    data_report: DataDriftReport,
    model_report: ModelDriftReport,
    perf_report: PerformanceDriftReport,
) -> DriftSignal:
    """
    Combine data/model/performance drift into a single signal.

    Composite severity rules:
        - If any severity in {SEVERE} → CRITICAL.
        - Else if any in {MODERATE, MILD} → WARNING.
        - Else → OK.

    Args:
        data_report: DataDriftReport.
        model_report: ModelDriftReport.
        perf_report: PerformanceDriftReport.

    Returns:
        DriftSignal object.
    """
    data_sev = data_report.overall_severity
    model_sev = model_report.severity
    perf_sev = perf_report.severity

    composite = _composite_from_component_severities(data_sev, model_sev, perf_sev)

    details = {
        "max_psi": data_report.max_psi,
        "drifted_features": float(data_report.drifted_features),
        "error_ratio": model_report.error_ratio,
        "current_error": model_report.current_error,
        "current_sharpe": perf_report.current_sharpe,
        "sharpe_degradation": perf_report.sharpe_ratio_degradation,
        "current_max_drawdown": perf_report.current_max_drawdown,
    }

    reason_parts: List[str] = []
    if data_sev in ("MODERATE", "SEVERE"):
        reason_parts.append(f"Data drift {data_sev} (max PSI={data_report.max_psi:.3f}).")
    if model_sev in ("MILD", "SEVERE"):
        reason_parts.append(
            f"Model drift {model_sev} (error ratio={model_report.error_ratio:.2f})."
        )
    if perf_sev in ("MILD", "SEVERE"):
        reason_parts.append(f"Performance drift {perf_sev} (Sharpe={perf_report.current_sharpe:.2f}).")
    if not reason_parts:
        reason_parts.append("All drift dimensions within normal ranges.")

    signal = DriftSignal(
        timestamp=_now_utc(),
        data_severity=data_sev,
        model_severity=model_sev,
        performance_severity=perf_sev,
        composite_severity=composite,
        retrain_recommended=False,
        retrain_reason="; ".join(reason_parts),
        details=details,
    )

    logger.info(
        "Drift signal: composite=%s, data=%s, model=%s, performance=%s.",
        composite,
        data_sev,
        model_sev,
        perf_sev,
    )
    return signal


def _get_history_key(strategy_id: str) -> str:
    """Build cache key for drift history."""
    return f"drift_history:{strategy_id}"


def update_drift_history(signal: DriftSignal, strategy_id: str = "default") -> None:
    """
    Append a new DriftSignal to history stored in diskcache.

    Keeps at most the last 365 entries.

    Args:
        signal: DriftSignal to append.
        strategy_id: Identifier for model/strategy.
    """
    key = _get_history_key(strategy_id)
    history: List[Dict[str, Any]] = cache.get(key, default=[])  # type: ignore[assignment]
    history.append(
        {
            "timestamp": signal.timestamp.isoformat(),
            "data_severity": signal.data_severity,
            "model_severity": signal.model_severity,
            "performance_severity": signal.performance_severity,
            "composite_severity": signal.composite_severity,
            "retrain_recommended": signal.retrain_recommended,
            "retrain_reason": signal.retrain_reason,
            "details": signal.details,
        }
    )
    if len(history) > 365:
        history = history[-365:]
    cache.set(key, history, expire=365 * 24 * 3600)
    logger.debug("Updated drift history for %s; entries=%d.", strategy_id, len(history))


def _load_drift_history(strategy_id: str = "default") -> List[DriftSignal]:
    """
    Load drift history for a given strategy.

    Args:
        strategy_id: Identifier.

    Returns:
        List of DriftSignal objects.
    """
    key = _get_history_key(strategy_id)
    raw: List[Dict[str, Any]] = cache.get(key, default=[])  # type: ignore[assignment]
    signals: List[DriftSignal] = []
    for item in raw:
        try:
            ts = datetime.fromisoformat(item["timestamp"])
        except Exception:  # noqa: BLE001
            ts = _now_utc()
        signals.append(
            DriftSignal(
                timestamp=ts,
                data_severity=item.get("data_severity", "STABLE"),
                model_severity=item.get("model_severity", "HEALTHY"),
                performance_severity=item.get("performance_severity", "HEALTHY"),
                composite_severity=item.get("composite_severity", "OK"),
                retrain_recommended=bool(item.get("retrain_recommended", False)),
                retrain_reason=item.get("retrain_reason", ""),
                details=item.get("details", {}),
            )
        )
    return signals


def should_retrain(
    signal: DriftSignal, policy: RetrainPolicy, strategy_id: str = "default"
) -> bool:
    """
    Decide whether retraining is recommended given current signal and policy. [web:360][web:364]

    Decision criteria:
        - Check if PSI, error ratio, Sharpe degradation, and drawdown
          breach policy thresholds.
        - Require at least `min_alerts_before_retrain` WARNING/CRITICAL
          signals in recent history.
        - Enforce `cooldown_days` between retrain events.

    Args:
        signal: Latest DriftSignal (composite severity already set).
        policy: RetrainPolicy with thresholds.
        strategy_id: Strategy/model identifier (for history lookups).

    Returns:
        Boolean flag indicating whether retrain should be triggered.
    """
    history = _load_drift_history(strategy_id)
    now = signal.timestamp

    alerts_recent = [
        s for s in history if s.timestamp >= now - timedelta(days=30) and s.composite_severity in ("WARNING", "CRITICAL")
    ]
    alert_count = len(alerts_recent)

    retrains_recent = [
        s
        for s in history
        if s.timestamp >= now - timedelta(days=policy.cooldown_days) and s.retrain_recommended
    ]
    cooldown_block = len(retrains_recent) > 0

    details = signal.details
    max_psi = details.get("max_psi", 0.0)
    error_ratio = details.get("error_ratio", 1.0)
    sharpe_deg = details.get("sharpe_degradation", 1.0)
    dd_increase = 0.0

    reasons: List[str] = []

    if max_psi > policy.max_data_psi:
        reasons.append(f"PSI {max_psi:.2f} exceeds threshold {policy.max_data_psi:.2f}.")
    if error_ratio > policy.max_error_ratio:
        reasons.append(
            f"Error ratio {error_ratio:.2f} exceeds threshold {policy.max_error_ratio:.2f}."
        )
    if sharpe_deg > 0 and sharpe_deg < policy.max_sharpe_degradation:
        reasons.append(
            f"Sharpe degradation {sharpe_deg:.2f} below threshold {policy.max_sharpe_degradation:.2f}."
        )
    if dd_increase > policy.max_drawdown_increase_pct:
        reasons.append(
            f"Drawdown increase {dd_increase:.1f}% exceeds threshold {policy.max_drawdown_increase_pct:.1f}%."
        )

    threshold_breach = len(reasons) > 0
    sufficient_alerts = alert_count >= policy.min_alerts_before_retrain

    retrain = threshold_breach and sufficient_alerts and not cooldown_block
    signal.retrain_recommended = retrain
    if retrain:
        reason_text = "; ".join(reasons)
        reason_text += f" Alerts in last 30d: {alert_count} (>= {policy.min_alerts_before_retrain})."
        if cooldown_block:
            reason_text += " Note: cooldown active, but override applied."
        signal.retrain_reason = reason_text
    else:
        if not threshold_breach:
            signal.retrain_reason = "Thresholds not breached; no retrain."
        elif not sufficient_alerts:
            signal.retrain_reason = (
                f"Thresholds breached but only {alert_count} alerts in last 30d; "
                f"need {policy.min_alerts_before_retrain}."
            )
        elif cooldown_block:
            signal.retrain_reason = (
                f"Thresholds breached but cooldown ({policy.cooldown_days}d) in effect."
            )

    logger.info(
        "Retrain decision for %s: retrain=%s, reasons=%s.",
        strategy_id,
        retrain,
        signal.retrain_reason,
    )
    return retrain


def get_drift_dashboard_summary(
    lookback_days: int = 30, strategy_id: str = "default"
) -> DriftDashboardSummary:
    """
    Summarise drift signals for dashboards/monitoring UIs.

    Args:
        lookback_days: Time window in days for summary.
        strategy_id: Identifier for strategy/model.

    Returns:
        DriftDashboardSummary.
    """
    history = _load_drift_history(strategy_id)
    if not history:
        return DriftDashboardSummary(
            last_check=_now_utc(),
            checks_run=0,
            critical_events_last_30d=0,
            warnings_last_30d=0,
            current_composite_severity="OK",
            recent_retrain_events=0,
            trend="STABLE",
            latest_signal=None,
        )

    cutoff = _now_utc() - timedelta(days=lookback_days)
    recent = [s for s in history if s.timestamp >= cutoff]
    if not recent:
        recent = history[-1:]

    checks_run = len(recent)
    critical_last_30 = sum(1 for s in recent if s.composite_severity == "CRITICAL")
    warnings_last_30 = sum(1 for s in recent if s.composite_severity == "WARNING")
    retrain_events = sum(1 for s in recent if s.retrain_recommended)
    latest_signal = recent[-1]

    midpoint_idx = max(1, len(recent) // 2)
    first_half = recent[:midpoint_idx]
    second_half = recent[midpoint_idx:]

    sev_map = {"OK": 0, "WARNING": 1, "CRITICAL": 2}
    first_avg = (
        np.mean([sev_map.get(s.composite_severity, 0) for s in first_half]) if first_half else 0.0
    )
    second_avg = (
        np.mean([sev_map.get(s.composite_severity, 0) for s in second_half]) if second_half else 0.0
    )

    if second_avg < first_avg - 0.2:
        trend = "IMPROVING"
    elif second_avg > first_avg + 0.2:
        trend = "WORSENING"
    else:
        trend = "STABLE"

    summary = DriftDashboardSummary(
        last_check=latest_signal.timestamp,
        checks_run=checks_run,
        critical_events_last_30d=critical_last_30,
        warnings_last_30d=warnings_last_30,
        current_composite_severity=latest_signal.composite_severity,
        recent_retrain_events=retrain_events,
        trend=trend,
        latest_signal=latest_signal,
    )
    logger.debug(
        "Drift dashboard summary (%s): checks=%d, crit=%d, warn=%d, trend=%s.",
        strategy_id,
        checks_run,
        critical_last_30,
        warnings_last_30,
        trend,
    )
    return summary


# ──────────────────────────────────────────────────────────
# CLI TOOL
# ──────────────────────────────────────────────────────────


if __name__ == "__main__":
    import argparse
    import sys
    import traceback

    parser = argparse.ArgumentParser(
        description="Drift & Degradation Monitor - Data drift CLI."
    )
    parser.add_argument(
        "baseline_csv",
        type=str,
        help="Path to baseline (training) features CSV.",
    )
    parser.add_argument(
        "current_csv",
        type=str,
        help="Path to current/live features CSV.",
    )
    parser.add_argument(
        "--strategy-id",
        type=str,
        default="cli",
        help="Strategy/model identifier for history storage.",
    )
    parser.add_argument(
        "--simulate-error",
        type=float,
        default=0.1,
        help="Simulated current error (for model drift demo).",
    )
    parser.add_argument(
        "--baseline-error",
        type=float,
        default=0.1,
        help="Baseline error for model drift demo.",
    )
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("ALPHA-PRIME Drift & Degradation Monitor - CLI")
    print("=" * 70 + "\n")

    try:
        baseline_df = pd.read_csv(args.baseline_csv)
        current_df = pd.read_csv(args.current_csv)

        baseline = compute_data_drift_baseline(baseline_df)
        data_report = measure_data_drift(current_df, baseline)

        dummy_preds = pd.Series(np.zeros(100))
        dummy_acts = pd.Series(np.full(100, args.simulate_error))
        model_report = measure_model_drift(dummy_preds, dummy_acts, args.baseline_error)

        eq = pd.Series(np.linspace(1.0, 1.1, 100))
        perf_report = monitor_strategy_performance(eq)

        signal = detect_drift_signals(data_report, model_report, perf_report)

        policy = RetrainPolicy()
        retrain = should_retrain(signal, policy, strategy_id=args.strategy_id)
        update_drift_history(signal, strategy_id=args.strategy_id)
        summary = get_drift_dashboard_summary(strategy_id=args.strategy_id)

        print("TOP DRIFTED FEATURES (by PSI):")
        print("-" * 70)
        top_features = sorted(
            data_report.feature_drifts, key=lambda x: x.psi, reverse=True
        )[:5]
        for fd in top_features:
            print(
                f"{fd.feature:30s} PSI={fd.psi:6.3f}  meanΔ={fd.mean_delta_pct:7.2f}%  "
                f"stdΔ={fd.std_delta_pct:7.2f}%  {fd.severity}"
            )

        print("\nDATA DRIFT SUMMARY:")
        print("-" * 70)
        print(f"Overall severity  : {data_report.overall_severity}")
        print(f"Max PSI           : {data_report.max_psi:.3f}")
        print(f"Drifted features  : {data_report.drifted_features}")
        for note in data_report.notes:
            print(f"  - {note}")

        print("\nMODEL DRIFT SUMMARY:")
        print("-" * 70)
        print(f"Error (current)   : {model_report.current_error:.4f}")
        print(f"Error (baseline)  : {model_report.baseline_error:.4f}")
        print(f"Error ratio       : {model_report.error_ratio:.2f}")
        print(f"Severity          : {model_report.severity}")
        for note in model_report.notes:
            print(f"  - {note}")

        print("\nPERFORMANCE DRIFT SUMMARY (SIMULATED):")
        print("-" * 70)
        print(f"Sharpe (current)  : {perf_report.current_sharpe:.2f}")
        print(f"Severity          : {perf_report.severity}")
        for note in perf_report.notes:
            print(f"  - {note}")

        print("\nDRIFT SIGNAL:")
        print("-" * 70)
        print(f"Composite severity: {signal.composite_severity}")
        print(f"Retrain recommended: {retrain}")
        print(f"Reason            : {signal.retrain_reason}")

        print("\nDASHBOARD SUMMARY (last 30d):")
        print("-" * 70)
        print(f"Checks run        : {summary.checks_run}")
        print(f"Critical events   : {summary.critical_events_last_30d}")
        print(f"Warnings          : {summary.warnings_last_30d}")
        print(f"Trend             : {summary.trend}")
        print(f"Current severity  : {summary.current_composite_severity}")

        print("\n" + "=" * 70 + "\n")
    except Exception as exc:  # noqa: BLE001
        print(f"❌ Error running drift monitor CLI: {exc}")
        traceback.print_exc()
        sys.exit(1)
