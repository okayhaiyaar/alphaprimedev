"""
============================================================
ALPHA-PRIME v2.0 - Data Quality Validator
============================================================

Validates OHLCV, feature, and signal data before backtesting
or live deployment to prevent garbage-in-garbage-out.

Data Types:
1. OHLCV (Open, High, Low, Close, Volume) time series.
2. Features (technicals, fundamentals, sentiment, etc.).
3. Signals (trade entry/exit recommendations).

Validation Goals:
- Detect missing or corrupt bars.
- Flag unrealistic prices, returns, or volumes.
- Identify problematic features (NaNs, leakage, collinearity).
- Sanity-check signals (value ranges, clustering, lookahead).

Usage:
    from data_quality import (
        validate_ohlcv,
        validate_features,
        validate_signals,
        fix_data,
        compute_data_quality_score,
        generate_data_fix_summary,
    )

CLI:
    python data_quality.py validate data.csv
    python data_quality.py fix data.csv --method=smart_fill --output=fixed.csv
    python data_quality.py score data.csv
============================================================
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import get_logger, get_settings

logger = get_logger(__name__)
settings = get_settings()


# ──────────────────────────────────────────────────────────
# DATACLASSES
# ──────────────────────────────────────────────────────────


@dataclass
class OHLCVQualityReport:
    """
    Quality report for OHLCV time series.

    Attributes:
        missing_pct: Percentage of missing rows or NaNs in key columns.
        ohlc_violations: Count of bars violating OHLC constraints.
        extreme_returns: List of (date, return_pct) for flagged moves.
        low_liquidity_days: Number of low-volume days.
        weekend_trades: Count of weekend bars.
        score: 0–100 quality score.
        pass_fail: Overall pass/fail for OHLCV.
        notes: Extra comments and fix suggestions.
    """

    missing_pct: float
    ohlc_violations: int
    extreme_returns: List[Tuple[datetime, float]]
    low_liquidity_days: int
    weekend_trades: int
    score: float
    pass_fail: bool
    notes: List[str] = field(default_factory=list)


@dataclass
class FeatureQualityReport:
    """
    Quality report for feature matrix.

    Attributes:
        missing_features: Features with critical NaN/inf issues.
        high_correlation_pairs: Pairs (f1, f2, corr) with |corr|>0.95.
        constant_features: Zero-variance features.
        outlier_features: Features with extreme outliers.
        high_missing_rate: Features with >20% NaNs.
        skewed_features: Features with |skew|>3.
        score: 0–100 score.
        pass_fail: Pass/fail suggestion.
        notes: Extra comments.
    """

    missing_features: List[str]
    high_correlation_pairs: List[Tuple[str, str, float]]
    constant_features: List[str]
    outlier_features: List[str]
    high_missing_rate: List[str]
    skewed_features: List[str]
    score: float
    pass_fail: bool
    notes: List[str] = field(default_factory=list)


@dataclass
class SignalQualityReport:
    """
    Quality report for trading signals.

    Attributes:
        invalid_signals: Count of invalid signal values.
        lookahead_detected: Heuristic flag for lookahead risk.
        trade_count: Estimated number of trades.
        signal_balance_pct: Percentage of non-zero signals.
        avg_confidence: Average confidence (if available).
        clustered_signals: True if >80% signals in one week.
        autocorr: First-lag autocorrelation of signal series.
        score: 0–100 score.
        pass_fail: Pass/fail suggestion.
        notes: Extra comments and recommendations.
    """

    invalid_signals: int
    lookahead_detected: bool
    trade_count: int
    signal_balance_pct: float
    avg_confidence: float
    clustered_signals: bool
    autocorr: float
    score: float
    pass_fail: bool
    notes: List[str] = field(default_factory=list)


@dataclass
class DataQualityReport:
    """
    Top-level data quality report.

    Attributes:
        timestamp: Time when validation ran.
        overall_score: 0–100 score (weighted).
        pass_fail: Overall pass/fail.
        critical_issues: Number of critical issues across sections.
        high_severity_issues: Number of high-severity issues.
        ohlcv_report: OHLCVQualityReport.
        feature_report: FeatureQualityReport.
        signal_report: SignalQualityReport.
        fix_recommendations: List of actionable fixes.
    """

    timestamp: datetime
    overall_score: float
    pass_fail: bool
    critical_issues: int
    high_severity_issues: int
    ohlcv_report: OHLCVQualityReport
    feature_report: FeatureQualityReport
    signal_report: SignalQualityReport
    fix_recommendations: List[str] = field(default_factory=list)


# ──────────────────────────────────────────────────────────
# UTILS
# ──────────────────────────────────────────────────────────


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _is_weekend(ts: pd.Timestamp) -> bool:
    return ts.weekday() >= 5


def _annualised_vol_from_returns(returns: pd.Series) -> float:
    r = returns.dropna()
    if r.empty:
        return 0.0
    return float(r.std() * np.sqrt(252))


# ──────────────────────────────────────────────────────────
# OHLCV VALIDATION
# ──────────────────────────────────────────────────────────


def validate_ohlcv(data: pd.DataFrame) -> OHLCVQualityReport:
    """
    Validate OHLCV time series for structural and statistical issues.

    Critical checks:
        - Required columns exist.
        - OHLC relationships hold.
        - No zero/negative prices, negative volumes.
        - Monotonic DatetimeIndex.
        - No future timestamps.

    High/Medium/Low severity checks:
        - Large gaps, extreme returns, volume spikes, weekend bars.
        - Missing rates, low liquidity, high volatility.

    Args:
        data: DataFrame with at least ['Open','High','Low','Close','Volume'].

    Returns:
        OHLCVQualityReport.
    """
    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    notes: List[str] = []
    missing_pct = 0.0
    ohlc_violations = 0
    extreme_returns: List[Tuple[datetime, float]] = []
    low_liq_days = 0
    weekend_trades = 0

    if data is None or data.empty:
        notes.append("Empty OHLCV data.")
        return OHLCVQualityReport(
            missing_pct=100.0,
            ohlc_violations=0,
            extreme_returns=[],
            low_liquidity_days=0,
            weekend_trades=0,
            score=0.0,
            pass_fail=False,
            notes=notes,
        )

    missing_cols = [c for c in required_cols if c not in data.columns]
    if missing_cols:
        notes.append(f"Missing OHLCV columns: {missing_cols}.")
        return OHLCVQualityReport(
            missing_pct=100.0,
            ohlc_violations=0,
            extreme_returns=[],
            low_liquidity_days=0,
            weekend_trades=0,
            score=0.0,
            pass_fail=False,
            notes=notes,
        )

    df = data.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            notes.append("Index cannot be converted to DatetimeIndex.")
            return OHLCVQualityReport(
                missing_pct=100.0,
                ohlc_violations=0,
                extreme_returns=[],
                low_liquidity_days=0,
                weekend_trades=0,
                score=0.0,
                pass_fail=False,
                notes=notes,
            )

    if not df.index.is_monotonic_increasing:
        notes.append("OHLCV index not strictly increasing.")
        ohlc_violations += 1

    now = _now_utc()
    future_bars = df.index[df.index > now]
    if not future_bars.empty:
        notes.append(f"{len(future_bars)} bars with timestamps in the future.")
        ohlc_violations += len(future_bars)

    key = df[required_cols]
    missing_pct = float(key.isna().any(axis=1).mean() * 100.0)

    high_ok = df["High"] >= df[["Open", "Close"]].max(axis=1)
    low_ok = df["Low"] <= df[["Open", "Close"]].min(axis=1)
    ohlc_bad = (~high_ok) | (~low_ok)
    ohlc_violations += int(ohlc_bad.sum())
    if ohlc_bad.any():
        notes.append(f"{int(ohlc_bad.sum())} bars violate OHLC high/low constraints.")

    invalid_price = (df["Close"] <= 0) | (df["Open"] <= 0) | (df["High"] <= 0) | (df["Low"] <= 0)
    invalid_vol = df["Volume"] < 0
    if invalid_price.any():
        notes.append(f"{int(invalid_price.sum())} bars with non-positive prices.")
        ohlc_violations += int(invalid_price.sum())
    if invalid_vol.any():
        notes.append(f"{int(invalid_vol.sum())} bars with negative volume.")
        ohlc_violations += int(invalid_vol.sum())

    ret = df["Close"].pct_change()
    ext_mask = ret.abs() > 0.3
    for ts, r in ret[ext_mask].items():
        extreme_returns.append((ts.to_pydatetime(), float(r * 100.0)))
    if extreme_returns:
        notes.append(f"{len(extreme_returns)} bars with |return| > 30% flagged for review.")

    vol = df["Volume"]
    if len(vol) >= 20:
        rolling_avg = vol.rolling(20).mean()
        spike = vol > 10 * rolling_avg
        spike_count = int(spike.sum())
        if spike_count > 0:
            notes.append(f"{spike_count} volume spikes >10x 20-day average.")

    for ts in df.index:
        if _is_weekend(ts):
            weekend_trades += 1
    if weekend_trades > 0:
        notes.append(f"{weekend_trades} weekend bars detected in daily data.")

    if len(df) > 50:
        avg_volume = float(vol.mean())
        if avg_volume < 100_000:
            low_liq_days = int((vol < 100_000).sum())
            notes.append(
                f"Average volume {avg_volume:.0f} < 100k; {low_liq_days} low-liquidity days."
            )

        daily_ret = df["Close"].pct_change()
        ann_vol = _annualised_vol_from_returns(daily_ret)
        if ann_vol > 1.0:
            notes.append(f"20-day annualised volatility {ann_vol*100:.1f}% > 100%.")

    score = 100.0
    critical_fail = ohlc_violations > 0 or missing_pct > 5.0
    if critical_fail:
        score -= 60.0
    if extreme_returns:
        score -= 10.0
    if weekend_trades > 0:
        score -= 5.0
    if low_liq_days > 0:
        score -= 5.0
    score = max(0.0, min(100.0, score))

    pass_fail = not critical_fail and score >= 70.0

    report = OHLCVQualityReport(
        missing_pct=missing_pct,
        ohlc_violations=ohlc_violations,
        extreme_returns=extreme_returns,
        low_liquidity_days=low_liq_days,
        weekend_trades=weekend_trades,
        score=score,
        pass_fail=pass_fail,
        notes=notes,
    )

    logger.info(
        "OHLCV validation: score=%.1f, missing_pct=%.2f, violations=%d.",
        score,
        missing_pct,
        ohlc_violations,
    )
    return report


# ──────────────────────────────────────────────────────────
# FEATURE VALIDATION
# ──────────────────────────────────────────────────────────


def validate_features(features: pd.DataFrame) -> FeatureQualityReport:
    """
    Validate feature matrix for missing values, leakage, collinearity, etc.

    Critical:
        - NaN/inf in technical columns used for signals.
        - Basic leakage heuristic: any 'future' column (e.g. *_lead) present.

    High:
        - Correlation > 0.95 between features (multicollinearity).
        - Extreme outliers (beyond 1.5*IQR).
        - Constant features.

    Medium:
        - Missing rate >20%.
        - |skew| >3 (heavy skew) for numeric features.

    Args:
        features: Feature DataFrame.

    Returns:
        FeatureQualityReport.
    """
    if features is None or features.empty:
        return FeatureQualityReport(
            missing_features=[],
            high_correlation_pairs=[],
            constant_features=[],
            outlier_features=[],
            high_missing_rate=[],
            skewed_features=[],
            score=0.0,
            pass_fail=False,
            notes=["Empty feature set."],
        )

    df = features.copy()
    numeric = df.select_dtypes(include=[np.number])

    missing_features: List[str] = []
    high_corr_pairs: List[Tuple[str, str, float]] = []
    constant_features: List[str] = []
    outlier_features: List[str] = []
    high_missing_rate: List[str] = []
    skewed_features: List[str] = []
    notes: List[str] = []

    finite_mask = np.isfinite(numeric)
    inf_cols = [col for col in numeric.columns if not finite_mask[col].all()]
    if inf_cols:
        missing_features.extend(inf_cols)
        notes.append(f"Features with NaN/inf values: {inf_cols}.")

    leak_cols = [c for c in df.columns if "future" in c.lower() or "lead" in c.lower()]
    if leak_cols:
        missing_features.extend(list(set(leak_cols) - set(missing_features)))
        notes.append(f"Potential leakage columns: {leak_cols}.")

    if numeric.shape[1] >= 2:
        corr = numeric.corr(method="spearman")
        cols = corr.columns
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                c = corr.iloc[i, j]
                if abs(c) > 0.95:
                    high_corr_pairs.append((cols[i], cols[j], float(c)))
        if high_corr_pairs:
            notes.append(f"High correlation pairs (|rho|>0.95): {len(high_corr_pairs)}.")

    for col in numeric.columns:
        series = numeric[col].dropna()
        if series.empty:
            continue
        if series.nunique() <= 1:
            constant_features.append(col)

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        if iqr > 0:
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outliers = ((series < lower) | (series > upper)).mean()
            if outliers > 0.05:
                outlier_features.append(col)

        miss_rate = float(series.isna().mean() * 100.0) if len(series) > 0 else 0.0
        if miss_rate > 20.0:
            high_missing_rate.append(col)

        skew = float(series.skew())
        if abs(skew) > 3.0:
            skewed_features.append(col)

    if constant_features:
        notes.append(f"Constant features: {constant_features}.")
    if outlier_features:
        notes.append(f"Features with many outliers: {outlier_features}.")
    if high_missing_rate:
        notes.append(f"Features with >20% NaNs: {high_missing_rate}.")
    if skewed_features:
        notes.append(f"Highly skewed features (|skew|>3): {skewed_features}.")

    score = 100.0
    if missing_features or leak_cols:
        score -= 40.0
    if high_corr_pairs or constant_features or outlier_features:
        score -= 20.0
    if high_missing_rate or skewed_features:
        score -= 10.0
    score = max(0.0, min(100.0, score))

    pass_fail = score >= 70.0 and not leak_cols and not inf_cols

    report = FeatureQualityReport(
        missing_features=list(sorted(set(missing_features))),
        high_correlation_pairs=high_corr_pairs,
        constant_features=constant_features,
        outlier_features=outlier_features,
        high_missing_rate=high_missing_rate,
        skewed_features=skewed_features,
        score=score,
        pass_fail=pass_fail,
        notes=notes,
    )

    logger.info(
        "Feature validation: score=%.1f, missing=%d, high_corr_pairs=%d.",
        score,
        len(missing_features),
        len(high_corr_pairs),
    )
    return report


# ──────────────────────────────────────────────────────────
# SIGNAL VALIDATION
# ──────────────────────────────────────────────────────────


def validate_signals(
    data: pd.DataFrame, signals: pd.Series | pd.DataFrame
) -> SignalQualityReport:
    """
    Validate signal series/DataFrame for value ranges and structure.

    Critical:
        - Invalid values (signals not in {-1,0,1}, confidence not [0,1]).
        - Lookahead: signal timestamp before corresponding data.
        - Always signaling (100% non-zero).

    High:
        - Trade count <20.
        - Clustered (>80% in one week).
        - High autocorrelation (signals too predictable).

    Medium:
        - Imbalance: |buy% - 50%| >30%.
        - Low confidence: avg confidence <30%.

    Args:
        data: DataFrame with feature/price index for alignment.
        signals: Series or DataFrame of signals.

    Returns:
        SignalQualityReport.
    """
    notes: List[str] = []
    invalid_signals = 0
    lookahead_detected = False
    trade_count = 0
    signal_balance_pct = 0.0
    avg_confidence = 0.0
    clustered_signals = False
    autocorr = 0.0

    if signals is None or len(signals) == 0:
        notes.append("No signals provided.")
        return SignalQualityReport(
            invalid_signals=0,
            lookahead_detected=False,
            trade_count=0,
            signal_balance_pct=0.0,
            avg_confidence=0.0,
            clustered_signals=False,
            autocorr=0.0,
            score=0.0,
            pass_fail=False,
            notes=notes,
        )

    if isinstance(signals, pd.DataFrame):
        sig = signals.get("signal", pd.Series(index=data.index, dtype=float))
        conf = signals.get("confidence", pd.Series(index=data.index, dtype=float))
    else:
        sig = signals
        conf = pd.Series(index=sig.index, data=np.nan, dtype=float)

    sig = sig.dropna()
    if not isinstance(sig.index, pd.DatetimeIndex):
        try:
            sig.index = pd.to_datetime(sig.index)
        except Exception:
            notes.append("Signal index is not a DatetimeIndex.")
    sig = sig.sort_index()

    invalid_mask = ~sig.isin([-1, 0, 1])
    invalid_signals = int(invalid_mask.sum())
    if invalid_signals > 0:
        notes.append(f"{invalid_signals} signals with invalid values (expected -1,0,1).")

    nonzero = sig[sig != 0]
    trade_count = int(nonzero.shape[0])
    signal_balance_pct = float(nonzero.shape[0] / sig.shape[0] * 100.0)

    if trade_count == sig.shape[0] and sig.shape[0] > 0:
        notes.append("Signals are always active (no HOLD periods); may be unrealistic.")

    if conf is not None and not conf.empty:
        conf_valid = conf[(conf >= 0) & (conf <= 1)]
        if not conf_valid.empty:
            avg_confidence = float(conf_valid.mean() * 100.0)
        else:
            avg_confidence = 0.0

    if not data.empty and isinstance(data.index, pd.DatetimeIndex):
        if sig.index.min() < data.index.min():
            lookahead_detected = True
            notes.append("Signal timestamps precede available data; possible lookahead.")

    if trade_count > 0 and isinstance(sig.index, pd.DatetimeIndex):
        df = sig[sig != 0]
        week = df.index.to_period("W")
        counts = df.groupby(week).size()
        if not counts.empty and counts.max() / counts.sum() > 0.8:
            clustered_signals = True
            notes.append("More than 80% of signals occur in a single week.")

    if len(sig) > 2:
        s = sig.astype(float)
        autocorr = float(s.autocorr(lag=1))
        if abs(autocorr) > 0.7:
            notes.append(f"High signal autocorrelation (lag-1={autocorr:.2f}); may be too predictable.")

    if trade_count < 20:
        notes.append("Total trades <20; backtest sample may be too small.")

    imbalance = abs(signal_balance_pct - 50.0)
    if imbalance > 30.0:
        notes.append(
            f"Signal imbalance |buy% - 50%|={imbalance:.1f}% >30%; strategy may be one-sided."
        )
    if conf is not None and not conf.empty and avg_confidence < 30.0:
        notes.append(f"Average confidence {avg_confidence:.1f}% <30%; signals may be weak.")

    score = 100.0
    if invalid_signals > 0 or lookahead_detected:
        score -= 40.0
    if trade_count < 20 or clustered_signals or abs(autocorr) > 0.7:
        score -= 20.0
    if imbalance > 30.0 or (avg_confidence > 0 and avg_confidence < 30.0):
        score -= 10.0
    score = max(0.0, min(100.0, score))

    pass_fail = score >= 70.0 and invalid_signals == 0 and not lookahead_detected

    report = SignalQualityReport(
        invalid_signals=invalid_signals,
        lookahead_detected=lookahead_detected,
        trade_count=trade_count,
        signal_balance_pct=signal_balance_pct,
        avg_confidence=avg_confidence,
        clustered_signals=clustered_signals,
        autocorr=autocorr,
        score=score,
        pass_fail=pass_fail,
        notes=notes,
    )

    logger.info(
        "Signal validation: score=%.1f, trades=%d, invalid=%d, lookahead=%s.",
        score,
        trade_count,
        invalid_signals,
        lookahead_detected,
    )
    return report


# ──────────────────────────────────────────────────────────
# FIXING FUNCTIONS
# ──────────────────────────────────────────────────────────


def fix_data(data: pd.DataFrame, method: str = "smart_fill") -> pd.DataFrame:
    """
    Apply non-destructive cleaning to OHLCV data.

    Methods:
        - 'drop_na': Drop rows with NaN Close.
        - 'forward_fill': Forward fill all OHLCV columns.
        - 'smart_fill': Forward fill Close/Volume, interpolate High/Low.
        - 'remove_outliers': Clip extreme returns (|ret|>30%).
        - 'market_hours': Remove weekend rows (for daily).

    Args:
        data: OHLCV-like DataFrame.
        method: Cleaning method string.

    Returns:
        Cleaned DataFrame (copy).
    """
    if data is None or data.empty:
        return data

    df = data.copy()

    if method == "drop_na":
        if "Close" in df.columns:
            df = df[df["Close"].notna()]
        else:
            df = df.dropna()
    elif method == "forward_fill":
        df = df.sort_index().ffill()
    elif method == "smart_fill":
        df = df.sort_index()
        for col in ["Close", "Open", "Volume"]:
            if col in df.columns:
                df[col] = df[col].ffill()
        for col in ["High", "Low"]:
            if col in df.columns:
                df[col] = df[col].interpolate(method="linear").ffill().bfill()
    elif method == "remove_outliers":
        if "Close" in df.columns:
            ret = df["Close"].pct_change()
            clip_mask = ret.abs() > 0.3
            df.loc[clip_mask, "Close"] = df["Close"].where(~clip_mask).shift(1)
    elif method == "market_hours":
        if isinstance(df.index, pd.DatetimeIndex):
            df = df[~df.index.map(_is_weekend)]
    else:
        logger.warning("Unknown fix method '%s'; returning original data.", method)

    return df


# ──────────────────────────────────────────────────────────
# SCORING & SUMMARY
# ──────────────────────────────────────────────────────────


def compute_data_quality_score(report: DataQualityReport) -> float:
    """
    Compute overall data quality score from component reports.

    Weights:
        OHLCV   = 50%
        Features= 30%
        Signals = 20%
    """
    o = report.ohlcv_report.score
    f = report.feature_report.score
    s = report.signal_report.score
    overall = 0.5 * o + 0.3 * f + 0.2 * s
    return float(overall)


def generate_data_fix_summary(report: DataQualityReport) -> str:
    """
    Generate a human-readable summary of data quality and fix suggestions.

    Includes:
        - Overall score and pass/fail.
        - Individual component scores.
        - Critical/high issues.
        - Recommended quick fixes.
    """
    lines: List[str] = []

    ticker = "DATASET"
    ohlcv = report.ohlcv_report
    start = "N/A"
    end = "N/A"

    lines.append(
        f"DATA QUALITY REPORT: {ticker} ({start} to {end})"
    )
    status_icon = "✅" if report.pass_fail else "❌"
    lines.append(
        f"OVERALL SCORE: {report.overall_score:.0f}/100 {status_icon} "
        f"{'PASS' if report.pass_fail else 'FAIL'}"
    )

    o_icon = "✅" if ohlcv.pass_fail else "⚠️"
    f_icon = "✅" if report.feature_report.pass_fail else "⚠️"
    s_icon = "✅" if report.signal_report.pass_fail else "⚠️"

    lines.append(
        f"OHLCV: {ohlcv.score:.0f}/100 {o_icon} "
        f"(missing={ohlcv.missing_pct:.2f}%, violations={ohlcv.ohlc_violations})"
    )
    lines.append(
        f"FEATURES: {report.feature_report.score:.0f}/100 {f_icon}"
    )
    lines.append(
        f"SIGNALS: {report.signal_report.score:.0f}/100 {s_icon}"
    )
    lines.append(f"CRITICAL ISSUES: {report.critical_issues}")
    lines.append(f"HIGH SEVERITY: {report.high_severity_issues}")

    if report.fix_recommendations:
        lines.append("RECOMMENDATIONS:")
        for rec in report.fix_recommendations:
            lines.append(f"- {rec}")

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────
# TOP-LEVEL GLUE: FULL DATA QUALITY REPORT
# ──────────────────────────────────────────────────────────


def _collect_fix_recommendations(
    ohlcv_report: OHLCVQualityReport,
    feature_report: FeatureQualityReport,
    signal_report: SignalQualityReport,
) -> Tuple[int, int, List[str]]:
    critical = 0
    high = 0
    recs: List[str] = []

    if ohlcv_report.missing_pct > 5.0 or ohlcv_report.ohlc_violations > 0:
        critical += 1
        recs.append("Fix OHLCV structural issues (missing bars, OHLC violations) before trading.")
    if ohlcv_report.weekend_trades > 0:
        high += 1
        recs.append("Remove weekend bars for daily equity markets.")
    if ohlcv_report.low_liquidity_days > 0:
        high += 1
        recs.append("Increase slippage assumptions for low-liquidity periods.")

    if feature_report.missing_features:
        critical += 1
        recs.append("Impute or drop rows with NaN/inf in critical features.")
    if feature_report.high_correlation_pairs:
        high += 1
        recs.append("Remove or combine highly correlated features to reduce multicollinearity.")
    if feature_report.high_missing_rate:
        high += 1
        recs.append("Address high missing rate features via imputation or removal.")

    if signal_report.invalid_signals or signal_report.lookahead_detected:
        critical += 1
        recs.append("Fix invalid signals and remove any lookahead bias.")
    if signal_report.trade_count < 20:
        high += 1
        recs.append("Increase backtest sample size (more trades) before trusting results.")

    recs = list(dict.fromkeys(recs))
    return critical, high, recs


def build_data_quality_report(
    ohlcv: pd.DataFrame,
    features: Optional[pd.DataFrame] = None,
    signals: Optional[pd.Series | pd.DataFrame] = None,
) -> DataQualityReport:
    """
    Convenience wrapper to run all validators and produce a full report.

    Args:
        ohlcv: OHLCV DataFrame.
        features: Optional feature DataFrame.
        signals: Optional signal Series/DataFrame.

    Returns:
        DataQualityReport.
    """
    o_rep = validate_ohlcv(ohlcv)
    f_rep = validate_features(features) if features is not None else FeatureQualityReport(
        missing_features=[],
        high_correlation_pairs=[],
        constant_features=[],
        outlier_features=[],
        high_missing_rate=[],
        skewed_features=[],
        score=100.0,
        pass_fail=True,
        notes=["No feature data provided; skipped feature validation."],
    )
    s_rep = validate_signals(ohlcv, signals) if signals is not None else SignalQualityReport(
        invalid_signals=0,
        lookahead_detected=False,
        trade_count=0,
        signal_balance_pct=0.0,
        avg_confidence=0.0,
        clustered_signals=False,
        autocorr=0.0,
        score=100.0,
        pass_fail=True,
        notes=["No signals provided; skipped signal validation."],
    )

    dummy = DataQualityReport(
        timestamp=_now_utc(),
        overall_score=0.0,
        pass_fail=True,
        critical_issues=0,
        high_severity_issues=0,
        ohlcv_report=o_rep,
        feature_report=f_rep,
        signal_report=s_rep,
        fix_recommendations=[],
    )
    overall = compute_data_quality_score(dummy)
    critical, high, recs = _collect_fix_recommendations(o_rep, f_rep, s_rep)

    pass_fail = overall >= 75.0 and critical == 0

    report = DataQualityReport(
        timestamp=_now_utc(),
        overall_score=overall,
        pass_fail=pass_fail,
        critical_issues=critical,
        high_severity_issues=high,
        ohlcv_report=o_rep,
        feature_report=f_rep,
        signal_report=s_rep,
        fix_recommendations=recs,
    )

    logger.info(
        "Data quality report built: overall=%.1f, pass=%s, critical=%d, high=%d.",
        overall,
        pass_fail,
        critical,
        high,
    )
    return report


# ──────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────


def _load_ohlcv_from_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols_map = {
        "date": "Date",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    }
    for k, v in cols_map.items():
        if k in df.columns and v not in df.columns:
            df[v] = df[k]
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
    return df


def main_cli() -> None:
    parser = argparse.ArgumentParser(
        description="ALPHA-PRIME v2.0 - Data Quality Validator CLI"
    )
    parser.add_argument(
        "mode",
        type=str,
        choices=["validate", "fix", "score"],
        help="Mode: validate | fix | score.",
    )
    parser.add_argument("data_csv", type=str, help="Path to OHLCV CSV (date,open,high,low,close,volume).")
    parser.add_argument(
        "--method",
        type=str,
        default="smart_fill",
        help="Fix method for 'fix' mode (drop_na, forward_fill, smart_fill, remove_outliers, market_hours).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="fixed.csv",
        help="Output CSV path for 'fix' mode.",
    )
    args = parser.parse_args()

    if args.mode in {"validate", "score"}:
        ohlcv = _load_ohlcv_from_csv(args.data_csv)
        report = build_data_quality_report(ohlcv)
        if args.mode == "validate":
            summary = generate_data_fix_summary(report)
            print("\n" + "=" * 70)
            print(summary)
            print("\n" + "=" * 70 + "\n")
        else:
            print(f"{report.overall_score:.2f}")
    elif args.mode == "fix":
        ohlcv = _load_ohlcv_from_csv(args.data_csv)
        fixed = fix_data(ohlcv, method=args.method)
        fixed.to_csv(args.output)
        print(f"Fixed data written to {args.output} using method '{args.method}'.")


if __name__ == "__main__":
    main_cli()
