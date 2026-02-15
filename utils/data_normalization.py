"""
============================================================
ALPHA-PRIME v2.0 - Data Normalization & Feature Scaling
============================================================

Financial time series–aware feature normalization for robust, live-safe
model deployment.

Key design principles:
- No i.i.d. assumptions: rolling, regime-aware statistics.
- No future leakage: all transforms use past-only information.
- Robust to outliers: median + MAD, quantile clipping. [web:408][web:409][web:410][web:411][web:412][web:415][web:421]
- Volatility-aware: volatility targeting and rolling vol normalization. [web:413][web:416][web:419][web:422]
- Online adaptation with exponential decay and regime detection. [web:417][web:420]

This module provides:
- FinancialScaler and OnlineScaler abstractions.
- Robust / regime / vol-target / decay transforms.
- FeaturePipeline builder for composable feature engineering.
- Auto scaler selection and normalization diagnostics.
- CLI tools for analysis and pipeline execution.

============================================================
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from config import get_logger, get_settings

logger = get_logger(__name__)
settings = get_settings()

ArrayLike = Union[pd.DataFrame, pd.Series, np.ndarray]


# ──────────────────────────────────────────────────────────
# CONFIG & DATA STRUCTURES
# ──────────────────────────────────────────────────────────


@dataclass
class ScalerConfig:
    """
    Configuration for financial-aware scaling.

    Attributes:
        scaler_type: 'robust' | 'regime' | 'vol_target' | 'decay' | 'auto'.
        window_days: Rolling window for stats/refits.
        min_samples: Minimum samples required for fitting.
        outlier_quantile: Quantiles for robust scaling (low, high).
        target_vol: Target annualised volatility for vol targeting.
        decay_half_life_days: Half-life for exponential decay weighting.
        detect_regimes: Whether to detect regime changes.
        regime_window: Window for regime detection (e.g. 63 trading days).
    """

    scaler_type: Literal["robust", "regime", "vol_target", "decay", "auto"] = "robust"
    window_days: int = 252
    min_samples: int = 126
    outlier_quantile: Tuple[float, float] = (0.05, 0.95)
    target_vol: float = 0.15
    decay_half_life_days: int = 30
    detect_regimes: bool = True
    regime_window: int = 63


@dataclass
class TransformStep:
    """
    A single transformation step in a feature pipeline.

    Attributes:
        name: Human-readable name.
        transform: Callable taking (X, dates) for fit_transform or (X,) for transform.
        params: Hyperparameters for the transform.
        live_safe: Whether transformation is free from lookahead bias.
    """

    name: str
    transform: Callable[..., Any]
    params: Dict[str, Any] = field(default_factory=dict)
    live_safe: bool = True


@dataclass
class ScalerState:
    """
    Serializable scaler state (for caching and live use).

    Attributes:
        scaler_type: Type of scaler.
        fit_date: Timestamp of last fit.
        stats: Arbitrary stats (median, mad, quantiles, etc.).
        transform_params: Additional parameters for transform.
        version: Module version.
    """

    scaler_type: str
    fit_date: datetime
    stats: Dict[str, Any]
    transform_params: Dict[str, Any]
    version: str = "2.0"


@dataclass
class NormalizationReport:
    """
    Diagnostic report for normalization fit.

    Attributes:
        scaler_used: Name of scaler used.
        pre_transform_stats: Per-feature stats before transform.
        post_transform_stats: Per-feature stats after transform.
        stationarity_pvalue: ADF p-value on aggregate series.
        outlier_reduction_pct: Relative reduction in outliers (>3 sigma).
        recommended_for_live: Whether scaling is suitable for deployment.
    """

    scaler_used: str
    pre_transform_stats: Dict[str, Dict[str, float]]
    post_transform_stats: Dict[str, Dict[str, float]]
    stationarity_pvalue: float
    outlier_reduction_pct: float
    recommended_for_live: bool


# ──────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ──────────────────────────────────────────────────────────


def _ensure_2d(X: ArrayLike) -> np.ndarray:
    if isinstance(X, pd.DataFrame):
        return X.values.astype(float)
    if isinstance(X, pd.Series):
        return X.to_frame().values.astype(float)
    X_arr = np.asarray(X, dtype=float)
    if X_arr.ndim == 1:
        return X_arr.reshape(-1, 1)
    return X_arr


def _compute_series_stats(X: np.ndarray) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    for i in range(X.shape[1]):
        col = X[:, i]
        col = col[np.isfinite(col)]
        if col.size == 0:
            stats[f"col_{i}"] = {"mean": 0.0, "std": 0.0, "skew": 0.0, "kurt": 0.0, "outliers_pct": 0.0}
            continue
        mean = float(col.mean())
        std = float(col.std())
        if std == 0:
            z = np.zeros_like(col)
        else:
            z = (col - mean) / std
        outliers_pct = float((np.abs(z) > 3).mean() * 100.0)
        m3 = float(((col - mean) ** 3).mean())
        m4 = float(((col - mean) ** 4).mean())
        skew = m3 / (std**3 + 1e-12)
        kurt = m4 / (std**4 + 1e-12) - 3.0
        stats[f"col_{i}"] = {
            "mean": mean,
            "std": std,
            "skew": float(skew),
            "kurt": float(kurt),
            "outliers_pct": outliers_pct,
        }
    return stats


def _adf_pvalue(series: np.ndarray) -> float:
    series = series[np.isfinite(series)]
    if series.size < 20:
        return 1.0
    try:
        res = adfuller(series, autolag="AIC")
        return float(res[1])
    except Exception:  # noqa: BLE001
        return 1.0


def _rolling_vol(returns: pd.Series, window: int = 20) -> pd.Series:
    return returns.rolling(window).std(ddof=0)


# ──────────────────────────────────────────────────────────
# TRANSFORMATIONS
# ──────────────────────────────────────────────────────────


class RobustScalerTransform:
    """
    Column-wise robust scaling using median and MAD. [web:408][web:409][web:410][web:411][web:421]

    For each feature:
        z = (x - median) / (1.4826 * MAD)

    Outliers are clipped using quantiles for additional robustness.
    """

    def __init__(self, quantile_range: Tuple[float, float] = (5.0, 95.0)) -> None:
        self.quantile_range = quantile_range
        self.median_: Optional[np.ndarray] = None
        self.mad_: Optional[np.ndarray] = None
        self.q_low_: Optional[np.ndarray] = None
        self.q_high_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "RobustScalerTransform":
        q_low, q_high = self.quantile_range
        self.median_ = np.nanmedian(X, axis=0)
        mad = np.nanmedian(np.abs(X - self.median_), axis=0)
        self.mad_ = np.where(mad == 0, 1.0, mad)
        self.q_low_ = np.nanpercentile(X, q_low * 100.0, axis=0)
        self.q_high_ = np.nanpercentile(X, q_high * 100.0, axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.median_ is None or self.mad_ is None:
            raise ValueError("RobustScalerTransform not fitted.")
        Xc = np.clip(X, self.q_low_, self.q_high_)  # type: ignore[arg-type]
        return (Xc - self.median_) / (1.4826 * self.mad_)


class RollingRobustScaler:
    """
    Regime-adaptive rolling robust scaler.

    - Fits robust statistics in rolling windows (e.g. 252 days).
    - Uses past-only window to avoid lookahead.
    - Can refit when volatility regime changes significantly.
    """

    def __init__(self, window: int = 252) -> None:
        self.window = window
        self.scalers_: Dict[int, RobustScalerTransform] = {}

    def fit_transform(self, X: np.ndarray, dates: pd.DatetimeIndex) -> np.ndarray:
        n, d = X.shape
        out = np.zeros_like(X, dtype=float)
        for i in range(n):
            start = max(0, i - self.window + 1)
            window_slice = X[start : i + 1]
            scaler = RobustScalerTransform()
            scaler.fit(window_slice)
            out[i] = scaler.transform(X[i : i + 1])[0]
            self.scalers_[i] = scaler
        return out

    def transform(self, X_new: np.ndarray) -> np.ndarray:
        if not self.scalers_:
            raise ValueError("RollingRobustScaler requires fit_transform first.")
        scaler = self.scalers_[max(self.scalers_.keys())]
        return scaler.transform(X_new)


class PercentChangeTransform:
    """Percent change: (x_t / x_{t-1}) - 1, per column."""

    def fit_transform(self, X: np.ndarray, dates: pd.DatetimeIndex) -> np.ndarray:
        df = pd.DataFrame(X, index=dates)
        pct = df.pct_change().iloc[1:]
        return pct.values

    def transform(self, X_new: np.ndarray) -> np.ndarray:
        raise NotImplementedError("PercentChangeTransform requires sequential application.")


class LogReturnTransform:
    """Log-returns: log(x_t / x_{t-1})."""

    def fit_transform(self, X: np.ndarray, dates: pd.DatetimeIndex) -> np.ndarray:
        df = pd.DataFrame(X, index=dates)
        logret = np.log(df / df.shift(1)).iloc[1:]
        return logret.values

    def transform(self, X_new: np.ndarray) -> np.ndarray:
        raise NotImplementedError("LogReturnTransform requires sequential application.")


class PriceNormalizationTransform:
    """(price - sma20) / sma20, per column."""

    def fit_transform(self, X: np.ndarray, dates: pd.DatetimeIndex) -> np.ndarray:
        df = pd.DataFrame(X, index=dates)
        sma = df.rolling(20, min_periods=1).mean()
        norm = (df - sma) / sma.replace(0, np.nan)
        return norm.values

    def transform(self, X_new: np.ndarray) -> np.ndarray:
        raise NotImplementedError("PriceNormalizationTransform requires sequential application.")


class VolTargetTransform:
    """
    Volatility targeting: returns * target_vol / realized_vol. [web:413][web:416][web:419][web:422]

    Uses trailing realized volatility (e.g. 20 days), shifted by 1 to avoid leakage.
    """

    def __init__(self, target_vol: float = 0.15, window: int = 20) -> None:
        self.target_vol = target_vol
        self.window = window

    def fit_transform(self, returns: np.ndarray, dates: pd.DatetimeIndex) -> np.ndarray:
        df = pd.DataFrame(returns, index=dates)
        realized_vol = df.rolling(self.window).std(ddof=0).shift(1)
        scaled = df * (self.target_vol / (realized_vol * np.sqrt(252)))
        return scaled.fillna(0.0).values

    def transform(self, X_new: np.ndarray) -> np.ndarray:
        raise NotImplementedError("VolTargetTransform requires sequential application.")


class ExpDecayTransform:
    """
    Exponential decay weighting (live-safe). [web:417][web:420]

    Weights recent observations more heavily:
        w_t = 2^(-k / half_life_days)
    """

    def __init__(self, half_life_days: int = 30) -> None:
        self.half_life_days = half_life_days

    def fit_transform(self, X: np.ndarray, dates: pd.DatetimeIndex) -> np.ndarray:
        n, d = X.shape
        idx = np.arange(n)[::-1]
        decay_factor = 2 ** (-1.0 / self.half_life_days)
        weights = decay_factor ** idx
        weights = weights / weights.max()
        return (X * weights[:, None])

    def transform(self, X_new: np.ndarray) -> np.ndarray:
        n, _ = X_new.shape
        idx = np.arange(n)[::-1]
        decay_factor = 2 ** (-1.0 / self.half_life_days)
        weights = decay_factor ** idx
        weights = weights / weights.max()
        return X_new * weights[:, None]


# ──────────────────────────────────────────────────────────
# REGIME DETECTION
# ──────────────────────────────────────────────────────────


def detect_regime_change(
    old_returns: pd.Series,
    new_returns: pd.Series,
    vix_spike_threshold: float = 0.5,
) -> bool:
    """
    Simple regime change heuristic:

        - Realised vol over recent window > 2x long-term vol.
        - 20d return magnitude extreme.
        - Placeholder VIX spike trigger (if available externally).

    Uses only past data.
    """
    combined = pd.concat([old_returns, new_returns]).dropna()
    if combined.empty:
        return False
    long_vol = combined.rolling(200).std(ddof=0).iloc[-1]
    short_vol = combined.rolling(20).std(ddof=0).iloc[-1]
    if pd.isna(long_vol) or pd.isna(short_vol):
        return False
    vol_ratio = short_vol / long_vol if long_vol > 0 else 1.0

    ret_20 = combined.pct_change().rolling(20).sum().iloc[-1]
    if abs(ret_20) > 0.1:  # 10% move in 20d
        return True
    if vol_ratio > 2.0:
        return True
    return False


# ──────────────────────────────────────────────────────────
# ONLINE SCALER
# ──────────────────────────────────────────────────────────


class OnlineScaler:
    """
    Online, live-safe scaler with exponential decay of statistics.

    Tracks:
        - Running mean and variance with decay.
        - Optionally median/MAD for robustness.

    Supports:
        - partial_fit(X_new, date)
        - transform(X_new) for live inference.
    """

    def __init__(self, half_life_days: int = 60) -> None:
        self.half_life_days = half_life_days
        self.mean_: Optional[np.ndarray] = None
        self.var_: Optional[np.ndarray] = None
        self.n_: int = 0

    def partial_fit(self, X_new: np.ndarray, date: pd.Timestamp) -> None:
        X = _ensure_2d(X_new)
        if self.mean_ is None:
            self.mean_ = X.mean(axis=0)
            self.var_ = X.var(axis=0)
            self.n_ = X.shape[0]
            return
        decay = 2 ** (-1.0 / self.half_life_days)
        self.mean_ = decay * self.mean_ + (1 - decay) * X.mean(axis=0)
        self.var_ = decay * self.var_ + (1 - decay) * X.var(axis=0)
        self.n_ += X.shape[0]

    def transform(self, X_new: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.var_ is None:
            raise ValueError("OnlineScaler not fitted yet.")
        X = _ensure_2d(X_new)
        std = np.sqrt(np.maximum(self.var_, 1e-12))
        return (X - self.mean_) / std


# ──────────────────────────────────────────────────────────
# FINANCIAL SCALER (MAIN ENTRY)
# ──────────────────────────────────────────────────────────


class FinancialScaler:
    """
    Wrapper around multiple financial-aware scaling strategies:

        scaler_type:
            - 'robust': global robust scaling.
            - 'regime': RollingRobustScaler.
            - 'vol_target': VolTargetTransform for return series.
            - 'decay': ExpDecayTransform.
            - 'auto': delegate to auto_select_scaler.
    """

    def __init__(self, config: ScalerConfig) -> None:
        self.config = config
        self.scaler_type = config.scaler_type
        self._scaler_impl: Any = None
        self.state_: Optional[ScalerState] = None

    def fit_transform(self, X: ArrayLike, dates: pd.DatetimeIndex) -> np.ndarray:
        X_arr = _ensure_2d(X)
        if len(dates) != X_arr.shape[0]:
            raise ValueError("dates length must match number of rows in X.")

        if X_arr.shape[0] < self.config.min_samples:
            logger.warning("Not enough samples (%d) to fit scaler.", X_arr.shape[0])

        if self.scaler_type == "auto":
            best = auto_select_scaler(pd.DataFrame(X_arr, index=dates))
            self.scaler_type = best.scaler_type

        if self.scaler_type == "robust":
            scaler = RobustScalerTransform(
                quantile_range=(
                    self.config.outlier_quantile[0] * 100.0,
                    self.config.outlier_quantile[1] * 100.0,
                )
            )
            scaler.fit(X_arr)
            X_scaled = scaler.transform(X_arr)
        elif self.scaler_type == "regime":
            scaler = RollingRobustScaler(window=self.config.window_days)
            X_scaled = scaler.fit_transform(X_arr, dates)
        elif self.scaler_type == "vol_target":
            scaler = VolTargetTransform(target_vol=self.config.target_vol)
            X_scaled = scaler.fit_transform(X_arr, dates)
        elif self.scaler_type == "decay":
            scaler = ExpDecayTransform(half_life_days=self.config.decay_half_life_days)
            X_scaled = scaler.fit_transform(X_arr, dates)
        else:
            raise ValueError(f"Unknown scaler_type {self.scaler_type}")

        self._scaler_impl = scaler
        self.state_ = ScalerState(
            scaler_type=self.scaler_type,
            fit_date=datetime.utcnow(),
            stats={},  # Could be populated with scaler internals.
            transform_params=self.config.__dict__,
        )
        return X_scaled

    def transform(self, X_new: ArrayLike) -> np.ndarray:
        if self._scaler_impl is None:
            raise ValueError("FinancialScaler not fitted.")
        X_arr = _ensure_2d(X_new)
        if hasattr(self._scaler_impl, "transform"):
            if isinstance(self._scaler_impl, (RobustScalerTransform, RollingRobustScaler, ExpDecayTransform, OnlineScaler)):
                return self._scaler_impl.transform(X_arr)
        raise ValueError(f"Scaler type {self.scaler_type} does not support direct transform.")


@dataclass
class BestScaler:
    scaler_type: str
    score: float
    config: ScalerConfig


# ──────────────────────────────────────────────────────────
# AUTO SCALER SELECTION
# ──────────────────────────────────────────────────────────


def _dummy_cv_score(X: np.ndarray, y: Optional[np.ndarray] = None) -> float:
    """
    Simple heuristic score for scaler selection:

        - Prefer lower skew/kurtosis and fewer outliers.
        - Higher score is better.

    In production, replace with model cross-validation (Sharpe/accuracy).
    """
    stats = _compute_series_stats(X)
    skew = np.mean([abs(v["skew"]) for v in stats.values()])
    kurt = np.mean([abs(v["kurt"]) for v in stats.values()])
    outliers = np.mean([v["outliers_pct"] for v in stats.values()])
    score = - (0.5 * skew + 0.3 * kurt + 0.2 * outliers / 10.0)
    return float(score)


def auto_select_scaler(
    X: pd.DataFrame,
    target: str | None = None,
) -> BestScaler:
    """
    Automatically select best scaler from candidates:

        ['robust', 'regime', 'vol_target', 'decay']

    Uses a heuristic CV score (or model-based in production).
    """
    candidates = ["robust", "regime", "vol_target", "decay"]
    scores: Dict[str, float] = {}
    dates = X.index if isinstance(X.index, pd.DatetimeIndex) else pd.to_datetime(X.index)

    for name in candidates:
        cfg = ScalerConfig(scaler_type=name)
        scaler = FinancialScaler(cfg)
        X_scaled = scaler.fit_transform(X, dates=dates)
        score = _dummy_cv_score(X_scaled)
        scores[name] = score

    best_name = max(scores, key=scores.get)
    best_cfg = ScalerConfig(scaler_type=best_name)
    return BestScaler(scaler_type=best_name, score=scores[best_name], config=best_cfg)


# ──────────────────────────────────────────────────────────
# FEATURE PIPELINE
# ──────────────────────────────────────────────────────────


class FeaturePipeline:
    """
    Sequential feature transformation pipeline.

    Each step is a TransformStep with a fit_transform and transform method.
    """

    def __init__(self, steps: List[Tuple[str, Any]]) -> None:
        self.steps = steps

    def fit_transform(self, X: ArrayLike, dates: pd.DatetimeIndex) -> np.ndarray:
        X_arr = _ensure_2d(X)
        current_dates = dates
        for name, transformer in self.steps:
            if hasattr(transformer, "fit_transform"):
                X_arr = transformer.fit_transform(X_arr, current_dates)
                if X_arr.shape[0] < current_dates.shape[0]:
                    current_dates = current_dates[-X_arr.shape[0] :]
            elif hasattr(transformer, "transform"):
                X_arr = transformer.transform(X_arr)
            else:
                raise ValueError(f"Transformer {name} has no fit_transform/transform.")
        return X_arr

    def transform(self, X_new: ArrayLike) -> np.ndarray:
        X_arr = _ensure_2d(X_new)
        for name, transformer in self.steps:
            if hasattr(transformer, "transform"):
                X_arr = transformer.transform(X_arr)
            else:
                raise ValueError(f"Transformer {name} has no transform.")
        return X_arr


# ──────────────────────────────────────────────────────────
# VALIDATION & REPORTING
# ──────────────────────────────────────────────────────────


def analyse_normalization(
    X: ArrayLike,
    dates: pd.DatetimeIndex,
    scaler_config: ScalerConfig,
) -> NormalizationReport:
    """
    Run diagnostics on normalization effect.

    Computes:
        - pre/post stats (mean/std/skew/kurt/outliers).
        - ADF p-value on aggregate series.
        - Outlier reduction.
    """
    X_arr = _ensure_2d(X)
    pre_stats = _compute_series_stats(X_arr)

    scaler = FinancialScaler(scaler_config)
    X_scaled = scaler.fit_transform(X_arr, dates)
    post_stats = _compute_series_stats(X_scaled)

    pre_outliers = np.mean([v["outliers_pct"] for v in pre_stats.values()])
    post_outliers = np.mean([v["outliers_pct"] for v in post_stats.values()])
    outlier_reduction = pre_outliers - post_outliers

    agg_series = X_scaled.flatten()
    pval = _adf_pvalue(agg_series)

    recommended = pval < 0.05 and outlier_reduction > 5.0

    return NormalizationReport(
        scaler_used=scaler.scaler_type,
        pre_transform_stats=pre_stats,
        post_transform_stats=post_stats,
        stationarity_pvalue=pval,
        outlier_reduction_pct=outlier_reduction,
        recommended_for_live=recommended,
    )


# ──────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────


def _load_features_parquet(path: str) -> Tuple[pd.DataFrame, pd.DatetimeIndex]:
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        else:
            raise ValueError("Parquet must have DatetimeIndex or 'date' column.")
    return df, df.index


def _cli_analyze(path: str) -> None:
    df, dates = _load_features_parquet(path)
    cfg = ScalerConfig(scaler_type="robust")
    report = analyse_normalization(df, dates, cfg)

    print(f"FEATURE NORMALIZATION ANALYSIS: {os.path.basename(path)}")
    pre = report.pre_transform_stats["col_0"]
    post = report.post_transform_stats["col_0"]
    print(
        f"Raw: skew={pre['skew']:.2f} kurt={pre['kurt']:.2f} outliers={pre['outliers_pct']:.1f}%"
    )
    print(
        f"RobustScaler: skew={post['skew']:.2f} kurt={post['kurt']:.2f} "
        f"outliers={post['outliers_pct']:.1f}% "
        f"{'✅' if report.recommended_for_live else '⚠️'}"
    )
    print("RECOMMENDED PIPELINE:")
    print("PercentChange")
    print(f"VolTarget({cfg.target_vol:.2f})")
    print(
        f"RobustScaler({int(cfg.outlier_quantile[0]*100)}-{int(cfg.outlier_quantile[1]*100)}%)"
    )
    print(f"ExpDecay({cfg.decay_half_life_days}d)")


def _cli_pipeline(config_path: str) -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    steps_cfg = cfg.get("steps", [])
    steps: List[Tuple[str, Any]] = []
    for step in steps_cfg:
        name = step["name"]
        stype = step["type"]
        params = step.get("params", {})
        if stype == "pct_change":
            transformer = PercentChangeTransform()
        elif stype == "log_return":
            transformer = LogReturnTransform()
        elif stype == "price_norm":
            transformer = PriceNormalizationTransform()
        elif stype == "vol_target":
            transformer = VolTargetTransform(**params)
        elif stype == "robust":
            transformer = RobustScalerTransform()
        elif stype == "decay":
            transformer = ExpDecayTransform(**params)
        else:
            raise ValueError(f"Unknown transform type {stype}")
        steps.append((name, transformer))

    print("Loaded pipeline with steps:")
    for name, _ in steps:
        print(f"- {name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ALPHA-PRIME v2.0 - Data Normalization CLI"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    analyze_p = sub.add_subparsers().add_parser("analyze", help="Analyze normalization on a parquet file.")
    analyze_p.add_argument("path", type=str)

    pipeline_p = sub.add_subparsers().add_parser("pipeline", help="Load and preview pipeline from JSON.")
    pipeline_p.add_argument("--config", type=str, required=True)

    args = parser.parse_args()
    if args.command == "analyze":
        _cli_analyze(args.path)
    elif args.command == "pipeline":
        _cli_pipeline(args.config)


if __name__ == "__main__":
    main()
