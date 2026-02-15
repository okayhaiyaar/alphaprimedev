"""
============================================================
ALPHA-PRIME v2.0 - Performance Calculator
============================================================

Vectorised performance analytics for systematic trading strategies.

Design goals:
- 40+ institutional-grade metrics (returns, risk, trade, time).
- Regime-aware decomposition (bull/bear/sideways/volatile/crisis). [web:429][web:432][web:435]
- Fast NumPy/Pandas implementation suitable for 100k+ rows. [web:430][web:436]
- Statistical testing: Sharpe significance, t-tests, deflated Sharpe. [web:428][web:431][web:434][web:437]
- Benchmark comparison and attribution vs SPY or custom indices.
- Robust handling of edge cases (few trades, flat series, short history).

============================================================
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats  # lightweight, widely used

from config import get_logger, get_settings

logger = get_logger(__name__)
settings = get_settings()

ArrayLike = Union[pd.Series, np.ndarray]


# ──────────────────────────────────────────────────────────
# DATA STRUCTURES
# ──────────────────────────────────────────────────────────


@dataclass
class TradeStats:
    total_trades: int
    win_rate: float
    profit_factor: float
    expectancy: float
    avg_win_pct: float
    avg_loss_pct: float
    largest_win_loss: Tuple[float, float]
    consec_wins_max: int
    consec_losses_max: int


@dataclass
class TimeStats:
    exposure_time: float
    profitable_time: float
    avg_trade_duration: float
    hold_time: float
    recovery_factor: float
    time_under_water: float


@dataclass
class DrawdownReport:
    max_drawdown: float
    max_dd_duration_days: int
    avg_drawdown: float
    drawdowns: pd.DataFrame
    drawdown_buckets: Dict[int, float]
    recovery_time_avg_days: float


@dataclass
class RegimeReport:
    regime_sharpe: Dict[str, float]
    regime_win_rate: Dict[str, float]
    regime_max_dd: Dict[str, float]
    regime_performance: pd.DataFrame
    worst_regime: str
    regime_consistency_score: float


@dataclass
class StatTestReport:
    sharpe_p_value: float
    ttest_p_value: float
    cohen_d: float
    deflated_sharpe: float
    probable_superiority: float


@dataclass
class ComparisonReport:
    alpha: float
    beta: float
    tracking_error: float
    information_ratio: float
    up_capture: float
    down_capture: float


@dataclass
class PerformanceReport:
    basic: Dict[str, float]
    risk_adjusted: Dict[str, float]
    trade_stats: TradeStats
    time_stats: TimeStats
    regime_analysis: RegimeReport
    stat_tests: StatTestReport
    benchmark_comparison: Optional[ComparisonReport]
    overall_grade: str
    deployment_score: float


# ──────────────────────────────────────────────────────────
# HELPER UTILITIES
# ──────────────────────────────────────────────────────────


def _to_series(arr: ArrayLike, name: str = "returns") -> pd.Series:
    if isinstance(arr, pd.Series):
        return arr.astype(float)
    return pd.Series(np.asarray(arr, dtype=float), name=name)


def _safe_std(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    s = float(np.std(x, ddof=1))
    return s if s > 0 else 0.0


def _annualize_ret(daily_ret: float, periods_per_year: int = 252) -> float:
    return daily_ret * periods_per_year


def _annualize_vol(daily_vol: float, periods_per_year: int = 252) -> float:
    return daily_vol * math.sqrt(periods_per_year)


def _sharpe_ratio(returns: np.ndarray, rf_rate: float = 0.0) -> float:
    r = returns[np.isfinite(returns)]
    if r.size == 0:
        return 0.0
    excess = r - rf_rate / 252.0
    mu = float(np.mean(excess))
    sigma = _safe_std(excess)
    if sigma == 0:
        return 0.0
    return mu / sigma * math.sqrt(252.0)


def _sortino_ratio(returns: np.ndarray, rf_rate: float = 0.0) -> float:
    r = returns[np.isfinite(returns)]
    if r.size == 0:
        return 0.0
    excess = r - rf_rate / 252.0
    downside = excess[excess < 0]
    if downside.size == 0:
        return float("inf")
    downside_std = float(np.std(downside, ddof=1))
    if downside_std == 0:
        return float("inf")
    mu = float(np.mean(excess))
    return mu / downside_std * math.sqrt(252.0)


def _cagr(returns: np.ndarray) -> float:
    r = returns[np.isfinite(returns)]
    if r.size == 0:
        return 0.0
    equity = np.cumprod(1.0 + r)
    total = float(equity[-1])
    years = len(r) / 252.0
    if years <= 0 or total <= 0:
        return 0.0
    return total ** (1.0 / years) - 1.0


def _total_return(returns: np.ndarray) -> float:
    r = returns[np.isfinite(returns)]
    if r.size == 0:
        return 0.0
    return float(np.prod(1.0 + r) - 1.0)


def _geometric_mean_return(returns: np.ndarray) -> float:
    r = returns[np.isfinite(returns)]
    if r.size == 0:
        return 0.0
    log_r = np.log1p(r)
    return float(np.expm1(np.mean(log_r)))


def _buy_hold_return(prices: np.ndarray) -> float:
    if prices.size < 2:
        return 0.0
    return float(prices[-1] / prices[0] - 1.0)


def _drawdown_series_from_equity(equity: np.ndarray) -> np.ndarray:
    peak = np.maximum.accumulate(equity)
    dd = equity / peak - 1.0
    return dd


def _max_drawdown(equity: np.ndarray) -> float:
    if equity.size == 0:
        return 0.0
    dd = _drawdown_series_from_equity(equity)
    return float(dd.min())


def _ulcer_index(equity: np.ndarray) -> float:
    if equity.size == 0:
        return 0.0
    dd = _drawdown_series_from_equity(equity) * 100.0
    return float(np.sqrt(np.mean(dd**2)))


def _max_drawdown_duration(dd: np.ndarray) -> int:
    underwater = dd < 0
    max_dur = cur = 0
    for u in underwater:
        if u:
            cur += 1
            max_dur = max(max_dur, cur)
        else:
            cur = 0
    return max_dur


def _avg_drawdown(dd: np.ndarray) -> float:
    underwater = dd[dd < 0]
    if underwater.size == 0:
        return 0.0
    return float(underwater.mean())


def _var_es(returns: np.ndarray, alpha: float = 0.95) -> Tuple[float, float]:
    r = returns[np.isfinite(returns)]
    if r.size == 0:
        return 0.0, 0.0
    var = float(np.quantile(r, 1 - alpha))
    tail = r[r <= var]
    es = float(tail.mean()) if tail.size > 0 else var
    return var, es


def _skew_kurt(returns: np.ndarray) -> Tuple[float, float]:
    r = returns[np.isfinite(returns)]
    if r.size == 0:
        return 0.0, 0.0
    return float(stats.skew(r)), float(stats.kurtosis(r, fisher=True))


def _tail_ratio(returns: np.ndarray, q: float = 0.95) -> float:
    r = returns[np.isfinite(returns)]
    if r.size == 0:
        return 0.0
    up = np.quantile(r, q)
    down = abs(np.quantile(r, 1 - q))
    if down == 0:
        return 0.0
    return float(up / down)


def _trade_stats_from_returns(returns: np.ndarray) -> TradeStats:
    r = returns[np.isfinite(returns)]
    total_trades = len(r)
    if total_trades == 0:
        return TradeStats(0, 0.0, 0.0, 0.0, 0.0, 0.0, (0.0, 0.0), 0, 0)

    wins = r[r > 0]
    losses = r[r < 0]

    win_rate = float((r > 0).mean() * 100.0)
    gross_profit = float(wins.sum())
    gross_loss = float(-losses.sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    expectancy = float(r.mean())

    avg_win_pct = float(wins.mean()) if wins.size > 0 else 0.0
    avg_loss_pct = float(losses.mean()) if losses.size > 0 else 0.0
    largest_win = float(wins.max()) if wins.size > 0 else 0.0
    largest_loss = float(losses.min()) if losses.size > 0 else 0.0

    consec_wins = consec_losses = 0
    consec_wins_max = consec_losses_max = 0
    for val in r:
        if val > 0:
            consec_wins += 1
            consec_losses = 0
        elif val < 0:
            consec_losses += 1
            consec_wins = 0
        else:
            consec_wins = consec_losses = 0
        consec_wins_max = max(consec_wins_max, consec_wins)
        consec_losses_max = max(consec_losses_max, consec_losses)

    return TradeStats(
        total_trades=total_trades,
        win_rate=win_rate,
        profit_factor=profit_factor,
        expectancy=expectancy,
        avg_win_pct=avg_win_pct,
        avg_loss_pct=avg_loss_pct,
        largest_win_loss=(largest_win, largest_loss),
        consec_wins_max=consec_wins_max,
        consec_losses_max=consec_losses_max,
    )


def _time_stats_from_returns(returns: np.ndarray) -> TimeStats:
    r = returns[np.isfinite(returns)]
    if r.size == 0:
        return TimeStats(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    equity = np.cumprod(1.0 + r)
    dd = _drawdown_series_from_equity(equity)

    exposure_time = float((r != 0).mean() * 100.0)
    profitable_time = float((r > 0).mean() * 100.0)
    avg_trade_duration = 1.0
    hold_time = 1.0
    peak_dd = float(dd.min())
    rec_factor = _total_return(r) / abs(peak_dd) if peak_dd < 0 else 0.0
    time_under_water = float((dd < 0).mean() * 100.0)

    return TimeStats(
        exposure_time=exposure_time,
        profitable_time=profitable_time,
        avg_trade_duration=avg_trade_duration,
        hold_time=hold_time,
        recovery_factor=rec_factor,
        time_under_water=time_under_water,
    )


def _drawdown_report(returns: np.ndarray, index: Optional[pd.DatetimeIndex]) -> DrawdownReport:
    r = returns[np.isfinite(returns)]
    if r.size == 0:
        empty_df = pd.DataFrame(columns=["peak", "trough", "recovery", "drawdown"])
        return DrawdownReport(0.0, 0, 0.0, empty_df, {}, 0.0)

    if index is None:
        idx = pd.RangeIndex(start=0, stop=len(r), step=1)
    else:
        idx = index

    equity = np.cumprod(1.0 + r)
    dd = _drawdown_series_from_equity(equity)

    max_dd = float(dd.min())
    max_dd_duration = _max_drawdown_duration(dd)
    avg_dd = _avg_drawdown(dd)

    df = pd.DataFrame({"equity": equity, "dd": dd}, index=idx)

    drawdowns = []
    in_dd = False
    peak_date = trough_date = recovery_date = None
    peak_val = trough_val = None

    for i, (dt, row) in enumerate(df.iterrows()):
        if not in_dd:
            if i == 0 or row["dd"] == 0:
                continue
            in_dd = True
            peak_date = dt
            peak_val = row["equity"] / (1.0 + row["dd"])
            trough_date = dt
            trough_val = row["equity"]
        else:
            if row["dd"] < df.loc[trough_date, "dd"]:
                trough_date = dt
                trough_val = row["equity"]
            if row["dd"] == 0:
                recovery_date = dt
                drawdowns.append(
                    {
                        "peak": peak_date,
                        "trough": trough_date,
                        "recovery": recovery_date,
                        "drawdown": trough_val / peak_val - 1.0,
                    }
                )
                in_dd = False

    dd_df = pd.DataFrame(drawdowns)
    buckets: Dict[int, float] = {}
    if not dd_df.empty:
        for threshold in [5, 10, 20, 30, 40]:
            count = (dd_df["drawdown"] <= -threshold / 100.0).sum()
            buckets[threshold] = float(count)
        dt_durations = (dd_df["recovery"] - dd_df["peak"]).dt.days.replace(0, 1)
        recovery_time_avg = float(dt_durations.mean())
    else:
        recovery_time_avg = 0.0

    return DrawdownReport(
        max_drawdown=max_dd,
        max_dd_duration_days=max_dd_duration,
        avg_drawdown=avg_dd,
        drawdowns=dd_df,
        drawdown_buckets=buckets,
        recovery_time_avg_days=recovery_time_avg,
    )


# ──────────────────────────────────────────────────────────
# REGIME DECOMPOSITION
# ──────────────────────────────────────────────────────────


def detect_regimes(spy_returns: pd.Series, vix: Optional[pd.Series] = None) -> pd.Series:
    """
    Detect market regimes based on SPY returns and volatility. [web:429][web:432][web:435]

    Regimes:
        BULL: 20d return > 1%, vol < 1.2x 200d vol.
        BEAR: 20d return < -1%.
        SIDEWAYS: |20d return| < 1%.
        VOLATILE: vol > 1.5x 200d vol.
        CRISIS: VIX > 30 (if provided).
    """
    spy_returns = spy_returns.astype(float).sort_index()
    spy_20d = spy_returns.rolling(20).sum()
    vol_20 = spy_returns.rolling(20).std()
    vol_200 = spy_returns.rolling(200).std()
    vol_ratio = vol_20 / vol_200.replace(0, np.nan)

    regimes = pd.Series(index=spy_returns.index, dtype="object")

    regimes[(spy_20d > 0.01) & (vol_ratio < 1.2)] = "BULL"
    regimes[(spy_20d < -0.01)] = "BEAR"
    regimes[(spy_20d.abs() < 0.01)] = "SIDEWAYS"
    regimes[(vol_ratio > 1.5)] = "VOLATILE"
    if vix is not None:
        vix = vix.reindex(regimes.index).fillna(method="ffill")
        regimes[vix > 30] = "CRISIS"

    regimes = regimes.fillna("UNKNOWN")
    return regimes.astype("category")


def _regime_report(
    strategy_returns: pd.Series,
    regimes: pd.Series,
) -> RegimeReport:
    df = pd.concat([strategy_returns, regimes], axis=1).dropna()
    if df.empty:
        empty = pd.DataFrame(columns=["regime", "sharpe", "win_rate", "max_dd"])
        return RegimeReport({}, {}, {}, empty, "UNKNOWN", 0.0)

    ret = df.iloc[:, 0]
    reg = df.iloc[:, 1]

    regime_sharpe: Dict[str, float] = {}
    regime_win_rate: Dict[str, float] = {}
    regime_max_dd: Dict[str, float] = {}
    rows = []
    for regime in reg.cat.categories:
        mask = reg == regime
        r_reg = ret[mask]
        if r_reg.empty:
            continue
        sharpe = _sharpe_ratio(r_reg.values)
        win_rate = float((r_reg > 0).mean() * 100.0)
        equity = np.cumprod(1.0 + r_reg.values)
        max_dd = _max_drawdown(equity)
        regime_sharpe[regime] = sharpe
        regime_win_rate[regime] = win_rate
        regime_max_dd[regime] = max_dd
        rows.append(
            {"regime": regime, "sharpe": sharpe, "win_rate": win_rate, "max_dd": max_dd}
        )

    regime_perf = pd.DataFrame(rows).set_index("regime") if rows else pd.DataFrame()

    if regime_sharpe:
        worst_regime = min(regime_sharpe, key=regime_sharpe.get)
        max_sharpe = max(regime_sharpe.values())
        min_sharpe = min(regime_sharpe.values())
        regime_consistency = min_sharpe / max_sharpe if max_sharpe != 0 else 0.0
    else:
        worst_regime = "UNKNOWN"
        regime_consistency = 0.0

    return RegimeReport(
        regime_sharpe=regime_sharpe,
        regime_win_rate=regime_win_rate,
        regime_max_dd=regime_max_dd,
        regime_performance=regime_perf,
        worst_regime=worst_regime,
        regime_consistency_score=regime_consistency,
    )


# ──────────────────────────────────────────────────────────
# STATISTICAL TESTS
# ──────────────────────────────────────────────────────────


def _bootstrap_sharpe_pvalue(
    returns: np.ndarray,
    n_bootstrap: int = 10000,
    rf_rate: float = 0.0,
) -> float:
    r = returns[np.isfinite(returns)]
    if r.size < 10:
        return 1.0
    obs_sharpe = _sharpe_ratio(r, rf_rate)
    boot = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(r, size=r.size, replace=True)
        boot.append(_sharpe_ratio(sample, rf_rate))
    boot = np.array(boot)
    p = float((boot >= obs_sharpe).mean())
    return p


def _ttest_vs_benchmark(
    strategy: np.ndarray,
    benchmark: np.ndarray,
) -> Tuple[float, float]:
    n = min(strategy.size, benchmark.size)
    if n < 10:
        return 1.0, 0.0
    s = strategy[-n:]
    b = benchmark[-n:]
    diff = s - b
    t_stat, p_val = stats.ttest_1samp(diff, 0.0)
    sd = np.std(diff, ddof=1)
    d = float(diff.mean() / (sd + 1e-12))
    return float(p_val), d


def _deflated_sharpe_ratio(
    sharpe: float,
    T: int,
    skew: float,
    kurt: float,
    n_trials: int = 1,
) -> float:
    """
    Approximate deflated Sharpe ratio. [web:428][web:431][web:434][web:437]

    Simplified version using variance of Sharpe estimator.
    """
    if T <= 1:
        return 0.0
    sigma_sr = math.sqrt(
        (1 - skew * sharpe + (kurt - 1) * sharpe**2 / 4.0) / (T - 1)
    )
    if sigma_sr == 0:
        return 0.0
    z = sharpe / sigma_sr
    return float(stats.norm.cdf(z))


def _probable_superiority(strategy: np.ndarray, benchmark: np.ndarray) -> float:
    n = min(strategy.size, benchmark.size)
    if n == 0:
        return 0.0
    s = strategy[-n:]
    b = benchmark[-n:]
    better = (s > b).mean()
    return float(better * 100.0)


def _stat_tests(
    strategy: np.ndarray,
    benchmark: Optional[np.ndarray],
) -> StatTestReport:
    sharpe_p = _bootstrap_sharpe_pvalue(strategy)
    if benchmark is not None:
        t_p, d = _ttest_vs_benchmark(strategy, benchmark)
        prob_sup = _probable_superiority(strategy, benchmark)
    else:
        t_p, d, prob_sup = 1.0, 0.0, 0.0

    s = strategy[np.isfinite(strategy)]
    T = s.size
    skew, kurt = _skew_kurt(s)
    sr = _sharpe_ratio(s)
    dsr = _deflated_sharpe_ratio(sr, T, skew, kurt)

    return StatTestReport(
        sharpe_p_value=sharpe_p,
        ttest_p_value=t_p,
        cohen_d=d,
        deflated_sharpe=dsr,
        probable_superiority=prob_sup,
    )


# ──────────────────────────────────────────────────────────
# BENCHMARK COMPARISON
# ──────────────────────────────────────────────────────────


def _benchmark_comparison(
    strategy: np.ndarray,
    benchmark: np.ndarray,
) -> ComparisonReport:
    s = strategy[np.isfinite(strategy)]
    b = benchmark[np.isfinite(benchmark)]
    n = min(s.size, b.size)
    if n < 2:
        return ComparisonReport(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    s = s[-n:]
    b = b[-n:]

    cov = np.cov(s, b)
    var_b = cov[1, 1]
    if var_b == 0:
        beta = 0.0
    else:
        beta = cov[0, 1] / var_b
    alpha_daily = np.mean(s - beta * b)
    te = float(np.std(s - b, ddof=1))
    ir = alpha_daily / te * math.sqrt(252.0) if te > 0 else 0.0

    up_mask = b > 0
    down_mask = b < 0
    up_capture = (
        (1 + s[up_mask]).prod() - 1
    ) / ((1 + b[up_mask]).prod() - 1 + 1e-12) if up_mask.any() else 0.0
    down_capture = (
        (1 + s[down_mask]).prod() - 1
    ) / ((1 + b[down_mask]).prod() - 1 + 1e-12) if down_mask.any() else 0.0

    return ComparisonReport(
        alpha=alpha_daily * 252.0,
        beta=float(beta),
        tracking_error=te * math.sqrt(252.0),
        information_ratio=float(ir),
        up_capture=float(up_capture),
        down_capture=float(down_capture),
    )


# ──────────────────────────────────────────────────────────
# PERFORMANCE CALCULATOR
# ──────────────────────────────────────────────────────────


class PerformanceCalculator:
    """
    Central performance analytics engine for ALPHA-PRIME.

    Args:
        returns: Strategy returns (daily).
        benchmark: Optional benchmark return series (e.g., SPY).
    """

    def __init__(
        self,
        returns: ArrayLike,
        benchmark: Optional[ArrayLike] = None,
        prices: Optional[pd.Series] = None,
        spy_returns: Optional[pd.Series] = None,
        vix: Optional[pd.Series] = None,
    ) -> None:
        self.returns = _to_series(returns, name="strategy")
        self.benchmark = _to_series(benchmark, name="benchmark") if benchmark is not None else None
        self.prices = prices
        self.spy_returns = spy_returns
        self.vix = vix

    # ---- main API ----------------------------------------------------------

    def calculate_all_metrics(self) -> PerformanceReport:
        r = self.returns.values

        total_ret = _total_return(r)
        ann_ret = _annualize_ret(r.mean())
        cagr = _cagr(r)
        geom_mean = _geometric_mean_return(r)

        if self.prices is not None:
            bh_ret = _buy_hold_return(self.prices.values)
        else:
            bh_ret = total_ret

        vol_daily = _safe_std(r)
        vol_ann = _annualize_vol(vol_daily)

        sharpe = _sharpe_ratio(r)
        sortino = _sortino_ratio(r)
        equity = np.cumprod(1.0 + r)
        max_dd = _max_drawdown(equity)
        max_dd_dur = _max_drawdown_duration(_drawdown_series_from_equity(equity))
        calmar = -cagr / max_dd if max_dd < 0 else 0.0
        sterling = -total_ret / max_dd if max_dd < 0 else 0.0

        ul_idx = _ulcer_index(equity)
        martin = (total_ret / ul_idx) if ul_idx > 0 else 0.0

        var95, es95 = _var_es(r, 0.95)
        var99, es99 = _var_es(r, 0.99)
        skew, kurt = _skew_kurt(r)
        tail = _tail_ratio(r)

        trade_stats = _trade_stats_from_returns(r)
        time_stats = _time_stats_from_returns(r)
        dd_report = _drawdown_report(r, self.returns.index)

        if self.benchmark is not None:
            cmp_report = _benchmark_comparison(r, self.benchmark.values)
            info_ratio = cmp_report.information_ratio
        else:
            cmp_report = None
            info_ratio = 0.0

        if self.spy_returns is not None:
            regimes = detect_regimes(self.spy_returns, self.vix)
            regime_report = _regime_report(self.returns, regimes)
        else:
            empty_reg = pd.Series(index=self.returns.index, dtype="object")
            regime_report = _regime_report(self.returns, empty_reg)

        stat_tests = _stat_tests(r, self.benchmark.values if self.benchmark is not None else None)

        basic = {
            "total_return": total_ret,
            "annualized_return": ann_ret,
            "cagr": cagr,
            "geometric_mean_return": geom_mean,
            "buy_hold_return": bh_ret,
            "vol_daily": vol_daily,
            "vol_annualized": vol_ann,
            "max_drawdown": max_dd,
            "max_drawdown_duration_days": max_dd_dur,
            "var_95": var95,
            "cvar_95": es95,
            "var_99": var99,
            "cvar_99": es99,
            "skewness": skew,
            "kurtosis": kurt,
            "tail_ratio": tail,
        }

        risk_adjusted = {
            "sharpe": sharpe,
            "sortino": sortino,
            "calmar": calmar,
            "sterling": sterling,
            "ulcer_index": ul_idx,
            "martin_ratio": martin,
            "information_ratio": info_ratio,
        }

        grade, deploy_score = self._grade_strategy(
            sharpe=sharpe,
            max_dd=max_dd,
            calmar=calmar,
            win_rate=trade_stats.win_rate,
            regime_consistency=regime_report.regime_consistency_score,
            sharpe_p=stat_tests.sharpe_p_value,
        )

        return PerformanceReport(
            basic=basic,
            risk_adjusted=risk_adjusted,
            trade_stats=trade_stats,
            time_stats=dd_report_to_time_stats(time_stats, dd_report),
            regime_analysis=regime_report,
            stat_tests=stat_tests,
            benchmark_comparison=cmp_report,
            overall_grade=grade,
            deployment_score=deploy_score,
        )

    def drawdown_analysis(self) -> DrawdownReport:
        return _drawdown_report(self.returns.values, self.returns.index)

    def regime_decomposition(self) -> RegimeReport:
        if self.spy_returns is None:
            spy_like = self.benchmark if self.benchmark is not None else self.returns
        else:
            spy_like = self.spy_returns
        regimes = detect_regimes(spy_like, self.vix)
        return _regime_report(self.returns, regimes)

    def statistical_tests(self, benchmark_returns: pd.Series) -> StatTestReport:
        return _stat_tests(self.returns.values, benchmark_returns.values)

    def benchmark_comparison(self, benchmark_returns: pd.Series) -> ComparisonReport:
        return _benchmark_comparison(self.returns.values, benchmark_returns.values)

    # ---- grading -----------------------------------------------------------

    @staticmethod
    def _grade_strategy(
        sharpe: float,
        max_dd: float,
        calmar: float,
        win_rate: float,
        regime_consistency: float,
        sharpe_p: float,
    ) -> Tuple[str, float]:
        """
        Simple scoring model to derive overall grade and deployment score.
        """
        s_sharpe = min(1.0, max(0.0, (sharpe - 0.5) / 1.5))
        s_dd = min(1.0, max(0.0, (-max_dd - 0.05) / 0.25))  # more negative is worse
        s_calmar = min(1.0, max(0.0, calmar / 3.0))
        s_win = min(1.0, max(0.0, (win_rate - 40.0) / 30.0))
        s_regime = min(1.0, max(0.0, regime_consistency))
        s_p = 1.0 - min(1.0, sharpe_p * 2.0)

        score = (
            0.30 * s_sharpe
            + 0.20 * s_dd
            + 0.20 * s_calmar
            + 0.10 * s_win
            + 0.10 * s_regime
            + 0.10 * s_p
        ) * 100.0

        if score >= 90:
            grade = "A"
        elif score >= 80:
            grade = "B"
        elif score >= 70:
            grade = "C"
        elif score >= 60:
            grade = "D"
        else:
            grade = "F"

        return grade, score


def dd_report_to_time_stats(time_stats: TimeStats, dd_report: DrawdownReport) -> TimeStats:
    """
    Enhance TimeStats using drawdown info.
    """
    rec_factor = (
        -_total_return_from_dd(dd_report.max_drawdown) / dd_report.max_drawdown
        if dd_report.max_drawdown < 0
        else time_stats.recovery_factor
    )
    return TimeStats(
        exposure_time=time_stats.exposure_time,
        profitable_time=time_stats.profitable_time,
        avg_trade_duration=time_stats.avg_trade_duration,
        hold_time=time_stats.hold_time,
        recovery_factor=rec_factor,
        time_under_water=time_stats.time_under_water,
    )


def _total_return_from_dd(max_dd: float) -> float:
    return max_dd


# ──────────────────────────────────────────────────────────
# VISUALISATION HOOK
# ──────────────────────────────────────────────────────────


def plot_performance_report(report: PerformanceReport) -> Any:
    """
    Placeholder for 6-panel matplotlib/plotly dashboard:

        1. Cumulative returns vs benchmark.
        2. Drawdown overlay.
        3. Rolling Sharpe.
        4. Regime performance heatmap.
        5. Underwater plot.
        6. Monthly returns.

    Implementation intentionally omitted; wire to your plotting layer.
    """
    return None


# ──────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────


def _load_returns_parquet(path: str) -> pd.Series:
    df = pd.read_parquet(path)
    if isinstance(df, pd.Series):
        return df.astype(float)
    if df.shape[1] == 1:
        return df.iloc[:, 0].astype(float)
    if "returns" in df.columns:
        return df["returns"].astype(float)
    raise ValueError("Parquet must contain a 'returns' column or single series.")


def _cli_analyze(path: str) -> None:
    ret = _load_returns_parquet(path)
    calc = PerformanceCalculator(ret)
    report = calc.calculate_all_metrics()

    start, end = ret.index[0], ret.index[-1]
    print(f"PERFORMANCE ANALYSIS: {path} ({start.date()}–{end.date()})")
    print(
        f"OVERALL GRADE: {report.overall_grade} ({report.deployment_score:.0f}/100) "
        f"{'✅ DEPLOYABLE' if report.deployment_score >= 80 else '⚠️ REVIEW'}"
    )
    b = report.basic
    ra = report.risk_adjusted
    print("\n📊 KEY METRICS:")
    print(
        f"Sharpe: {ra['sharpe']:.2f} | Sortino: {ra['sortino']:.2f} | Max DD: {b['max_drawdown']*100:.1f}%"
    )
    print(
        f"Calmar: {ra['calmar']:.2f} | Win Rate: {report.trade_stats.win_rate:.1f}% "
        f"| Profit Factor: {report.trade_stats.profit_factor:.2f}"
    )
    print("\n📉 RISK:")
    print(
        f"VaR 95%: {b['var_95']*100:.2f}% | CVaR: {b['cvar_95']*100:.2f}% | Ulcer: {ra['ulcer_index']:.2f}"
    )

    reg = report.regime_analysis
    if reg.regime_sharpe:
        bull = reg.regime_sharpe.get("BULL", 0.0)
        bear = reg.regime_sharpe.get("BEAR", 0.0)
        crisis = reg.regime_sharpe.get("CRISIS", 0.0)
        print("\n🎯 REGIMES:")
        print(
            f"BULL: Sharpe {bull:.2f} | BEAR: {bear:.2f} | CRISIS: {crisis:.2f}"
        )
        print(f"Regime Consistency: {reg.regime_consistency_score*100:.0f}%")

    st = report.stat_tests
    print("\n🔬 STAT TESTS:")
    print(
        f"Sharpe p={st.sharpe_p_value:.3f} | DSR: {st.deflated_sharpe:.2f} | "
        f"Prob. Superiority: {st.probable_superiority:.1f}%"
    )


def _cli_compare(strategy_path: str, bench_path: str) -> None:
    strat = _load_returns_parquet(strategy_path)
    bench = _load_returns_parquet(bench_path)
    cmp_report = _benchmark_comparison(strat.values, bench.values)
    print(f"COMPARISON: {strategy_path} vs {bench_path}")
    print(
        f"Alpha: {cmp_report.alpha*100:.2f}% | Beta: {cmp_report.beta:.2f} | "
        f"TE: {cmp_report.tracking_error*100:.2f}% | IR: {cmp_report.information_ratio:.2f}"
    )
    print(
        f"Up Capture: {cmp_report.up_capture*100:.1f}% | Down Capture: {cmp_report.down_capture*100:.1f}%"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ALPHA-PRIME v2.0 - Performance Calculator CLI"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    analyze_p = sub.add_parser("analyze", help="Analyze strategy performance from parquet returns.")
    analyze_p.add_argument("returns_path", type=str)

    compare_p = sub.add_parser("compare", help="Compare strategy vs benchmark.")
    compare_p.add_argument("strategy_path", type=str)
    compare_p.add_argument("benchmark_path", type=str)

    args = parser.parse_args()
    if args.command == "analyze":
        _cli_analyze(args.returns_path)
    elif args.command == "compare":
        _cli_compare(args.strategy_path, args.benchmark_path)


if __name__ == "__main__":
    main()
