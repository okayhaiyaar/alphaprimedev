"""
============================================================
ALPHA-PRIME v2.0 - Helpers (Utility Toolkit)
============================================================

Vectorised, production-grade helper utilities for financial research
and trading systems. [web:430][web:436]

Categories:
- Date & trading calendar utilities (US markets).
- Financial math (returns, risk, position sizing).
- Data processing helpers (NaN, winsorising, z-scores).
- Symbol/string utilities.
- Signal and position helpers.
- Validation and safety checks.
- Core technical indicators (EMA, RSI, ATR, pivots).

Design notes:
- NumPy/Pandas vectorised; no explicit Python loops over rows. [web:430][web:436]
- Conservative holiday calendar for US equity markets.
- Minimal dependencies (numpy, pandas, datetime, math).

============================================================
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Dict, Iterable, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Type aliases
DateLike = Union[date, datetime]
SeriesLike = Union[pd.Series, np.ndarray]


# ──────────────────────────────────────────────────────────
# DATE & TRADING CALENDAR (US)
# ──────────────────────────────────────────────────────────


def _us_holidays(year: int) -> List[date]:
    """
    Approximate major US stock market holidays (no early closes):

    New Year's, MLK, Presidents, Good Friday, Memorial, Juneteenth,
    Independence, Labor, Thanksgiving, Christmas.
    """
    # Helper for nth weekday
    def nth_weekday(month: int, weekday: int, n: int) -> date:
        d = date(year, month, 1)
        while d.weekday() != weekday:
            d += timedelta(days=1)
        return d + timedelta(days=(n - 1) * 7)

    def last_monday(month: int) -> date:
        d = date(year, month + 1, 1) - timedelta(days=1)
        while d.weekday() != 0:
            d -= timedelta(days=1)
        return d

    holidays: List[date] = []

    # New Year's Day
    nyd = date(year, 1, 1)
    if nyd.weekday() == 5:
        nyd = date(year, 12, 31)
    elif nyd.weekday() == 6:
        nyd = date(year, 1, 2)
    holidays.append(nyd)

    # MLK: 3rd Monday in Jan
    holidays.append(nth_weekday(1, 0, 3))

    # Presidents: 3rd Monday in Feb
    holidays.append(nth_weekday(2, 0, 3))

    # Good Friday (approximation using known date pattern not implemented; skip)

    # Memorial Day: last Monday in May
    holidays.append(last_monday(5))

    # Juneteenth: June 19
    jt = date(year, 6, 19)
    if jt.weekday() == 5:
        jt = date(year, 6, 18)
    elif jt.weekday() == 6:
        jt = date(year, 6, 20)
    holidays.append(jt)

    # Independence Day: July 4
    ind = date(year, 7, 4)
    if ind.weekday() == 5:
        ind = date(year, 7, 3)
    elif ind.weekday() == 6:
        ind = date(year, 7, 5)
    holidays.append(ind)

    # Labor Day: 1st Monday in Sep
    holidays.append(nth_weekday(9, 0, 1))

    # Thanksgiving: 4th Thursday in Nov
    holidays.append(nth_weekday(11, 3, 4))

    # Christmas
    xmas = date(year, 12, 25)
    if xmas.weekday() == 5:
        xmas = date(year, 12, 24)
    elif xmas.weekday() == 6:
        xmas = date(year, 12, 26)
    holidays.append(xmas)

    return holidays


def is_trading_day(d: date) -> bool:
    """Return True if date is a US trading day (Mon–Fri, excluding holidays)."""
    if d.weekday() >= 5:
        return False
    return d not in _us_holidays(d.year)


def trading_days_between(start: date, end: date) -> int:
    """Number of trading days between two dates (inclusive of start, exclusive of end)."""
    if start > end:
        start, end = end, start
    days = (end - start).days
    rng = np.arange(days)
    dates = np.array([start + timedelta(days=int(i)) for i in rng])
    weekdays = np.array([d.weekday() for d in dates])
    mask_weekday = weekdays < 5
    years = np.array([d.year for d in dates])
    holidays = {h for y in np.unique(years) for h in _us_holidays(int(y))}
    mask_holiday = np.array([d not in holidays for d in dates])
    return int((mask_weekday & mask_holiday).sum())


def next_business_day(d: date) -> date:
    """Next trading business day after given date."""
    cur = d + timedelta(days=1)
    while not is_trading_day(cur):
        cur += timedelta(days=1)
    return cur


def get_next_friday(d: date) -> date:
    """Next calendar Friday on or after given date."""
    offset = (4 - d.weekday()) % 7
    return d + timedelta(days=offset)


def business_days_ahead(d: date, days: int) -> date:
    """Add N trading days to date (can be negative)."""
    step = 1 if days >= 0 else -1
    remaining = abs(days)
    cur = d
    while remaining > 0:
        cur += timedelta(days=step)
        if is_trading_day(cur):
            remaining -= 1
    return cur


def add_trading_days(d: date, days: int) -> date:
    """Alias for business_days_ahead."""
    return business_days_ahead(d, days)


def trading_days_in_month(year: int, month: int) -> List[date]:
    """List of trading days in a given year-month."""
    start = date(year, month, 1)
    if month == 12:
        end = date(year + 1, 1, 1)
    else:
        end = date(year, month + 1, 1)
    days = (end - start).days
    rng = [start + timedelta(days=i) for i in range(days)]
    return [d for d in rng if is_trading_day(d)]


def get_rebalance_dates(start: date, end: date, freq: str) -> List[date]:
    """
    Return rebalance dates between start and end.

    freq: 'M' (monthly), 'Q' (quarterly), 'A' (annual).
    """
    freq = freq.upper()
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if freq == "M":
        idx = pd.date_range(start_ts, end_ts, freq="M")
    elif freq == "Q":
        idx = pd.date_range(start_ts, end_ts, freq="Q")
    elif freq == "A":
        idx = pd.date_range(start_ts, end_ts, freq="A")
    else:
        raise ValueError("freq must be one of 'M','Q','A'")
    dates = [d.date() for d in idx]
    return [d if is_trading_day(d) else business_days_ahead(d, -1) for d in dates]


def is_market_open(
    dt: datetime, timezone: str = "US/Eastern"
) -> bool:
    """
    Approximate US equity market hours: 9:30–16:00 local (no DST handling).

    Args:
        dt: Datetime (assumed already in desired tz).
        timezone: Placeholder argument for future tz integration.
    """
    if not is_trading_day(dt.date()):
        return False
    hour = dt.hour
    minute = dt.minute
    if hour < 9 or hour > 16:
        return False
    if hour == 9 and minute < 30:
        return False
    return True


def get_trading_session(dt: datetime) -> str:
    """
    Return trading session label: 'PRE', 'INTRADAY', 'POST', or 'CLOSED'."""
    if not is_trading_day(dt.date()):
        return "CLOSED"
    h = dt.hour
    m = dt.minute
    if 4 <= h < 9 or (h == 9 and m < 30):
        return "PRE"
    if (h == 9 and m >= 30) or (10 <= h < 16):
        return "INTRADAY"
    if h == 16 and m <= 30 or 16 < h < 20:
        return "POST"
    return "CLOSED"


# ──────────────────────────────────────────────────────────
# FINANCIAL CALCULATIONS
# ──────────────────────────────────────────────────────────


def pct_change_safe(series: pd.Series, fill_value: float = 0.0) -> pd.Series:
    """
    Vectorised percent change with safe fill.

    result[0] = fill_value
    result[i] = series[i] / series[i-1] - 1
    """
    arr = series.astype(float).values
    out = np.full_like(arr, fill_value, dtype=float)
    if arr.size > 1:
        prev = arr[:-1]
        out[1:] = np.where(prev != 0, arr[1:] / prev - 1.0, fill_value)
    return pd.Series(out, index=series.index, name=series.name)


def log_returns(series: pd.Series) -> pd.Series:
    """Log-returns: log(price_t / price_{t-1})."""
    arr = series.astype(float).values
    out = np.full_like(arr, np.nan, dtype=float)
    if arr.size > 1:
        prev = arr[:-1]
        out[1:] = np.where(prev > 0, np.log(arr[1:] / prev), np.nan)
    return pd.Series(out, index=series.index, name=series.name)


def sharpe_ratio(returns: pd.Series, rf: float = 0.0) -> float:
    """Annualised Sharpe ratio using daily returns and rf as annual rate."""
    r = returns.astype(float).values
    mask = np.isfinite(r)
    r = r[mask]
    if r.size == 0:
        return 0.0
    excess = r - rf / 252.0
    mu = excess.mean()
    sigma = excess.std(ddof=1)
    if sigma == 0:
        return 0.0
    return float(mu / sigma * math.sqrt(252.0))


def max_drawdown(returns: pd.Series) -> float:
    """Maximum drawdown (as negative fraction) from daily returns."""
    r = returns.astype(float).values
    if r.size == 0:
        return 0.0
    equity = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(equity)
    dd = equity / peak - 1.0
    return float(dd.min())


def calmar_ratio(returns: pd.Series) -> float:
    """Calmar ratio: CAGR / |max drawdown|."""
    r = returns.astype(float).values
    if r.size == 0:
        return 0.0
    equity = np.cumprod(1.0 + r)
    total = float(equity[-1])
    years = len(r) / 252.0
    if years <= 0 or total <= 0:
        return 0.0
    cagr = total ** (1.0 / years) - 1.0
    mdd = max_drawdown(returns)
    if mdd >= 0:
        return 0.0
    return float(cagr / abs(mdd))


def kelly_criterion(win_prob: float, avg_win: float, avg_loss: float) -> float:
    """
    Kelly fraction with average win/loss.

    f* = p - (1-p)/b, b = avg_win/|avg_loss|.
    """
    if avg_loss >= 0 or avg_win <= 0:
        return 0.0
    b = avg_win / abs(avg_loss)
    if b <= 0:
        return 0.0
    p = win_prob
    if p <= 0 or p >= 1:
        return 0.0
    f = p - (1 - p) / b
    return float(max(0.0, min(1.0, f)))


def position_size(capital: float, risk_pct: float, stop_distance_pct: float) -> float:
    """
    Dollar position size given capital, risk fraction, and stop distance.

    pos_size = capital * risk_pct / stop_distance_pct
    """
    if stop_distance_pct <= 0:
        return 0.0
    return float(capital * risk_pct / stop_distance_pct)


def optimal_f(returns: pd.Series) -> float:
    """
    Approximate optimal f (Tharp) via grid search on fraction f in [0,1].

    Vectorised evaluation over candidate f values.
    """
    r = returns.astype(float).values
    if r.size == 0:
        return 0.0
    f_grid = np.linspace(0.0, 1.0, 101)
    # equity(f) = prod(1 + f * r)
    eq = np.prod(1.0 + np.outer(r, f_grid), axis=0)
    idx = int(np.argmax(eq))
    return float(f_grid[idx])


def var_historical(returns: pd.Series, alpha: float = 0.05) -> float:
    """
    Historical VaR at level alpha (default 5%).

    Returns negative number (loss).
    """
    r = returns.astype(float).values
    r = r[np.isfinite(r)]
    if r.size == 0:
        return 0.0
    q = np.quantile(r, alpha)
    return float(q)


# ──────────────────────────────────────────────────────────
# DATA PROCESSING
# ──────────────────────────────────────────────────────────


def safe_divide(a: pd.Series, b: pd.Series, fill_value: float = 0.0) -> pd.Series:
    """Elementwise division a / b with safe handling of zero/NaN."""
    a_aligned, b_aligned = a.align(b, join="outer")
    num = a_aligned.astype(float)
    den = b_aligned.astype(float)
    out = np.where(np.isfinite(num) & np.isfinite(den) & (den != 0), num / den, fill_value)
    return pd.Series(out, index=a_aligned.index, name=a.name)


def fillna_forward_limit(series: pd.Series, limit: int = 5) -> pd.Series:
    """Forward-fill NaNs with a maximum of `limit` steps."""
    return series.ffill(limit=limit)


def clip_extreme_returns(returns: pd.Series, max_pct: float = 0.5) -> pd.Series:
    """Clip returns to +/- max_pct."""
    r = returns.astype(float)
    return r.clip(lower=-max_pct, upper=max_pct)


def remove_weekends(df: pd.DataFrame) -> pd.DataFrame:
    """Filter out weekend rows based on DatetimeIndex."""
    if not isinstance(df.index, pd.DatetimeIndex):
        return df
    mask = df.index.dayofweek < 5
    return df.loc[mask]


def resample_to_trading_days(series: pd.Series, freq: str) -> pd.Series:
    """
    Resample series to trading-frequency close values.

    freq examples: 'B' (business), 'W', 'M'.
    """
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("Series index must be DatetimeIndex.")
    resampled = series.resample(freq).last().dropna()
    return resampled


def zscore(series: pd.Series, window: int = 20) -> pd.Series:
    """Rolling z-score over window."""
    s = series.astype(float)
    mean = s.rolling(window).mean()
    std = s.rolling(window).std(ddof=1)
    return (s - mean) / std.replace(0, np.nan)


def rank_data(df: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    """
    Rolling cross-sectional rank normalised to [0,1].

    For each date, rank assets.
    """
    def _rank_row(row: pd.Series) -> pd.Series:
        return row.rank(pct=True)
    return df.rolling(window).apply(lambda x: _rank_row(pd.Series(x, index=df.columns)), raw=False)


def winsorize(
    df: pd.DataFrame, limits: Tuple[float, float] = (0.01, 0.99)
) -> pd.DataFrame:
    """Column-wise winsorisation at given lower/upper quantiles."""
    low, high = limits
    q_low = df.quantile(low)
    q_high = df.quantile(high)
    return df.clip(lower=q_low, upper=q_high, axis=1)


# ──────────────────────────────────────────────────────────
# SYMBOL & STRING UTILITIES
# ──────────────────────────────────────────────────────────


def is_valid_ticker(symbol: str) -> bool:
    """Check if symbol contains only A–Z, 0–9, '.', '-'."""
    if not symbol:
        return False
    for ch in symbol:
        if not (ch.isalnum() or ch in ".-"):
            return False
    return True


def normalize_ticker(symbol: str) -> str:
    """Normalise ticker to upper-case and strip whitespace."""
    return symbol.strip().upper()


def ticker_to_figi(ticker: str) -> str:
    """
    Placeholder FIGI mapping: deterministic hash-based pseudo FIGI.

    In production, integrate OpenFIGI API.
    """
    import hashlib

    base = normalize_ticker(ticker)
    h = hashlib.sha1(base.encode("utf-8")).hexdigest().upper()
    return f"BBG{h[:9]}"


def parse_pnl_str(pnl_str: str) -> float:
    """Parse PnL string like '$1,234.56' into float."""
    s = pnl_str.strip().replace(",", "").replace("$", "")
    try:
        return float(s)
    except ValueError:
        return 0.0


def humanize_pnl(amount: float, currency: str = "$") -> str:
    """Format amount into human-readable PnL string."""
    sign = "-" if amount < 0 else ""
    amt = abs(amount)
    if amt >= 1e9:
        val = f"{amt/1e9:.2f}B"
    elif amt >= 1e6:
        val = f"{amt/1e6:.2f}M"
    elif amt >= 1e3:
        val = f"{amt/1e3:.2f}K"
    else:
        val = f"{amt:.2f}"
    return f"{sign}{currency}{val}"


def parse_percentage(pct_str: str) -> float:
    """Parse percentage string like '2.5%' into decimal 0.025."""
    s = pct_str.strip().replace("%", "")
    try:
        return float(s) / 100.0
    except ValueError:
        return 0.0


# ──────────────────────────────────────────────────────────
# SIGNAL & POSITION HELPERS
# ──────────────────────────────────────────────────────────


def signal_to_direction(signal: Union[float, int]) -> Literal[-1, 0, 1]:
    """Map raw signal to discrete direction {-1,0,1}."""
    if signal > 0:
        return 1
    if signal < 0:
        return -1
    return 0


def position_direction_to_signal(pos: float) -> Literal[-1, 0, 1]:
    """Map position size to direction {-1,0,1}."""
    if pos > 0:
        return 1
    if pos < 0:
        return -1
    return 0


def calculate_stop_loss(entry: float, atr_val: float, risk_reward: float = 2.0) -> float:
    """
    Stop-loss based on ATR:

        stop = entry - k * ATR (long)
    """
    k = 1.0 / risk_reward if risk_reward > 0 else 1.0
    return float(entry - k * atr_val)


def calculate_target(entry: float, stop: float, rr_ratio: float = 2.0) -> float:
    """Calculate target price given entry, stop, and R:R."""
    risk = entry - stop
    return float(entry + rr_ratio * risk)


def risk_reward_ratio(entry: float, stop: float, target: float) -> float:
    """Compute risk-reward ratio."""
    risk = entry - stop
    reward = target - entry
    if risk <= 0:
        return 0.0
    return float(reward / risk)


# ──────────────────────────────────────────────────────────
# VALIDATION & SAFETY
# ──────────────────────────────────────────────────────────


def validate_returns(returns: pd.Series, max_daily: float = 0.5) -> bool:
    """
    Validate daily returns are finite and within +/- max_daily.

    Returns True if all checks pass.
    """
    r = returns.astype(float)
    if r.empty:
        return False
    if not np.isfinite(r.values).all():
        return False
    if (r.abs() > max_daily).any():
        return False
    return True


def validate_ohlc(df: pd.DataFrame) -> List[str]:
    """
    Validate OHLC structure; return list of violation messages.

    Conditions:
        HIGH >= max(OPEN,CLOSE)
        LOW <= min(OPEN,CLOSE)
    """
    violations: List[str] = []
    required = {"open", "high", "low", "close"}
    if not required.issubset(df.columns.str.lower()):
        missing = required.difference(set(df.columns.str.lower()))
        violations.append(f"Missing columns: {missing}")
        return violations
    cols = {c.lower(): c for c in df.columns}
    o = df[cols["open"]].astype(float)
    h = df[cols["high"]].astype(float)
    l = df[cols["low"]].astype(float)
    c = df[cols["close"]].astype(float)
    if (h < np.maximum(o, c)).any():
        violations.append("HIGH < max(OPEN,CLOSE)")
    if (l > np.minimum(o, c)).any():
        violations.append("LOW > min(OPEN,CLOSE)")
    return violations


def assert_no_nan(obj: Union[pd.Series, pd.DataFrame], name: str) -> None:
    """Raise ValueError if any NaN present."""
    if obj.isna().any().any():
        raise ValueError(f"{name} contains NaN values.")


def assert_positive(prices: pd.Series, name: str) -> None:
    """Raise ValueError if any price <= 0."""
    if (prices <= 0).any():
        raise ValueError(f"{name} contains non-positive values.")


# ──────────────────────────────────────────────────────────
# TECHNICAL HELPERS
# ──────────────────────────────────────────────────────────


def pivot_points(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classic daily pivot points.

    Assumes columns: high, low, close.
    """
    cols = {c.lower(): c for c in df.columns}
    h = df[cols["high"]].astype(float)
    l = df[cols["low"]].astype(float)
    c = df[cols["close"]].astype(float)
    pivot = (h + l + c) / 3.0
    r1 = 2 * pivot - l
    s1 = 2 * pivot - h
    r2 = pivot + (h - l)
    s2 = pivot - (h - l)
    out = pd.DataFrame(
        {
            "pivot": pivot,
            "r1": r1,
            "s1": s1,
            "r2": r2,
            "s2": s2,
        },
        index=df.index,
    )
    return out


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Average True Range (ATR) over given period."""
    h = high.astype(float)
    l = low.astype(float)
    c = close.astype(float)
    prev_close = c.shift(1)
    tr1 = h - l
    tr2 = (h - prev_close).abs()
    tr3 = (l - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average."""
    return series.astype(float).ewm(span=period, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (RSI)."""
    delta = close.astype(float).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_series = 100 - (100 / (1 + rs))
    return rsi_series


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
    raise ValueError("Parquet must contain 'returns' column or single series.")


def _cli_validate_returns(path: str) -> None:
    ret = _load_returns_parquet(path)
    valid = validate_returns(ret)
    total = len(ret)
    exceeded = ret[ret.abs() > 0.5]
    print(f"RETURNS VALIDATION: {path}")
    if valid:
        print(f"✅ {total:,} valid daily returns")
    else:
        print(f"⚠️ Validation failed for {total:,} returns")
    if not exceeded.empty:
        dates = ", ".join(d.strftime("%Y-%m-%d") for d in exceeded.index[:5])
        print(f"⚠️ {len(exceeded)} extreme days (>50%): {dates}")


def _cli_calc_sharpe(path: str, rf: float) -> None:
    ret = _load_returns_parquet(path)
    s = sharpe_ratio(ret, rf=rf)
    mdd = max_drawdown(ret)
    print(f"Sharpe: {s:.2f} | Max DD: {mdd*100:.1f}%")


def _cli_dates_between(start_str: str, end_str: str) -> None:
    start = date.fromisoformat(start_str)
    end = date.fromisoformat(end_str)
    days = trading_days_between(start, end)
    print(f"Trading days between: {days} days")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ALPHA-PRIME v2.0 - Helpers CLI"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    vret = sub.add_parser("validate_returns", help="Validate returns parquet.")
    vret.add_argument("path", type=str)

    calc = sub.add_parser("calc", help="Calculate metrics.")
    calc.add_argument("--sharpe", type=str, help="Returns parquet path for Sharpe.")
    calc.add_argument("--rf", type=float, default=0.0)

    dates_cmd = sub.add_parser("dates", help="Date utilities.")
    dates_cmd.add_argument("--between", nargs=2, metavar=("START", "END"))

    args = parser.parse_args()
    if args.command == "validate_returns":
        _cli_validate_returns(args.path)
    elif args.command == "calc":
        if args.sharpe:
            _cli_calc_sharpe(args.sharpe, args.rf)
    elif args.command == "dates":
        if args.between:
            _cli_dates_between(args.between[0], args.between[1])


if __name__ == "__main__":
    main()
