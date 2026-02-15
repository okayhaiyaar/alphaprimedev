#!/usr/bin/env python3
"""
ALPHA-PRIME v2.0 - Trade Export Script
======================================
Export trade data with flexible filtering and formatting options.

Usage:
    python scripts/export_trades.py --format csv --year 2024
    python scripts/export_trades.py --format excel --strategy EMA_Cross_v2
    python scripts/export_trades.py --tax-report --year 2024
    python scripts/export_trades.py --reconcile --broker zerodha --month 2024-01
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple

import json
import pandas as pd
import numpy as np
from decimal import Decimal

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from database.queries import TradeQueries
from database.models import Trade, Order, Position  # noqa: F401 (used via to_dict)
from core.tax_calculator import TaxCalculator
from integrations.broker_clients import get_broker_client
from utils.logger import setup_logger

# === CONFIGURATION ===
EXPORTS_DIR = PROJECT_ROOT / "exports"
EXPORT_FORMATS = ["csv", "excel", "json", "parquet", "sql"]

# === DATA SCHEMAS ===
SUMMARY_COLUMNS = [
    "trade_id",
    "date",
    "symbol",
    "side",
    "quantity",
    "entry_price",
    "exit_price",
    "pnl",
    "pnl_pct",
    "holding_days",
]

DETAILED_COLUMNS = [
    "trade_id",
    "order_id",
    "timestamp",
    "symbol",
    "exchange",
    "side",
    "quantity",
    "price",
    "commission",
    "slippage",
    "strategy_id",
    "signal_id",
    "pnl",
    "exit_reason",
    "tags",
]

FULL_COLUMNS = DETAILED_COLUMNS + [
    "entry_timestamp",
    "exit_timestamp",
    "avg_entry_price",
    "avg_exit_price",
    "total_commission",
    "total_slippage",
    "gross_pnl",
    "net_pnl",
    "stop_loss",
    "take_profit",
    "max_favorable_excursion",
    "max_adverse_excursion",
]


# === ARGUMENT PARSING ===
def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="ALPHA-PRIME v2.0 Trade Export Utility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Export Modes:
  Standard       - Export trades with standard columns
  Tax Report     - Generate tax-compliant trade report
  Reconciliation - Match trades with broker statements
  Attribution    - Performance attribution by dimension
  Audit Trail    - Full audit trail with all details

Examples:
  # Export all 2024 trades to CSV
  python scripts/export_trades.py --format csv --year 2024

  # Detailed Excel export for specific strategy
  python scripts/export_trades.py --format excel --strategy EMA_Cross_v2 --detailed

  # Tax report (FIFO method)
  python scripts/export_trades.py --tax-report --year 2024 --method fifo

  # Broker reconciliation
  python scripts/export_trades.py --reconcile --broker zerodha --month 2024-01

  # Performance attribution by strategy
  python scripts/export_trades.py --attribution --by strategy --year 2024
        """,
    )

    # Export format
    parser.add_argument(
        "--format",
        type=str,
        choices=EXPORT_FORMATS,
        default="csv",
        help="Export format (default: csv)",
    )

    # Date filtering
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--year", type=int, help="Calendar year (shortcut for full year)")
    parser.add_argument("--month", type=str, help="Month (YYYY-MM)")
    parser.add_argument("--quarter", type=str, help="Quarter (YYYY-Q1/Q2/Q3/Q4)")

    # Filtering options
    parser.add_argument("--strategy", type=str, help="Filter by strategy ID")
    parser.add_argument(
        "--symbol",
        type=str,
        help="Filter by symbol (comma-separated for multiple)",
    )
    parser.add_argument(
        "--side",
        type=str,
        choices=["LONG", "SHORT"],
        help="Filter by trade side",
    )
    parser.add_argument(
        "--status",
        type=str,
        choices=["OPEN", "CLOSED", "CANCELLED"],
        help="Filter by trade status",
    )
    parser.add_argument("--min-pnl", type=float, help="Minimum PnL filter")
    parser.add_argument("--max-pnl", type=float, help="Maximum PnL filter")

    # Data granularity
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Export summary data only",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Export detailed data",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Export full audit trail",
    )

    # Special modes
    parser.add_argument(
        "--tax-report",
        action="store_true",
        help="Generate tax-compliant report",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["fifo", "lifo", "specific_lot", "average_cost"],
        default="fifo",
        help="Tax accounting method (default: fifo)",
    )
    parser.add_argument(
        "--reconcile",
        action="store_true",
        help="Reconcile with broker statements",
    )
    parser.add_argument("--broker", type=str, help="Broker name for reconciliation")
    parser.add_argument(
        "--attribution",
        action="store_true",
        help="Performance attribution analysis",
    )
    parser.add_argument(
        "--by",
        type=str,
        choices=["strategy", "symbol", "month", "day"],
        help="Attribution dimension",
    )

    # Output options
    parser.add_argument(
        "--output",
        type=Path,
        help="Output file path (auto-generated if not provided)",
    )
    parser.add_argument(
        "--compress",
        action="store_true",
        help="Compress output file (gzip)",
    )
    parser.add_argument(
        "--split",
        type=int,
        help="Split into chunks of N rows (per-file limit)",
    )

    # Data options
    parser.add_argument(
        "--include-fees",
        action="store_true",
        help="Include commission and fees in PnL (default: enabled)",
    )
    parser.add_argument(
        "--include-slippage",
        action="store_true",
        help="Include slippage in calculations",
    )
    parser.add_argument(
        "--currency",
        type=str,
        default="USD",
        help="Currency for amounts (default: USD)",
    )

    # Incremental export
    parser.add_argument(
        "--since-last-export",
        action="store_true",
        help="Export only trades since last export",
    )
    parser.add_argument(
        "--checkpoint-file",
        type=Path,
        default=Path(".last_export_checkpoint"),
        help="Checkpoint file for incremental exports",
    )

    # Validation
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate data before export",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview export without writing file",
    )

    # Logging
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose logging",
    )

    return parser.parse_args()


# === DATE RANGE HANDLING ===
def get_date_range(config: Dict[str, Any]) -> Tuple[datetime, datetime]:
    """Determine date range from config."""
    if config.get("year"):
        start = datetime(config["year"], 1, 1)
        end = datetime(config["year"], 12, 31, 23, 59, 59)

    elif config.get("month"):
        year, month = map(int, config["month"].split("-"))
        start = datetime(year, month, 1)
        next_month = month + 1 if month < 12 else 1
        next_year = year if month < 12 else year + 1
        end = datetime(next_year, next_month, 1) - timedelta(seconds=1)

    elif config.get("quarter"):
        year_str, q_str = config["quarter"].split("-Q")
        year = int(year_str)
        quarter = int(q_str)
        start_month = (quarter - 1) * 3 + 1
        start = datetime(year, start_month, 1)
        end_month = start_month + 2
        if end_month == 12:
            end = datetime(year, 12, 31, 23, 59, 59)
        else:
            end = datetime(year, end_month + 1, 1) - timedelta(seconds=1)

    elif config.get("start") and config.get("end"):
        start = datetime.fromisoformat(config["start"])
        end = datetime.fromisoformat(config["end"])

    else:
        end = datetime.now()
        start = end - timedelta(days=30)

    # Incremental override: since last export timestamp
    if config.get("since_last_export") and config.get("checkpoint_file"):
        chk = config["checkpoint_file"]
        if chk.exists():
            last_ts_str = chk.read_text(encoding="utf-8").strip()
            try:
                last_ts = datetime.fromisoformat(last_ts_str)
                if last_ts > start:
                    start = last_ts
            except Exception:
                pass

    return start, end


# === DATA FETCHING & FILTERING ===
async def fetch_trades(config: Dict[str, Any], logger: logging.Logger) -> pd.DataFrame:
    """Fetch and filter trades based on configuration."""
    logger.info("Fetching trade data...")

    queries = TradeQueries()
    start_date, end_date = get_date_range(config)
    logger.info("Date range: %s to %s", start_date, end_date)

    symbols: Optional[List[str]] = None
    if config.get("symbol"):
        symbols = [s.strip() for s in config["symbol"].split(",") if s.strip()]

    trades: List[Trade] = await queries.get_trades(
        start_date=start_date,
        end_date=end_date,
        strategy_id=config.get("strategy"),
        symbol=symbols,
        side=config.get("side"),
        status=config.get("status"),
    )

    if not trades:
        logger.warning("No trades found matching criteria")
        return pd.DataFrame()

    df = pd.DataFrame([t.to_dict() for t in trades])
    logger.info("Fetched %d trades", len(df))

    if config.get("min_pnl") is not None and "pnl" in df.columns:
        df = df[df["pnl"] >= config["min_pnl"]]

    if config.get("max_pnl") is not None and "pnl" in df.columns:
        df = df[df["pnl"] <= config["max_pnl"]]

    logger.info("After PnL filtering: %d trades", len(df))

    # Normalize timestamp column name
    if "timestamp" not in df.columns:
        for alt in ("exit_timestamp", "close_timestamp", "closed_at"):
            if alt in df.columns:
                df.rename(columns={alt: "timestamp"}, inplace=True)
                break

    return df


# === EXPORT HANDLER ===
class ExportHandler:
    """Handle different export formats."""

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger

    def export_csv(self, df: pd.DataFrame, output_path: Path) -> Path:
        """Export to CSV format."""
        self.logger.info("Exporting to CSV: %s", output_path)
        df.to_csv(output_path, index=False, encoding="utf-8")

        if self.config.get("compress"):
            import gzip

            compressed_path = Path(str(output_path) + ".gz")
            with open(output_path, "rb") as f_in, gzip.open(
                compressed_path, "wb"
            ) as f_out:
                f_out.writelines(f_in)
            output_path.unlink()
            output_path = compressed_path

        return output_path

    def export_excel(self, df: pd.DataFrame, output_path: Path) -> Path:
        """Export to Excel format with summary sheet."""
        self.logger.info("Exporting to Excel: %s", output_path)

        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Trades", index=False)

            summary = self._create_summary(df)
            summary.to_excel(writer, sheet_name="Summary", index=False)

        return output_path

    def export_json(self, df: pd.DataFrame, output_path: Path) -> Path:
        """Export to JSON format with metadata."""
        self.logger.info("Exporting to JSON: %s", output_path)

        if not df.empty and "timestamp" in df.columns:
            ts_min = pd.to_datetime(df["timestamp"]).min().isoformat()
            ts_max = pd.to_datetime(df["timestamp"]).max().isoformat()
        else:
            ts_min = ts_max = None

        data = {
            "metadata": {
                "exported_at": datetime.now().isoformat(),
                "total_trades": int(len(df)),
                "date_range": {
                    "start": ts_min,
                    "end": ts_max,
                },
                "filters": self._get_active_filters(),
            },
            "trades": df.to_dict(orient="records"),
            "summary": self._create_summary(df).to_dict(orient="records")
            if not df.empty
            else [],
        }

        output_path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        return output_path

    def export_parquet(self, df: pd.DataFrame, output_path: Path) -> Path:
        """Export to Parquet format."""
        self.logger.info("Exporting to Parquet: %s", output_path)
        df.to_parquet(output_path, engine="pyarrow", compression="snappy")
        return output_path

    def export_sql(self, df: pd.DataFrame, output_path: Path) -> Path:
        """Export as SQL INSERT statements."""
        self.logger.info("Exporting to SQL: %s", output_path)

        with output_path.open("w", encoding="utf-8") as f:
            f.write("-- ALPHA-PRIME Trade Export\n")
            f.write(f"-- Generated: {datetime.now().isoformat()}\n\n")

            for _, row in df.iterrows():
                cols = ", ".join(df.columns)
                vals = []
                for v in row.values:
                    if isinstance(v, (datetime, pd.Timestamp)):
                        vals.append(f"'{v.isoformat()}'")
                    elif isinstance(v, str):
                        vals.append("'" + v.replace("'", "''") + "'")
                    elif isinstance(v, (float, int, np.number, Decimal)):
                        vals.append(str(v))
                    elif pd.isna(v):
                        vals.append("NULL")
                    else:
                        vals.append("'" + str(v).replace("'", "''") + "'")
                f.write(
                    f"INSERT INTO trades ({cols}) VALUES ({', '.join(vals)});\n"
                )

        return output_path

    def _create_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create summary statistics."""
        if df.empty or "pnl" not in df.columns:
            return pd.DataFrame([{"Total Trades": len(df)}])

        pnl = df["pnl"]
        wins = pnl[pnl > 0]
        losses = pnl[pnl < 0]
        win_rate = float(len(wins) / len(pnl) * 100) if len(pnl) else 0.0

        summary = {
            "Total Trades": int(len(df)),
            "Winning Trades": int(len(wins)),
            "Losing Trades": int(len(losses)),
            "Win Rate (%)": win_rate,
            "Total PnL": float(pnl.sum()),
            "Average PnL": float(pnl.mean()),
            "Max Win": float(pnl.max()),
            "Max Loss": float(pnl.min()),
            "Total Commission": float(df["commission"].sum())
            if "commission" in df.columns
            else 0.0,
        }

        return pd.DataFrame([summary])

    def _get_active_filters(self) -> Dict[str, Any]:
        """Return non-null filters for metadata."""
        keys = [
            "strategy",
            "symbol",
            "side",
            "status",
            "min_pnl",
            "max_pnl",
            "year",
            "month",
            "quarter",
        ]
        return {k: self.config.get(k) for k in keys if self.config.get(k) is not None}


# === TAX REPORTING ===
class TaxReportGenerator:
    """Generate tax-compliant trade reports."""

    def __init__(self, method: str = "fifo"):
        self.method = method
        self.tax_calculator = TaxCalculator()

    async def generate_tax_report(
        self, df: pd.DataFrame, year: int, logger: logging.Logger
    ) -> pd.DataFrame:
        """Generate tax report with lot matching and wash sales."""
        logger.info("Generating tax report for %s using %s method...", year, self.method)

        if df.empty:
            return df

        if "status" in df.columns:
            closed_trades = df[df["status"] == "CLOSED"].copy()
        else:
            closed_trades = df.copy()

        matched_trades = await self._match_lots(closed_trades)
        wash_sales = await self._calculate_wash_sales(matched_trades)

        matched_trades["holding_days"] = matched_trades.get("holding_days", 0)
        matched_trades["short_term"] = matched_trades["holding_days"] <= 365
        matched_trades["long_term"] = matched_trades["holding_days"] > 365

        if not wash_sales.empty:
            adj_map = wash_sales.set_index("trade_id")["adjustment"]
            matched_trades["wash_sale_adjustment"] = matched_trades["trade_id"].map(
                adj_map
            ).fillna(0.0)
        else:
            matched_trades["wash_sale_adjustment"] = 0.0

        matched_trades["adjusted_pnl"] = (
            matched_trades["pnl"] - matched_trades["wash_sale_adjustment"]
        )

        ts_col = "exit_timestamp" if "exit_timestamp" in matched_trades.columns else "timestamp"
        matched_trades = matched_trades.sort_values(ts_col)

        logger.info("Tax report trades: %d", len(matched_trades))
        logger.info("  Short-term: %d", matched_trades["short_term"].sum())
        logger.info("  Long-term: %d", matched_trades["long_term"].sum())
        logger.info("  Wash sales: %d", len(wash_sales))

        return matched_trades

    async def _match_lots(self, df: pd.DataFrame) -> pd.DataFrame:
        """Match lots via configured tax method."""
        if self.method == "fifo":
            return self.tax_calculator.match_fifo(df)
        if self.method == "lifo":
            return self.tax_calculator.match_lifo(df)
        if self.method == "specific_lot":
            return self.tax_calculator.match_specific_lot(df)
        return self.tax_calculator.match_average_cost(df)

    async def _calculate_wash_sales(self, df: pd.DataFrame) -> pd.DataFrame:
        """Identify potential wash sales (simplified)."""
        records: List[Dict[str, Any]] = []
        if df.empty:
            return pd.DataFrame(records)

        if "exit_timestamp" in df.columns:
            ts_col = "exit_timestamp"
        elif "timestamp" in df.columns:
            ts_col = "timestamp"
        else:
            return pd.DataFrame(records)

        for symbol in df["symbol"].unique():
            s_df = df[df["symbol"] == symbol].copy()
            s_df[ts_col] = pd.to_datetime(s_df[ts_col])
            s_df = s_df.sort_values(ts_col)

            for _, trade in s_df.iterrows():
                pnl = trade.get("pnl", 0.0)
                if pnl >= 0:
                    continue
                exit_ts = trade[ts_col]
                window_start = exit_ts - timedelta(days=30)
                window_end = exit_ts + timedelta(days=30)

                repurchases = s_df[
                    (s_df.get("entry_timestamp", s_df[ts_col]) >= window_start)
                    & (s_df.get("entry_timestamp", s_df[ts_col]) <= window_end)
                    & (s_df.get("side") == "LONG")
                ]
                if not repurchases.empty:
                    records.append(
                        {
                            "trade_id": trade["trade_id"],
                            "symbol": symbol,
                            "loss": pnl,
                            "adjustment": float(abs(pnl)),
                        }
                    )

        return pd.DataFrame(records)


# === BROKER RECONCILIATION ===
def match_trades(
    internal_df: pd.DataFrame, broker_df: pd.DataFrame
) -> Tuple[pd.DataFrame, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Basic matching by symbol, side, quantity, and close timestamp (rounded)."""
    if internal_df.empty or broker_df.empty:
        return pd.DataFrame(), [], []

    df_int = internal_df.copy()
    df_brk = broker_df.copy()

    for col in ("timestamp", "exit_timestamp", "trade_time"):
        if col in df_int.columns:
            df_int["match_ts"] = pd.to_datetime(df_int[col]).dt.floor("min")
            break
    for col in ("timestamp", "trade_time", "exec_time"):
        if col in df_brk.columns:
            df_brk["match_ts"] = pd.to_datetime(df_brk[col]).dt.floor("min")
            break

    key_cols = ["symbol", "side", "quantity", "match_ts"]
    for col in key_cols:
        if col not in df_int.columns:
            df_int[col] = None
        if col not in df_brk.columns:
            df_brk[col] = None

    merged = df_int.merge(
        df_brk,
        on=key_cols,
        how="outer",
        suffixes=("_int", "_brk"),
        indicator=True,
    )

    matched = merged[merged["_merge"] == "both"]
    unmatched_internal = merged[merged["_merge"] == "left_only"]
    unmatched_broker = merged[merged["_merge"] == "right_only"]

    unmatched_internal_list = unmatched_internal.to_dict(orient="records")
    unmatched_broker_list = unmatched_broker.to_dict(orient="records")

    matched_ids = matched["trade_id_int"].dropna().tolist()
    matched_df = internal_df[internal_df["trade_id"].isin(matched_ids)].copy()

    return matched_df, unmatched_internal_list, unmatched_broker_list


async def reconcile_with_broker(
    df: pd.DataFrame, broker_name: str, config: Dict[str, Any], logger: logging.Logger
) -> pd.DataFrame:
    """Reconcile internal trades with broker statements."""
    if not broker_name:
        raise ValueError("--broker is required when using --reconcile")

    logger.info("Reconciling with broker: %s", broker_name)

    broker_client = get_broker_client(broker_name)
    start_date, end_date = get_date_range(config)

    broker_trades = await broker_client.get_trade_history(
        start_date=start_date, end_date=end_date
    )
    broker_df = pd.DataFrame(broker_trades)

    matched, unmatched_internal, unmatched_broker = match_trades(df, broker_df)

    logger.info("Reconciliation Summary:")
    logger.info("  Matched trades:        %d", len(matched))
    logger.info("  Unmatched internal:    %d", len(unmatched_internal))
    logger.info("  Unmatched broker:      %d", len(unmatched_broker))

    if unmatched_internal:
        logger.warning("Internal trades not found in broker:")
        for t in unmatched_internal[:10]:
            logger.warning(
                "  %s: %s @ %s",
                t.get("trade_id_int"),
                t.get("symbol"),
                t.get("timestamp_int"),
            )

    if unmatched_broker:
        logger.warning("Broker trades not found internally:")
        for t in unmatched_broker[:10]:
            logger.warning(
                "  %s: %s @ %s",
                t.get("broker_trade_id"),
                t.get("symbol"),
                t.get("timestamp"),
            )

    df = df.copy()
    df["reconciled"] = df["trade_id"].isin(matched["trade_id"])
    return df


# === PERFORMANCE ATTRIBUTION ===
def calculate_attribution(df: pd.DataFrame, by: str, logger: logging.Logger) -> pd.DataFrame:
    """Calculate performance attribution by dimension."""
    if df.empty:
        logger.warning("No trades for attribution analysis")
        return df

    logger.info("Calculating performance attribution by %s...", by)

    work = df.copy()
    if by == "strategy":
        grouped = work.groupby("strategy_id")
    elif by == "symbol":
        grouped = work.groupby("symbol")
    elif by == "month":
        work["month"] = pd.to_datetime(work["timestamp"]).dt.to_period("M")
        grouped = work.groupby("month")
    elif by == "day":
        work["day"] = pd.to_datetime(work["timestamp"]).dt.date
        grouped = work.groupby("day")
    else:
        raise ValueError(f"Unknown attribution dimension: {by}")

    agg = grouped.agg(
        {
            "trade_id": "count",
            "pnl": ["sum", "mean", "std"],
            "quantity": "sum",
            "commission": "sum" if "commission" in work.columns else "sum",
        }
    )
    agg.columns = ["_".join([c for c in col if c]).strip("_") for col in agg.columns]
    agg = agg.reset_index()

    total_pnl = agg["pnl_sum"].sum()
    agg["contribution_pct"] = (
        agg["pnl_sum"] / total_pnl * 100 if total_pnl != 0 else 0.0
    )

    agg = agg.sort_values("pnl_sum", ascending=False)

    logger.info("Top 5 contributors by %s:", by)
    display_col = by if by in agg.columns else agg.columns[0]
    for _, row in agg.head(5).iterrows():
        logger.info(
            "  %s: %.2f (%.1f%%)",
            row[display_col],
            row["pnl_sum"],
            row["contribution_pct"],
        )

    return agg


# === DATA VALIDATION ===
def validate_export_data(df: pd.DataFrame, logger: logging.Logger) -> bool:
    """Validate data integrity before export."""
    logger.info("Validating export data...")

    issues: List[str] = []
    required_cols = ["trade_id", "symbol", "timestamp", "pnl"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        issues.append(f"Missing columns: {missing_cols}")

    for col in required_cols:
        if col in df.columns:
            null_count = int(df[col].isnull().sum())
            if null_count > 0:
                issues.append(f"Null values in {col}: {null_count} rows")

    if "trade_id" in df.columns and df["trade_id"].duplicated().any():
        dup_count = int(df["trade_id"].duplicated().sum())
        issues.append(f"Duplicate trade IDs: {dup_count}")

    if "timestamp" in df.columns:
        try:
            pd.to_datetime(df["timestamp"])
        except Exception as exc:
            issues.append(f"Invalid timestamps: {exc}")

    if all(c in df.columns for c in ("entry_price", "exit_price", "quantity", "pnl")):
        try:
            calc_pnl = (df["exit_price"] - df["entry_price"]) * df["quantity"]
            if not np.allclose(calc_pnl.fillna(0), df["pnl"].fillna(0), rtol=0.01, atol=0.01):
                issues.append("PnL calculation mismatch detected (entry/exit/quantity vs pnl)")
        except Exception as exc:
            issues.append(f"Error checking PnL consistency: {exc}")

    if issues:
        logger.warning("Validation issues found:")
        for issue in issues:
            logger.warning("  - %s", issue)
        return False

    logger.info("Data validation passed ✓")
    return True


# === CHECKPOINT HANDLING ===
def update_checkpoint(checkpoint_file: Path, last_timestamp: Any) -> None:
    """Update checkpoint file with latest exported timestamp."""
    if last_timestamp is None:
        return
    ts = pd.to_datetime(last_timestamp).isoformat()
    checkpoint_file.write_text(ts, encoding="utf-8")


# === MAIN EXECUTION ===
async def main() -> int:
    args = parse_arguments()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logger("trade_export", level=log_level)

    try:
        config: Dict[str, Any] = vars(args)

        df = await fetch_trades(config, logger)
        if df.empty:
            logger.warning("No trades to export")
            return 0

        # Apply granularity
        if args.summary:
            cols = [c for c in SUMMARY_COLUMNS if c in df.columns]
            df = df[cols]
        elif args.detailed:
            cols = [c for c in DETAILED_COLUMNS if c in df.columns]
            df = df[cols]
        elif args.full:
            cols = [c for c in FULL_COLUMNS if c in df.columns]
            df = df[cols]

        # Tax report
        if args.tax_report:
            if not args.year:
                raise ValueError("--tax-report requires --year")
            tax_gen = TaxReportGenerator(method=args.method)
            df = await tax_gen.generate_tax_report(df, args.year, logger)

        # Reconciliation
        if args.reconcile:
            df = await reconcile_with_broker(df, args.broker, config, logger)

        # Attribution
        if args.attribution:
            if not args.by:
                raise ValueError("--attribution requires --by")
            df = calculate_attribution(df, args.by, logger)

        # Validation
        if args.validate:
            if not validate_export_data(df, logger):
                logger.error("Validation failed. Use --dry-run to inspect output.")
                return 1

        if args.dry_run:
            logger.info("Dry run mode - previewing first 10 rows:")
            print(df.head(10).to_string())
            logger.info("Total rows: %d", len(df))
            return 0

        EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

        if not args.output:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            ext = "csv" if args.format == "csv" else args.format
            filename = f"trades_{ts}.{ext}"
            if args.format == "excel":
                filename = f"trades_{ts}.xlsx"
            output_path = EXPORTS_DIR / filename
        else:
            output_path = args.output

        handler = ExportHandler(config, logger)

        # Optional splitting
        if args.split and len(df) > args.split:
            logger.info(
                "Splitting export into chunks of %d rows (total %d rows)",
                args.split,
                len(df),
            )
            base = output_path
            paths: List[Path] = []
            for idx, start in enumerate(range(0, len(df), args.split), start=1):
                chunk = df.iloc[start : start + args.split]
                if base.suffix:
                    chunk_path = base.with_name(
                        f"{base.stem}_part{idx}{base.suffix}"
                    )
                else:
                    chunk_path = base.with_name(f"{base.name}_part{idx}")
                paths.append(_export_single(handler, args.format, chunk, chunk_path))
            logger.info("Exported %d chunk(s)", len(paths))
            last_ts = df["timestamp"].max() if "timestamp" in df.columns else None
            if args.since_last_export and last_ts is not None:
                update_checkpoint(args.checkpoint_file, last_ts)
            return 0

        output_path = _export_single(handler, args.format, df, output_path)

        if args.since_last_export and "timestamp" in df.columns:
            update_checkpoint(args.checkpoint_file, df["timestamp"].max())

        size_kb = output_path.stat().st_size / 1024.0
        logger.info("Export completed: %s", output_path)
        logger.info("  Rows exported: %d", len(df))
        logger.info("  File size: %.2f KB", size_kb)

        return 0

    except Exception as exc:
        logger.error("Export failed: %s", exc, exc_info=True)
        return 1


def _export_single(
    handler: ExportHandler, fmt: str, df: pd.DataFrame, path: Path
) -> Path:
    """Dispatch to specific export method."""
    if fmt == "csv":
        return handler.export_csv(df, path)
    if fmt == "excel":
        return handler.export_excel(df, path)
    if fmt == "json":
        return handler.export_json(df, path)
    if fmt == "parquet":
        return handler.export_parquet(df, path)
    if fmt == "sql":
        return handler.export_sql(df, path)
    raise ValueError(f"Unsupported format: {fmt}")


if __name__ == "__main__":
    import asyncio

    sys.exit(asyncio.run(main()))
