#!/usr/bin/env python3
"""
ALPHA-PRIME v2.0 - Backtest Execution Script
============================================
Run strategy backtests with comprehensive analysis and reporting.

Usage:
    python scripts/run_backtest.py --strategy EMA_Cross_v2 --start 2024-01-01 --end 2025-12-31
    python scripts/run_backtest.py --config backtest_config.yaml
    python scripts/run_backtest.py --optimize --param-grid params.json
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import json

import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.backtest_engine import BacktestEngine
from core.strategy_registry import StrategyRegistry
from core.data_manager import DataManager
from core.performance_analyzer import PerformanceAnalyzer
from core.report_generator import ReportGenerator
from core.optimizer import GridSearchOptimizer, WalkForwardOptimizer
from core.strategies.base import Strategy
from utils.logger import setup_logger

# === CONFIGURATION ===
DEFAULT_INITIAL_CAPITAL = 100_000.0
DEFAULT_COMMISSION = 0.001  # 0.1%
DEFAULT_SLIPPAGE = 0.0005   # 0.05%


# === 1. ARGUMENT PARSING ===
def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="ALPHA-PRIME v2.0 Backtest Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic backtest
  python scripts/run_backtest.py --strategy EMA_Cross_v2 --start 2024-01-01 --end 2025-12-31

  # With custom parameters
  python scripts/run_backtest.py --strategy EMA_Cross_v2 --capital 50000 --commission 0.002

  # Multiple symbols
  python scripts/run_backtest.py --strategy EMA_Cross_v2 --symbols AAPL,MSFT,GOOGL

  # Walk-forward optimization
  python scripts/run_backtest.py --strategy EMA_Cross_v2 --optimize --walk-forward

  # Load from config file
  python scripts/run_backtest.py --config backtest_config.yaml
        """
    )

    # Strategy selection
    parser.add_argument(
        "--strategy",
        type=str,
        required=False,
        help="Strategy ID from registry"
    )
    parser.add_argument(
        "--strategy-file",
        type=Path,
        help="Path to custom strategy Python file"
    )

    # Date range
    parser.add_argument(
        "--start",
        type=str,
        help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end",
        type=str,
        help="End date (YYYY-MM-DD, default: today)"
    )

    # Symbols
    parser.add_argument(
        "--symbols",
        type=str,
        help="Comma-separated list of symbols (e.g., AAPL,MSFT)"
    )

    # Capital & costs
    parser.add_argument(
        "--capital",
        type=float,
        default=DEFAULT_INITIAL_CAPITAL,
        help=f"Initial capital (default: {DEFAULT_INITIAL_CAPITAL})"
    )
    parser.add_argument(
        "--commission",
        type=float,
        default=DEFAULT_COMMISSION,
        help=f"Commission rate (default: {DEFAULT_COMMISSION})"
    )
    parser.add_argument(
        "--slippage",
        type=float,
        default=DEFAULT_SLIPPAGE,
        help=f"Slippage rate (default: {DEFAULT_SLIPPAGE})"
    )

    # Optimization
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Run parameter optimization"
    )
    parser.add_argument(
        "--param-grid",
        type=Path,
        help="Parameter grid JSON file for optimization"
    )
    parser.add_argument(
        "--walk-forward",
        action="store_true",
        help="Use walk-forward analysis"
    )

    # Output options
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("backtest_results"),
        help="Output directory for results"
    )
    parser.add_argument(
        "--format",
        choices=["html", "pdf", "json", "csv", "all"],
        default="html",
        help="Report format"
    )
    parser.add_argument(
        "--no-charts",
        action="store_true",
        help="Skip chart generation"
    )

    # Configuration file
    parser.add_argument(
        "--config",
        type=Path,
        help="YAML/JSON config file (overrides other args)"
    )

    # Logging
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose logging"
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Minimal output"
    )

    return parser.parse_args()


# === 2. CONFIGURATION LOADING ===
def load_configuration(args: argparse.Namespace) -> Dict[str, Any]:
    """Load and merge configuration from file and CLI args."""
    config: Dict[str, Any] = {}

    # Load from file if provided
    if args.config:
        if not args.config.exists():
            raise FileNotFoundError(f"Config file not found: {args.config}")
        with open(args.config, "r", encoding="utf-8") as f:
            if args.config.suffix.lower() in {".yaml", ".yml"}:
                import yaml

                config = yaml.safe_load(f) or {}
            else:
                config = json.load(f)

    # CLI args override config file
    config.update(
        {
            "strategy_id": args.strategy or config.get("strategy_id"),
            "strategy_file": args.strategy_file or config.get("strategy_file"),
            "start_date": args.start or config.get("start_date"),
            "end_date": args.end
            or config.get("end_date", datetime.now().strftime("%Y-%m-%d")),
            "symbols": (
                args.symbols.split(",")
                if args.symbols
                else config.get("symbols", ["AAPL"])
            ),
            "initial_capital": args.capital,
            "commission": args.commission,
            "slippage": args.slippage,
            "optimize": args.optimize or config.get("optimize", False),
            "param_grid": args.param_grid or config.get("param_grid"),
            "walk_forward": args.walk_forward or config.get("walk_forward", False),
            "output_dir": args.output_dir,
            "report_format": args.format,
            "generate_charts": not args.no_charts,
            "verbose": args.verbose,
            "quiet": args.quiet,
        }
    )

    validate_configuration(config)

    return config


def validate_configuration(config: Dict[str, Any]) -> None:
    """Validate backtest configuration."""
    if not config.get("strategy_id") and not config.get("strategy_file"):
        raise ValueError("Either --strategy or --strategy-file must be provided")

    if not config.get("start_date"):
        raise ValueError("--start date is required")

    # Validate date formats
    try:
        datetime.strptime(config["start_date"], "%Y-%m-%d")
        datetime.strptime(config["end_date"], "%Y-%m-%d")
    except Exception as exc:
        raise ValueError("Invalid date format, expected YYYY-MM-DD") from exc

    if config["initial_capital"] <= 0:
        raise ValueError("Initial capital must be positive")

    if not (0 <= config["commission"] < 1):
        raise ValueError("Commission must be between 0 and 1")

    if not (0 <= config["slippage"] < 1):
        raise ValueError("Slippage must be between 0 and 1")

    if config["optimize"] and config["strategy_file"]:
        # Optimization is defined for registered strategies (by ID)
        raise ValueError(
            "Optimization is only supported for registered strategies (--strategy), "
            "not for custom strategy files"
        )


# === 3. DATA PREPARATION ===
async def prepare_data(config: Dict[str, Any], logger: logging.Logger) -> pd.DataFrame:
    """Fetch and prepare historical data for backtest."""
    logger.info("Fetching historical data...")

    data_manager = DataManager()

    all_data: List[pd.DataFrame] = []
    for symbol in config["symbols"]:
        logger.info("  Loading %s...", symbol)

        data = await data_manager.get_historical_data(
            symbol=symbol,
            start_date=config["start_date"],
            end_date=config["end_date"],
            interval="1d",
        )

        if data is None or data.empty:
            logger.warning("  No data for %s, skipping", symbol)
            continue

        if "symbol" not in data.columns:
            data["symbol"] = symbol

        all_data.append(data)

    if not all_data:
        raise ValueError("No data available for backtest (all symbols empty)")

    combined_data = pd.concat(all_data, ignore_index=True)

    if "timestamp" in combined_data.columns:
        combined_data["timestamp"] = pd.to_datetime(combined_data["timestamp"])

    logger.info(
        "Loaded %d bars across %d symbols",
        len(combined_data),
        len({s for s in combined_data["symbol"]}),
    )
    if "timestamp" in combined_data.columns:
        logger.info(
            "Date range: %s to %s",
            combined_data["timestamp"].min(),
            combined_data["timestamp"].max(),
        )

    return combined_data


# === 4. BACKTEST EXECUTION ===
async def run_backtest(
    config: Dict[str, Any],
    data: pd.DataFrame,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """Execute backtest with configured strategy."""
    logger.info("Initializing backtest engine...")

    # Load strategy
    if config.get("strategy_id"):
        strategy_registry = StrategyRegistry()
        strategy_cls = strategy_registry.get_strategy(config["strategy_id"])
        strategy = strategy_cls(config) if callable(strategy_cls) else strategy_cls
    else:
        strategy = load_strategy_from_file(config["strategy_file"])

    engine = BacktestEngine(
        strategy=strategy,
        initial_capital=config["initial_capital"],
        commission=config["commission"],
        slippage=config["slippage"],
    )

    logger.info("Running backtest for strategy: %s", getattr(strategy, "name", strategy))
    logger.info("  Initial capital: %.2f", config["initial_capital"])
    logger.info("  Commission: %.4f", config["commission"])
    logger.info("  Slippage: %.4f", config["slippage"])

    start_time = datetime.now()

    def progress_cb(pct: float) -> None:
        if pct <= 0:
            return
        # Log every 10%
        if int(pct) % 10 == 0:
            logger.info("  Progress: %.1f%%", pct)

    results = await engine.run(data, progress_callback=progress_cb)

    duration = (datetime.now() - start_time).total_seconds()

    logger.info("✓ Backtest complete in %.2fs", duration)
    logger.info("  Total trades: %s", len(results.trades))
    logger.info("  Final equity: %.2f", results.final_equity)
    logger.info("  Total return: %.2f%%", results.total_return)

    return {
        "strategy": strategy,
        "results": results,
        "duration": duration,
        "config": config,
    }


# === 5. PERFORMANCE ANALYSIS ===
def analyze_performance(
    backtest_results: Dict[str, Any],
    logger: logging.Logger,
) -> Dict[str, Any]:
    """Analyze backtest performance and calculate metrics."""
    logger.info("Analyzing performance...")

    results = backtest_results["results"]
    analyzer = PerformanceAnalyzer()

    metrics = analyzer.calculate_all_metrics(
        equity_curve=results.equity_curve,
        trades=results.trades,
        benchmark=getattr(results, "benchmark", None),
        regimes=getattr(results, "regimes", None),
    )

    logger.info("\n" + "=" * 60)
    logger.info("PERFORMANCE SUMMARY")
    logger.info("=" * 60)
    logger.info("Total Return:        %10.2f%%", metrics.get("total_return", 0.0))
    logger.info("Sharpe Ratio:        %10.2f", metrics.get("sharpe_ratio", 0.0))
    logger.info("Sortino Ratio:       %10.2f", metrics.get("sortino_ratio", 0.0))
    logger.info("Max Drawdown:        %10.2f%%", metrics.get("max_drawdown_pct", 0.0))
    logger.info("Win Rate:            %10.2f%%", metrics.get("win_rate", 0.0))
    logger.info("Profit Factor:       %10.2f", metrics.get("profit_factor", 0.0))
    logger.info("Total Trades:        %10d", metrics.get("total_trades", 0))
    logger.info("Avg Trade:           %10.2f", metrics.get("avg_trade_pnl", 0.0))
    logger.info("=" * 60 + "\n")

    # Optional regime analysis logging
    regime_stats = metrics.get("regime_stats")
    if regime_stats:
        logger.info("PERFORMANCE BY REGIME")
        for regime, rs in regime_stats.items():
            logger.info(
                "  %-10s  Return: %7.2f%%  Sharpe: %5.2f  Trades: %4d",
                regime,
                rs.get("total_return", 0.0),
                rs.get("sharpe_ratio", 0.0),
                rs.get("total_trades", 0),
            )
        logger.info("")

    return metrics


# === 6. REPORT GENERATION & EXPORT ===
async def generate_reports(
    backtest_results: Dict[str, Any],
    metrics: Dict[str, Any],
    config: Dict[str, Any],
    logger: logging.Logger,
) -> None:
    """Generate backtest reports in specified format(s)."""
    logger.info("Generating reports...")

    output_dir: Path = config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    strategy_name = getattr(backtest_results["strategy"], "name", "strategy").replace(
        " ", "_"
    )
    base_filename = f"{strategy_name}_{timestamp}"

    report_generator = ReportGenerator(
        strategy=backtest_results["strategy"],
        results=backtest_results["results"],
        metrics=metrics,
        config=config,
    )

    if config["report_format"] == "all":
        formats: List[str] = ["html", "pdf", "json", "csv"]
    else:
        formats = [config["report_format"]]

    for fmt in formats:
        if fmt == "html":
            filepath = output_dir / f"{base_filename}.html"
            await report_generator.generate_html_report(
                filepath, include_charts=config["generate_charts"]
            )
            logger.info("  ✓ HTML report: %s", filepath)

        elif fmt == "pdf":
            filepath = output_dir / f"{base_filename}.pdf"
            await report_generator.generate_pdf_report(filepath)
            logger.info("  ✓ PDF report: %s", filepath)

        elif fmt == "json":
            filepath = output_dir / f"{base_filename}.json"
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "strategy": serialize_strategy(backtest_results["strategy"]),
                        "metrics": metrics,
                        "config": serialize_config(config),
                        "trades": [t.to_dict() for t in backtest_results["results"].trades],
                    },
                    f,
                    indent=2,
                    default=str,
                )
            logger.info("  ✓ JSON report: %s", filepath)

        elif fmt == "csv":
            trades_path = output_dir / f"{base_filename}_trades.csv"
            equity_path = output_dir / f"{base_filename}_equity.csv"

            trades_df = pd.DataFrame(
                [t.to_dict() for t in backtest_results["results"].trades]
            )
            trades_df.to_csv(trades_path, index=False)

            equity_df = backtest_results["results"].equity_curve
            if not isinstance(equity_df, pd.DataFrame):
                equity_df = pd.DataFrame(equity_df)
            equity_df.to_csv(equity_path, index=False)

            logger.info("  ✓ CSV trades: %s", trades_path)
            logger.info("  ✓ CSV equity: %s", equity_path)

    logger.info("Reports generated successfully ✓")


def serialize_strategy(strategy: Any) -> Dict[str, Any]:
    """Serialize strategy object to a JSON-friendly dict."""
    if hasattr(strategy, "to_dict"):
        return strategy.to_dict()
    return {
        "name": getattr(strategy, "name", strategy.__class__.__name__),
        "class": strategy.__class__.__name__,
        "module": strategy.__class__.__module__,
    }


def serialize_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Convert Path and other non-serializable items in config."""
    serialized: Dict[str, Any] = {}
    for k, v in config.items():
        if isinstance(v, Path):
            serialized[k] = str(v)
        else:
            serialized[k] = v
    return serialized


# === 7. PARAMETER OPTIMIZATION ===
async def run_optimization(
    config: Dict[str, Any],
    data: pd.DataFrame,
    logger: logging.Logger,
) -> Any:
    """Run parameter optimization (grid search or walk-forward)."""
    logger.info("Starting parameter optimization...")

    # Load parameter grid
    if config["param_grid"]:
        if not Path(config["param_grid"]).exists():
            raise FileNotFoundError(f"Param grid file not found: {config['param_grid']}")
        with open(config["param_grid"], "r", encoding="utf-8") as f:
            param_grid = json.load(f)
    else:
        # Default parameter ranges for EMA + RSI style strategies
        param_grid = {
            "ema_fast": [5, 10, 20],
            "ema_slow": [50, 100, 200],
            "rsi_period": [14, 21],
            "rsi_oversold": [25, 30],
            "rsi_overbought": [70, 75],
        }

    if config["walk_forward"]:
        optimizer = WalkForwardOptimizer(
            param_grid=param_grid,
            train_window=252,
            test_window=63,
        )
    else:
        optimizer = GridSearchOptimizer(param_grid=param_grid)

    logger.info("  Parameter combinations: %s", optimizer.total_combinations)

    # Strategy must be registry-based for optimizer (strategy_id)
    if not config.get("strategy_id"):
        raise ValueError(
            "Optimization requires a registered strategy (--strategy), "
            "not a custom strategy file"
        )

    optimization_results = await optimizer.optimize(
        strategy_id=config["strategy_id"],
        data=data,
        initial_capital=config["initial_capital"],
        commission=config["commission"],
        slippage=config["slippage"],
        metric="sharpe_ratio",
        progress_callback=lambda pct, current, total: logger.info(
            "  Optimization progress: %d/%d (%.1f%%)", current, total, pct
        ),
    )

    logger.info("\n" + "=" * 60)
    logger.info("OPTIMIZATION RESULTS")
    logger.info("=" * 60)
    logger.info("Best Sharpe Ratio:   %.2f", optimization_results.best_score)
    logger.info("Best Parameters:")
    for param, value in optimization_results.best_params.items():
        logger.info("  %s: %s", param, value)
    logger.info("=" * 60 + "\n")

    return optimization_results


# === HELPER FUNCTIONS ===
def load_strategy_from_file(filepath: Path):
    """Dynamically load strategy from Python file."""
    import importlib.util

    if filepath is None:
        raise ValueError("strategy_file path is required")

    filepath = filepath.expanduser().resolve()
    if not filepath.exists():
        raise FileNotFoundError(f"Strategy file not found: {filepath}")

    spec = importlib.util.spec_from_file_location("custom_strategy_module", filepath)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load strategy module from {filepath}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]

    for name in dir(module):
        obj = getattr(module, name)
        if (
            isinstance(obj, type)
            and issubclass(obj, Strategy)
            and obj is not Strategy
        ):
            return obj()

    raise ValueError(f"No Strategy subclass found in {filepath}")


def print_progress_bar(
    iteration: int,
    total: int,
    prefix: str = "",
    length: int = 50,
) -> None:
    """Print console progress bar."""
    if total <= 0:
        return
    percent = 100 * (iteration / float(total))
    filled_length = int(length * iteration // total)
    bar = "█" * filled_length + "-" * (length - filled_length)
    print(f"\r{prefix} |{bar}| {percent:.1f}% Complete", end="\r")
    if iteration >= total:
        print("")


# === MAIN EXECUTION ===
async def main() -> int:
    """Main execution flow."""
    args = parse_arguments()

    log_level = (
        logging.DEBUG
        if args.verbose
        else logging.ERROR
        if args.quiet
        else logging.INFO
    )
    logger = setup_logger("backtest", level=log_level)

    try:
        config = load_configuration(args)

        data = await prepare_data(config, logger)

        if config["optimize"]:
            optimization_results = await run_optimization(config, data, logger)
            config.update(optimization_results.best_params)

        backtest_results = await run_backtest(config, data, logger)

        metrics = analyze_performance(backtest_results, logger)

        await generate_reports(backtest_results, metrics, config, logger)

        logger.info("✓ Backtest completed successfully")
        return 0
    except Exception as exc:
        logger.error("Backtest failed: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    import asyncio

    sys.exit(asyncio.run(main()))
