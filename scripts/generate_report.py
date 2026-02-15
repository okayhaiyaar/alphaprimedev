#!/usr/bin/env python3
"""
ALPHA-PRIME v2.0 - Report Generation Script
===========================================
Generate professional trading reports with comprehensive analytics.

Usage:
    python scripts/generate_report.py --type performance --period monthly
    python scripts/generate_report.py --type portfolio --format pdf --email
    python scripts/generate_report.py --type custom --template my_report.html
    python scripts/generate_report.py --schedule daily --recipients team@company.com
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import json

import pandas as pd
from jinja2 import Environment, FileSystemLoader
import plotly.graph_objects as go

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.portfolio import PortfolioManager
from core.performance_analyzer import PerformanceAnalyzer
from core.risk_manager import RiskManager
from integrations.email_client import EmailClient
from utils.logger import setup_logger
from database.queries import ReportQueries

# === CONFIGURATION ===
REPORTS_DIR = PROJECT_ROOT / "reports"
TEMPLATES_DIR = PROJECT_ROOT / "templates" / "reports"
CHARTS_DIR = REPORTS_DIR / "charts"

# === REPORT TYPES ===
REPORT_TYPES: Dict[str, str] = {
    "performance": "PerformanceReport",
    "portfolio": "PortfolioReport",
    "strategy": "StrategyReport",
    "risk": "RiskReport",
    "trading": "TradingActivityReport",
    "compliance": "ComplianceReport",
    "executive": "ExecutiveSummaryReport",
    "custom": "CustomReport",
}


# === ARGUMENT PARSING ===
def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="ALPHA-PRIME v2.0 Report Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Report Types:
  performance   - Performance metrics, returns, Sharpe ratio
  portfolio     - Portfolio positions, allocation, exposure
  strategy      - Strategy-specific performance analysis
  risk          - Risk metrics, VaR, drawdowns, limits
  trading       - Trade log, execution quality, slippage
  compliance    - Regulatory compliance checks
  executive     - High-level executive summary
  custom        - Custom report from template

Examples:
  # Monthly performance report
  python scripts/generate_report.py --type performance --period monthly --format pdf

  # Portfolio snapshot with email delivery
  python scripts/generate_report.py --type portfolio --format html --email --recipients team@company.com

  # Custom report from template
  python scripts/generate_report.py --type custom --template my_report.html --data custom_data.json

  # Quarterly risk report
  python scripts/generate_report.py --type risk --period quarterly --format excel
        """,
    )

    # Report type
    parser.add_argument(
        "--type",
        type=str,
        choices=list(REPORT_TYPES.keys()),
        required=True,
        help="Type of report to generate",
    )

    # Time period
    parser.add_argument(
        "--period",
        type=str,
        choices=["daily", "weekly", "monthly", "quarterly", "ytd", "custom"],
        default="monthly",
        help="Report time period",
    )
    parser.add_argument(
        "--start",
        type=str,
        help="Custom start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        type=str,
        help="Custom end date (YYYY-MM-DD)",
    )

    # Filters
    parser.add_argument(
        "--strategy",
        type=str,
        help="Filter by strategy ID",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        help="Filter by symbol",
    )
    parser.add_argument(
        "--account",
        type=str,
        help="Filter by account ID",
    )

    # Output options
    parser.add_argument(
        "--format",
        type=str,
        choices=["html", "pdf", "excel", "json", "csv"],
        default="html",
        help="Output format",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Custom output file path",
    )
    parser.add_argument(
        "--no-charts",
        action="store_true",
        help="Skip chart generation",
    )

    # Template options
    parser.add_argument(
        "--template",
        type=Path,
        help="Custom Jinja2 template file",
    )
    parser.add_argument(
        "--data",
        type=Path,
        help="Additional data JSON file for custom reports",
    )

    # Delivery options
    parser.add_argument(
        "--email",
        action="store_true",
        help="Send report via email",
    )
    parser.add_argument(
        "--recipients",
        type=str,
        help="Comma-separated email recipients",
    )
    parser.add_argument(
        "--subject",
        type=str,
        help="Email subject line",
    )

    # Scheduling
    parser.add_argument(
        "--schedule",
        type=str,
        choices=["daily", "weekly", "monthly"],
        help="Schedule recurring report (prints cron entry)",
    )

    # Logging
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose logging",
    )

    return parser.parse_args()


# === COMMON HELPERS ===
def build_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Build normalized config dict from CLI args."""
    start_date, end_date = resolve_date_range(args.period, args.start, args.end)

    recipients: List[str] = []
    if args.recipients:
        recipients = [r.strip() for r in args.recipients.split(",") if r.strip()]

    return {
        "type": args.type,
        "period": args.period,
        "start_date": start_date,
        "end_date": end_date,
        "strategy": args.strategy,
        "symbol": args.symbol,
        "account": args.account,
        "format": args.format,
        "output": args.output,
        "no_charts": args.no_charts,
        "template": args.template,
        "data_path": args.data,
        "email": args.email,
        "recipients": recipients,
        "subject": args.subject,
        "schedule": args.schedule,
    }


def resolve_date_range(
    period: str, start: Optional[str], end: Optional[str]
) -> Tuple[datetime, datetime]:
    """Resolve reporting date range based on period and optional explicit dates."""
    today = datetime.utcnow().date()

    if period == "custom":
        if not start or not end:
            raise ValueError("Custom period requires --start and --end")
        start_date = datetime.strptime(start, "%Y-%m-%d")
        end_date = datetime.strptime(end, "%Y-%m-%d")
        return start_date, end_date

    if period == "daily":
        start_date = datetime.combine(today, datetime.min.time())
        end_date = datetime.combine(today, datetime.max.time())
    elif period == "weekly":
        # ISO weekday: Monday=1
        monday = today - timedelta(days=today.isoweekday() - 1)
        start_date = datetime.combine(monday, datetime.min.time())
        end_date = datetime.combine(monday + timedelta(days=6), datetime.max.time())
    elif period == "monthly":
        first = today.replace(day=1)
        if first.month == 12:
            next_month = first.replace(year=first.year + 1, month=1)
        else:
            next_month = first.replace(month=first.month + 1)
        start_date = datetime.combine(first, datetime.min.time())
        end_date = datetime.combine(next_month - timedelta(days=1), datetime.max.time())
    elif period == "quarterly":
        quarter = (today.month - 1) // 3 + 1
        first_month = 3 * (quarter - 1) + 1
        first = today.replace(month=first_month, day=1)
        if quarter == 4:
            next_q = first.replace(year=first.year + 1, month=1)
        else:
            next_q = first.replace(month=first.month + 3)
        start_date = datetime.combine(first, datetime.min.time())
        end_date = datetime.combine(next_q - timedelta(days=1), datetime.max.time())
    elif period == "ytd":
        first = today.replace(month=1, day=1)
        start_date = datetime.combine(first, datetime.min.time())
        end_date = datetime.combine(today, datetime.max.time())
    else:
        raise ValueError(f"Unsupported period: {period}")

    return start_date, end_date


def ensure_dirs() -> None:
    """Ensure reports and charts directories exist."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)


# === PERFORMANCE REPORT ===
class PerformanceReport:
    """Generate performance analysis report."""

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.analyzer = PerformanceAnalyzer()
        self.queries = ReportQueries()

    async def generate(self) -> Dict[str, Any]:
        """Generate performance report data."""
        self.logger.info("Generating performance report...")

        start_date: datetime = self.config["start_date"]
        end_date: datetime = self.config["end_date"]

        equity_curve = await self.queries.get_equity_curve(
            start_date, end_date, strategy_id=self.config.get("strategy")
        )
        trades = await self.queries.get_trades(
            start_date,
            end_date,
            strategy_id=self.config.get("strategy"),
            symbol=self.config.get("symbol"),
            account_id=self.config.get("account"),
        )

        metrics = self.analyzer.calculate_all_metrics(equity_curve, trades)
        period_returns = self._calculate_period_returns(equity_curve)

        charts: Dict[str, str] = {}
        if not self.config.get("no_charts"):
            charts["equity_curve"] = self._create_equity_chart(equity_curve)
            charts["drawdown"] = self._create_drawdown_chart(equity_curve)
            charts["monthly_returns"] = self._create_returns_heatmap(period_returns)
            charts["win_loss_distribution"] = self._create_pnl_distribution(trades)

        return {
            "report_type": "performance",
            "title": f"Performance Report - {self.config['period'].title()}",
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
                "name": self.config["period"],
            },
            "filters": {
                "strategy": self.config.get("strategy"),
                "symbol": self.config.get("symbol"),
                "account": self.config.get("account"),
            },
            "metrics": metrics,
            "period_returns": period_returns,
            "trades_summary": self._summarize_trades(trades),
            "charts": charts,
            "generated_at": datetime.utcnow().isoformat(),
        }

    def _calculate_period_returns(self, equity_curve: pd.DataFrame) -> pd.DataFrame:
        """Calculate monthly returns table from equity curve."""
        if equity_curve.empty:
            return pd.DataFrame()

        df = equity_curve.copy()
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        elif "timestamp" in df.columns:
            df["date"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("date").sort_index()

        df["returns"] = df["equity"].pct_change()
        monthly = df["returns"].resample("M").apply(lambda x: (1 + x).prod() - 1)
        monthly_df = monthly.to_frame(name="return")
        monthly_df["year"] = monthly_df.index.year
        monthly_df["month"] = monthly_df.index.month
        return monthly_df.reset_index(drop=True)

    def _create_equity_chart(self, equity_curve: pd.DataFrame) -> str:
        """Create equity curve chart."""
        if equity_curve.empty:
            return ""

        df = equity_curve.copy()
        x_col = "date" if "date" in df.columns else "timestamp"
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=df[x_col],
                y=df["equity"],
                mode="lines",
                name="Portfolio Equity",
                line=dict(color="#2E86AB", width=2),
            )
        )

        fig.update_layout(
            title="Equity Curve",
            xaxis_title="Date",
            yaxis_title="Equity",
            template="plotly_white",
            height=400,
        )

        chart_path = CHARTS_DIR / f"equity_curve_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.html"
        fig.write_html(str(chart_path))
        return str(chart_path)

    def _create_drawdown_chart(self, equity_curve: pd.DataFrame) -> str:
        """Create drawdown chart."""
        if equity_curve.empty:
            return ""

        df = equity_curve.copy()
        x_col = "date" if "date" in df.columns else "timestamp"
        if x_col not in df.columns:
            return ""

        df[x_col] = pd.to_datetime(df[x_col])
        df = df.sort_values(x_col)
        running_max = df["equity"].cummax()
        df["drawdown"] = df["equity"] / running_max - 1.0

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df[x_col],
                y=df["drawdown"] * 100,
                mode="lines",
                name="Drawdown",
                line=dict(color="#C0392B", width=2),
            )
        )
        fig.update_layout(
            title="Drawdown (%)",
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            template="plotly_white",
            height=400,
        )

        chart_path = CHARTS_DIR / f"drawdown_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.html"
        fig.write_html(str(chart_path))
        return str(chart_path)

    def _create_returns_heatmap(self, period_returns: pd.DataFrame) -> str:
        """Create monthly returns heatmap chart."""
        if period_returns.empty:
            return ""

        df = period_returns.copy()
        pivot = df.pivot(index="year", columns="month", values="return").fillna(0.0)

        fig = go.Figure(
            data=go.Heatmap(
                z=pivot.values * 100,
                x=[str(m) for m in pivot.columns],
                y=[str(y) for y in pivot.index],
                colorscale="RdYlGn",
                reversescale=True,
                colorbar=dict(title="Return (%)"),
            )
        )
        fig.update_layout(
            title="Monthly Returns Heatmap",
            xaxis_title="Month",
            yaxis_title="Year",
            template="plotly_white",
            height=400,
        )

        chart_path = CHARTS_DIR / f"monthly_returns_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.html"
        fig.write_html(str(chart_path))
        return str(chart_path)

    def _create_pnl_distribution(self, trades: List[Any]) -> str:
        """Create win/loss PnL distribution chart."""
        if not trades:
            return ""

        pnl = [t.realized_pnl for t in trades if getattr(t, "realized_pnl", None) is not None]
        if not pnl:
            return ""

        fig = go.Figure()
        fig.add_trace(go.Histogram(x=pnl, nbinsx=30, marker_color="#34495E"))
        fig.update_layout(
            title="Trade PnL Distribution",
            xaxis_title="PnL",
            yaxis_title="Count",
            template="plotly_white",
            height=400,
        )

        chart_path = CHARTS_DIR / f"pnl_distribution_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.html"
        fig.write_html(str(chart_path))
        return str(chart_path)

    def _summarize_trades(self, trades: List[Any]) -> Dict[str, Any]:
        """Summarize trades list."""
        if not trades:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
            }

        pnl = [t.realized_pnl for t in trades if getattr(t, "realized_pnl", None) is not None]
        wins = [p for p in pnl if p > 0]
        losses = [p for p in pnl if p < 0]

        total = len(pnl)
        win_rate = len(wins) / total * 100 if total > 0 else 0.0
        avg_win = sum(wins) / len(wins) if wins else 0.0
        avg_loss = sum(losses) / len(losses) if losses else 0.0

        return {
            "total_trades": total,
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
        }


# === PORTFOLIO REPORT ===
class PortfolioReport:
    """Generate portfolio summary report."""

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.portfolio_manager = PortfolioManager()

    async def generate(self) -> Dict[str, Any]:
        """Generate portfolio report data."""
        self.logger.info("Generating portfolio report...")

        portfolio = await self.portfolio_manager.get_current_portfolio(
            account_id=self.config.get("account")
        )

        sector_allocation = self._calculate_sector_allocation(portfolio.positions)
        asset_class_allocation = self._calculate_asset_class_allocation(portfolio.positions)

        risk_manager = RiskManager()
        risk_metrics = await risk_manager.calculate_portfolio_risk(portfolio)

        charts: Dict[str, str] = {}
        if not self.config.get("no_charts"):
            charts["sector_pie"] = self._create_allocation_pie_chart(
                sector_allocation, "Sector"
            )
            charts["asset_class_pie"] = self._create_allocation_pie_chart(
                asset_class_allocation, "Asset Class"
            )
            charts["position_bars"] = self._create_position_bar_chart(portfolio.positions)

        return {
            "report_type": "portfolio",
            "title": "Portfolio Summary Report",
            "snapshot_date": datetime.utcnow().isoformat(),
            "summary": {
                "total_value": portfolio.total_value,
                "cash": portfolio.cash,
                "invested": portfolio.invested_value,
                "unrealized_pnl": portfolio.unrealized_pnl,
                "positions_count": len(portfolio.positions),
            },
            "positions": [p.to_dict() for p in portfolio.positions],
            "allocations": {
                "by_sector": sector_allocation,
                "by_asset_class": asset_class_allocation,
            },
            "risk_metrics": risk_metrics,
            "charts": charts,
            "generated_at": datetime.utcnow().isoformat(),
        }

    def _calculate_sector_allocation(self, positions: List[Any]) -> Dict[str, float]:
        """Calculate allocation by sector."""
        totals: Dict[str, float] = {}
        total_mv = 0.0
        for p in positions:
            sector = getattr(p, "sector", "Unknown")
            mv = getattr(p, "market_value", 0.0)
            totals[sector] = totals.get(sector, 0.0) + mv
            total_mv += mv
        if total_mv <= 0:
            return totals
        return {k: v / total_mv for k, v in totals.items()}

    def _calculate_asset_class_allocation(self, positions: List[Any]) -> Dict[str, float]:
        """Calculate allocation by asset class."""
        totals: Dict[str, float] = {}
        total_mv = 0.0
        for p in positions:
            asset_class = getattr(p, "asset_class", "Equity")
            mv = getattr(p, "market_value", 0.0)
            totals[asset_class] = totals.get(asset_class, 0.0) + mv
            total_mv += mv
        if total_mv <= 0:
            return totals
        return {k: v / total_mv for k, v in totals.items()}

    def _create_allocation_pie_chart(
        self, allocation: Dict[str, float], title: str
    ) -> str:
        """Create allocation pie chart."""
        if not allocation:
            return ""

        fig = go.Figure(
            data=[
                go.Pie(
                    labels=list(allocation.keys()),
                    values=list(allocation.values()),
                    hole=0.3,
                )
            ]
        )

        fig.update_layout(
            title=f"{title} Allocation",
            template="plotly_white",
            height=400,
        )

        slug = title.lower().replace(" ", "_")
        chart_path = CHARTS_DIR / f"{slug}_pie_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.html"
        fig.write_html(str(chart_path))
        return str(chart_path)

    def _create_position_bar_chart(self, positions: List[Any]) -> str:
        """Create bar chart of top positions by market value."""
        if not positions:
            return ""

        sorted_pos = sorted(
            positions, key=lambda p: getattr(p, "market_value", 0.0), reverse=True
        )[:20]
        symbols = [p.symbol for p in sorted_pos]
        values = [p.market_value for p in sorted_pos]

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=symbols,
                y=values,
                marker_color="#2E86AB",
            )
        )
        fig.update_layout(
            title="Top Positions by Market Value",
            xaxis_title="Symbol",
            yaxis_title="Market Value",
            template="plotly_white",
            height=400,
        )

        chart_path = CHARTS_DIR / f"positions_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.html"
        fig.write_html(str(chart_path))
        return str(chart_path)


# === STRATEGY REPORT ===
class StrategyReport:
    """Generate strategy-specific performance report."""

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.analyzer = PerformanceAnalyzer()
        self.queries = ReportQueries()

    async def generate(self) -> Dict[str, Any]:
        """Generate strategy report data."""
        self.logger.info("Generating strategy report...")

        start_date: datetime = self.config["start_date"]
        end_date: datetime = self.config["end_date"]

        strategy_id = self.config.get("strategy")
        if not strategy_id:
            raise ValueError("Strategy report requires --strategy filter")

        equity_curve = await self.queries.get_equity_curve(
            start_date, end_date, strategy_id=strategy_id
        )
        trades = await self.queries.get_trades(
            start_date, end_date, strategy_id=strategy_id
        )

        metrics = self.analyzer.calculate_all_metrics(equity_curve, trades)
        by_symbol = self.analyzer.group_by_symbol(trades)
        by_regime = self.analyzer.group_by_regime(trades)

        charts: Dict[str, str] = {}
        if not self.config.get("no_charts"):
            charts["equity_curve"] = PerformanceReport(self.config, self.logger)._create_equity_chart(
                equity_curve
            )

        return {
            "report_type": "strategy",
            "title": f"Strategy Report - {strategy_id}",
            "strategy_id": strategy_id,
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
            },
            "metrics": metrics,
            "by_symbol": by_symbol,
            "by_regime": by_regime,
            "trades": [t.to_dict() for t in trades],
            "charts": charts,
            "generated_at": datetime.utcnow().isoformat(),
        }


# === RISK REPORT ===
class RiskReport:
    """Generate risk analysis report."""

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.risk_manager = RiskManager()

    async def generate(self) -> Dict[str, Any]:
        """Generate risk report data."""
        self.logger.info("Generating risk report...")

        start_date: datetime = self.config["start_date"]
        end_date: datetime = self.config["end_date"]

        var_95 = await self.risk_manager.calculate_var(confidence=0.95)
        var_99 = await self.risk_manager.calculate_var(confidence=0.99)
        cvar = await self.risk_manager.calculate_cvar()

        max_drawdown = await self.risk_manager.calculate_max_drawdown(
            start_date, end_date
        )
        current_drawdown = await self.risk_manager.get_current_drawdown()

        position_risks = await self.risk_manager.calculate_position_risks()
        concentration_metrics = await self.risk_manager.analyze_concentration()
        limit_breaches = await self.risk_manager.get_limit_breaches(
            start_date, end_date
        )

        return {
            "report_type": "risk",
            "title": "Risk Analysis Report",
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
            },
            "var_metrics": {
                "var_95": var_95,
                "var_99": var_99,
                "cvar": cvar,
            },
            "drawdown": {
                "max_drawdown": max_drawdown,
                "current_drawdown": current_drawdown,
            },
            "position_risks": position_risks,
            "concentration": concentration_metrics,
            "limit_breaches": limit_breaches,
            "generated_at": datetime.utcnow().isoformat(),
        }


# === TRADING ACTIVITY REPORT ===
class TradingActivityReport:
    """Generate trading activity and execution quality report."""

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.queries = ReportQueries()

    async def generate(self) -> Dict[str, Any]:
        """Generate trading activity report data."""
        self.logger.info("Generating trading activity report...")

        start_date: datetime = self.config["start_date"]
        end_date: datetime = self.config["end_date"]

        trades = await self.queries.get_trades(
            start_date,
            end_date,
            strategy_id=self.config.get("strategy"),
            symbol=self.config.get("symbol"),
            account_id=self.config.get("account"),
        )
        orders = await self.queries.get_orders(
            start_date,
            end_date,
            strategy_id=self.config.get("strategy"),
            symbol=self.config.get("symbol"),
            account_id=self.config.get("account"),
        )

        execution_metrics = self._calculate_execution_metrics(orders)
        trade_analysis = self._analyze_trades(trades)
        symbol_breakdown = self._breakdown_by_symbol(trades)

        return {
            "report_type": "trading",
            "title": "Trading Activity Report",
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
            },
            "summary": {
                "total_trades": len(trades),
                "total_orders": len(orders),
                "avg_slippage": execution_metrics["avg_slippage"],
                "avg_commission": execution_metrics["avg_commission"],
                "order_fill_rate": execution_metrics["fill_rate"],
            },
            "execution_quality": execution_metrics,
            "trade_analysis": trade_analysis,
            "by_symbol": symbol_breakdown,
            "trade_log": [t.to_dict() for t in trades],
            "generated_at": datetime.utcnow().isoformat(),
        }

    def _calculate_execution_metrics(self, orders: List[Any]) -> Dict[str, Any]:
        """Compute basic execution quality metrics from orders."""
        if not orders:
            return {"avg_slippage": 0.0, "avg_commission": 0.0, "fill_rate": 0.0}

        filled = [o for o in orders if getattr(o, "filled_quantity", 0) > 0]
        total_qty = sum(getattr(o, "quantity", 0) for o in orders)
        filled_qty = sum(getattr(o, "filled_quantity", 0) for o in orders)

        commissions = [getattr(o, "commission", 0.0) for o in orders]
        slippages = [getattr(o, "slippage", 0.0) for o in orders]

        avg_commission = float(sum(commissions) / len(commissions)) if commissions else 0.0
        avg_slippage = float(sum(slippages) / len(slippages)) if slippages else 0.0
        fill_rate = (filled_qty / total_qty * 100) if total_qty > 0 else 0.0

        return {
            "avg_slippage": avg_slippage,
            "avg_commission": avg_commission,
            "fill_rate": fill_rate,
            "total_orders": len(orders),
            "filled_orders": len(filled),
        }

    def _analyze_trades(self, trades: List[Any]) -> Dict[str, Any]:
        """Analyze trades (direction, holding time, etc.)."""
        long_trades = [t for t in trades if getattr(t, "side", "").upper() == "LONG"]
        short_trades = [t for t in trades if getattr(t, "side", "").upper() == "SHORT"]

        return {
            "total_trades": len(trades),
            "long_trades": len(long_trades),
            "short_trades": len(short_trades),
        }

    def _breakdown_by_symbol(self, trades: List[Any]) -> Dict[str, Any]:
        """Aggregate trades by symbol."""
        breakdown: Dict[str, Dict[str, Any]] = {}
        for t in trades:
            symbol = t.symbol
            pnl = getattr(t, "realized_pnl", 0.0)
            d = breakdown.setdefault(
                symbol,
                {"trades": 0, "gross_pnl": 0.0},
            )
            d["trades"] += 1
            d["gross_pnl"] += pnl
        return breakdown


# === COMPLIANCE REPORT ===
class ComplianceReport:
    """Generate compliance and regulatory report."""

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.queries = ReportQueries()
        self.risk_manager = RiskManager()

    async def generate(self) -> Dict[str, Any]:
        """Generate compliance report data."""
        self.logger.info("Generating compliance report...")

        start_date: datetime = self.config["start_date"]
        end_date: datetime = self.config["end_date"]

        breaches = await self.risk_manager.get_limit_breaches(start_date, end_date)
        trade_flags = await self.queries.get_compliance_flags(start_date, end_date)
        restricted_list_hits = await self.queries.get_restricted_list_hits(
            start_date, end_date
        )

        return {
            "report_type": "compliance",
            "title": "Compliance Report",
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
            },
            "limit_breaches": breaches,
            "trade_flags": trade_flags,
            "restricted_list_hits": restricted_list_hits,
            "generated_at": datetime.utcnow().isoformat(),
        }


# === EXECUTIVE SUMMARY REPORT ===
class ExecutiveSummaryReport:
    """Generate high-level executive summary report."""

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.performance = PerformanceAnalyzer()
        self.queries = ReportQueries()
        self.portfolio_manager = PortfolioManager()

    async def generate(self) -> Dict[str, Any]:
        """Generate executive summary report data."""
        self.logger.info("Generating executive summary report...")

        start_date: datetime = self.config["start_date"]
        end_date: datetime = self.config["end_date"]

        equity_curve = await self.queries.get_equity_curve(start_date, end_date)
        trades = await self.queries.get_trades(start_date, end_date)
        portfolio = await self.portfolio_manager.get_current_portfolio()

        metrics = self.performance.calculate_all_metrics(equity_curve, trades)
        top_strategies = await self.queries.get_top_strategies(start_date, end_date)
        top_positions = sorted(
            portfolio.positions,
            key=lambda p: getattr(p, "market_value", 0.0),
            reverse=True,
        )[:10]

        return {
            "report_type": "executive",
            "title": "Executive Summary Report",
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
            },
            "headline_metrics": {
                "total_return": metrics.get("total_return"),
                "sharpe_ratio": metrics.get("sharpe_ratio"),
                "max_drawdown_pct": metrics.get("max_drawdown_pct"),
                "win_rate": metrics.get("win_rate"),
                "total_trades": metrics.get("total_trades"),
            },
            "portfolio_snapshot": {
                "total_value": portfolio.total_value,
                "cash": portfolio.cash,
                "positions_count": len(portfolio.positions),
            },
            "top_strategies": top_strategies,
            "top_positions": [p.to_dict() for p in top_positions],
            "generated_at": datetime.utcnow().isoformat(),
        }


# === CUSTOM REPORT ===
class CustomReport:
    """Generate custom report from template and JSON data."""

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger

    async def generate(self) -> Dict[str, Any]:
        """Prepare data for custom report."""
        self.logger.info("Generating custom report...")

        if not self.config.get("template"):
            raise ValueError("Custom report requires --template")

        extra_data: Dict[str, Any] = {}
        data_path: Optional[Path] = self.config.get("data_path")
        if data_path:
            if not data_path.exists():
                raise FileNotFoundError(f"Custom data file not found: {data_path}")
            with open(data_path, "r", encoding="utf-8") as f:
                extra_data = json.load(f)

        start_date: datetime = self.config["start_date"]
        end_date: datetime = self.config["end_date"]

        return {
            "report_type": "custom",
            "title": extra_data.get("title", "Custom Report"),
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
            },
            "data": extra_data,
            "generated_at": datetime.utcnow().isoformat(),
        }


# === RENDERER ===
class ReportRenderer:
    """Render reports in various formats."""

    def __init__(self, template_dir: Path = TEMPLATES_DIR):
        self.env = Environment(loader=FileSystemLoader(template_dir))
        self.env.globals.update(
            {
                "format_currency": self._format_currency,
                "format_percent": self._format_percent,
                "format_date": self._format_date,
            }
        )

    def render_html(
        self, report_data: Dict[str, Any], template_path: Optional[Path] = None
    ) -> str:
        """Render report as HTML."""
        if template_path:
            # Allow external template path
            loader = FileSystemLoader(template_path.parent)
            env = Environment(loader=loader)
            env.globals.update(self.env.globals)
            template = env.get_template(template_path.name)
        else:
            template_name = f"{report_data.get('report_type', 'default')}.html"
            template = self.env.get_template(template_name)

        return template.render(**report_data)

    def render_pdf(self, html_content: str, output_path: Path) -> Path:
        """Convert HTML to PDF using WeasyPrint."""
        from weasyprint import HTML

        HTML(string=html_content).write_pdf(str(output_path))
        return output_path

    def render_excel(self, report_data: Dict[str, Any], output_path: Path) -> Path:
        """Render report as Excel workbook."""
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            if "summary" in report_data:
                pd.DataFrame([report_data["summary"]]).to_excel(
                    writer, sheet_name="Summary", index=False
                )

            if "metrics" in report_data:
                pd.DataFrame([report_data["metrics"]]).to_excel(
                    writer, sheet_name="Metrics", index=False
                )

            if "positions" in report_data:
                pd.DataFrame(report_data["positions"]).to_excel(
                    writer, sheet_name="Positions", index=False
                )

            if "trade_log" in report_data:
                pd.DataFrame(report_data["trade_log"]).to_excel(
                    writer, sheet_name="Trades", index=False
                )

        return output_path

    @staticmethod
    def _format_currency(value: float) -> str:
        return f"${value:,.2f}"

    @staticmethod
    def _format_percent(value: float) -> str:
        return f"{value:.2f}%"

    @staticmethod
    def _format_date(date_str: str) -> str:
        dt = datetime.fromisoformat(date_str)
        return dt.strftime("%Y-%m-%d")


# === EMAIL DELIVERY ===
async def send_report_email(
    report_path: Path,
    config: Dict[str, Any],
    logger: logging.Logger,
) -> None:
    """Send report via email."""
    if not config.get("recipients"):
        logger.warning("Email requested but no recipients provided, skipping email")
        return

    logger.info("Sending report via email...")

    email_client = EmailClient()

    subject = config.get("subject") or (
        f"ALPHA-PRIME {config['type'].title()} Report - {datetime.utcnow().strftime('%Y-%m-%d')}"
    )

    body = (
        f"Attached is the {config['type']} report for period: "
        f"{config['period']} ({config['start_date'].date()} to {config['end_date'].date()}).\n\n"
        f"Report generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        "ALPHA-PRIME v2.0 Trading System"
    )

    await email_client.send(
        to=config["recipients"],
        subject=subject,
        body=body,
        attachments=[report_path],
    )

    logger.info("Report sent to %d recipient(s)", len(config["recipients"]))


# === SCHEDULING SUPPORT ===
def setup_scheduled_report(config: Dict[str, Any], logger: logging.Logger) -> None:
    """Output cron entry for scheduled reports."""
    logger.info("Setting up %s scheduled report (cron snippet)...", config["schedule"])

    if config["schedule"] == "daily":
        cron_expr = "0 8 * * *"
    elif config["schedule"] == "weekly":
        cron_expr = "0 8 * * 1"
    elif config["schedule"] == "monthly":
        cron_expr = "0 8 1 * *"
    else:
        raise ValueError(f"Unknown schedule: {config['schedule']}")

    cmd_args = [
        f"--type {config['type']}",
        f"--period {config.get('period', 'monthly')}",
        f"--format {config.get('format', 'pdf')}",
    ]

    if config.get("email") and config.get("recipients"):
        cmd_args.append("--email")
        cmd_args.append(f"--recipients \"{','.join(config['recipients'])}\"")

    command = (
        f"cd {PROJECT_ROOT} && "
        f"python scripts/generate_report.py {' '.join(cmd_args)}"
    )

    crontab_entry = f"{cron_expr} {command}\n"

    logger.info("Add the following line to your crontab (crontab -e):")
    logger.info(crontab_entry.strip())


# === MAIN EXECUTION ===
async def main() -> int:
    args = parse_arguments()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logger("report_generator", level=log_level)

    try:
        config = build_config(args)

        if args.schedule:
            setup_scheduled_report(config, logger)
            return 0

        ensure_dirs()

        report_class_name = REPORT_TYPES[args.type]
        report_cls = globals()[report_class_name]
        report_generator = report_cls(config, logger)
        report_data = await report_generator.generate()

        renderer = ReportRenderer()

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        default_name = f"{args.type}_report_{timestamp}"

        if args.format == "html":
            html = renderer.render_html(report_data, args.template)
            output_path = args.output or REPORTS_DIR / f"{default_name}.html"
            output_path.write_text(html, encoding="utf-8")

        elif args.format == "pdf":
            html = renderer.render_html(report_data, args.template)
            output_path = args.output or REPORTS_DIR / f"{default_name}.pdf"
            renderer.render_pdf(html, output_path)

        elif args.format == "excel":
            output_path = args.output or REPORTS_DIR / f"{default_name}.xlsx"
            renderer.render_excel(report_data, output_path)

        elif args.format == "json":
            output_path = args.output or REPORTS_DIR / f"{default_name}.json"
            output_path.write_text(json.dumps(report_data, indent=2, default=str), encoding="utf-8")

        elif args.format == "csv":
            output_path = args.output or REPORTS_DIR / f"{default_name}.csv"
            df = pd.json_normalize(report_data)
            df.to_csv(output_path, index=False)

        logger.info("✓ Report generated: %s", output_path)

        if args.email:
            await send_report_email(output_path, config, logger)

        return 0

    except Exception as exc:
        logger.error("Report generation failed: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    import asyncio

    sys.exit(asyncio.run(main()))
