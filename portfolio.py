"""
============================================================
ALPHA-PRIME v2.0 - Portfolio Manager (Paper Trading)
============================================================
Module 6: Paper trading ledger with atomic persistence.

Manages:
- Cash and position tracking
- Trade execution (BUY/SELL)
- Realized and unrealized P&L
- Trade history logging (CSV)
- Performance metrics (Sharpe, win rate, max DD)
- Atomic state persistence (corruption-proof)

State Persistence:
- Portfolio state: data/portfolio.json (atomic writes)
- Trade history: data/trade_history.csv (append-only)
- Backups: backups/portfolio_backup_YYYYMMDD_HHMMSS.json

Usage:
    from portfolio import PaperTrader

    trader = PaperTrader()

    # Execute trade
    result = trader.execute_trade(
        action="BUY",
        ticker="AAPL",
        price=150.00,
        quantity=10,
    )

    # Get state
    portfolio = trader.get_portfolio_state()
    print(f"Cash: ${portfolio.cash:.2f}")
    print(f"Total Value: ${portfolio.total_value:.2f}")

    # Get metrics
    metrics = trader.calculate_metrics()
    print(f"Win Rate: {metrics.win_rate:.1f}%")
    print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")

Thread Safety:
- Uses file locking for concurrent access
- Atomic writes prevent corruption
- Safe for scheduler + dashboard concurrent access
============================================================
"""

from __future__ import annotations

import csv
import json
import logging
import os
import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fcntl
import numpy as np

from config import get_logger, get_settings

logger = get_logger(__name__)
settings = get_settings()


# ──────────────────────────────────────────────────────────
# DATA SCHEMAS
# ──────────────────────────────────────────────────────────


@dataclass
class Position:
    """
    Single stock position.

    Attributes:
        ticker: Symbol of the asset.
        quantity: Number of shares currently held.
        avg_entry_price: Volume-weighted average entry price.
        entry_timestamp_utc: ISO timestamp when position was opened.
        last_updated_utc: ISO timestamp of last modification.
        realized_pnl: Realized P&L accumulated on partial exits.
    """

    ticker: str
    quantity: float
    avg_entry_price: float
    entry_timestamp_utc: str
    last_updated_utc: str
    realized_pnl: float = 0.0

    @property
    def cost_basis(self) -> float:
        """Total cost basis of the position (quantity * avg_entry_price)."""
        return self.quantity * self.avg_entry_price

    def unrealized_pnl(self, current_price: float) -> float:
        """
        Compute unrealized P&L at a given current_price.

        Args:
            current_price: Latest market price.

        Returns:
            Unrealized P&L in currency units.
        """
        return (current_price - self.avg_entry_price) * self.quantity

    def unrealized_pnl_pct(self, current_price: float) -> float:
        """
        Compute percentage unrealized P&L at a given current_price.

        Args:
            current_price: Latest market price.

        Returns:
            Unrealized P&L percentage.
        """
        if self.avg_entry_price == 0:
            return 0.0
        return (current_price - self.avg_entry_price) / self.avg_entry_price * 100.0


@dataclass
class Trade:
    """
    Single trade record.

    Attributes:
        timestamp_utc: ISO timestamp of execution.
        ticker: Symbol of the asset.
        action: "BUY" or "SELL".
        quantity: Number of shares executed.
        price: Execution price.
        commission: Commission paid on this trade.
        pnl: Realized P&L for this trade (0 for BUY).
        portfolio_value_after: Portfolio total value immediately after trade.
        notes: Optional free-form notes.
    """

    timestamp_utc: str
    ticker: str
    action: str  # BUY | SELL
    quantity: float
    price: float
    commission: float
    pnl: float
    portfolio_value_after: float
    notes: str = ""


@dataclass
class Portfolio:
    """
    Complete portfolio state.

    Attributes:
        cash: Free cash available.
        positions: Mapping of ticker → Position.
        starting_cash: Initial capital.
        total_trades: Total trade count.
        total_commission: Total commissions paid.
        created_at_utc: ISO timestamp when portfolio was created.
        last_updated_utc: ISO timestamp of last modification.
    """

    cash: float
    positions: Dict[str, Position]
    starting_cash: float
    total_trades: int
    total_commission: float
    created_at_utc: str
    last_updated_utc: str

    @property
    def total_position_value(self) -> float:
        """
        Sum of all position cost bases.

        Note:
            This uses cost basis, not mark-to-market. For MTM you must
            revalue using current prices externally.
        """
        return float(sum(pos.cost_basis for pos in self.positions.values()))

    @property
    def total_value(self) -> float:
        """Total portfolio value (cash + positions at cost basis)."""
        return float(self.cash + self.total_position_value)

    @property
    def total_pnl(self) -> float:
        """Total P&L relative to starting cash."""
        return float(self.total_value - self.starting_cash)

    @property
    def total_pnl_pct(self) -> float:
        """Total P&L percentage relative to starting cash."""
        if self.starting_cash == 0:
            return 0.0
        return self.total_pnl / self.starting_cash * 100.0

    @property
    def position_count(self) -> int:
        """Number of open positions."""
        return len(self.positions)


@dataclass
class TradeResult:
    """
    Result of trade execution.

    Attributes:
        success: True if trade executed successfully.
        trade: Trade object if success, else None.
        message: Human-readable summary.
        portfolio_snapshot: Portfolio snapshot after trade (if success).
        error: Optional error message when success is False.
    """

    success: bool
    trade: Optional[Trade]
    message: str
    portfolio_snapshot: Optional[Portfolio] = None
    error: Optional[str] = None


@dataclass
class PerformanceMetrics:
    """
    Portfolio performance metrics.

    Attributes:
        total_trades: Number of closed trades considered.
        winning_trades: Count of trades with positive P&L.
        losing_trades: Count of trades with negative P&L.
        win_rate: Winning trades percentage.
        average_win: Mean P&L of winning trades.
        average_loss: Mean P&L of losing trades.
        profit_factor: Total gains / total losses.
        total_pnl: Total P&L in currency units.
        total_pnl_pct: Total P&L percentage vs starting cash.
        max_drawdown_pct: Maximum drawdown in percent.
        sharpe_ratio: Simplified Sharpe ratio (per-trade).
        current_streak: Consecutive wins (>0) or losses (<0).
        largest_win: Largest winning trade P&L.
        largest_loss: Largest losing trade P&L (negative).
    """

    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    average_win: float
    average_loss: float
    profit_factor: float
    total_pnl: float
    total_pnl_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    current_streak: int
    largest_win: float
    largest_loss: float


@dataclass
class EquityPoint:
    """
    Single point on the portfolio equity curve.

    Attributes:
        timestamp_utc: ISO timestamp of this equity snapshot.
        portfolio_value: Total portfolio value after the trade.
        cash: Cash at that time (if recorded).
        position_value: Position value at that time (if recorded).
        cumulative_pnl: P&L vs starting cash.
    """

    timestamp_utc: str
    portfolio_value: float
    cash: float
    position_value: float
    cumulative_pnl: float


# ──────────────────────────────────────────────────────────
# FILE OPERATIONS (Thread-Safe, Atomic)
# ──────────────────────────────────────────────────────────


@contextmanager
def file_lock(file_path: Path):
    """
    Context manager providing an exclusive advisory file lock.

    Uses a separate ".lock" file alongside the target file. Ensures that
    only one process/thread performs critical section operations at a time.

    Args:
        file_path: Path for which to acquire lock.

    Yields:
        A file handle on the lock file (do not close it manually).
    """
    lock_file = file_path.parent / f"{file_path.name}.lock"
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    lock_file.touch(exist_ok=True)

    with open(lock_file, "r") as fh:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            yield fh
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def atomic_write_json(file_path: Path, data: Dict[str, object]) -> None:
    """
    Write JSON atomically to a file to prevent corruption.

    Implementation:
        - Serialize JSON into a temp file in the same directory.
        - Flush and fsync temp file.
        - Atomically replace target using os.replace.

    Args:
        file_path: Destination path.
        data: Serializable dictionary to write.

    Raises:
        IOError: If the write or replace operation fails.
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(
        dir=str(file_path.parent),
        prefix=f".{file_path.name}.",
        suffix=".tmp",
    )
    tmp_file = Path(tmp_path)

    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(data, fh, indent=2, default=str)
            fh.flush()
            os.fsync(fh.fileno())

        os.replace(tmp_file, file_path)
    except Exception as exc:  # noqa: BLE001
        try:
            if tmp_file.exists():
                tmp_file.unlink()
        except Exception:  # noqa: BLE001
            pass
        raise IOError(f"Failed to atomically write {file_path}: {exc}") from exc


def append_trade_to_csv(trade: Trade, csv_path: Path) -> None:
    """
    Append a trade record to the CSV trade log.

    Creates the file and headers if it doesn't exist.

    Args:
        trade: Trade to log.
        csv_path: Path to CSV file.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()

    with open(csv_path, "a", newline="") as fh:
        fieldnames = [
            "timestamp_utc",
            "ticker",
            "action",
            "quantity",
            "price",
            "commission",
            "pnl",
            "portfolio_value_after",
            "notes",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(asdict(trade))


# ──────────────────────────────────────────────────────────
# PAPER TRADER CLASS
# ──────────────────────────────────────────────────────────


class PaperTrader:
    """
    Paper trading portfolio manager.

    Responsibilities:
        - Maintain a persistent portfolio ledger.
        - Execute BUY/SELL trades with validation.
        - Keep append-only trade history CSV.
        - Compute performance metrics from trade history.
        - Provide equity curve for analytics.

    The class is safe for concurrent use via advisory file locks
    around portfolio JSON writes.
    """

    def __init__(self, portfolio_path: Optional[str] = None) -> None:
        """
        Initialize the paper trader and load / create portfolio.

        Args:
            portfolio_path: Optional override for portfolio JSON path.
        """
        self.portfolio_path: Path = Path(portfolio_path or settings.portfolio_path)
        self.trade_history_path: Path = Path(settings.trade_history_path)
        self.backup_dir: Path = Path(settings.backup_dir)

        self.portfolio_path.parent.mkdir(parents=True, exist_ok=True)
        self.trade_history_path.parent.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        self._portfolio: Portfolio = self._load_or_initialize()

    # ──────────────────────────────────────────────────────
    # INTERNAL LOAD/SAVE
    # ──────────────────────────────────────────────────────

    def _load_or_initialize(self) -> Portfolio:
        """
        Load portfolio from disk or create a new one.

        Returns:
            Portfolio instance representing current state.
        """
        if self.portfolio_path.exists():
            try:
                with open(self.portfolio_path, "r") as fh:
                    data = json.load(fh)

                positions: Dict[str, Position] = {}
                for ticker, pos_data in data.get("positions", {}).items():
                    positions[ticker] = Position(**pos_data)

                portfolio = Portfolio(
                    cash=float(data["cash"]),
                    positions=positions,
                    starting_cash=float(data["starting_cash"]),
                    total_trades=int(data["total_trades"]),
                    total_commission=float(data["total_commission"]),
                    created_at_utc=str(data["created_at_utc"]),
                    last_updated_utc=str(data["last_updated_utc"]),
                )

                logger.info(
                    "Loaded portfolio: value=$%.2f (%d positions).",
                    portfolio.total_value,
                    portfolio.position_count,
                )
                return portfolio
            except Exception as exc:  # noqa: BLE001
                logger.error("Error loading portfolio from %s: %s", self.portfolio_path, exc)
                logger.warning("Falling back to new portfolio initialization.")

        now = datetime.now(timezone.utc).isoformat()
        starting_cash = float(getattr(settings, "starting_cash", 10000.0))

        portfolio = Portfolio(
            cash=starting_cash,
            positions={},
            starting_cash=starting_cash,
            total_trades=0,
            total_commission=0.0,
            created_at_utc=now,
            last_updated_utc=now,
        )

        self._save_portfolio(portfolio)
        logger.info("Created new portfolio with starting cash $%.2f.", starting_cash)
        return portfolio

    def _save_portfolio(self, portfolio: Portfolio) -> None:
        """
        Persist portfolio snapshot to disk atomically with locking.

        Args:
            portfolio: Portfolio to save.

        Raises:
            IOError: If persistence fails.
        """
        positions_dict = {t: asdict(p) for t, p in portfolio.positions.items()}
        data = {
            "cash": portfolio.cash,
            "positions": positions_dict,
            "starting_cash": portfolio.starting_cash,
            "total_trades": portfolio.total_trades,
            "total_commission": portfolio.total_commission,
            "created_at_utc": portfolio.created_at_utc,
            "last_updated_utc": datetime.now(timezone.utc).isoformat(),
        }

        try:
            with file_lock(self.portfolio_path):
                atomic_write_json(self.portfolio_path, data)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to save portfolio: %s", exc, exc_info=True)
            raise

    # ──────────────────────────────────────────────────────
    # PUBLIC API
    # ──────────────────────────────────────────────────────

    def get_portfolio_state(self) -> Portfolio:
        """
        Retrieve current portfolio state.

        Returns:
            Portfolio snapshot (in-memory object).
        """
        return self._portfolio

    def execute_trade(
        self,
        action: str,
        ticker: str,
        price: float,
        quantity: float,
        notes: str = "",
    ) -> TradeResult:
        """
        Execute a paper trade.

        Validation:
            - action must be BUY or SELL.
            - price and quantity must be positive.
            - BUY: check sufficient cash.
            - SELL: check position exists and has sufficient shares.

        Args:
            action: "BUY" or "SELL".
            ticker: Stock symbol.
            price: Execution price.
            quantity: Number of shares.
            notes: Optional trade notes.

        Returns:
            TradeResult describing outcome.
        """
        action = action.upper().strip()
        ticker = ticker.upper().strip()

        logger.info(
            "Executing trade: %s %s %s @ %.4f",
            action,
            quantity,
            ticker,
            price,
        )

        if action not in ("BUY", "SELL"):
            return TradeResult(
                success=False,
                trade=None,
                message="Action must be BUY or SELL.",
                error="invalid_action",
            )

        if quantity <= 0:
            return TradeResult(
                success=False,
                trade=None,
                message="Quantity must be positive.",
                error="invalid_quantity",
            )

        if price <= 0:
            return TradeResult(
                success=False,
                trade=None,
                message="Price must be positive.",
                error="invalid_price",
            )

        commission = float(getattr(settings, "commission_per_trade", 0.0))

        try:
            if action == "BUY":
                return self._execute_buy(
                    ticker=ticker,
                    price=price,
                    quantity=quantity,
                    commission=commission,
                    notes=notes,
                )
            return self._execute_sell(
                ticker=ticker,
                price=price,
                quantity=quantity,
                commission=commission,
                notes=notes,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Trade execution error: %s", exc, exc_info=True)
            return TradeResult(
                success=False,
                trade=None,
                message="Trade execution failed.",
                error=str(exc),
            )

    # ──────────────────────────────────────────────────────
    # TRADE HELPERS
    # ──────────────────────────────────────────────────────

    def _execute_buy(
        self,
        ticker: str,
        price: float,
        quantity: float,
        commission: float,
        notes: str,
    ) -> TradeResult:
        """Internal BUY execution with validation and state updates."""
        total_cost = price * quantity + commission

        if self._portfolio.cash < total_cost:
            msg = (
                f"Insufficient cash: need ${total_cost:.2f}, "
                f"available ${self._portfolio.cash:.2f}."
            )
            logger.warning(msg)
            return TradeResult(
                success=False,
                trade=None,
                message=msg,
                error="insufficient_cash",
            )

        self._portfolio.cash -= total_cost
        self._portfolio.total_commission += commission

        now = datetime.now(timezone.utc).isoformat()

        if ticker in self._portfolio.positions:
            pos = self._portfolio.positions[ticker]
            total_qty = pos.quantity + quantity
            total_cost_basis = pos.avg_entry_price * pos.quantity + price * quantity
            pos.quantity = total_qty
            pos.avg_entry_price = total_cost_basis / total_qty
            pos.last_updated_utc = now
        else:
            self._portfolio.positions[ticker] = Position(
                ticker=ticker,
                quantity=quantity,
                avg_entry_price=price,
                entry_timestamp_utc=now,
                last_updated_utc=now,
                realized_pnl=0.0,
            )

        self._portfolio.total_trades += 1

        trade = Trade(
            timestamp_utc=now,
            ticker=ticker,
            action="BUY",
            quantity=quantity,
            price=price,
            commission=commission,
            pnl=0.0,
            portfolio_value_after=self._portfolio.total_value,
            notes=notes,
        )

        self._save_portfolio(self._portfolio)
        append_trade_to_csv(trade, self.trade_history_path)

        logger.info("BUY executed: %s x %s @ %.4f.", quantity, ticker, price)

        return TradeResult(
            success=True,
            trade=trade,
            message=f"BUY {quantity} {ticker} @ ${price:.4f}",
            portfolio_snapshot=self._portfolio,
        )

    def _execute_sell(
        self,
        ticker: str,
        price: float,
        quantity: float,
        commission: float,
        notes: str,
    ) -> TradeResult:
        """Internal SELL execution with validation and state updates."""
        if ticker not in self._portfolio.positions:
            msg = f"No open position in {ticker}."
            logger.warning(msg)
            return TradeResult(
                success=False,
                trade=None,
                message=msg,
                error="no_position",
            )

        pos = self._portfolio.positions[ticker]

        if pos.quantity < quantity:
            msg = (
                f"Insufficient shares: trying to sell {quantity}, "
                f"holding {pos.quantity}."
            )
            logger.warning(msg)
            return TradeResult(
                success=False,
                trade=None,
                message=msg,
                error="insufficient_shares",
            )

        gross_proceeds = price * quantity
        net_proceeds = gross_proceeds - commission
        cost_basis = pos.avg_entry_price * quantity
        realized_pnl = net_proceeds - cost_basis

        self._portfolio.cash += net_proceeds
        self._portfolio.total_commission += commission

        now = datetime.now(timezone.utc).isoformat()
        pos.quantity -= quantity
        pos.realized_pnl += realized_pnl
        pos.last_updated_utc = now

        if pos.quantity <= 0:
            del self._portfolio.positions[ticker]

        self._portfolio.total_trades += 1

        trade = Trade(
            timestamp_utc=now,
            ticker=ticker,
            action="SELL",
            quantity=quantity,
            price=price,
            commission=commission,
            pnl=realized_pnl,
            portfolio_value_after=self._portfolio.total_value,
            notes=notes,
        )

        self._save_portfolio(self._portfolio)
        append_trade_to_csv(trade, self.trade_history_path)

        logger.info(
            "SELL executed: %s x %s @ %.4f (realized P&L=%.2f).",
            quantity,
            ticker,
            price,
            realized_pnl,
        )

        return TradeResult(
            success=True,
            trade=trade,
            message=f"SELL {quantity} {ticker} @ ${price:.4f} (P&L: ${realized_pnl:+.2f})",
            portfolio_snapshot=self._portfolio,
        )

    # ──────────────────────────────────────────────────────
    # PERFORMANCE METRICS & EQUITY
    # ──────────────────────────────────────────────────────

    def calculate_metrics(self) -> PerformanceMetrics:
        """
        Calculate performance metrics from trade history.

        Only SELL trades are considered for per-trade P&L metrics.

        Returns:
            PerformanceMetrics instance.
        """
        if not self.trade_history_path.exists():
            return PerformanceMetrics(
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                average_win=0.0,
                average_loss=0.0,
                profit_factor=0.0,
                total_pnl=self._portfolio.total_pnl,
                total_pnl_pct=self._portfolio.total_pnl_pct,
                max_drawdown_pct=0.0,
                sharpe_ratio=0.0,
                current_streak=0,
                largest_win=0.0,
                largest_loss=0.0,
            )

        trade_pnls: List[float] = []
        with open(self.trade_history_path, "r") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                if row.get("action") == "SELL":
                    try:
                        trade_pnls.append(float(row["pnl"]))
                    except (KeyError, ValueError):
                        continue

        if not trade_pnls:
            return PerformanceMetrics(
                total_trades=self._portfolio.total_trades,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                average_win=0.0,
                average_loss=0.0,
                profit_factor=0.0,
                total_pnl=self._portfolio.total_pnl,
                total_pnl_pct=self._portfolio.total_pnl_pct,
                max_drawdown_pct=0.0,
                sharpe_ratio=0.0,
                current_streak=0,
                largest_win=0.0,
                largest_loss=0.0,
            )

        wins = [p for p in trade_pnls if p > 0]
        losses = [p for p in trade_pnls if p < 0]

        winning_trades = len(wins)
        losing_trades = len(losses)
        total_trades = len(trade_pnls)
        win_rate = winning_trades / total_trades * 100.0 if total_trades else 0.0

        average_win = float(sum(wins) / len(wins)) if wins else 0.0
        average_loss = float(sum(losses) / len(losses)) if losses else 0.0

        total_gains = float(sum(wins))
        total_losses_abs = float(abs(sum(losses)))
        profit_factor = total_gains / total_losses_abs if total_losses_abs > 0 else 0.0

        current_streak = 0
        for pnl in reversed(trade_pnls):
            if pnl > 0:
                if current_streak >= 0:
                    current_streak += 1
                else:
                    break
            elif pnl < 0:
                if current_streak <= 0:
                    current_streak -= 1
                else:
                    break
            else:
                break

        equity_curve = self.get_equity_curve()
        max_dd_pct = self._calculate_max_drawdown(
            [point.portfolio_value for point in equity_curve]
        )

        returns = [
            pnl / self._portfolio.starting_cash
            for pnl in trade_pnls
            if self._portfolio.starting_cash > 0
        ]
        sharpe = self._calculate_sharpe_ratio(returns)

        largest_win = float(max(wins)) if wins else 0.0
        largest_loss = float(min(losses)) if losses else 0.0

        return PerformanceMetrics(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            average_win=average_win,
            average_loss=average_loss,
            profit_factor=profit_factor,
            total_pnl=self._portfolio.total_pnl,
            total_pnl_pct=self._portfolio.total_pnl_pct,
            max_drawdown_pct=max_dd_pct,
            sharpe_ratio=sharpe,
            current_streak=current_streak,
            largest_win=largest_win,
            largest_loss=largest_loss,
        )

    @staticmethod
    def _calculate_max_drawdown(equity: List[float]) -> float:
        """
        Calculate maximum drawdown in percentages from an equity curve.

        Args:
            equity: List of portfolio values over time.

        Returns:
            Maximum drawdown in percent (positive number).
        """
        if not equity:
            return 0.0

        peak = equity[0]
        max_dd = 0.0
        for value in equity:
            if value > peak:
                peak = value
            dd = (peak - value) / peak * 100.0 if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd
        return max_dd

    @staticmethod
    def _calculate_sharpe_ratio(
        returns: List[float],
        risk_free_rate: float = 0.0,
    ) -> float:
        """
        Calculate a simplified Sharpe ratio for a list of returns.

        Args:
            returns: List of per-period returns (e.g., per trade).
            risk_free_rate: Risk-free rate per period (default 0).

        Returns:
            Sharpe ratio (dimensionless).
        """
        if len(returns) < 2:
            return 0.0

        arr = np.array(returns, dtype=float)
        excess = arr - risk_free_rate
        std = excess.std(ddof=1)
        if std == 0:
            return 0.0
        return float(excess.mean() / std)

    def get_equity_curve(self) -> List[EquityPoint]:
        """
        Build an equity curve from trade history.

        Uses portfolio_value_after at each trade as equity series.

        Returns:
            List of EquityPoint objects ordered by time.
        """
        if not self.trade_history_path.exists():
            return []

        points: List[EquityPoint] = []

        with open(self.trade_history_path, "r") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                try:
                    value = float(row["portfolio_value_after"])
                    ts = row.get("timestamp_utc", "")
                except (KeyError, ValueError):
                    continue

                cumulative_pnl = value - self._portfolio.starting_cash
                points.append(
                    EquityPoint(
                        timestamp_utc=ts,
                        portfolio_value=value,
                        cash=0.0,
                        position_value=0.0,
                        cumulative_pnl=cumulative_pnl,
                    )
                )

        return points

    # ──────────────────────────────────────────────────────
    # BACKUP & RESTORE
    # ──────────────────────────────────────────────────────

    def backup_portfolio(self) -> bool:
        """
        Create a timestamped backup copy of the portfolio JSON.

        Returns:
            True if backup succeeded; False otherwise.
        """
        try:
            if not self.portfolio_path.exists():
                logger.warning(
                    "No portfolio file at %s; backup skipped.",
                    self.portfolio_path,
                )
                return False

            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            backup_path = self.backup_dir / f"portfolio_backup_{timestamp}.json"

            shutil.copy2(self.portfolio_path, backup_path)
            logger.info("Portfolio backed up to %s.", backup_path)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("Portfolio backup failed: %s", exc, exc_info=True)
            return False

    def restore_portfolio(self, backup_path: Path) -> bool:
        """
        Restore portfolio from a backup file.

        Automatically creates a backup of current portfolio before restore.

        Args:
            backup_path: Path to backup JSON file.

        Returns:
            True if restore succeeded; False otherwise.
        """
        if not backup_path.exists():
            logger.error("Backup file not found: %s", backup_path)
            return False

        try:
            self.backup_portfolio()
            shutil.copy2(backup_path, self.portfolio_path)
            self._portfolio = self._load_or_initialize()
            logger.info("Portfolio restored from %s.", backup_path)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("Portfolio restore failed: %s", exc, exc_info=True)
            return False


# ──────────────────────────────────────────────────────────
# CLI TOOL
# ──────────────────────────────────────────────────────────


def _print_portfolio_summary(trader: PaperTrader) -> None:
    """Helper to print basic portfolio summary to stdout."""
    pf = trader.get_portfolio_state()
    print("\n" + "=" * 70)
    print("ALPHA-PRIME Portfolio Manager")
    print("=" * 70)
    print(f"\nCash: ${pf.cash:,.2f}")
    print(f"Positions: {pf.position_count}")
    print(f"Total Value: ${pf.total_value:,.2f}")
    print(f"Total P&L: ${pf.total_pnl:+,.2f} ({pf.total_pnl_pct:+.2f}%)")

    if pf.positions:
        print("\nOpen Positions:")
        for ticker, pos in pf.positions.items():
            print(
                f"  {ticker}: {pos.quantity} @ ${pos.avg_entry_price:.4f} "
                f"(realized P&L=${pos.realized_pnl:+.2f})"
            )

    print("\nCommands:")
    print("  python portfolio.py metrics    # Show performance metrics")
    print("  python portfolio.py backup     # Create backup")
    print("  python portfolio.py reset      # Reset portfolio (WARNING)")
    print("=" * 70 + "\n")


def _print_metrics(trader: PaperTrader) -> None:
    """Helper to print performance metrics to stdout."""
    metrics = trader.calculate_metrics()
    print("\n" + "=" * 70)
    print("Performance Metrics")
    print("=" * 70)
    print(f"Total Trades: {metrics.total_trades}")
    print(f"Win Rate: {metrics.win_rate:.1f}%")
    print(f"Wins / Losses: {metrics.winning_trades} / {metrics.losing_trades}")
    print(f"Average Win: ${metrics.average_win:.2f}")
    print(f"Average Loss: ${metrics.average_loss:.2f}")
    print(f"Profit Factor: {metrics.profit_factor:.2f}")
    print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    print(f"Max Drawdown: {metrics.max_drawdown_pct:.2f}%")
    print(
        f"Total P&L: ${metrics.total_pnl:+,.2f} "
        f"({metrics.total_pnl_pct:+.2f}%)"
    )
    print(f"Current Streak: {metrics.current_streak}")
    print(f"Largest Win: ${metrics.largest_win:.2f}")
    print(f"Largest Loss: ${metrics.largest_loss:.2f}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    import sys

    trader = PaperTrader()

    if len(sys.argv) < 2:
        _print_portfolio_summary(trader)
        sys.exit(0)

    command = sys.argv[1].lower()

    if command == "metrics":
        _print_metrics(trader)
    elif command == "backup":
        if trader.backup_portfolio():
            print("✅ Backup created successfully.")
        else:
            print("❌ Backup failed.")
    elif command == "reset":
        confirm = input(
            "⚠️  Reset portfolio to starting cash and clear positions? (yes/no): "
        )
        if confirm.strip().lower() == "yes":
            now = datetime.now(timezone.utc).isoformat()
            starting_cash = float(getattr(settings, "starting_cash", 10000.0))
            trader._portfolio = Portfolio(
                cash=starting_cash,
                positions={},
                starting_cash=starting_cash,
                total_trades=0,
                total_commission=0.0,
                created_at_utc=now,
                last_updated_utc=now,
            )
            trader._save_portfolio(trader._portfolio)
            print("✅ Portfolio reset.")
        else:
            print("Cancelled.")
    else:
        _print_portfolio_summary(trader)
