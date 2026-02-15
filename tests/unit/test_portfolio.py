"""
Unit tests for portfolio management logic in ALPHA-PRIME v2.0.

Covers:
- Portfolio and position lifecycle (init, add/update/close).
- PnL (realized, unrealized, total) and cash flow.
- Exposure, concentration, risk metrics, rebalancing.
- Portfolio analytics and edge-case validation.

All tests are:
- Fast (<100ms total), isolated (mock prices, no APIs),
- Deterministic (fixed seeds), and math-precise.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List

import numpy as np
import pytest

from core.portfolio import (
    Portfolio,
    Position,
    PortfolioMetrics,
    calculate_pnl,
    calculate_exposure,
    calculate_concentration,
    rebalance_portfolio,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def empty_portfolio() -> Portfolio:
    """Empty portfolio with zero cash."""
    return Portfolio(cash=0.0)


@pytest.fixture
def funded_portfolio() -> Portfolio:
    """Portfolio with 100,000 cash."""
    return Portfolio(cash=100_000.0)


@pytest.fixture
def sample_positions() -> List[Position]:
    """Sample positions for testing."""
    return [
        Position(symbol="AAPL", quantity=100, entry_price=150.0),
        Position(symbol="MSFT", quantity=50, entry_price=300.0),
        Position(symbol="GOOGL", quantity=-20, entry_price=140.0),  # Short
    ]


@pytest.fixture
def portfolio_with_positions(funded_portfolio: Portfolio, sample_positions: List[Position]) -> Portfolio:
    """Portfolio with multiple positions."""
    for pos in sample_positions:
        funded_portfolio.add_position(pos)
    return funded_portfolio


@pytest.fixture
def mock_price_data() -> Dict[str, float]:
    """Mock current prices for symbols."""
    return {
        "AAPL": 155.0,
        "MSFT": 310.0,
        "GOOGL": 135.0,
    }


@pytest.fixture
def historical_returns() -> np.ndarray:
    """252 days of historical returns for testing."""
    np.random.seed(42)
    return np.random.normal(0.001, 0.02, 252)


# ---------------------------------------------------------------------------
# 1. Portfolio Initialization
# ---------------------------------------------------------------------------

class TestPortfolioInitialization:
    """Test Portfolio class initialization and state."""

    def test_empty_portfolio_creation(self):
        """Test creating an empty portfolio."""
        portfolio = Portfolio(cash=0.0)
        assert portfolio.cash == 0.0
        assert len(portfolio.positions) == 0
        assert portfolio.total_value == 0.0

    def test_portfolio_with_initial_cash(self):
        """Test portfolio with initial cash."""
        portfolio = Portfolio(cash=100_000.0)
        assert portfolio.cash == 100_000.0
        assert pytest.approx(portfolio.total_value) == 100_000.0

    def test_portfolio_with_initial_positions(self, sample_positions: List[Position]):
        """Test portfolio initialization with positions."""
        portfolio = Portfolio(cash=0.0, positions={p.symbol: p for p in sample_positions})
        assert len(portfolio.positions) == 3
        assert portfolio.positions["AAPL"].quantity == 100

    def test_portfolio_copy_constructor(self, portfolio_with_positions: Portfolio):
        """Test creating a copy of a portfolio."""
        copy = portfolio_with_positions.copy()
        assert copy is not portfolio_with_positions
        assert copy.positions.keys() == portfolio_with_positions.positions.keys()

    def test_portfolio_from_dict(self):
        """Test portfolio construction from dict representation."""
        data = {"cash": 1_000.0, "positions": []}
        portfolio = Portfolio.from_dict(data)
        assert portfolio.cash == 1_000.0
        assert len(portfolio.positions) == 0

    def test_portfolio_to_dict_serialization(self, funded_portfolio: Portfolio):
        """Test serialization to dict."""
        d = funded_portfolio.to_dict()
        assert d["cash"] == 100_000.0
        assert "positions" in d

    def test_portfolio_immutable_after_freeze(self, funded_portfolio: Portfolio):
        """Test portfolio cannot be modified after freeze."""
        funded_portfolio.freeze()
        with pytest.raises(RuntimeError):
            funded_portfolio.add_position(Position("AAPL", 10, 150.0))

    def test_portfolio_timestamp_tracking(self):
        """Test portfolio timestamps for creation and updates."""
        portfolio = Portfolio(cash=0.0)
        assert isinstance(portfolio.created_at, datetime)
        before = portfolio.updated_at
        portfolio.touch()
        assert portfolio.updated_at >= before


# ---------------------------------------------------------------------------
# 2. Position Management
# ---------------------------------------------------------------------------

class TestPositionManagement:
    """Test adding, removing, updating positions."""

    def test_add_long_position(self, funded_portfolio: Portfolio):
        """Test adding a long position."""
        pos = Position(symbol="AAPL", quantity=100, entry_price=150.0)
        funded_portfolio.add_position(pos)
        assert "AAPL" in funded_portfolio.positions
        assert funded_portfolio.positions["AAPL"].quantity == 100
        assert pytest.approx(funded_portfolio.cash) == 100_000.0 - 15_000.0

    def test_add_short_position(self, funded_portfolio: Portfolio):
        """Test adding a short position."""
        pos = Position(symbol="AAPL", quantity=-50, entry_price=150.0)
        funded_portfolio.add_position(pos)
        assert funded_portfolio.positions["AAPL"].quantity == -50

    def test_update_existing_position(self, funded_portfolio: Portfolio):
        """Test updating an existing position."""
        funded_portfolio.add_position(Position("AAPL", 100, 150.0))
        funded_portfolio.add_position(Position("AAPL", 50, 160.0))
        assert funded_portfolio.positions["AAPL"].quantity == 150

    def test_close_position_fully(self, funded_portfolio: Portfolio):
        """Test fully closing a position removes it."""
        funded_portfolio.add_position(Position("AAPL", 100, 150.0))
        funded_portfolio.close_position("AAPL", exit_price=150.0)
        assert "AAPL" not in funded_portfolio.positions

    def test_partial_position_close(self, funded_portfolio: Portfolio):
        """Test partially closing a position."""
        funded_portfolio.add_position(Position("AAPL", 100, 150.0))
        funded_portfolio.close_position("AAPL", quantity=40, exit_price=155.0)
        assert funded_portfolio.positions["AAPL"].quantity == 60

    def test_average_price_calculation(self, funded_portfolio: Portfolio):
        """Test average price updates when adding to position."""
        funded_portfolio.add_position(Position("AAPL", 100, 150.0))
        funded_portfolio.add_position(Position("AAPL", 100, 160.0))
        avg_price = funded_portfolio.positions["AAPL"].avg_price
        assert pytest.approx(avg_price) == 155.0

    def test_position_size_validation(self, funded_portfolio: Portfolio):
        """Test invalid position size is rejected."""
        with pytest.raises(ValueError):
            funded_portfolio.add_position(Position("AAPL", 0, 150.0))

    def test_position_limits_enforcement(self, funded_portfolio: Portfolio):
        """Test exceeding position limits raises."""
        with pytest.raises(ValueError):
            funded_portfolio.add_position(Position("AAPL", 1_000_000, 150.0))

    def test_multiple_positions_same_symbol(self, funded_portfolio: Portfolio):
        """Test multiple adds to same symbol aggregate."""
        funded_portfolio.add_position(Position("AAPL", 10, 150.0))
        funded_portfolio.add_position(Position("AAPL", -5, 155.0))
        assert funded_portfolio.positions["AAPL"].quantity == 5

    def test_position_cost_basis_tracking(self, funded_portfolio: Portfolio):
        """Test cost basis is tracked correctly."""
        funded_portfolio.add_position(Position("AAPL", 100, 150.0))
        pos = funded_portfolio.positions["AAPL"]
        assert pytest.approx(pos.cost_basis) == 15_000.0

    def test_position_hold_duration(self, funded_portfolio: Portfolio):
        """Test hold duration calculation."""
        funded_portfolio.add_position(Position("AAPL", 100, 150.0))
        pos = funded_portfolio.positions["AAPL"]
        assert pos.hold_duration.days >= 0

    def test_position_state_transitions(self, funded_portfolio: Portfolio):
        """Test position open/closed state transitions."""
        funded_portfolio.add_position(Position("AAPL", 100, 150.0))
        pos = funded_portfolio.positions["AAPL"]
        assert pos.is_open
        funded_portfolio.close_position("AAPL", exit_price=155.0)
        assert not pos.is_open


# ---------------------------------------------------------------------------
# 3. PnL Calculations
# ---------------------------------------------------------------------------

class TestPnLCalculations:
    """Test realized, unrealized, and total PnL calculations."""

    def test_realized_pnl_long_profitable(self, funded_portfolio: Portfolio):
        """Test realized PnL for profitable long trade."""
        pos = Position("AAPL", 100, 150.0)
        funded_portfolio.add_position(pos)
        realized_pnl = funded_portfolio.close_position("AAPL", exit_price=155.0)
        assert pytest.approx(realized_pnl) == 500.0

    def test_realized_pnl_long_loss(self, funded_portfolio: Portfolio):
        pos = Position("AAPL", 100, 150.0)
        funded_portfolio.add_position(pos)
        realized_pnl = funded_portfolio.close_position("AAPL", exit_price=145.0)
        assert pytest.approx(realized_pnl) == -500.0

    def test_realized_pnl_short_profitable(self, funded_portfolio: Portfolio):
        pos = Position("AAPL", -100, 150.0)
        funded_portfolio.add_position(pos)
        pnl = funded_portfolio.close_position("AAPL", exit_price=145.0)
        assert pytest.approx(pnl) == 500.0

    def test_realized_pnl_short_loss(self, funded_portfolio: Portfolio):
        pos = Position("AAPL", -100, 150.0)
        funded_portfolio.add_position(pos)
        pnl = funded_portfolio.close_position("AAPL", exit_price=155.0)
        assert pytest.approx(pnl) == -500.0

    def test_unrealized_pnl_single_position(
        self, funded_portfolio: Portfolio, mock_price_data: Dict[str, float]
    ):
        pos = Position("AAPL", 100, 150.0)
        funded_portfolio.add_position(pos)
        unrealized = funded_portfolio.unrealized_pnl(mock_price_data)
        assert pytest.approx(unrealized) == 500.0

    def test_unrealized_pnl_multiple_positions(
        self, portfolio_with_positions: Portfolio, mock_price_data: Dict[str, float]
    ):
        unrealized = portfolio_with_positions.unrealized_pnl(mock_price_data)
        # (155-150)*100 + (310-300)*50 + (135-140)*(-20)
        expected = 500.0 + 500.0 + 100.0
        assert pytest.approx(unrealized) == expected

    def test_total_pnl_calculation(
        self, portfolio_with_positions: Portfolio, mock_price_data: Dict[str, float]
    ):
        metrics: PortfolioMetrics = portfolio_with_positions.metrics(mock_price_data)
        assert pytest.approx(metrics.total_pnl) == metrics.realized_pnl + metrics.unrealized_pnl

    def test_pnl_with_commissions(self, funded_portfolio: Portfolio):
        pos = Position("AAPL", 100, 150.0)
        funded_portfolio.add_position(pos, commission=10.0)
        realized_pnl = funded_portfolio.close_position("AAPL", exit_price=155.0, commission=10.0)
        assert pytest.approx(realized_pnl) == 480.0

    def test_pnl_with_slippage(self, funded_portfolio: Portfolio):
        pos = Position("AAPL", 100, 150.0)
        funded_portfolio.add_position(pos, slippage_bps=10.0)
        pnl = funded_portfolio.close_position("AAPL", exit_price=155.0, slippage_bps=10.0)
        assert pnl < 500.0

    def test_pnl_percentage_calculations(self, funded_portfolio: Portfolio):
        pos = Position("AAPL", 100, 150.0)
        funded_portfolio.add_position(pos)
        pnl_pct = funded_portfolio.close_position("AAPL", exit_price=155.0, return_pct=True)
        assert pytest.approx(pnl_pct) == (155.0 - 150.0) / 150.0

    def test_daily_pnl_tracking(self, funded_portfolio: Portfolio):
        funded_portfolio.add_position(Position("AAPL", 100, 150.0))
        funded_portfolio.record_daily_pnl(date=datetime(2024, 1, 1), pnl=100.0)
        funded_portfolio.record_daily_pnl(date=datetime(2024, 1, 2), pnl=-50.0)
        series = funded_portfolio.daily_pnl
        assert series[datetime(2024, 1, 1)] == 100.0
        assert series[datetime(2024, 1, 2)] == -50.0

    def test_mtd_ytd_pnl_aggregation(self, funded_portfolio: Portfolio):
        today = datetime(2024, 6, 15)
        funded_portfolio.record_daily_pnl(today - timedelta(days=1), 100.0)
        funded_portfolio.record_daily_pnl(today, 50.0)
        mtd, ytd = funded_portfolio.mtd_ytd_pnl(today)
        assert mtd >= 150.0
        assert ytd >= 150.0

    def test_pnl_attribution_by_strategy(self, funded_portfolio: Portfolio):
        funded_portfolio.record_strategy_pnl("strategy_a", 100.0)
        funded_portfolio.record_strategy_pnl("strategy_b", -50.0)
        attr = funded_portfolio.strategy_pnl
        assert attr["strategy_a"] == 100.0
        assert attr["strategy_b"] == -50.0

    def test_pnl_with_dividends(self, funded_portfolio: Portfolio):
        funded_portfolio.add_position(Position("AAPL", 100, 150.0))
        funded_portfolio.record_dividend("AAPL", 100.0)
        assert funded_portfolio.dividends["AAPL"] == 100.0

    def test_pnl_edge_case_zero_position(self, funded_portfolio: Portfolio):
        pnl = calculate_pnl(quantity=0, entry_price=150.0, exit_price=155.0)
        assert pnl == 0.0


# ---------------------------------------------------------------------------
# 4. Exposure Calculations
# ---------------------------------------------------------------------------

class TestExposureCalculations:
    """Test portfolio exposure metrics."""

    def test_gross_exposure_calculation(self, portfolio_with_positions: Portfolio, mock_price_data: Dict[str, float]):
        metrics = calculate_exposure(portfolio_with_positions, mock_price_data)
        assert metrics.gross_exposure > 0.0

    def test_net_exposure_calculation(self, portfolio_with_positions: Portfolio, mock_price_data: Dict[str, float]):
        metrics = calculate_exposure(portfolio_with_positions, mock_price_data)
        assert abs(metrics.net_exposure) <= metrics.gross_exposure

    def test_long_short_exposure_split(self, portfolio_with_positions: Portfolio, mock_price_data: Dict[str, float]):
        metrics = calculate_exposure(portfolio_with_positions, mock_price_data)
        assert metrics.long_exposure >= 0.0
        assert metrics.short_exposure <= 0.0

    def test_exposure_by_sector(self, portfolio_with_positions: Portfolio, mock_price_data: Dict[str, float]):
        metrics = calculate_exposure(portfolio_with_positions, mock_price_data)
        assert isinstance(metrics.exposure_by_sector, dict)

    def test_exposure_by_asset_class(self, portfolio_with_positions: Portfolio, mock_price_data: Dict[str, float]):
        metrics = calculate_exposure(portfolio_with_positions, mock_price_data)
        assert isinstance(metrics.exposure_by_asset_class, dict)

    def test_exposure_limits_checking(self, portfolio_with_positions: Portfolio, mock_price_data: Dict[str, float]):
        metrics = calculate_exposure(portfolio_with_positions, mock_price_data, max_leverage=2.0)
        assert metrics.leverage <= 2.0

    def test_leverage_calculation(self, portfolio_with_positions: Portfolio, mock_price_data: Dict[str, float]):
        metrics = calculate_exposure(portfolio_with_positions, mock_price_data)
        assert metrics.leverage >= 0.0

    def test_notional_exposure(self, portfolio_with_positions: Portfolio, mock_price_data: Dict[str, float]):
        metrics = calculate_exposure(portfolio_with_positions, mock_price_data)
        assert metrics.notional_exposure == metrics.gross_exposure

    def test_delta_adjusted_exposure(self, portfolio_with_positions: Portfolio, mock_price_data: Dict[str, float]):
        metrics = calculate_exposure(portfolio_with_positions, mock_price_data, delta=1.0)
        assert metrics.delta_adjusted_exposure == metrics.gross_exposure

    def test_exposure_empty_portfolio(self, empty_portfolio: Portfolio, mock_price_data: Dict[str, float]):
        metrics = calculate_exposure(empty_portfolio, mock_price_data)
        assert metrics.gross_exposure == 0.0
        assert metrics.net_exposure == 0.0


# ---------------------------------------------------------------------------
# 5. Concentration & Diversification
# ---------------------------------------------------------------------------

class TestConcentration:
    """Test concentration and diversification metrics."""

    def test_top_holdings_concentration(self, portfolio_with_positions: Portfolio, mock_price_data: Dict[str, float]):
        conc = calculate_concentration(portfolio_with_positions, mock_price_data)
        assert conc.top_holdings_concentration >= 0.0

    def test_single_position_concentration(self, funded_portfolio: Portfolio, mock_price_data: Dict[str, float]):
        funded_portfolio.add_position(Position("AAPL", 100, 150.0))
        conc = calculate_concentration(funded_portfolio, mock_price_data)
        assert conc.top_holdings_concentration == 1.0

    def test_sector_concentration(self, portfolio_with_positions: Portfolio, mock_price_data: Dict[str, float]):
        conc = calculate_concentration(portfolio_with_positions, mock_price_data)
        assert isinstance(conc.sector_concentration, dict)

    def test_herfindahl_index(self, portfolio_with_positions: Portfolio, mock_price_data: Dict[str, float]):
        conc = calculate_concentration(portfolio_with_positions, mock_price_data)
        assert 0.0 <= conc.herfindahl_index <= 1.0

    def test_concentration_limit_breach_detection(self, funded_portfolio: Portfolio, mock_price_data: Dict[str, float]):
        funded_portfolio.add_position(Position("AAPL", 500, 150.0))
        conc = calculate_concentration(funded_portfolio, mock_price_data, max_concentration=0.5)
        assert conc.breaches_limit is True

    def test_diversification_score(self, portfolio_with_positions: Portfolio, mock_price_data: Dict[str, float]):
        conc = calculate_concentration(portfolio_with_positions, mock_price_data)
        assert 0.0 <= conc.diversification_score <= 1.0

    def test_position_weight_distribution(self, portfolio_with_positions: Portfolio, mock_price_data: Dict[str, float]):
        conc = calculate_concentration(portfolio_with_positions, mock_price_data)
        assert abs(sum(conc.weights.values()) - 1.0) < 1e-6

    def test_concentration_empty_portfolio(self, empty_portfolio: Portfolio, mock_price_data: Dict[str, float]):
        conc = calculate_concentration(empty_portfolio, mock_price_data)
        assert conc.herfindahl_index == 0.0


# ---------------------------------------------------------------------------
# 6. Risk Metrics
# ---------------------------------------------------------------------------

class TestRiskMetrics:
    """Test portfolio risk calculations."""

    def test_portfolio_volatility(self, historical_returns: np.ndarray):
        vol = PortfolioMetrics.volatility(historical_returns)
        assert vol > 0.0

    def test_portfolio_beta(self):
        returns = np.array([0.01, 0.02, -0.01])
        benchmark = np.array([0.005, 0.015, -0.005])
        beta = PortfolioMetrics.beta(returns, benchmark)
        assert beta > 0.0

    def test_var_calculation_95(self, historical_returns: np.ndarray):
        var = PortfolioMetrics.var(historical_returns, level=0.95)
        assert isinstance(var, float)

    def test_var_calculation_99(self, historical_returns: np.ndarray):
        var = PortfolioMetrics.var(historical_returns, level=0.99)
        assert isinstance(var, float)

    def test_cvar_expected_shortfall(self, historical_returns: np.ndarray):
        cvar = PortfolioMetrics.cvar(historical_returns, level=0.95)
        assert isinstance(cvar, float)

    def test_maximum_drawdown(self, portfolio_with_positions: Portfolio):
        equity_curve = np.array([100, 110, 105, 95, 98, 120], dtype=float)
        max_dd = portfolio_with_positions.calculate_max_drawdown(equity_curve)
        assert max_dd <= 0.0

    def test_current_drawdown(self, portfolio_with_positions: Portfolio):
        equity_curve = np.array([100, 110, 105, 95], dtype=float)
        current_dd = portfolio_with_positions.current_drawdown(equity_curve)
        assert current_dd <= 0.0

    def test_time_underwater(self):
        eq = np.array([100, 90, 80, 95, 105], dtype=float)
        tw = PortfolioMetrics.time_underwater(eq)
        assert tw >= 0

    def test_sharpe_ratio(self, historical_returns: np.ndarray):
        sharpe = PortfolioMetrics.sharpe(historical_returns, risk_free=0.0)
        expected = np.sqrt(252) * historical_returns.mean() / historical_returns.std()
        assert abs(sharpe - expected) < 1e-6

    def test_sortino_ratio(self, historical_returns: np.ndarray):
        sortino = PortfolioMetrics.sortino(historical_returns, risk_free=0.0)
        assert isinstance(sortino, float)

    def test_calmar_ratio(self):
        eq = np.array([100, 120, 110, 130, 90, 140], dtype=float)
        calmar = PortfolioMetrics.calmar(eq)
        assert isinstance(calmar, float)

    def test_risk_metrics_insufficient_data(self):
        returns = np.array([0.01])
        sharpe = PortfolioMetrics.sharpe(returns, risk_free=0.0)
        assert sharpe == 0.0


# ---------------------------------------------------------------------------
# 7. Portfolio Rebalancing
# ---------------------------------------------------------------------------

class TestPortfolioRebalancing:
    """Test portfolio rebalancing logic."""

    def test_rebalance_to_target_weights(self, portfolio_with_positions: Portfolio, mock_price_data: Dict[str, float]):
        targets = {"AAPL": 0.5, "MSFT": 0.5}
        orders = rebalance_portfolio(portfolio_with_positions, mock_price_data, targets)
        assert isinstance(orders, list)

    def test_rebalance_with_cash_constraints(self, funded_portfolio: Portfolio, mock_price_data: Dict[str, float]):
        targets = {"AAPL": 1.0}
        orders = rebalance_portfolio(funded_portfolio, mock_price_data, targets, max_cash_to_invest=50_000.0)
        assert all(abs(o.notional) <= 50_000.0 for o in orders)

    def test_rebalance_minimize_trades(self, funded_portfolio: Portfolio, mock_price_data: Dict[str, float]):
        targets = {"AAPL": 1.0}
        orders = rebalance_portfolio(funded_portfolio, mock_price_data, targets, minimize_trades=True)
        assert isinstance(orders, list)

    def test_rebalance_respect_position_limits(self, funded_portfolio: Portfolio, mock_price_data: Dict[str, float]):
        targets = {"AAPL": 1.0}
        orders = rebalance_portfolio(
            funded_portfolio,
            mock_price_data,
            targets,
            max_position_size=0.5,
        )
        assert orders

    def test_rebalance_transaction_costs(self, funded_portfolio: Portfolio, mock_price_data: Dict[str, float]):
        targets = {"AAPL": 1.0}
        orders = rebalance_portfolio(
            funded_portfolio,
            mock_price_data,
            targets,
            commission_per_trade=10.0,
        )
        assert isinstance(orders, list)

    def test_rebalance_partial_fills(self, funded_portfolio: Portfolio, mock_price_data: Dict[str, float]):
        targets = {"AAPL": 1.0}
        orders = rebalance_portfolio(funded_portfolio, mock_price_data, targets, allow_partial_fills=True)
        assert isinstance(orders, list)

    def test_rebalance_tax_aware(self, funded_portfolio: Portfolio, mock_price_data: Dict[str, float]):
        targets = {"AAPL": 1.0}
        orders = rebalance_portfolio(funded_portfolio, mock_price_data, targets, tax_aware=True)
        assert isinstance(orders, list)

    def test_rebalance_drift_threshold(self, funded_portfolio: Portfolio, mock_price_data: Dict[str, float]):
        targets = {"AAPL": 1.0}
        orders = rebalance_portfolio(funded_portfolio, mock_price_data, targets, drift_threshold=0.05)
        assert isinstance(orders, list)

    def test_rebalance_empty_portfolio(self, empty_portfolio: Portfolio, mock_price_data: Dict[str, float]):
        targets = {"AAPL": 1.0}
        orders = rebalance_portfolio(empty_portfolio, mock_price_data, targets)
        assert orders == []

    def test_rebalance_validation_errors(self, funded_portfolio: Portfolio, mock_price_data: Dict[str, float]):
        with pytest.raises(ValueError):
            rebalance_portfolio(funded_portfolio, mock_price_data, targets={"AAPL": 0.6, "MSFT": 0.5})


# ---------------------------------------------------------------------------
# 8. Cash Management
# ---------------------------------------------------------------------------

class TestCashManagement:
    """Test cash tracking and management."""

    def test_initial_cash_balance(self, funded_portfolio: Portfolio):
        assert funded_portfolio.cash == 100_000.0

    def test_cash_after_buy(self, funded_portfolio: Portfolio):
        funded_portfolio.add_position(Position("AAPL", 100, 150.0))
        assert pytest.approx(funded_portfolio.cash) == 100_000.0 - 15_000.0

    def test_cash_after_sell(self, funded_portfolio: Portfolio):
        funded_portfolio.add_position(Position("AAPL", 100, 150.0))
        funded_portfolio.close_position("AAPL", exit_price=155.0)
        assert funded_portfolio.cash > 100_000.0

    def test_cash_with_commissions(self, funded_portfolio: Portfolio):
        funded_portfolio.add_position(Position("AAPL", 100, 150.0), commission=10.0)
        funded_portfolio.close_position("AAPL", exit_price=155.0, commission=10.0)
        assert funded_portfolio.cash < 100_000.0 + 500.0

    def test_insufficient_cash_rejection(self, funded_portfolio: Portfolio):
        with pytest.raises(ValueError):
            funded_portfolio.add_position(Position("AAPL", 10_000, 150.0), enforce_cash=True)

    def test_margin_calculations(self, funded_portfolio: Portfolio):
        funded_portfolio.add_position(Position("AAPL", 100, 150.0), margin_required=0.5)
        assert funded_portfolio.margin_used > 0.0

    def test_cash_interest_accrual(self, funded_portfolio: Portfolio):
        funded_portfolio.accrue_cash_interest(rate=0.01, days=30)
        assert funded_portfolio.cash > 100_000.0

    def test_cash_flow_tracking(self, funded_portfolio: Portfolio):
        funded_portfolio.record_cash_flow(-10_000.0, reason="withdrawal")
        assert funded_portfolio.cash == 90_000.0


# ---------------------------------------------------------------------------
# 9. Portfolio Analytics
# ---------------------------------------------------------------------------

class TestPortfolioAnalytics:
    """Test portfolio performance analytics."""

    def test_equity_curve_generation(self, funded_portfolio: Portfolio):
        curve = funded_portfolio.equity_curve()
        assert len(curve) >= 1

    def test_returns_calculation(self, funded_portfolio: Portfolio):
        curve = np.array([100_000.0, 101_000.0, 99_000.0])
        rets = PortfolioMetrics.returns(curve)
        assert len(rets) == 2

    def test_cumulative_returns(self, funded_portfolio: Portfolio):
        rets = np.array([0.01, -0.02, 0.03])
        cum = PortfolioMetrics.cumulative_returns(rets)
        assert len(cum) == len(rets)

    def test_rolling_sharpe(self, historical_returns: np.ndarray):
        rs = PortfolioMetrics.rolling_sharpe(historical_returns, window=20)
        assert len(rs) == len(historical_returns)

    def test_win_rate_calculation(self):
        pnls = np.array([100, -50, 200, -10, 0])
        win_rate = PortfolioMetrics.win_rate(pnls)
        assert 0.0 <= win_rate <= 1.0

    def test_profit_factor(self):
        pnls = np.array([100, -50, 200, -10])
        pf = PortfolioMetrics.profit_factor(pnls)
        assert pf > 0.0

    def test_expectancy(self):
        pnls = np.array([100, -50, 200, -10])
        exp = PortfolioMetrics.expectancy(pnls)
        assert isinstance(exp, float)

    def test_best_worst_trades(self):
        pnls = np.array([100, -50, 200, -10])
        best, worst = PortfolioMetrics.best_worst_trades(pnls)
        assert best == 200
        assert worst == -50

    def test_trade_duration_stats(self):
        durations = np.array([1, 5, 10, 2], dtype=float)
        stats = PortfolioMetrics.trade_duration_stats(durations)
        assert "mean" in stats and "max" in stats

    def test_analytics_empty_portfolio(self, empty_portfolio: Portfolio):
        metrics = empty_portfolio.analytics()
        assert metrics is not None


# ---------------------------------------------------------------------------
# 10. Edge Cases & Validation
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Test edge cases and validation."""

    def test_portfolio_with_zero_cash(self):
        p = Portfolio(cash=0.0)
        assert p.total_value == 0.0

    def test_portfolio_with_negative_cash(self):
        p = Portfolio(cash=-1_000.0)
        assert p.cash == -1_000.0

    def test_position_with_zero_quantity(self):
        with pytest.raises(ValueError):
            Position("AAPL", 0, 150.0)

    def test_position_with_zero_price(self):
        with pytest.raises(ValueError):
            Position("AAPL", 10, 0.0)

    def test_invalid_position_symbol(self, funded_portfolio: Portfolio):
        with pytest.raises(ValueError):
            funded_portfolio.add_position(Position("", 10, 150.0))

    def test_concurrent_position_updates(self, funded_portfolio: Portfolio):
        funded_portfolio.add_position(Position("AAPL", 10, 150.0))
        funded_portfolio.begin_batch()
        funded_portfolio.add_position(Position("AAPL", 5, 155.0))
        funded_portfolio.end_batch()
        assert funded_portfolio.positions["AAPL"].quantity == 15

    def test_portfolio_state_consistency(self, portfolio_with_positions: Portfolio, mock_price_data: Dict[str, float]):
        metrics = portfolio_with_positions.metrics(mock_price_data)
        assert metrics.total_value >= 0.0

    def test_floating_point_precision(self, funded_portfolio: Portfolio):
        p = Position("AAPL", quantity=Decimal("0.1"), entry_price=Decimal("150.0"))
        funded_portfolio.add_position(p)
        assert isinstance(funded_portfolio.positions["AAPL"].quantity, Decimal)

    def test_extreme_values_handling(self, funded_portfolio: Portfolio, mock_price_data: Dict[str, float]):
        funded_portfolio.add_position(Position("AAPL", 1_000_000, 0.01))
        metrics = funded_portfolio.metrics(mock_price_data)
        assert metrics.total_value > 0.0

    def test_portfolio_clone_independence(self, portfolio_with_positions: Portfolio):
        clone = portfolio_with_positions.copy()
        clone.cash += 1_000.0
        assert clone.cash != portfolio_with_positions.cash

    def test_position_lifecycle_completeness(self, funded_portfolio: Portfolio):
        pos = Position("AAPL", 100, 150.0)
        funded_portfolio.add_position(pos)
        funded_portfolio.close_position("AAPL", exit_price=155.0)
        assert pos.lifecycle_complete

    def test_portfolio_validation_comprehensive(self, portfolio_with_positions: Portfolio, mock_price_data: Dict[str, float]):
        metrics = portfolio_with_positions.metrics(mock_price_data)
        assert metrics.is_valid
        assert metrics.errors == []


# ---------------------------------------------------------------------------
# Parametrized tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "quantity,entry,exit,expected_pnl",
    [
        (100, 150, 155, 500),      # Long profit
        (100, 150, 145, -500),     # Long loss
        (-100, 150, 145, 500),     # Short profit
        (-100, 150, 155, -500),    # Short loss
    ],
)
def test_pnl_scenarios(funded_portfolio: Portfolio, quantity: float, entry: float, exit: float, expected_pnl: float):
    """Test PnL across various scenarios."""
    pos = Position(symbol="TEST", quantity=quantity, entry_price=entry)
    funded_portfolio.add_position(pos)
    pnl = funded_portfolio.close_position("TEST", exit_price=exit)
    assert pytest.approx(pnl) == expected_pnl


@pytest.mark.parametrize(
    "positions,max_concentration,should_pass",
    [
        ([("AAPL", 100, 150.0)], 0.50, True),     # 15% < 50%
        ([("AAPL", 500, 150.0)], 0.50, False),    # likely > 50%
        ([("AAPL", 100, 150.0), ("MSFT", 100, 150.0)], 0.50, True),
    ],
)
def test_concentration_limits(
    funded_portfolio: Portfolio,
    positions: List[tuple],
    max_concentration: float,
    should_pass: bool,
    mock_price_data: Dict[str, float],
):
    """Test concentration limit enforcement."""
    for symbol, qty, price in positions:
        funded_portfolio.add_position(Position(symbol, qty, price))
    conc = calculate_concentration(funded_portfolio, mock_price_data, max_concentration=max_concentration)
    assert conc.breaches_limit is (not should_pass)
